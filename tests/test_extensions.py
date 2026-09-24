"""Contract tests use synthetic fragments, not new source/device physics."""
from importlib import resources
from pathlib import Path
import shutil

import numpy as np
import pytest
from pyRadPlan import generate_stf

from minibeam import TOPASPhotonEngine
from minibeam.geometry.devices import GeometryInclude, validate_device_fragment
from minibeam.sources.beamlet import PointBeamletSource
from minibeam.topas.contracts import ParameterFragment, SourceDescription
from minibeam.topas.manifest import implementation_hashes, load_manifest, sha256


class AlternateSource:
    """Parameter-only test double: explicitly no physics or phase-space reader."""
    def __init__(self, assets=(), **description):
        self.assets = assets
        self.description = description

    def describe(self, context):
        return SourceDescription("world", ((-2000, -20, -30), (-1900, 20, 30)),
                                 source_name="Alternate", **self.description)

    def render(self, context):
        return ParameterFragment('s:So/Alternate/Type = "TestOnly"', self.assets,
                                 {"model": "test_only"})


class TestDevice:
    __test__ = False

    def __init__(self, name, position=(0, 0, -100), assets=()):
        self.name, self.position, self.assets = name, position, assets

    def render(self, context):
        text = [f's:Ge/{self.name}/Type = "Group"',
                f's:Ge/{self.name}/Parent = "BeamFrame"']
        text += [f'd:Ge/{self.name}/Trans{axis} = {value} mm'
                 for axis, value in zip("XYZ", self.position)]
        return ParameterFragment("\n".join(text), self.assets,
                                 {"position_in_beam_mm": self.position})


def test_energy_composition(case):
    ct, _, _, stf, engine = case
    from minibeam.sources.energy import MonoenergeticEnergy
    class Energy(MonoenergeticEnergy):
        def render(self, context):
            return ParameterFragment('d:So/Beam/BeamEnergy = 2 MeV', record={"energy_mev": 2})
    engine.source_model = PointBeamletSource(energy_model=Energy())
    definition = engine._definition(ct, stf)[0]
    assert definition['jobs'][0]['source']['energy_mev'] == 2
    assert 'PhaseSpaceFileName' in definition['sources'][0]
    assert 'BeamEnergySpread' not in definition['sources'][0]


def test_source_assets_relocation_and_extent(case, tmp_path, monkeypatch):
    ct, _, cst, stf, engine = case
    paths = [tmp_path/'sample.phsp', tmp_path/'sample.header']
    for path in paths:
        path.write_bytes(b'test asset\n' * 1000)
    engine.source_model = AlternateSource(tuple(('inputs/'+p.name, p) for p in paths))
    # Large source assets must never pass through read_bytes().
    original = Path.read_bytes
    def guarded(path):
        assert path not in paths
        return original(path)
    monkeypatch.setattr(Path, 'read_bytes', guarded)
    root = engine.prepare_jobs(ct, cst, stf)
    manifest = load_manifest(root)
    text = (root/manifest['jobs'][0]['parameter_file']).read_text()
    assert 'BeamEnergy' not in text and 'BeamPosition' not in text
    assert 'So/Alternate/NumberOfHistoriesInRun = 10' in text
    assert 'World/HLX = 2500 mm' in (root/'inputs/common.txt').read_text()
    for path in paths:
        assert sha256(root/'inputs'/path.name) == sha256(path)
    moved = tmp_path/'relocated'
    shutil.move(root, moved)
    assert load_manifest(moved)['bundle_id'] == manifest['bundle_id']
    paths[0].write_bytes(b'changed')
    with pytest.raises(ValueError, match='different inputs'):
        engine.prepare_jobs(ct, cst, stf, bundle_dir=moved)


@pytest.mark.parametrize('description, error', [
    ({'job_modes': ('full_field',)}, 'per-beamlet'),
    ({'normalization': 'recorded_particle'}, 'normalization'),
    ({'independent_histories': False}, 'dependent histories'),
])
def test_unsupported_source_capabilities(case, description, error):
    ct, _, cst, stf, engine = case
    engine.source_model = AlternateSource(**description)
    with pytest.raises(ValueError, match=error):
        engine.prepare_jobs(ct, cst, stf)
    assert not Path(engine.bundle_dir).exists()


def test_devices_fixed_across_rays(case):
    ct, plan, cst, _, engine = case
    plan.prop_stf.update(generator='photonIMRT', bixel_width=3.)
    stf = generate_stf(ct, cst, plan)
    engine.devices = (TestDevice('Upstream', (1, 2, -200)), TestDevice('Downstream', (3, 4, -100)))
    definition = engine._definition(ct, stf)[0]
    first_beam = [(job, text) for job, text in zip(definition['jobs'], definition['sources']) if job['beam_index'] == 1]
    fixed = [text.split('s:Ge/Source/Type')[0] for _, text in first_beam]
    assert len(first_beam) > 1 and len(set(fixed)) == 1
    directions = [job['source']['local_to_world'] for job, _ in first_beam]
    np.testing.assert_allclose(directions[0], directions[-1])
    assert first_beam[0][0]['source']['selections'] != first_beam[-1][0]['source']['selections']
    assert fixed[0].index('Upstream/Type') < fixed[0].index('Downstream/Type')
    assert [j['bixel_index'] for j in definition['jobs']] == list(range(1, stf.total_number_of_bixels+1))
    for job in definition['jobs']:
        beam = stf.beams[job['beam_index']-1]
        assert job['gantry_angle_deg'] == beam.gantry_angle
        assert job['couch_angle_deg'] == beam.couch_angle


def test_device_conflicts(case, tmp_path):
    ct, plan, cst, stf, engine = case
    with pytest.raises(ValueError, match='not both'):
        TOPASPhotonEngine(plan, geometry=GeometryInclude(''), devices=(TestDevice('A'),))
    engine.devices = (TestDevice('Same'), TestDevice('Same'))
    with pytest.raises(ValueError, match='unique'):
        engine.prepare_jobs(ct, cst, stf)
    asset = tmp_path/'asset'; asset.write_text('x')
    engine.devices = (TestDevice('A', assets=(('inputs/data', asset),)),)
    engine.source_model = AlternateSource((('inputs/data', asset),))
    with pytest.raises(ValueError, match='Conflicting asset'):
        engine.prepare_jobs(ct, cst, stf)
    engine.devices = ()
    engine.source_model = AlternateSource((('inputs/common.txt', asset),))
    with pytest.raises(ValueError, match='unique flat paths'):
        engine.prepare_jobs(ct, cst, stf)


def test_recursive_fingerprints_and_resources(tmp_path):
    (tmp_path/'nested').mkdir()
    (tmp_path/'nested/source.py').write_text('first')
    before = implementation_hashes(tmp_path)
    (tmp_path/'nested/source.py').write_text('second')
    assert before['nested/source.py'] != implementation_hashes(tmp_path)['nested/source.py']
    assert 'sources/beamlet.py' in implementation_hashes()
    material = resources.files('minibeam.materials').joinpath('data/HUtoMaterialSchneider.txt')
    assert 'Schneider' in material.read_text()


@pytest.mark.parametrize("position", [(0, 0), (0, 0, float("nan"))])
def test_device_position_required(position):
    with pytest.raises(ValueError, match="explicit finite XYZ"):
        validate_device_fragment("A", TestDevice("A", position).render(None))


def test_device_asset_invalidates_bundle(case, tmp_path):
    ct, _, cst, stf, engine = case
    asset = tmp_path/'device.txt'
    asset.write_text('original')
    engine.devices = (TestDevice('A', assets=(('inputs/device.txt', asset),)), TestDevice('B'))
    root = engine.prepare_jobs(ct, cst, stf)
    assert sha256(root/'inputs/device.txt') == sha256(asset)
    assert [d['name'] for d in load_manifest(root)['jobs'][0]['devices']] == ['A', 'B']
    asset.write_text('modified')
    with pytest.raises(ValueError, match='different inputs'):
        engine.prepare_jobs(ct, cst, stf)
