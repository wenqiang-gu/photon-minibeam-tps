"""Outside-field source roulette and its original-history normalization."""
import json
import os
from time import perf_counter
from unittest.mock import Mock

import numpy as np
import pytest

from conftest import physical_replay_records
from minibeam import TOPASPhotonEngine
from minibeam.geometry.spatial import beam_basis
from minibeam.sources import iaea
from minibeam.sources.bixels import rectangle_bounds
from minibeam.sources.phase_space import PhaseSpaceBeamletSource, _roulette_survivors, _roulette_threshold
from minibeam.topas.manifest import load_manifest, sha256
from minibeam.workflow.artifacts import save_planning_snapshot
from minibeam.workflow.collection import collect_bundle
from projects import add_batches
from test_combined import two_squares
from test_phase_space import make_iaea, context
from test_results import fake_results
from test_topas import execute_jobs


def prepare(case, bases, mode, probability=1.0, seed=12345, histories=100):
    source = PhaseSpaceBeamletSource(file_bases=bases, selection=mode,
                                    outside_field_survival_probability=probability)
    source.prepare(case[3], histories, execution='combined', seed=seed)
    fragment = source.render(context(case))
    return source, fragment, physical_replay_records(fragment.assets[1][1], iaea.TOPAS_DTYPE)


def source_parts(tmp_path):
    bases = []
    for part in range(2):
        records = []
        for history in range(1, 25):
            energy = 1 + (part*24+history)/100
            # Half have no in-field particle. The lower square edge is included
            # and the upper edge excluded. Test all three supported species.
            weight = 0.125 if history % 3 else 0.375
            if history % 2:
                records.append((1, -energy, -.25, 0, 0, 0, weight, 2, 0))
                start = 0
            else:
                start = 2
            records.append((2, -energy if start else energy, .25, 0, 0, 0, weight, start, 0))
            records.append((3, energy, -.4, 0, 0, 0, weight, 0, 0))
        records += [(-1, -2, 0, 0, 0, 0, 1, 1, 0), (1, 2, 0, 0, 1, 0, 1, 0, 0)]
        bases.append(make_iaea(tmp_path/f'part{part}', records, original=50))
    return bases


@pytest.mark.parametrize('mode', ['all_forward', 'outside_field'])
def test_probability_one_matches_existing_replay(case, tmp_path, mode):
    bases = source_parts(tmp_path)
    source = PhaseSpaceBeamletSource(file_bases=bases, selection=mode)
    source.prepare(case[3], 100, execution='combined')
    explicit, fragment, _ = prepare(case, bases, mode)
    previous = source.render(context(case))
    assert previous.record == fragment.record
    assert len(explicit.selections) == 1
    assert [item[0] for item in previous.assets] == [item[0] for item in fragment.assets]
    for (_, a), (_, b) in zip(previous.assets, fragment.assets):
        assert a.read_bytes() == b.read_bytes()


def test_modes_share_survivors_keep_histories_and_are_chunk_invariant(case, tmp_path, monkeypatch):
    bases = source_parts(tmp_path)
    _, _, baseline = prepare(case, bases, 'all_forward')
    _, _, field = prepare(case, bases, 'field')
    source, fragment, combined = prepare(case, bases, 'all_forward', .5)
    _, outside_fragment, outside = prepare(case, bases, 'outside_field', .5)
    assert len(source.selections) == 1
    assert len(source.render(context(case, 1)).assets) == 2
    assert source.render(context(case, 1)).assets == fragment.assets
    properties = [name for name in combined.dtype.names if name != 'new_history']
    assert np.array_equal(combined[combined['pdg'] == 22][properties], field[properties])
    assert np.array_equal(combined[combined['pdg'] != 22][properties], outside[properties])
    assert len(outside) > 0 and set(outside['pdg']) == {11, -11}
    assert len(combined) < len(baseline)
    original_histories = np.cumsum(baseline['new_history'])
    by_particle = {(row['pdg'], row['energy']): (row, history)
                   for row, history in zip(baseline, original_histories)}
    for output in (combined, outside):
        selected_history = []
        for row in output:
            original, history = by_particle[row['pdg'], row['energy']]
            for key in ('x', 'y', 'z', 'u', 'v', 'negative_z'):
                assert row[key] == original[key]
            assert row['weight'] == original['weight'] * (1 if row['pdg'] == 22 else 2)
            selected_history.append(history)
        np.testing.assert_array_equal(output['new_history'], np.diff(selected_history, prepend=0) != 0)
    assert fragment.record['empty_histories'] > 100 - np.count_nonzero(baseline['new_history'])
    assert fragment.record['excluded_backward_or_tangent_particles'] == 4
    assert fragment.record['represented_original_histories'] == 100
    stats = fragment.record['outside_field_roulette']
    assert stats == outside_fragment.record['outside_field_roulette']
    assert stats['eligible_particles'] == 96
    assert stats['discarded_particles'] == 96-len(outside)
    assert stats['retained_particles'] == len(outside)
    assert stats['survivor_weight_multiplier'] == 2.0
    assert stats['retained_reweighted_weight_sum'] == outside['weight'].sum()
    assert outside_fragment.record['rejected_inside_field_particles'] == len(field)
    assert outside_fragment.record['retained_outside_field_particles'] == len(outside)
    _, _, different_seed = prepare(case, bases, 'all_forward', .5, seed=29)
    assert not np.array_equal(combined, different_seed)

    read_chunks = iaea.validated_chunks
    monkeypatch.setattr(iaea, 'validated_chunks', lambda info, audit: read_chunks(info, audit, chunk_size=3))
    _, _, chunked = prepare(case, bases, 'all_forward', .5)
    assert np.array_equal(combined, chunked)


def test_irregular_unions_have_distinct_replay_assets(case, tmp_path):
    stf = two_squares(case)
    beam = stf.beams[1]
    beam.rays[1].ray_pos += 5 * beam_basis(beam)[:,0]
    base = make_iaea(tmp_path/'gap', [(1, -1, .25, 0, 0, 0, 1, 1, 0)]*100, original=100)
    source = PhaseSpaceBeamletSource(base, selection='all_forward', outside_field_survival_probability=.5)
    source.prepare(stf, 100, execution='combined')
    assert len(source.selections) == 2
    states = list(source.selections.values())
    assert states[0]['rectangle'] == (-5., 5., -2.5, 2.5)
    assert states[1]['rectangle'] is None
    assert states[0]['particles'] == 100
    assert 0 < states[1]['particles'] < 100
    assert states[1]['record']['outside_field_roulette']['eligible_particles'] == 100


def test_rectangle_requires_every_cell_and_exact_boundaries():
    squares = [dict(center_xy_mm=[x, y], width_mm=5) for x in (-2.5, 2.5) for y in (-2.5, 2.5)]
    assert rectangle_bounds(squares) == (-5, 5, -5, 5)
    assert rectangle_bounds(squares[:-1]) is None
    squares[-1]['center_xy_mm'][0] += 1e-8
    assert rectangle_bounds(squares) is None


def test_roulette_can_represent_probabilities_smaller_than_one_random_word():
    threshold = _roulette_threshold(2.**-65)
    assert threshold == (0, 2**63)
    generator = Mock()
    generator.random_raw.return_value = np.array([[0, 0], [0, 2**63], [1, 0]], dtype=np.uint64)
    np.testing.assert_array_equal(_roulette_survivors(generator, 3, threshold), [True, False, False])
    generator.random_raw.assert_called_once_with((3, 2))


@pytest.mark.parametrize('bad', [0, -1, 1.01, True, np.bool_(False), np.nan, np.inf, '0.5', None])
def test_invalid_probability_rejected_before_reading_files(bad):
    with pytest.raises(ValueError, match='SURVIVAL_PROBABILITY'):
        PhaseSpaceBeamletSource('/missing', selection='all_forward', outside_field_survival_probability=bad)
    active = PhaseSpaceBeamletSource('/missing', selection='outside_field', outside_field_survival_probability=.5)
    with pytest.raises(ValueError, match='requires combined'):
        active.prepare(None, 2)


def test_empty_outside_selection_and_weight_overflow(case, tmp_path):
    center = make_iaea(tmp_path/'center', [(1, -1, 0, 0, 0, 0, 1, 1, 0)])
    with pytest.raises(ValueError, match='no survivors'):
        prepare(case, [center], 'outside_field', .5, histories=12)
    maximum = float(np.finfo(np.float32).max)
    outside = make_iaea(tmp_path/'heavy', [(1, -1, 1, 0, 0, 0, maximum, 1, 0)]*100, original=100)
    with pytest.raises(ValueError, match='float32'):
        prepare(case, [outside], 'outside_field', .5)


@pytest.mark.parametrize('workflow', ['patient', 'water'])
def test_root_settings_and_inactive_modes(workflow, monkeypatch, tmp_path):
    import patient_workflow
    import water_phantom
    from minibeam.workflow.planning import topas_settings

    module = patient_workflow if workflow == 'patient' else water_phantom
    monkeypatch.setattr(module, 'SOURCE_TYPE', 'phase_space')
    monkeypatch.setattr(module, 'PHASE_SPACE_SELECTION', 'all_forward')
    monkeypatch.setattr(module, 'PHASE_SPACE_OUTSIDE_FIELD_SURVIVAL_PROBABILITY', .25)
    if workflow == 'patient':
        monkeypatch.setattr(module, 'BEAMLET_EXECUTION', 'combined')
        plan = module.patient.configure_plan(module.planning_settings(), project_dir=tmp_path)
    else:
        plan = module.water.build_planning(module.planning_settings(), tmp_path)[3]
    assert plan.prop_dose_calc['source_config']['outside_field_survival_probability'] == .25
    for source_type in ('phase_space', 'point'):
        settings = topas_settings(project_dir=tmp_path, source_type=source_type, phase_space_file_bases=[],
                                  histories=10, num_threads=1, execution='separate', water=True,
                                  enable_opengl=False, dose_spacing_mm=None, phase_space_selection='field',
                                  outside_field_survival_probability='ignored')
        assert 'outside_field_survival_probability' not in settings['source_config']


@pytest.mark.parametrize('source_config', [dict(type='point'),
    dict(type='phase_space', file_base='/not-read', selection='field')])
def test_inactive_engine_option_is_ignored(case, source_config):
    plan = case[1]
    baseline = TOPASPhotonEngine(plan, source_config=source_config)
    explicit = TOPASPhotonEngine(
        plan, source_config=dict(source_config, outside_field_survival_probability='ignored'))
    assert explicit._plan_info['source_configuration'] == baseline._plan_info['source_configuration']


def test_manifest_batches_and_saved_collection(case, tmp_path):
    ct, plan, cst, stf, _ = case
    bases = source_parts(tmp_path)
    config = dict(type='phase_space', file_bases=[str(base) for base in bases], selection='all_forward')
    engine = TOPASPhotonEngine(plan, water=True, histories=100, beamlet_execution='combined',
                               source_config=dict(config, outside_field_survival_probability=.5))
    root = engine.prepare_jobs(ct, cst, stf)
    for options in (dict(config, outside_field_survival_probability=.75),
                    dict(config, outside_field_survival_probability=.5)):
        other = TOPASPhotonEngine(plan, water=True, histories=100, beamlet_execution='combined',
                                  source_config=options, seed=42 if options['outside_field_survival_probability']==.5 else 12345)
        with pytest.raises(ValueError, match='different inputs/settings'):
            other.prepare_jobs(ct, cst, stf)

    save_planning_snapshot(ct, cst, plan, stf, root, {})
    manifest = fake_results(root)
    replay_hashes = {p: sha256(p) for p in (root/'inputs').glob('phase_*.phsp')}
    for base in bases:
        for extension in ('.phsp', '.header'):
            base.with_suffix(extension).unlink()
    add_batches.add(root, 1)
    assert {p: sha256(p) for p in (root/'inputs').glob('phase_*.phsp')} == replay_hashes
    appended = add_batches.inventory(root, manifest)[1]
    for saved, job in zip(manifest['jobs'], appended['jobs']):
        assert job['source'] == saved['source']
        assert job['normalization_histories'] == 100
        assert 'PhaseSpaceFileName' in (root/job['parameter_file']).read_text()
    collected = engine.collect_results(root)
    np.testing.assert_allclose(collected.physical_dose.flat[0].toarray()[0], [1e-10, 2e-10])
    collect_bundle(root, 'collect')
    collect_bundle(root, 'forward')
    for stage in ('collect', 'forward'):
        meta = json.loads((root/'derived'/f'{stage}_metadata.json').read_text())
        assert [record['survival_probability'] for record in meta['outside_field_roulette']] == [.5, .5]
    report = (root/'derived/indexing.md').read_text()
    assert 'Outside-field roulette' in report and 'uncertainty is conditional' in report


def test_expected_survival_and_weighted_fluence(case, tmp_path):
    count = 300
    weights = np.tile([.125, .25, .5], count//3)
    base = make_iaea(tmp_path/'weighted', [(1, -1, 1, 0, 0, 0, w, 1, 0) for w in weights], original=count)
    probability = .25
    samples = []
    for seed in range(48):
        _, _, rows = prepare(case, [base], 'outside_field', probability, seed=seed, histories=count)
        samples.append([len(rows), rows['weight'].sum()])
    expected = [count*probability, weights.sum()]
    variances = [count*probability*(1-probability), (weights**2).sum()*(1-probability)/probability]
    standard_error = np.sqrt(np.array(variances)/len(samples))
    assert np.all(np.abs(np.mean(samples, axis=0)-expected) < 4*standard_error)


@pytest.mark.topas
def test_roulette_water_benchmark(case, tmp_path):
    executable = os.environ.get('PHOTON_TPS_TOPAS')
    if not executable:
        pytest.skip('Set PHOTON_TPS_TOPAS for real transport')
    ct, plan, cst, stf, _ = case
    stf = stf.model_copy(deep=True)
    stf.beams = stf.beams[:1]
    beam = stf.beams[0]
    beam.bixel_width = 20
    beam.iso_center = np.zeros(3)
    rng = np.random.default_rng(5)
    count = 24000
    positions = rng.uniform([-2.7, -.8], [2.7, .8], size=(count, 2))
    base = make_iaea(tmp_path/'source',
                     [(1, -2, x, y, 0, 0, .05, 1, 0) for x, y in positions], original=count)
    samples = {}
    for p in (1., .5, .25):
        samples[p] = []
        for seed in (510, 511, 512, 513, 514, 515):
            engine = TOPASPhotonEngine(plan, water=True, histories=count, seed=seed, beamlet_execution='combined',
                                       source_config=dict(type='phase_space', file_base=str(base), selection='all_forward',
                                                          outside_field_survival_probability=p))
            start = perf_counter()
            root = engine.prepare_jobs(ct, cst, stf, bundle_dir=tmp_path/f'p{p}-seed{seed}')
            prepare_seconds = perf_counter()-start
            start = perf_counter()
            execute_jobs(root, executable)
            transport_seconds = perf_counter()-start
            manifest = load_manifest(root)
            assert manifest['jobs'][0]['normalization_histories'] == count
            dose = engine.collect_results(root).physical_dose.flat[0].toarray().reshape(12,20,20)
            abs_x = np.abs(np.arange(20)*3-28.5)
            regional = [dose[:,1:10,mask].mean() for mask in
                        (abs_x < 4, (abs_x >= 7) & (abs_x < 13), abs_x >= 19)]
            assert min(regional) > 0
            samples[p].append([*regional, prepare_seconds, transport_seconds, manifest['jobs'][0]['histories']])
    baseline = np.asarray(samples[1.])
    for p, values in samples.items():
        data = np.asarray(values)
        mean, sd = data.mean(axis=0), data.std(axis=0, ddof=1)
        print(f'BENCHMARK p={p:g}: mean Gy/history {mean[:3]}, SD/mean {sd[:3]/mean[:3]}, '
              f'mean prep {mean[3]:.3f}s, TOPAS {mean[4]:.3f}s, transported {mean[5]:.0f}')
        if p < 1:
            baseline_mean = baseline[:,:3].mean(axis=0)
            standard_error = np.sqrt((sd[:3]**2+baseline[:,:3].var(axis=0,ddof=1))/len(baseline))
            assert np.all(np.abs(mean[:3]-baseline_mean) < 4*standard_error)
            assert mean[5] < baseline[:,5].mean()
