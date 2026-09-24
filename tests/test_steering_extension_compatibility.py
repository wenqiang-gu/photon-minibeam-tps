"""Production steering subclasses through native pyRadPlan and MATLAB APIs."""
from copy import deepcopy
import numpy as np
import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic_core import PydanticSerializationError
from scipy.io import loadmat, savemat
from pyRadPlan import calc_dose_influence, calc_dose_forward
from pyRadPlan.stf import Beam, SteeringInformation, validate_stf
from minibeam import TOPASPhotonEngine
from minibeam.topas.manifest import json_data


from minibeam.geometry.models import BeamGeometry
from minibeam.steering import (MinibeamBeam, MinibeamSteeringInformation, enrich_stf,
                               matrad_steering, validate_minibeam_stf)


@pytest.fixture
def extended(case):
    config = BeamGeometry.from_config().replace(assembly={'enabled': False})
    return enrich_stf(case[3], config, overrides=[{'aperture': {'lateral_shift_mm': float(i+1)}} for i in range(2)])


def test_live_validation_preserves_subclasses(extended):
    assert validate_stf(extended) is extended
    assert SteeringInformation.model_validate(extended) is extended
    native_container = SteeringInformation(beams=extended.beams)
    assert native_container.beams[0] is extended.beams[0]
    assert isinstance(native_container.beams[0], MinibeamBeam)


def test_copy_and_reordering_keep_associations(extended):
    for copied in (deepcopy(extended), extended.model_copy(deep=True)):
        copied.beams.reverse()
        assert copied.beams[0].gantry_angle == extended.beams[1].gantry_angle
        assert copied.beams[0].geometry.aperture.lateral_shift_mm == 2.
        copied.beams[0].geometry = copied.beams[0].geometry.replace(aperture={'lateral_shift_mm': 3.})
        assert extended.beams[1].geometry.aperture.lateral_shift_mm == 2.
    # Shallow copies share mutable nested models, as expected in Pydantic.
    assert extended.model_copy().beams[0].geometry is extended.beams[0].geometry


def test_typed_dictionary_roundtrip_and_native_field_loss(extended):
    dump = extended.model_dump(exclude_computed_fields=True)
    assert dump['beams'][0]['geometry']['aperture']['lateral_shift_mm'] == 1.
    restored = MinibeamSteeringInformation.model_validate(deepcopy(dump))
    assert isinstance(restored.beams[0], MinibeamBeam)
    assert restored.beams[0].geometry == extended.beams[0].geometry
    native = validate_stf(deepcopy(dump))
    assert type(native.beams[0]) is Beam
    assert not hasattr(native.beams[0], 'geometry')


def test_geometry_validation(extended):
    with pytest.raises(ValidationError):
        extended.beams[0].geometry.aperture.lateral_shift_mm = -1.
    dump = extended.model_dump(exclude_computed_fields=True)
    dump['beams'][0]['geometry']['aperture']['slit_entrance_width_mm'] = -1.
    with pytest.raises(ValidationError):
        MinibeamSteeringInformation.model_validate(dump)
    with pytest.raises(ValidationError):
        extended.beams[0].geometry.replace(unknown=1)


def test_native_json_limitation_is_not_subclass_specific(case, extended):
    # The upstream beams serializer returns Python-mode NumPy arrays, which
    # model_dump_json cannot encode. Extension persistence needs an explicit codec.
    for steering in (case[3], extended):
        with pytest.raises(PydanticSerializationError, match='numpy.ndarray'):
            steering.model_dump_json()
        # The strict manifest converter handles arrays, but native beamlets also
        # have infinite defaults (e.g. max_mu). A full-steering JSON codec must
        # explicitly represent these rather than silently alter their meaning.
        with pytest.raises(ValueError, match='Out of range float'):
            json_data(steering.model_dump(exclude_computed_fields=True))


def test_explicit_standard_matrad_export(case, extended, tmp_path):
    # An inherited serializer exports geometry too; it is NOT standard-only.
    assert 'geometry' in extended.to_matrad().dtype.names
    native = SteeringInformation(beams=[Beam.model_validate(beam.model_dump(
        exclude={'geometry'}, exclude_computed_fields=True)) for beam in extended.beams])
    standard = native.to_matrad()
    assert standard.dtype.names == case[3].to_matrad().dtype.names
    assert 'geometry' not in standard.dtype.names
    companion = [dict(beam_index=i+1, **b.geometry.model_dump()) for i,b in enumerate(extended.beams)]
    path = tmp_path/'steering.mat'
    savemat(path, {'stf': standard, 'beam_geometry': companion})
    saved = loadmat(path, simplify_cells=True)
    assert saved['beam_geometry'][0]['beam_index'] == 1
    assert saved['beam_geometry'][1]['aperture']['lateral_shift_mm'] == 2.
    for before,after in zip(case[3].beams, saved['stf']):
        assert before.gantry_angle == after['gantryAngle']
        np.testing.assert_array_equal(before.source_point, after['sourcePoint'])
        np.testing.assert_array_equal(before.rays[0].ray_pos, after['ray']['rayPos'])
    assert extended.beams[0].geometry.aperture.lateral_shift_mm == 1.


def test_native_matrad_single_ray_import_limitation(case, tmp_path):
    # This failure occurs even without subclasses: simplified single-ray structs
    # need shape normalization before the upstream MATLAB importer can read them.
    path = tmp_path/'native.mat'
    savemat(path, {'stf': case[3].to_matrad()})
    payload = loadmat(path, simplify_cells=True)['stf']
    with pytest.raises(ValidationError, match="All values in the 'ray' dictionary"):
        validate_stf(payload)


def test_native_dose_calls_receive_extended_objects(case, extended, monkeypatch):
    from test_results import fake_results
    ct,plan,cst,_,engine = case
    plan.prop_dose_calc.update(water=True)
    root = engine.prepare_jobs(ct,cst,extended)
    fake_results(root)
    seen = []
    original = TOPASPhotonEngine.prepare_jobs
    def inspect(self,ct,cst,stf,**kwargs):
        assert stf is extended
        assert isinstance(stf.beams[0], MinibeamBeam)
        assert stf.beams[0].geometry.aperture.lateral_shift_mm == 1.
        seen.append(True)
        return original(self,ct,cst,stf,**kwargs)
    monkeypatch.setattr(TOPASPhotonEngine,'prepare_jobs',inspect)
    dij = calc_dose_influence(ct,cst,extended,plan)
    result = calc_dose_forward(ct,cst,extended,plan,weights=[1.,1.])
    assert dij.physical_dose.flat[0].shape[1] == 2
    assert 'physical_dose' in result and len(seen) == 0
