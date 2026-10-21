"""Device-specific fixed envelopes and shared-dimension migration."""
import copy
import numpy as np
import pytest
from minibeam.geometry.configuration import GeometryConfig
from minibeam.geometry.models import BeamGeometry
from minibeam.geometry.assembly import resolve, TreatmentHead
from minibeam.geometry.diagnostics import material_solids


@pytest.mark.parametrize('rounded', [True, False])
def test_device_envelopes_and_solids(case, rounded):
    geometry = BeamGeometry.from_config().replace(mlc={'round_tip_enabled':rounded}, jaws={'round_tip_enabled':rounded})
    head = TreatmentHead(geometry.as_config())
    _,_,envelopes,_ = head.resolved(case[3].beams[0])
    np.testing.assert_allclose(envelopes[0]['half_size'], [155,80,45])
    np.testing.assert_allclose(envelopes[1]['half_size'], [115,155,38.5])
    solids = material_solids(head,case[3].beams[0])
    for prefix,expected in [('MLC',[155,80]), ('LowerJaw',[115,155])]:
        vertices = np.concatenate([v for n,v in solids if n.startswith(prefix)])
        np.testing.assert_allclose(np.abs(vertices[:,:2]).max(axis=0),expected)


def test_independent_dimensions_and_field_openings():
    g = BeamGeometry.from_config()
    baseline = resolve(g.as_config(),1000)
    for changed,unchanged in [('mlc','LowerJaw'),('jaws','MLC')]:
        other = resolve(g.replace(**{changed:{'outer_width_mm':300.0}}).as_config(),1000)
        assert [s for s in baseline[3].splitlines() if '/'+unchanged in s] == [s for s in other[3].splitlines() if '/'+unchanged in s]
    for field in [{'width_mm':150.0},{'height_mm':150.0}]:
        other=resolve(g.replace(field=field).as_config(),1000)
        assert other[2]==baseline[2]
    assert g.as_config().sha256 != g.replace(mlc={'outer_width_mm':300.0}).as_config().sha256


def test_legacy_snapshot_migration_without_mutation():
    data=GeometryConfig.load().data
    for section in ('mlc','jaws'):
        data[section].pop('outer_width_mm');data[section].pop('outer_height_mm')
    data['field'].update(outer_width_mm=400.0,outer_height_mm=400.0)
    original=copy.deepcopy(data)
    snapshot=BeamGeometry.model_validate(data)
    assert data==original
    assert snapshot.mlc.outer_width_mm==snapshot.jaws.outer_width_mm==400
    canonical=GeometryConfig.from_data(data).data
    assert resolve(GeometryConfig.from_data(canonical),1000)[3]==resolve(GeometryConfig.from_data(data),1000)[3]
    data['mlc']['outer_width_mm']=310.0
    with pytest.raises(ValueError,match='Mixed'):
        BeamGeometry.model_validate(data)
    with pytest.raises(ValueError,match='Mixed'):
        GeometryConfig.from_data(data).validate()


@pytest.mark.parametrize('value',[0,-1,True,float('inf'),float('nan')])
def test_invalid_dimensions(value):
    with pytest.raises(ValueError):
        BeamGeometry.from_config().replace(mlc={'outer_width_mm':value})


def test_containment_and_disabled():
    small=BeamGeometry.from_config().replace(mlc={'outer_width_mm':30.0})
    with pytest.raises(ValueError,match='opening exceeds'):
        resolve(small.as_config(),1000)
    _,_,envelopes,text=resolve(small.replace(mlc={'enabled':False}).as_config(),1000)
    assert 'Ge/MLC' not in text
    assert all(e['name']!='mlc' for e in envelopes)
