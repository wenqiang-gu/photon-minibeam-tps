"""Source-to-body and source-to-curvature references must not be confused."""
from importlib.resources import files
import numpy as np
import pytest
from minibeam.geometry.configuration import GeometryConfig
from minibeam.geometry.models import BeamGeometry
from minibeam.geometry.assembly import TreatmentHead, serializable
from minibeam.geometry.diagnostics import material_solids


@pytest.mark.parametrize('filename',['config.toml','water_config.toml'])
@pytest.mark.parametrize('model',['simple','leaves'])
def test_corrected_axial_references(case,filename,model):
    config=GeometryConfig.load(files('minibeam.geometry').joinpath(filename))
    geometry=BeamGeometry.from_config(config).replace(mlc={'model':model})
    head=TreatmentHead(geometry.as_config());beam=case[3].beams[0]
    g,_,envelopes,text=head.resolved(beam)
    expected=dict(body_center_from_source_mm=356.8,entrance_from_source_mm=311.8,
                  exit_from_source_mm=401.8,tip_curvature_center_from_source_mm=349.3)
    assert serializable(g)['mlc_axial_reference']==pytest.approx(expected)
    points=np.concatenate([v for n,v in material_solids(head,beam) if n.startswith('MLC')])
    np.testing.assert_allclose(points[:,2].min()+beam.sad,311.8)
    np.testing.assert_allclose(points[:,2].max()+beam.sad,401.8)
    envelope=next(e for e in envelopes if e['name']=='mlc')
    assert envelope['center'][2]+beam.sad==pytest.approx(356.8)
    if model=='simple':
        assert g.mlc_round_tip.circle_center_z+beam.sad==pytest.approx(349.3)
        tip=g.mlc_round_tip
        assert tip.tangent_x/(tip.tangent_z+beam.sad)==pytest.approx(geometry.field.width_mm/2/beam.sad)
    else:
        assert all(l['center_z']+beam.sad==pytest.approx(356.8) for l in g.leaf_shape['leaves'])
        assert all(2*l['y_half']==pytest.approx(5*356.8/beam.sad) for l in g.leaf_shape['leaves'])
        assert min(p['resolved_gap_mm'] for p in g.leaf_shape['pair_gaps'])>=1-1e-10
        assert 'd:Ge/MLCLeaf001Negative/TransZ = -643.2 mm' in text
    # Explicit old/custom placement remains authoritative, without migration.
    old=geometry.replace(mlc={'source_to_center_mm':350.})
    previous=TreatmentHead(old.as_config()).resolved(beam)[0]
    assert previous.mlc_axial_reference['body_center_from_source_mm']==350
    assert previous.mlc_axial_reference['tip_curvature_center_from_source_mm']==342.5
