"""Water overviews project full rotated solids, not edge-on center sections."""
import numpy as np
import pytest
from scipy.spatial import ConvexHull
from minibeam.geometry.report import slit_projections, write_geometry_report
from minibeam.geometry.diagnostics import material_solids
from minibeam.geometry.assembly import TreatmentHead
from minibeam.geometry.models import BeamGeometry
from minibeam.geometry.configuration import GeometryConfig


@pytest.mark.parametrize('angle,tilt,shift', [(-90,0,0),(0,0,0),(-45,0,3),(25,12,4.5)])
def test_solid_projections(case,angle,tilt,shift):
    geometry=BeamGeometry.from_config(GeometryConfig.load()).replace(aperture=dict(
        enabled=True,rotation_z_deg=angle,rotation_x_deg=tilt,rotation_y_deg=tilt/2,lateral_shift_mm=shift))
    head=TreatmentHead(geometry.as_config());beam=case[3].beams[0]
    solids=dict(material_solids(head,beam))
    for axis in (0,1):
        polygons=slit_projections(head,beam,axis)
        assert len(polygons)==22
        for name,polygon in polygons:
            source=solids[name][:,[2,axis]]
            np.testing.assert_allclose(polygon.min(axis=0),source.min(axis=0))
            np.testing.assert_allclose(polygon.max(axis=0),source.max(axis=0))
            hull=ConvexHull(polygon)
            assert hull.volume>0
            assert np.all(source @ hull.equations[:,:2].T + hull.equations[:,2] <= 1e-8)


def test_disabled_slits(case):
    geometry=BeamGeometry.from_config(GeometryConfig.load()).replace(aperture={'enabled':False})
    head=TreatmentHead(geometry.as_config())
    assert slit_projections(head,case[3].beams[0],0)==[]
    assert slit_projections(head,case[3].beams[0],1)==[]


def test_both_panels_draw_solids(case,tmp_path,monkeypatch):
    from matplotlib.axes import Axes
    seen=[];original=Axes.add_patch
    def record(ax,patch):
        if patch.get_alpha() in (.18,.5):seen.append((ax,patch))
        return original(ax,patch)
    monkeypatch.setattr(Axes,'add_patch',record)
    ct,_,_,stf,_=case
    stf=stf.model_copy(deep=True);stf.beams=stf.beams[:1]
    head=TreatmentHead(GeometryConfig.load())
    write_geometry_report(ct,stf,head,tmp_path/'report.pdf')
    assert len(seen)==44
    assert len({id(ax) for ax,_ in seen})==2
