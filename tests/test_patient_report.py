import re
import numpy as np
import pytest
import SimpleITK as sitk
from pyRadPlan import generate_stf
from minibeam.geometry.patient_report import (patient_scene, slice_groups, projected_polygon,
    axial_data, clipped_segment, write_patient_geometry_report)
from minibeam.geometry.assembly import TreatmentHead
from minibeam.geometry.configuration import GeometryConfig
from minibeam.geometry.spatial import beam_basis
from minibeam.geometry.diagnostics import material_solids, beam_diagnostics


def test_scene_native_beam_transform(case):
    ct,plan,cst,_,_=case
    plan.prop_stf.update(gantry_angles=[0.,90.,180.,270.,37.],couch_angles=[0.,0.,0.,0.,21.])
    stf=generate_stf(ct,cst,plan)
    head=TreatmentHead(GeometryConfig.load())
    scene=patient_scene(ct,stf,head)
    np.testing.assert_allclose(scene['ct_box'].min(axis=0),[-30,-30,-18])
    np.testing.assert_allclose(scene['ct_box'].max(axis=0),[30,30,18])
    for item,beam in zip(scene['beams'],stf.beams):
        np.testing.assert_allclose(item['source'],beam.iso_center+beam.source_point)
        np.testing.assert_allclose(item['iso']-item['source'],-beam.source_point,atol=1e-10)
        assert np.linalg.norm(item['iso']-item['source'])==pytest.approx(beam.sad)
        for (name,actual),(expected_name,local) in zip(item['solids'],material_solids(head,beam)):
            assert name==expected_name
            np.testing.assert_allclose((actual-item['iso'])@beam_basis(beam),local,atol=1e-10)
        assert all(not name.startswith(('Frame','Blade')) for name,_ in item['solids']) == (not head.config.data['aperture']['enabled'])
    assert abs(scene['beams'][-1]['source'][2]-scene['beams'][-1]['iso'][2])>1


def test_native_slice_extent_and_target(case):
    ct,_,cst,stf,_=case
    scene=patient_scene(ct,stf,None)
    assert not np.allclose(scene['ct_center'],stf.beams[0].iso_center)
    index=scene['beams'][0]['slice']['slice_index_1based']
    pixels,extent,overlays=axial_data(ct,cst,index,('TARGET',))
    np.testing.assert_array_equal(pixels,sitk.GetArrayViewFromImage(ct.cube_hu)[index-1])
    assert extent==[-30,30,-30,30]
    assert overlays[0][0:2]==('TARGET',True)
    np.testing.assert_array_equal(overlays[0][2],sitk.GetArrayViewFromImage(cst.vois[0].mask)[index-1])
    stf.beams[1].iso_center[2]=ct.origin[2]
    assert len(slice_groups(patient_scene(ct,stf,None)))==2


def test_projection_preserves_oblique_shape():
    from minibeam.geometry.spatial import corners,placement_rotation
    points=corners({'center':[10,20,30],'half_size':[2,3,4],
                   'rotation':placement_rotation(0,0,37)})
    polygon=projected_polygon(points,(0,1))
    assert len(polygon)==4
    edges=np.roll(polygon,-1,axis=0)-polygon
    assert np.all(np.abs(edges)>1e-6)  # rotated edges, not an enclosing rectangle


def test_closeup_clipping():
    np.testing.assert_allclose(clipped_segment([-10,0,0],[0,0,0],[-1]*3,[1]*3),[[-1,0,0],[0,0,0]])
    assert clipped_segment([-10,2,0],[0,2,0],[-1]*3,[1]*3) is None


def test_patient_pdf_pages_and_collision_pages(case,tmp_path):
    ct,_,cst,stf,_=case
    stf.beams[1].iso_center[2]=ct.origin[2]
    head=TreatmentHead(GeometryConfig.load())
    records=[beam_diagnostics(ct,b,head,i) for i,b in enumerate(stf.beams,1)]
    output=tmp_path/'geometry.pdf'
    write_patient_geometry_report(ct,cst,stf,head,output,target_names=('TARGET',),diagnostics=records)
    data=output.read_bytes()
    assert data.startswith(b'%PDF')
    # Two slice overviews, one 3D overview, two detail pages, two collision pages.
    assert len(re.findall(rb'/Type /Page\b',data))==7


def test_patient_and_water_dispatch(case,tmp_path,monkeypatch):
    ct,_,cst,stf,engine=case
    calls=[]
    monkeypatch.setattr('minibeam.geometry.patient_report.write_patient_geometry_report',
                        lambda *args,**kw: calls.append(('patient',kw['target_names'],args[1])))
    monkeypatch.setattr('minibeam.geometry.report.write_geometry_report',
                        lambda *args,**kw: calls.append(('water',)))
    writer=engine._writer();writer.water=False
    writer._report(ct,cst,stf,tmp_path)
    assert calls[0][0:2]==('patient',('TARGET',)) and calls[0][2] is cst
    writer.water=True;writer.geometry_snapshot=GeometryConfig.load()
    writer._report(ct,cst,stf,tmp_path)
    assert calls[-1]==('water',)


@pytest.mark.parametrize('enabled',[True,False])
def test_aperture_switch_controls_rendered_solids(case,enabled):
    text=GeometryConfig.load().text
    text=text.replace('[aperture]\nenabled = false','[aperture]\nenabled = true')
    if not enabled:
        text=text.replace('[aperture]\nenabled = true','[aperture]\nenabled = false')
    scene=patient_scene(case[0],case[3],TreatmentHead(GeometryConfig(text)))
    names=[name for name,_ in scene['beams'][0]['solids']]
    assert sum(name.startswith('Blade_') for name in names)==(18 if enabled else 0)
    assert sum(name.startswith('Frame') for name in names)==(4 if enabled else 0)
