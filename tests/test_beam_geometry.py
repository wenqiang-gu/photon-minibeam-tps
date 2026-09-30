from copy import deepcopy
import os
import shutil
import subprocess
import numpy as np
import pytest
from pydantic import ValidationError
from pyRadPlan import generate_stf
from scipy.io import savemat, loadmat
from minibeam import TOPASPhotonEngine
from minibeam.geometry.models import BeamGeometry
from minibeam.geometry.assembly import TreatmentHead
from minibeam.geometry.diagnostics import material_solids
from minibeam.geometry.spatial import beam_basis
from minibeam.steering import enrich_stf, validate_minibeam_stf, matrad_steering
from minibeam.topas.manifest import load_manifest


@pytest.mark.parametrize('shift',[0.,1.5,3.,4.5])
def test_rigid_shifts_across_angles(case,shift):
    ct,plan,cst,_,_=case
    plan.prop_stf.update(generator='photonIMRT',gantry_angles=[45.,135.,225.,315.],couch_angles=[0.]*4)
    native=generate_stf(ct,cst,plan)
    base=BeamGeometry.from_config().replace(aperture={"rotation_z_deg":0.})
    zero=enrich_stf(native,base.replace(aperture={'lateral_shift_mm':0.}))
    stf=enrich_stf(native,base.replace(aperture={'lateral_shift_mm':shift}))
    head=TreatmentHead(None)
    for b,b0 in zip(stf.beams,zero.beams):
        ap=head.resolved(b)[1];ap0=head.resolved(b0)[1]
        assert ap.collimator_blades == ap0.collimator_blades
        assert ap.entrance_slit_ctcs == ap0.entrance_slit_ctcs
        # For couch=0, fixed beam +X is [cos(g),sin(g),0] in LPS.
        angle=np.deg2rad(b.gantry_angle)
        np.testing.assert_allclose(beam_basis(b)@np.array([shift,0,0]),[shift*np.cos(angle),shift*np.sin(angle),0],atol=1e-10)
        for (name,vertices),(_,original) in zip(material_solids(head,b),material_solids(head,b0)):
            expected=[shift,0,0] if name.startswith(('Frame','Blade')) else [0,0,0]
            # Both brass frame pieces and blades translate.
            np.testing.assert_allclose(vertices-original,np.broadcast_to(expected,vertices.shape),atol=1e-10)
        assert f'd:Ge/Collimator/TransX = {shift:.6f} mm' in head.resolved(b)[3]
    definition=TOPASPhotonEngine(plan,water=True)._definition(ct,stf)[0]
    for bi in range(1,5):
        jobs=[j for j in definition['jobs'] if j['beam_index']==bi]
        assert len(jobs)>1
        assert all(j['devices']==jobs[0]['devices'] for j in jobs)
    np.testing.assert_array_equal(native.beams[0].rays[0].ray_pos,stf.beams[0].rays[0].ray_pos)


def test_authoritative_snapshots_and_boundaries(case,monkeypatch):
    ct,plan,cst,native,engine=case
    stf=enrich_stf(native,BeamGeometry.from_config())
    monkeypatch.setattr('minibeam.geometry.configuration.GeometryConfig.load',lambda *a,**k: pytest.fail('Attached geometry must not load TOML'))
    engine._definition(ct,stf.model_dump(exclude_computed_fields=True))
    bad=deepcopy(stf)
    bad.beams[0]=bad.beams[0].model_copy(update={'geometry': bad.beams[0].geometry.model_copy(update={'aperture': {'bogus': 1}})})
    with pytest.raises(ValidationError):engine._definition(ct,bad)
    with pytest.raises(ValueError,match='conflicts'):
        TOPASPhotonEngine(plan,devices=())._definition(ct,stf)
    with pytest.raises(ValidationError):stf.beams[0].geometry.aperture.lateral_shift_mm=2.
    with pytest.raises(ValueError):enrich_stf(stf,BeamGeometry.model_validate(stf.beams[0].geometry.model_dump()))


def test_matrad_cache_collection_and_relocation(case,tmp_path):
    from test_results import fake_results
    ct,plan,cst,native,engine=case
    stf=enrich_stf(native,BeamGeometry.from_config())
    root=engine.prepare_jobs(ct,cst,stf)
    manifest=load_manifest(root)
    assert len(manifest['beam_geometry'])==2
    changed=enrich_stf(stf,overrides=[{'aperture':{'lateral_shift_mm':1.5}},{}])
    with pytest.raises(ValueError,match='different inputs'):engine.prepare_jobs(ct,cst,changed)
    second=engine.prepare_jobs(ct,cst,changed,bundle_dir=tmp_path/'shift25')
    assert load_manifest(second)['request_id']!=manifest['request_id']
    destination=tmp_path/'relocated';shutil.move(root,destination)
    fake_results(destination)
    matrix=engine.collect_results(destination).physical_dose.flat[0]
    import SimpleITK as sitk
    dose=engine.collect_forward([2.,3.],destination)['physical_dose']
    np.testing.assert_allclose(matrix@np.array([2.,3.]),sitk.GetArrayFromImage(dose).ravel())
    stf.beams.reverse()
    exported=matrad_steering(stf)
    assert 'geometry' not in exported['stf'].dtype.names
    savemat(tmp_path/'stf.mat',exported,long_field_names=True)
    saved=loadmat(tmp_path/'stf.mat',simplify_cells=True)
    assert saved['beam_geometry'][0]['beam_index']==1
    assert saved['beam_geometry'][0]['gantry_angle_deg']==stf.beams[0].gantry_angle


@pytest.mark.topas
@pytest.mark.parametrize("rotation",[-90.,0.])
def test_shifted_local_transport(case,tmp_path,rotation):
    executable=os.environ.get('PHOTON_TPS_TOPAS')
    if not executable:pytest.skip('Set PHOTON_TPS_TOPAS for local transport')
    ct,plan,cst,stf,_=case
    stf=enrich_stf(stf,BeamGeometry.from_config().replace(aperture={'lateral_shift_mm':1.5,'rotation_z_deg':rotation}))
    engine=TOPASPhotonEngine(plan,water=True,histories=2000)
    root=engine.prepare_jobs(ct,cst,stf,bundle_dir=tmp_path/'shifted')
    for job in load_manifest(root)['jobs']:
        result=subprocess.run([executable,job['parameter_file']],cwd=root,capture_output=True,text=True,check=True)
        assert 'Overlap is detected' not in result.stdout+result.stderr
    assert engine.collect_results(root).physical_dose.flat[0].nnz>0


def test_collision_diagnostics_use_attached_shifts(case,tmp_path):
    import json
    from minibeam.geometry.assembly import GeometryCollisionError
    from minibeam.steering import beam_geometry_records
    ct,plan,cst,native,engine=case
    stf=enrich_stf(native,BeamGeometry.from_config().replace(aperture={'lateral_shift_mm':4.5}))
    # Bring the collimator into this small test CT, without altering its shape.
    for b in stf.beams:
        b.iso_center = b.iso_center + beam_basis(b)@np.array([0.,0.,float(b.sad)-b.geometry.aperture.source_to_center_mm])
    root=tmp_path/'blocked'
    with pytest.raises(GeometryCollisionError) as error:
        engine.prepare_jobs(ct,cst,stf,bundle_dir=root)
    assert not (root/'manifest.json').exists()
    diagnostics=error.value.diagnostics_dir
    record=json.loads((diagnostics/'collisions.json').read_text())
    assert record['beam_geometry']==beam_geometry_records(stf)
    assert all(any(hit['intersects'] for hit in b['material_solids']) for b in record['beams'])
    assert (diagnostics/'geometry.pdf').is_file()


def test_nonzero_couch_shift_and_zero_legacy_parameters(case):
    from minibeam.geometry.configuration import GeometryConfig
    from minibeam.geometry.assembly import resolve
    from minibeam.geometry.patient_report import patient_scene
    ct,_,_,native,_=case
    config=GeometryConfig.load()
    stf=enrich_stf(native,BeamGeometry.from_config(config))
    shifted=enrich_stf(stf,overrides=[{'aperture':{'lateral_shift_mm':3.}}]*len(stf.beams))
    head=TreatmentHead(None)
    for beam,zero in zip(shifted.beams,stf.beams):
        assert head.resolved(zero)[3]==resolve(config,float(zero.sad))[3]
        for (name,points),(_,points0) in zip(material_solids(head,beam),material_solids(head,zero)):
            delta=(points-points0)@beam_basis(beam).T
            expected=beam_basis(beam)@np.asarray(head.resolved(beam)[1].translation_beam_mm) if name.startswith(('Frame','Blade')) else np.zeros(3)
            np.testing.assert_allclose(delta,np.broadcast_to(expected,delta.shape),atol=1e-10)

@pytest.mark.parametrize('angle',[0.,90.,37.,-25.])
@pytest.mark.parametrize('shift',[0.,1.5,3.,4.5])
def test_rotated_slit_transform(case,angle,shift):
    from minibeam.workflow.geometry import resolve_geometry
    from minibeam.geometry.spatial import placement_rotation
    from minibeam.geometry.apertures.slits import SlitAperture
    ct,plan,cst,_,_=case
    plan.prop_stf.update(gantry_angles=[45.,135.,225.,315.],couch_angles=[0.,20.,0.,15.])
    native=generate_stf(ct,cst,plan)
    geo=resolve_geometry(config_path=None,enable_collimator=True,collimator_rotation_deg=angle)
    stf=enrich_stf(native,geo.replace(aperture={'lateral_shift_mm':shift}))
    base=enrich_stf(native,geo.replace(aperture={'rotation_z_deg':0.,'lateral_shift_mm':0.}))
    head=TreatmentHead(None);rot=placement_rotation(z=angle-90)
    for beam,old in zip(stf.beams,base.beams):
        ap=head.resolved(beam)[1];ap0=head.resolved(old)[1]
        np.testing.assert_allclose(ap.translation_beam_mm,rot[:,0]*shift,atol=1e-12)
        assert ap.collimator_blades==ap0.collimator_blades
        for (name,v),(_,v0) in zip(material_solids(head,beam),material_solids(head,old)):
            expected=(v0-[0,0,ap0.collimator_center_z])@rot.T+ap.center_beam_mm if name.startswith(('Frame','Blade')) else v0
            np.testing.assert_allclose(v,expected,atol=1e-10)
            np.testing.assert_allclose(v@beam_basis(beam).T,expected@beam_basis(beam).T,atol=1e-10)
        np.testing.assert_allclose(SlitAperture.envelope(ap)['center'],ap.center_beam_mm)
        text=head.resolved(beam)[3]
        assert f'd:Ge/Collimator/RotZ = {angle-90:.6f} deg' in text
        for axis,value in zip('XYZ',ap.center_beam_mm):assert f'd:Ge/Collimator/Trans{axis} = {value:.6f} mm' in text
        if angle==0:
            np.testing.assert_allclose(rot[:,0],[0,1,0],atol=1e-12)
            np.testing.assert_allclose(abs(rot[:,1]),[1,0,0],atol=1e-12)
        if angle==90:np.testing.assert_allclose(rot,np.eye(3))


def test_rotation_validation_tilts_and_exports(case):
    from minibeam.workflow.geometry import resolve_geometry
    from minibeam.geometry.spatial import placement_rotation
    from minibeam.steering import beam_geometry_records
    for invalid in [True,None,'90',float('inf'),float('nan')]:
        with pytest.raises(ValueError):resolve_geometry(config_path=None,enable_collimator=True,collimator_rotation_deg=invalid)
        assert not resolve_geometry(config_path=None,enable_collimator=False,collimator_rotation_deg=invalid).aperture.enabled
    _,_,_,native,_=case
    geo=resolve_geometry(config_path=None,enable_collimator=True,collimator_rotation_deg=37.)
    geo=geo.replace(aperture={'rotation_x_deg':2.,'rotation_y_deg':3.,'lateral_shift_mm':3.})
    stf=enrich_stf(native,geo);ap=TreatmentHead(None).resolved(stf.beams[0])[1]
    rot=placement_rotation(2,3,-53)
    np.testing.assert_allclose(ap.translation_beam_mm,rot[:,0]*3)
    record=beam_geometry_records(stf)[0]
    assert record['resolved']['aperture']['rotation_matrix']==ap.rotation_matrix
    assert record['resolved']['aperture']['translation_beam_mm']==ap.translation_beam_mm
    other=enrich_stf(native,geo.replace(aperture={'rotation_z_deg':0.}))
    assert record['configuration_sha256']!=beam_geometry_records(other)[0]['configuration_sha256']


def test_custom_baseline_helper(tmp_path):
    from minibeam.geometry.configuration import GeometryConfig
    from minibeam.workflow.geometry import resolve_geometry
    text=GeometryConfig.load().text.replace('rotation_z_deg = -90.0','rotation_z_deg = 12.0')
    path=tmp_path/'custom.toml';path.write_text(text)
    zero=resolve_geometry(config_path=path,enable_collimator=True,collimator_rotation_deg=0.)
    assert zero==resolve_geometry(config_path=path,enable_collimator=True)
    assert resolve_geometry(config_path=path,enable_collimator=True,collimator_rotation_deg=45.).aperture.rotation_z_deg==57.
