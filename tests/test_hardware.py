import json
import os
from pathlib import Path
import shutil
import subprocess
import numpy as np
import pytest
from pyRadPlan import generate_stf,calc_dose_influence
from minibeam import TOPASPhotonEngine
from minibeam.geometry.configuration import GeometryConfig
from minibeam.geometry.assembly import TreatmentHead,resolve,serializable
from minibeam.geometry.mlc_jaws import _rounded_mlc_polygon
from minibeam.topas.manifest import load_manifest


def test_legacy_geometry_agreement():
    gold=json.loads((Path(__file__).parent/'fixtures/legacy_devices.json').read_text())
    g,ap,_,_=resolve(GeometryConfig.load(),1000.)
    for key,expected in gold.items():
        if key=='rounded_mlc_positive_polygon':np.testing.assert_allclose(_rounded_mlc_polygon(g,True),expected,atol=1e-12)
        else:
            actual=serializable(getattr(g,key) if hasattr(g,key) else getattr(ap,key))
            assert actual==expected


def test_rectangular_fields_and_sad():
    config=GeometryConfig.load()
    small=GeometryConfig(config.text.replace('width_mm = 100.0','width_mm = 80.0',1).replace('height_mm = 100.0','height_mm = 120.0',1))
    g,ap,_,_=resolve(small,1100.)
    tip=g.mlc_round_tip;slope=40/1100
    assert (tip.circle_center_x-slope*(tip.circle_center_z+1100))/np.sqrt(1+slope*slope)==pytest.approx(tip.radius)
    assert g.lower_jaw.exit_opening==pytest.approx(120*(550+77/2)/1100)
    assert ap.collimator_center_z==707-1100


@pytest.mark.parametrize('old,new,message',[
    ('type = "slits"','type = "holes"','Unknown aperture'),
    ('enabled = true','enabled = 1','boolean'),
    ('slit_count = 17','slit_count = 16','odd'),
    ('blade_thickness_mm = 4.0','blade_thickness_mm = 8.0','CTC'),
    ('tip_radius_mm = 170.0','tip_radius_mm = 10.0','radius'),
    ('width_mm = 100.0','width_mm = 2000.0','MLC|mlc'),
    ('source_to_center_mm = 550.0','source_to_center_mm = 470.0','overlap'),
    ('brass_frame_thickness_x_mm = 16.0','brass_frame_thickness_x_mm = 75.0','cavity wall'),
    ('rotation_x_deg = 0.0','rotation_x_deg = nan','finite'),
])
def test_invalid_hardware(tmp_path,old,new,message):
    path=tmp_path/'config.toml';path.write_text(GeometryConfig.load().text.replace(old,new,1))
    with pytest.raises(ValueError,match=message):resolve(GeometryConfig.load(path),1000.)


def test_fixed_head_and_collision(case):
    ct,plan,cst,_,_=case
    plan.prop_stf.update(generator='photonIMRT',gantry_angles=[37.,215.],couch_angles=[21.,-33.])
    stf=generate_stf(ct,cst,plan)
    engine=TOPASPhotonEngine(plan,water=True,geometry_config=None)
    definition=engine._definition(ct,stf)[0]
    for bi in [1,2]:
        jobs=[j for j in definition['jobs'] if j['beam_index']==bi]
        assert len(jobs)>1
        assert all(j['devices']==jobs[0]['devices'] for j in jobs)
    for beam in stf.beams:
        beam.iso_center=np.asarray(beam.iso_center)+np.array([0,0,0])
    # Enlarge the conservative CT box until a hardware stage intersects it.
    from minibeam.geometry.spatial import overlaps_box
    assert overlaps_box([0,0,-293],np.eye(3),[78,87,50],[400,400,400])
    head=engine.devices[0]
    from minibeam.geometry.spatial import beam_basis
    stf.beams[0].iso_center=stf.beams[0].iso_center+beam_basis(stf.beams[0])[:,2]*293
    with pytest.raises(ValueError,match='Beam 1.*CT transport box'):
        head.validate_patient(ct,stf.beams[0],1)


def test_snapshot_relocation_and_report(case,tmp_path):
    ct,plan,cst,stf,_=case
    config=tmp_path/'head.toml';config.write_text(GeometryConfig.load().text)
    engine=TOPASPhotonEngine(plan,water=True,geometry_config=str(config))
    config.write_text(config.read_text().replace('width_mm = 100.0','width_mm = 80.0',1))
    root=engine.prepare_jobs(ct,cst,stf)
    manifest=load_manifest(root)
    assert 'width_mm = 100.0' in (root/'inputs/geometry-config.toml').read_text()
    assert (root/'derived/geometry.pdf').read_bytes().startswith(b'%PDF')
    assert not any(p.startswith('derived/') for p in manifest['files'])
    assert engine.prepare_jobs(ct,cst,stf)==root
    from test_results import fake_results
    fake_results(root)
    assert engine.collect_results(root).physical_dose.flat[0].shape[1]==2
    moved=tmp_path/'moved';shutil.move(root,moved)
    assert load_manifest(moved)['bundle_id']==manifest['bundle_id']
    changed=TOPASPhotonEngine(plan,water=True,geometry_config=str(config))
    with pytest.raises(ValueError,match='different inputs/settings'):changed.prepare_jobs(ct,cst,stf,bundle_dir=moved)


def test_disable_and_override(case,tmp_path):
    ct,plan,cst,stf,_=case
    assert not TOPASPhotonEngine(plan,devices=(),geometry_config=None).devices
    config=tmp_path/'disabled.toml';config.write_text(GeometryConfig.load().text.replace('enabled = true','enabled = false',1))
    engine=TOPASPhotonEngine(plan,water=True,geometry_config=str(config))
    assert not engine.devices
    root=engine.prepare_jobs(ct,cst,stf)
    assert 'geometry-config.toml' in load_manifest(root)['geometry_configuration']['input']
    # Native factory reads the same configuration; collection must match preparation.
    from test_results import fake_results
    fake_results(root)
    plan.prop_dose_calc.update(geometry_config=str(config),water=True)
    assert calc_dose_influence(ct,cst,stf,plan).physical_dose.flat[0].shape[1]==2


@pytest.mark.topas
def test_hardware_transport(case,tmp_path):
    executable=os.environ.get('PHOTON_TPS_TOPAS')
    if not executable:pytest.skip('Set PHOTON_TPS_TOPAS for hardware transport')
    ct,plan,cst,_,_=case
    # Narrow cones: central open slit, an adjacent brass blade, and a ray outside jaws.
    plan.prop_stf.update(gantry_angles=[0.],couch_angles=[0.],bixel_width=.25)
    stf=generate_stf(ct,cst,plan)
    central=stf.beams[0].rays[0]
    for x in [6.,70.]:
        ray=central.model_copy(deep=True);ray.ray_pos=np.array([x,0.,0.]);stf.beams[0].rays.append(ray)
    # A wide water box receives all three rays even without hardware.
    import SimpleITK as sitk
    from pyRadPlan.ct import CT
    from pyRadPlan.cst import StructureSet,create_voi
    image=sitk.GetImageFromArray(np.zeros((12,80,80),dtype=np.float32))
    image.SetSpacing((3.,3.,3.));image.SetOrigin((-118.5,-118.5,-16.5))
    ct=CT(cube_hu=image)
    mask=np.zeros((12,80,80),dtype=np.uint8);mask[6,40,40]=1
    cst=StructureSet(ct_image=ct,vois=[create_voi(name='TARGET',grid=ct.grid,mask=mask,voi_type='TARGET')])
    engine=TOPASPhotonEngine(plan,water=True,geometry_config=None,histories=10000)
    root=engine.prepare_jobs(ct,cst,stf,bundle_dir=tmp_path/'hardware')
    for job in load_manifest(root)['jobs']:
        result=subprocess.run([executable,job['parameter_file']],cwd=root,capture_output=True,text=True,check=True)
        assert 'Overlap is detected' not in result.stdout+result.stderr
    dij=engine.collect_results(root);matrix=dij.physical_dose.flat[0]
    totals=np.asarray(matrix.sum(axis=0)).ravel()
    # Finite-thickness brass transmits/scatters some high-energy photons.
    assert totals[0]>0 and totals[1]<totals[0]*.5 and totals[2]<totals[0]*.15
    import SimpleITK as sitk
    weights=np.array([1.,2.,3.])
    np.testing.assert_allclose(matrix@weights,sitk.GetArrayFromImage(engine.collect_forward(weights,root)['physical_dose']).ravel())


@pytest.mark.parametrize('disabled', ['mlc','jaws','aperture'])
def test_individual_device_switches(disabled):
    config=GeometryConfig.load()
    text=config.text.replace(f'[{disabled}]\nenabled = true',f'[{disabled}]\nenabled = false')
    g,ap,envelopes,parameters=resolve(GeometryConfig(text),1000.)
    assert not any(e['name']==('SlitCollimator' if disabled=='aperture' else disabled) for e in envelopes)
    if disabled=='mlc':assert g.mlc is None and 'Ge/MLCPositive/' not in parameters
    if disabled=='jaws':assert g.lower_jaw is None and 'Ge/LowerJawPositive/' not in parameters
    if disabled=='aperture':assert ap is None and 'Ge/Collimator/' not in parameters


def test_flat_banks_and_rotated_bounds():
    config=GeometryConfig.load()
    text=config.text.replace('round_tip_enabled = true','round_tip_enabled = false').replace('rotation_x_deg = 0.0','rotation_x_deg = 10.0').replace('rotation_y_deg = 0.0','rotation_y_deg = 5.0')
    g,ap,envelopes,parameters=resolve(GeometryConfig(text),1000.)
    assert g.mlc_round_tip is None and 'G4ExtrudedSolid' not in parameters
    assert 'Ge/MLCPositive/Type = "G4GTrap"' in parameters
    from minibeam.geometry.spatial import corners
    points=corners(envelopes[-1])
    assert np.ptp(points[:,2])>100


@pytest.mark.topas
def test_oblique_hardware_transport(case,tmp_path):
    executable=os.environ.get('PHOTON_TPS_TOPAS')
    if not executable:pytest.skip('Set PHOTON_TPS_TOPAS for hardware transport')
    ct,plan,cst,stf,_=case
    text=GeometryConfig.load().text.replace('width_mm = 100.0','width_mm = 80.0',1).replace('height_mm = 100.0','height_mm = 120.0',1)
    text=text.replace('rotation_x_deg = 0.0','rotation_x_deg = 10.0').replace('rotation_y_deg = 0.0','rotation_y_deg = 5.0')
    config=tmp_path/'rotated.toml';config.write_text(text)
    engine=TOPASPhotonEngine(plan,water=True,geometry_config=str(config),histories=3000)
    root=engine.prepare_jobs(ct,cst,stf,bundle_dir=tmp_path/'oblique')
    for job in load_manifest(root)['jobs']:
        result=subprocess.run([executable,job['parameter_file']],cwd=root,capture_output=True,text=True,check=True)
        assert 'Overlap is detected' not in result.stdout+result.stderr
    assert engine.collect_results(root).physical_dose.flat[0].nnz>0


@pytest.mark.patient
@pytest.mark.topas
def test_patient_head_clearance_and_jaw_transport(case,tmp_path):
    executable=os.environ.get('PHOTON_TPS_TOPAS');directory=os.environ.get('PHOTON_TPS_DICOM')
    if not executable or not directory:pytest.skip('Set TOPAS and DICOM paths for patient hardware integration')
    from pyRadPlan import load_patient
    from minibeam.workflow.patient import select_target
    ct,cst=load_patient(directory)
    select_target(cst,'PTV2017fw')
    plan=case[1]
    plan.prop_stf.update(gantry_angles=[45.,135.,225.,315.],couch_angles=[0.,0.,0.,0.])
    stf=generate_stf(ct,cst,plan)
    head=TreatmentHead(GeometryConfig.load())
    for index,beam in enumerate(stf.beams,1):
        with pytest.raises(ValueError,match='SlitCollimator.*CT transport box'):
            head.validate_patient(ct,beam,index)
    # Retain actual CT and reference positions; remove just the
    # overlapping slit assembly in this explicit test configuration.
    text=GeometryConfig.load().text.replace('[aperture]\nenabled = true','[aperture]\nenabled = false')
    config=tmp_path/'jaw-only.toml';config.write_text(text)
    engine=TOPASPhotonEngine(plan,geometry_config=str(config),dicom_dir=directory,
                             histories=1000,dose_spacing_mm=(10.,10.,10.))
    root=engine.prepare_jobs(ct,cst,stf,bundle_dir=tmp_path/'patient-head')
    for job in load_manifest(root)['jobs']:
        result=subprocess.run([executable,job['parameter_file']],cwd=root,capture_output=True,text=True,check=True)
        assert 'Overlap is detected' not in result.stdout+result.stderr
    assert engine.collect_results(root).physical_dose.flat[0].nnz>0


def test_native_configured_head_collection(case):
    ct,plan,cst,stf,_=case
    plan.prop_dose_calc.update(geometry_config=None,water=True)
    engine=TOPASPhotonEngine(plan)
    root=engine.prepare_jobs(ct,cst,stf)
    from test_results import fake_results
    fake_results(root)
    dij=calc_dose_influence(ct,cst,stf,plan)
    assert dij.physical_dose.flat[0].shape[1]==stf.total_number_of_bixels
    from pyRadPlan import calc_dose_forward
    import SimpleITK as sitk
    weights=np.array([2.,3.])
    result=calc_dose_forward(ct,cst,stf,plan,weights=weights)
    np.testing.assert_allclose(dij.physical_dose.flat[0]@weights,sitk.GetArrayFromImage(result['physical_dose']).ravel())
