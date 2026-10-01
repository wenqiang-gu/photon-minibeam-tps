"""Physical crop/grid mapping, saved-only collection, and real TOPAS checks."""
import os
import subprocess
import re
import numpy as np
import pytest
import SimpleITK as sitk
from scipy.io import loadmat
from pyRadPlan.ct import CT
from minibeam import TOPASPhotonEngine
from minibeam.geometry.cropping import resolve_grids, map_structure_mask, coverage_image
from minibeam.geometry.coordinates import patient_center
from minibeam.topas.manifest import load_manifest

CROP = {'x': (3,17), 'y': (4,18), 'z': (2,11)}


def air_case(case):
    ct, plan, cst, stf, _ = case
    a = np.full((12,20,20), -1000, dtype=np.float32)
    a[4:9,7:14,6:12] = 0
    a[6,10,8] = 500
    image = sitk.GetImageFromArray(a); image.CopyInformation(ct.cube_hu)
    ct = CT(cube_hu=image); cst.ct_image = ct
    return ct, plan, cst, stf


def test_crop_and_fine_grid(case):
    ct, _, cst, _ = air_case(case)
    transport, dose, meta = resolve_grids(ct,CROP,(1.5,1.5,1.5),cst)
    assert transport.size == (14,14,9)
    assert dose.dimensions == (28,28,18)
    np.testing.assert_allclose(transport.origin, np.array(ct.origin)+[9,12,6])
    np.testing.assert_allclose(dose.origin,np.array(transport.origin)-.75)
    np.testing.assert_allclose(np.array(dose.origin)+np.array(dose.dimensions)*1.5-.75,
                               np.array(transport.origin)+np.array(transport.size)*3-1.5)
    mask, status = map_structure_mask(cst.vois[0].mask,dose)
    assert status['complete']
    assert sitk.GetArrayFromImage(mask).sum() == sitk.GetArrayFromImage(cst.vois[0].mask).sum()*8
    assert meta['retained_ranges']=={a:list(p) for a,p in CROP.items()}
    assert ct.size == (20,20,12)


@pytest.mark.parametrize('crop',[{'x':(True,17),'y':(4,18),'z':(2,11)},
    {'x':(-1,17),'y':(4,18),'z':(2,11)}, {'x':(3,17)},
    {'x':(3,3),'y':(4,18),'z':(2,11)}])
def test_bad_ranges(case,crop):
    ct,_,cst,_=air_case(case)
    with pytest.raises(ValueError):resolve_grids(ct,crop,None,cst)


def test_reject_anatomy_and_roi(case):
    ct,_,cst,_=air_case(case)
    with pytest.raises(ValueError,match='structure'):resolve_grids(ct,dict(CROP,x=(9,19)),None,cst)
    with pytest.raises(ValueError,match='anatomy'):resolve_grids(ct,dict(CROP,y=(10,20)))
    # An enclosed low-HU region is protected even without an ROI.
    a=np.zeros((12,20,20),np.float32);a[1:-1,1:-1,1:-1]=-1000
    im=sitk.GetImageFromArray(a);im.CopyInformation(ct.cube_hu)
    with pytest.raises(ValueError):resolve_grids(CT(cube_hu=im),CROP)


def prepare(case,dicom_series,tmp_path):
    ct,plan,cst,stf=air_case(case)
    engine=TOPASPhotonEngine(plan,water=False,ct_crop_voxels=CROP,dose_spacing_mm=(1.5,1.5,1.5),
        dicom_dir=str(dicom_series(ct.cube_hu)),bundle_dir=str(tmp_path/'crop'))
    root=engine.prepare_jobs(ct,cst,stf)
    return ct,plan,cst,stf,engine,root


def test_bundle_crop(case,dicom_series,tmp_path):
    ct,plan,cst,stf,engine,root=prepare(case,dicom_series,tmp_path)
    m=load_manifest(root)
    assert m['ct_grid']['dimensions']==[20,20,12]
    assert m['transport_grid']['dimensions']==[14,14,9]
    assert m['dose_grid']['dimensions']==[28,28,18]
    text=(root/'inputs/common.txt').read_text()
    assert 'RestrictVoxelsXMin = 4' in text and 'RestrictVoxelsXMax = 17' in text
    assert 'Patient/TransY = 3 mm' in text
    from minibeam.workflow.artifacts import save_planning_snapshot
    save_planning_snapshot(ct,cst,plan,stf,root,{})
    from minibeam.workflow.collection import planning_snapshot,indexing_report
    saved,_=planning_snapshot(root,m)
    assert saved['crop_metadata']['applied']
    assert 'excluded CT voxels have no rows' in indexing_report(m,'forward')
    from minibeam.topas.validation import validate_request
    validate_request(m,ct,stf,(1.5,1.5,1.5),CROP)
    with pytest.raises(ValueError):validate_request(m,ct,stf,(1.5,1.5,1.5))


@pytest.mark.topas
def test_crop_actual_transport(case,dicom_series,tmp_path):
    executable=os.environ.get('PHOTON_TPS_TOPAS')
    if not executable:pytest.skip('Set PHOTON_TPS_TOPAS')
    ct,plan,cst,stf=air_case(case)
    from minibeam.geometry.devices import GeometryInclude
    engine=TOPASPhotonEngine(plan,histories=2000,water=False,ct_crop_voxels=CROP,
        dose_spacing_mm=(1.5,1.5,1.5),dicom_dir=str(dicom_series(ct.cube_hu)),
        bundle_dir=str(tmp_path/'actual'),geometry=GeometryInclude('b:Ge/Patient/DumpImagingValues = "True"'))
    root=engine.prepare_jobs(ct,cst,stf)
    for job in load_manifest(root)['jobs']:
        result=subprocess.run([executable,job['parameter_file']],cwd=root,capture_output=True,text=True)
        assert result.returncode==0,result.stdout+result.stderr
        decoded=np.array([int(v) for v in re.findall(r'Pixel: \d+ has value: (-?\d+)',result.stdout)])
        np.testing.assert_array_equal(decoded,sitk.GetArrayFromImage(ct.cube_hu)[2:11,4:18,3:17].ravel())
    dij=engine.collect_results(root)
    assert dij.physical_dose.flat[0].shape==(28*28*18,2)
    assert dij.physical_dose.flat[0].nnz>0
    values=dij.physical_dose.flat[0][:,0].toarray().reshape(18,28,28)>0
    subdivisions=values.reshape(9,2,14,2,14,2).sum(axis=(1,3,5))
    assert subdivisions.max() >= 2  # Hits in distinct bins within one CT material voxel.


def test_saved_crop_collection(case,dicom_series,tmp_path,monkeypatch):
    import shutil
    from test_results import fake_results
    from minibeam.workflow.artifacts import save_planning_snapshot
    from minibeam.workflow.collection import collect_bundle
    ct,plan,cst,stf,engine,root=prepare(case,dicom_series,tmp_path)
    save_planning_snapshot(ct,cst,plan,stf,root,{})
    fake_results(root)
    shutil.rmtree(engine.dicom_dir)
    moved=tmp_path/'moved'; shutil.move(root,moved)
    def forbidden(*a,**k):raise AssertionError('Must not prepare during collection')
    monkeypatch.setattr(TOPASPhotonEngine,'prepare_jobs',forbidden)
    collect_bundle(moved,'collect')
    data=loadmat(moved/'derived/result.mat',simplify_cells=True)
    assert data['dij']['physicalDose'].shape==(28*28*18,2)
    cube=data['dij']['physicalDose'][:,0].toarray().reshape((28,28,18),order='F')
    assert cube[3,2,4]==pytest.approx((1+2+30+400)*1e-10)
    assert data['crop_metadata']['applied']
    collect_bundle(moved,'forward',weight_per_bixel=[2.,3.])
    fine=sitk.ReadImage(str(moved/'derived/dose_scoring_grid.mha'))
    full=sitk.ReadImage(str(moved/'derived/dose.mha'))
    coverage=sitk.GetArrayFromImage(sitk.ReadImage(str(moved/'derived/dose_coverage.mha')))
    assert fine.GetSize()==(28,28,18) and full.GetSize()==ct.size
    assert coverage.sum()==14*14*9
    assert not np.any(sitk.GetArrayFromImage(full)[coverage==0])
    expected=(1+2+30+400)*1e-10*(2+2*3)
    assert sitk.GetArrayFromImage(fine)[4,3,2]==pytest.approx(expected)


def test_cropped_outline_and_spacing(case):
    from minibeam.geometry.patient import patient_parameters
    ct,_,cst,_=air_case(case)
    transport,dose,meta=resolve_grids(ct,CROP,(2.,2.,2.),cst)
    np.testing.assert_allclose(dose.resolution_vector,[2.,2.,27/14])
    text=patient_parameters(ct,water=False,num_threads=1,world_half=1200,enable_opengl=True,
        transport_ct=transport,crop_metadata=meta)
    assert 'CTOutline/HLX = 30' in text
    assert 'TransportOutline/HLX = 21' in text
    assert 'TransportOutline/TransY = 3' in text
    assert 'TransportOutline/Material' not in text


def test_patient_workflow_crop(case,dicom_series,tmp_path,monkeypatch):
    import patient_workflow as w
    from test_workflows import setup_workflow
    ct,plan,cst,stf=air_case(case)
    setup_workflow(monkeypatch,(ct,plan,cst,stf,None),tmp_path, water=False)
    monkeypatch.setattr(w,'ENABLE_COLLIMATOR',False)
    monkeypatch.setattr(w,'CT_CROP_VOXELS',CROP)
    monkeypatch.setattr(w,'DOSE_SPACING_MM',(1.5,1.5,1.5))
    monkeypatch.setattr(w,'DICOM_DIR',str(dicom_series(ct.cube_hu)))
    root=tmp_path/'patient-crop'
    w.prepare(root)
    manifest=load_manifest(root)
    assert manifest['dose_grid']['dimensions']==[28,28,18]
    data=loadmat(root/'derived/steering.mat',simplify_cells=True)
    assert data['crop_metadata']['retained_ranges']['x'].tolist()==[3,17]


def test_invalid_saved_crop(case,dicom_series,tmp_path):
    import copy
    from minibeam.geometry.cropping import validate_saved_grids
    *_,root=prepare(case,dicom_series,tmp_path)
    m=load_manifest(root)
    for field in ('transport_grid','dose_grid'):
        bad=copy.deepcopy(m);bad[field]['origin'][0]+=1
        with pytest.raises(ValueError):validate_saved_grids(bad)
