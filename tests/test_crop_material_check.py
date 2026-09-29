import json
import numpy as np
import pytest
import SimpleITK as sitk
from pyRadPlan.ct import CT
from minibeam.geometry.cropping import resolve_grids, CropValidationError
from test_cropping import air_case,CROP


def flagged_case(case):
    ct,plan,cst,stf=air_case(case)
    a=sitk.GetArrayFromImage(ct.cube_hu)
    # A sealed shell outside the crop, including a low-HU interior.
    a[0:3,0:3,0:3]=0; a[1,1,1]=-1000
    im=sitk.GetImageFromArray(a);im.CopyInformation(ct.cube_hu)
    ct=CT(cube_hu=im);cst.ct_image=ct
    return ct,plan,cst,stf


def test_enforcement_and_shared_masks(case):
    ct,_,cst,_=flagged_case(case)
    masks={}
    with pytest.raises(CropValidationError) as exc:
        resolve_grids(ct,CROP,cst=cst,diagnostic_masks=masks)
    assert masks['hu_above_threshold'].sum()==26
    assert masks['enclosed_low_hu'].sum()==1
    assert exc.value.audit['material_flagged_voxels']==27
    _,_,audit=resolve_grids(ct,CROP,cst=cst,enforce_protection=False)
    assert audit['protection_bypassed'] and audit['protection_counts']==exc.value.audit['protection_counts']
    with pytest.raises(CropValidationError,match='TARGET'):
        resolve_grids(ct,dict(CROP,x=(9,19)),cst=cst,enforce_protection=False)



@pytest.mark.parametrize('value',[0,1,'False',None])
def test_boolean_required(case,value):
    ct,_,cst,_=flagged_case(case)
    with pytest.raises(ValueError,match='Boolean'):resolve_grids(ct,CROP,cst=cst,enforce_protection=value)


def test_preview_out_of_slice_flags(case,tmp_path):
    from minibeam.workflow.crop_preview import preview_crop
    ct,_,cst,_=flagged_case(case)
    preview_crop(ct,cst,tmp_path/'preview',CROP,target_names=['TARGET'],enforce_protection=False)
    summary=json.loads((tmp_path/'preview-crop-preview/clipping_summary.json').read_text())
    assert summary['validation']=='accepted' and summary['protection_bypassed']
    xy=summary['plot']['planes'][0]
    assert xy['slice_category_counts']['hu_above_threshold']==0
    assert xy['projection_pixel_counts']['hu_above_threshold']==9
    assert xy['projection_pixel_counts']['enclosed_low_hu']==1


def test_saved_bypass_and_fingerprint(case,tmp_path,dicom_series):
    from minibeam import TOPASPhotonEngine
    from minibeam.topas.manifest import load_manifest
    from minibeam.workflow.artifacts import save_artifacts
    from minibeam.workflow.collection import collect_bundle
    from test_results import fake_results
    from scipy.io import loadmat
    ct,plan,cst,stf=flagged_case(case)
    engine=TOPASPhotonEngine(plan,bundle_dir=str(tmp_path/'bundle'),ct_crop_voxels=CROP,
        enforce_ct_crop_protection=False,dicom_dir=str(dicom_series(ct.cube_hu)))
    root=engine.prepare_jobs(ct,cst,stf)
    save_artifacts(ct,cst,plan,stf,root,'prepare',{})
    m=load_manifest(root);assert m['crop_metadata']['protection_bypassed']
    fake_results(root)
    collect_bundle(root,'collect');collect_bundle(root,'forward')
    assert not loadmat(root/'derived/result.mat',simplify_cells=True)['crop_metadata']['enforce_protection']
    assert 'BYPASSED' in (root/'derived/indexing.md').read_text()
    # With no flagged voxels, changing enforcement still invalidates reuse.
    clean,plan,cst,stf=air_case(case)
    clean_engine=TOPASPhotonEngine(plan,water=True,bundle_dir=str(tmp_path/'clean'))
    clean_engine.prepare_jobs(clean,cst,stf)
    clean_engine.enforce_ct_crop_protection=False
    with pytest.raises(ValueError,match='different inputs/settings'):clean_engine.prepare_jobs(clean,cst,stf)


@pytest.mark.topas
def test_bypass_transport(case,dicom_series,tmp_path):
    import os,subprocess
    from minibeam import TOPASPhotonEngine
    from minibeam.topas.manifest import load_manifest
    executable=os.environ.get('PHOTON_TPS_TOPAS')
    if not executable:pytest.skip('Set PHOTON_TPS_TOPAS')
    ct,plan,cst,stf=flagged_case(case)
    engine=TOPASPhotonEngine(plan,histories=2000,ct_crop_voxels=CROP,
        enforce_ct_crop_protection=False,dicom_dir=str(dicom_series(ct.cube_hu)),bundle_dir=str(tmp_path/'transport'))
    root=engine.prepare_jobs(ct,cst,stf)
    for job in load_manifest(root)['jobs']:
        run=subprocess.run([executable,job['parameter_file']],cwd=root,capture_output=True,text=True)
        assert run.returncode==0,run.stdout+run.stderr
    assert engine.collect_results(root).physical_dose.flat[0].nnz>0


def test_non_target_roi_bypassed(case):
    from test_crop_permissions import couch_case
    ct,_,cst,_=couch_case(case)
    _,_,audit=resolve_grids(ct,CROP,cst=cst,enforce_protection=False)
    assert any(r['name']=='CouchSurface' and r['disposition']=='bypassed' for r in audit['roi_clipping'])
