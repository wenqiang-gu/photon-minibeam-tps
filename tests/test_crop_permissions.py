import numpy as np
import pytest
import SimpleITK as sitk
from pyRadPlan.cst import create_voi
from pyRadPlan.ct import CT
from minibeam.geometry.cropping import resolve_grids,CropValidationError
from test_cropping import air_case,CROP


def couch_case(case):
    ct,plan,cst,stf=air_case(case)
    a=sitk.GetArrayFromImage(ct.cube_hu);a[:,18:,:]=0
    image=sitk.GetImageFromArray(a);image.CopyInformation(ct.cube_hu)
    ct=CT(cube_hu=image);cst.ct_image=ct
    mask=np.zeros(a.shape,np.uint8);mask[:,18:,:]=1
    cst.vois.append(create_voi(name='CouchSurface',grid=ct.grid,mask=mask,voi_type='OAR'))
    return ct,plan,cst,stf


def test_non_target_clipping_and_unchanged_cst(case):
    ct,_,cst,_=couch_case(case)
    with pytest.raises(CropValidationError):resolve_grids(ct,CROP,None,cst)
    transport,grid,audit=resolve_grids(ct,CROP,None,cst,False)
    row=next(r for r in audit['roi_clipping'] if r['name']=='CouchSurface')
    assert row['disposition']=='bypassed' and row['removed_fraction']==1
    assert row['clipping_extent']=='complete' and not row['complete_dose_coverage']
    assert cst.ct_image.size==(20,20,12) and transport.size==(14,14,9)


@pytest.mark.parametrize('option', ['ct_crop_allow_clipping_rois', 'enforce_ct_crop_material_check'])
def test_obsolete_engine_options(case,option):
    from minibeam import TOPASPhotonEngine
    with pytest.raises(ValueError,match='Obsolete crop options'):
        TOPASPhotonEngine(**{option: False})


def test_obsolete_helper_options(case):
    ct,_,cst,_=couch_case(case)
    with pytest.raises(ValueError,match='Obsolete crop options'):
        resolve_grids(ct,CROP,cst=cst,allow_clipping_rois=['CouchSurface'])


def test_target_clipping_always_blocks(case):
    ct,_,cst,_=couch_case(case)
    for enforce in (False,True):
        with pytest.raises(CropValidationError,match='TARGET'):
            resolve_grids(ct,dict(CROP,x=(9,19)),cst=cst,enforce_protection=enforce)


def test_preview_on_failure(case,tmp_path):
    import json
    from minibeam.workflow.crop_preview import preview_crop
    ct,_,cst,_=couch_case(case)
    with pytest.raises(CropValidationError):preview_crop(ct,cst,tmp_path/'project',CROP,target_names=['TARGET'])
    folder=tmp_path/'project-crop-preview'
    assert (folder/'crop_preview.png').is_file() and not (tmp_path/'project').exists()
    data=json.loads((folder/'clipping_summary.json').read_text())
    assert data['validation']=='rejected'
    assert data['plot']['planes'][0]['extent_mm']==[-30,30,-30,30]
    assert data['plot']['reference']=='selected target centroid'
    preview_crop(ct,cst,tmp_path/'allowed',CROP,enforce_protection=False)
    data=json.loads((tmp_path/'allowed-crop-preview/clipping_summary.json').read_text())
    assert data['validation']=='accepted' and data['plot']['reference']=='CT center'


def test_saved_allowed_clipping(case,tmp_path,dicom_series):
    from minibeam import TOPASPhotonEngine
    from minibeam.workflow.artifacts import save_artifacts
    from minibeam.workflow.collection import collect_bundle
    from minibeam.topas.manifest import load_manifest
    from test_results import fake_results
    from scipy.io import loadmat
    ct,plan,cst,stf=couch_case(case)
    engine=TOPASPhotonEngine(plan,ct_crop_voxels=CROP,enforce_ct_crop_protection=False,
        dicom_dir=str(dicom_series(ct.cube_hu)),bundle_dir=str(tmp_path/'run'))
    root=engine.prepare_jobs(ct,cst,stf)
    save_artifacts(ct,cst,plan,stf,root,'prepare',{})
    fake_results(root)
    collect_bundle(root,'collect');collect_bundle(root,'forward')
    data=loadmat(root/'derived/result.mat',simplify_cells=True)
    assert data['crop_metadata']['enforce_protection']==False
    assert 'incomplete dose-grid coverage' in (root/'derived/indexing.md').read_text()
    assert load_manifest(root)['crop_metadata']['enforce_protection']==False


@pytest.mark.topas
def test_allowed_couch_transport(case,dicom_series,tmp_path):
    import os,subprocess
    from minibeam import TOPASPhotonEngine
    from minibeam.topas.manifest import load_manifest
    executable=os.environ.get('PHOTON_TPS_TOPAS')
    if not executable:pytest.skip('Set PHOTON_TPS_TOPAS')
    ct,plan,cst,stf=couch_case(case)
    engine=TOPASPhotonEngine(plan,histories=2000,ct_crop_voxels=CROP,
        enforce_ct_crop_protection=False,dose_spacing_mm=(1.5,1.5,1.5),
        dicom_dir=str(dicom_series(ct.cube_hu)),bundle_dir=str(tmp_path/'transport'))
    root=engine.prepare_jobs(ct,cst,stf)
    for job in load_manifest(root)['jobs']:
        run=subprocess.run([executable,job['parameter_file']],cwd=root,capture_output=True,text=True)
        assert run.returncode==0,run.stdout+run.stderr
    assert engine.collect_results(root).physical_dose.flat[0].nnz>0
    assert engine.calc_dose_forward(ct,cst,stf,w=np.ones(2))['physical_dose'].GetSize()==ct.size


def test_inspect_without_target_writes_preview(case,tmp_path,monkeypatch):
    import patient_workflow as w
    ct,_,cst,_=couch_case(case)
    monkeypatch.setattr(w,'load_patient',lambda _: (ct,cst))
    monkeypatch.setattr(w,'read_roi_metadata',lambda *a: {'omitted_rois':[], 'original_dicom_rois':[]})
    monkeypatch.setattr(w,'TARGET',None)
    monkeypatch.setattr(w,'CT_CROP_VOXELS',CROP)
    monkeypatch.setattr(w,'ENFORCE_CT_CROP_PROTECTION',False)
    monkeypatch.setattr(w,'ENABLE_COLLIMATOR',False)
    w.inspect(tmp_path/'inspect')
    assert (tmp_path/'inspect-crop-preview/crop_preview.png').is_file()
    assert not (tmp_path/'inspect').exists()


def test_legacy_policy_collection_after_relocation(case,tmp_path,dicom_series):
    import json,shutil
    from minibeam import TOPASPhotonEngine
    from minibeam.topas.manifest import load_manifest,digest_json
    from minibeam.workflow.artifacts import save_artifacts
    from minibeam.workflow.collection import collect_bundle
    from test_results import fake_results
    ct,plan,cst,stf=couch_case(case)
    engine=TOPASPhotonEngine(plan,ct_crop_voxels=CROP,enforce_ct_crop_protection=False,
        dicom_dir=str(dicom_series(ct.cube_hu)),bundle_dir=str(tmp_path/'legacy'))
    root=engine.prepare_jobs(ct,cst,stf)
    m=load_manifest(root);crop=m['crop_metadata']
    crop.pop('protection_policy');crop.pop('enforce_protection');crop.pop('protection_bypassed')
    crop.update(allow_clipping_rois=['CouchSurface'],enforce_material_check=False,
                material_check_bypassed=True,unauthorized_material_voxels=crop.pop('material_flagged_voxels'))
    for row in crop['roi_clipping']:
        row['authorized']=row['name']=='CouchSurface'
        for key in ('blocking','disposition','clipping_extent','is_target'):row.pop(key)
    m.pop('bundle_id');m['bundle_id']=digest_json(m)
    (root/'manifest.json').write_text(json.dumps(m))
    save_artifacts(ct,cst,plan,stf,root,'prepare',{})
    fake_results(root)
    shutil.rmtree(engine.dicom_dir)
    moved=tmp_path/'relocated';shutil.move(root,moved)
    for stage in ('collect','forward'):
        collect_bundle(moved,stage)
        report=(moved/'derived/indexing.md').read_text()
        assert 'Legacy material protection' in report
        assert 'legacy target and unlisted-ROI checks remain enabled' in report
        assert 'CouchSurface' in report and '100.00%' in report
    # Native consumption compares saved grids without regenerating legacy policy.
    engine.bundle_dir=str(moved)
    assert engine._ready(ct,cst,stf)==moved.resolve()


def test_partial_clipping_counts(case):
    ct,_,cst,_=couch_case(case)
    _,_,audit=resolve_grids(ct,dict(CROP,x=(3,18),y=(4,19)),cst=cst,enforce_protection=False)
    row=next(r for r in audit['roi_clipping'] if r['name']=='CouchSurface')
    assert row['total_voxels']==480 and row['removed_voxels']==345
    assert row['clipping_extent']=='partial' and row['removed_fraction']==345/480
