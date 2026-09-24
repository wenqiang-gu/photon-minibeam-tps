import sys
from pathlib import Path
import json
import numpy as np
import pytest
import SimpleITK as sitk
from scipy.io import loadmat
from pyRadPlan.cst import create_voi
import patient_workflow as workflow


def setup_workflow(monkeypatch, case, tmp_path, target='TARGET'):
    ct,_,cst,_,_=case
    monkeypatch.setattr(workflow, 'load_patient', lambda _: (ct,cst))
    for key,value in dict(COLLIMATOR_SHIFT_FRACTIONS=[],TARGET=target,GANTRY_ANGLES=[0.,90.],WATER=True,DICOM_DIR=str(tmp_path/'no-dicom'),HISTORIES_PER_JOB=10,ONLY_CENTRAL_BEAMLET=True,BEAMLET_EXECUTION='separate',SOURCE_TYPE='point').items():
        monkeypatch.setattr(workflow,key,value)
    monkeypatch.setattr(sys,'argv',['patient_workflow.py','prepare','--project',str(tmp_path/'workflow')])
    return ct,cst


def test_explicit_target_only_and_native_export(monkeypatch,case,tmp_path):
    ct,cst=setup_workflow(monkeypatch,case,tmp_path)
    mask=np.zeros(ct.size[::-1],dtype=np.uint8);mask[1:3,1:3,1:3]=1
    cst.vois.append(create_voi(name='OTHER',grid=ct.grid,mask=mask,voi_type='TARGET'))
    expected=cst.vois[0].mask
    center=ct.cube_hu.TransformContinuousIndexToPhysicalPoint(tuple(np.argwhere(sitk.GetArrayFromImage(expected)).mean(axis=0)[::-1]))
    workflow.main()
    assert [v.name for v in cst.vois if v.voi_type == 'TARGET']==['TARGET']
    manifest=json.loads((tmp_path/'workflow/manifest.json').read_text())
    np.testing.assert_allclose(manifest['jobs'][0]['source']['aim_lps_mm'],center)
    data=loadmat(tmp_path/'workflow/derived/steering.mat',simplify_cells=True)
    np.testing.assert_array_equal(data['ct']['cubeDim'],ct.size)


@pytest.mark.parametrize('target',['missing','EMPTY'])
def test_target_must_exist_nonempty(monkeypatch,case,tmp_path,target):
    ct,cst=setup_workflow(monkeypatch,case,tmp_path,target)
    cst.vois.append(create_voi(name='EMPTY',grid=ct.grid,mask=np.zeros(ct.size[::-1],dtype=np.uint8),voi_type='OAR'))
    with pytest.raises(ValueError,match='nonempty'): workflow.main()


def test_nonsquare_matrad_rejected(monkeypatch,case,tmp_path):
    setup_workflow(monkeypatch,case,tmp_path)
    monkeypatch.setattr(workflow,'DOSE_SPACING_MM',(3.,6.,3.))
    with pytest.raises(ValueError,match='square transverse'): workflow.main()


def test_roi_metadata_and_omission_report(monkeypatch,case,tmp_path,capsys):
    import pydicom
    from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, RTStructureSetStorage, generate_uid
    setup_workflow(monkeypatch,case,tmp_path)
    directory=tmp_path/'no-dicom';directory.mkdir()
    meta=FileMetaDataset();meta.TransferSyntaxUID=ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID=RTStructureSetStorage;meta.MediaStorageSOPInstanceUID=generate_uid()
    ds=FileDataset(str(directory/'struct.dcm'),{},file_meta=meta,preamble=b'\0'*128)
    ds.Modality='RTSTRUCT';ds.StructureSetROISequence=[]
    for name,number in [('TARGET',42),('OMITTED',99)]:
        roi=Dataset();roi.ROIName=name;roi.ROINumber=number
        ds.StructureSetROISequence.append(roi)
    ds.save_as(directory/'struct.dcm',enforce_file_format=True)
    workflow.main()
    metadata=json.loads((tmp_path/'workflow/derived/metadata.json').read_text())
    assert metadata['original_dicom_rois']==[{'name':'TARGET','number':42},{'name':'OMITTED','number':99}]
    assert metadata['omitted_rois']==[{'name':'OMITTED','number':99}]
    assert 'OMITTED' in capsys.readouterr().out


@pytest.mark.parametrize('source_type', ['point','phase_space'])
def test_globals_used_through_collection_and_forward(monkeypatch,case,tmp_path,source_type):
    from test_results import fake_results
    setup_workflow(monkeypatch,case,tmp_path)
    monkeypatch.setattr(workflow,'SOURCE_TYPE',source_type)
    if source_type == 'phase_space':
        from test_phase_space import make_iaea
        monkeypatch.setattr(workflow,'PHASE_SPACE_FILE_BASE',str(make_iaea(tmp_path/'source')))
    calls=[]
    original=workflow.TOPASPhotonEngine.prepare_jobs
    def counted_prepare(self,*args,**kwargs):
        calls.append(True)
        return original(self,*args,**kwargs)
    monkeypatch.setattr(workflow.TOPASPhotonEngine,'prepare_jobs',counted_prepare)
    workflow.main()
    assert len(calls)==1
    root=tmp_path/'workflow'
    manifest=fake_results(root)
    assert all(job['histories']==workflow.HISTORIES_PER_JOB for job in manifest['jobs'])
    if source_type == 'phase_space':
        for suffix in ('.header', '.phsp'):
            Path(workflow.PHASE_SPACE_FILE_BASE + suffix).unlink()
    for stage in ['collect','forward']:
        calls.clear()
        monkeypatch.setattr(sys,'argv',['patient_workflow.py',stage,'--project',str(root)])
        workflow.main()
        assert len(calls)==0, f'{stage} must not prepare source inputs'
    data=loadmat(root/'derived/result.mat',simplify_cells=True)
    assert data['dij']['physicalDose'].shape[1]==2
    dose=sitk.GetArrayFromImage(sitk.ReadImage(str(root/'derived/dose.mha')))
    # MATLAB's matrix rows use the native serializer's MATLAB voxel order.
    np.testing.assert_allclose(data['dij']['physicalDose']@np.ones(2),dose.transpose(1,2,0).ravel(order='F'))


def test_inspect_uses_global_directory_without_planning(monkeypatch,case,tmp_path):
    setup_workflow(monkeypatch,case,tmp_path)
    seen=[]
    monkeypatch.setattr(workflow,'load_patient',lambda directory: (seen.append(directory) or case[0],case[2]))
    monkeypatch.setattr(workflow,'TARGET',None)
    monkeypatch.setattr(sys,'argv',['patient_workflow.py','inspect'])
    workflow.main()
    assert seen==[workflow.DICOM_DIR]
    assert not (tmp_path/'workflow').exists()


@pytest.mark.parametrize('execution', ['separate', 'combined'])
@pytest.mark.parametrize('only_central,generator',[(False,'photonIMRT'),(True,'photonSingleBixel')])
def test_native_bixel_modes(monkeypatch,case,tmp_path,only_central,generator,execution):
    from pyRadPlan import generate_stf
    ct,cst=setup_workflow(monkeypatch,case,tmp_path)
    monkeypatch.setattr(workflow,'ONLY_CENTRAL_BEAMLET',only_central)
    monkeypatch.setattr(workflow,'BEAMLET_EXECUTION',execution)
    plan=workflow.configure_plan(project_dir=tmp_path/'mode-test')
    assert plan.prop_dose_calc['beamlet_execution']==execution
    assert plan.prop_stf['generator']==generator
    stf=generate_stf(ct,cst,plan)
    if only_central:
        assert all(beam.total_number_of_bixels==1 for beam in stf.beams)
    else:
        assert stf.total_number_of_bixels>len(stf.beams)

    from test_results import fake_results
    workflow.main()
    root=tmp_path/'workflow'
    manifest=fake_results(root)
    expected=len(stf.beams) if execution=='combined' else stf.total_number_of_bixels
    assert len(manifest['jobs'])==expected
    # Saved collection must ignore even an invalid current selection setting.
    monkeypatch.setattr(workflow,'ONLY_CENTRAL_BEAMLET',None)
    for stage in ('collect','forward'):
        workflow.main([stage,'--project',str(root)])
    data=loadmat(root/'derived/result.mat',simplify_cells=True)
    assert data['dij']['physicalDose'].shape[1]==expected


@pytest.mark.parametrize('count',[None,0,1,2,5,-1,1.0,'True',np.bool_(True)])
def test_non_boolean_selection_rejected(monkeypatch,tmp_path,count):
    monkeypatch.setattr(workflow,'ONLY_CENTRAL_BEAMLET',count)
    with pytest.raises(ValueError,match='ONLY_CENTRAL_BEAMLET'):
        workflow.configure_plan(project_dir=tmp_path/'unused')


def test_automatic_job_parameters_and_mat_export(monkeypatch,case,tmp_path):
    setup_workflow(monkeypatch,case,tmp_path)
    monkeypatch.setattr(workflow,'COUCH_ANGLES',[5.,15.])
    workflow.main()
    root=tmp_path/'workflow'
    manifest=json.loads((root/'manifest.json').read_text())
    assert [j['gantry_angle_deg'] for j in manifest['jobs']]==[0.,90.]
    assert [j['couch_angle_deg'] for j in manifest['jobs']]==[5.,15.]
    data=loadmat(root/'derived/steering.mat',simplify_cells=True)
    assert 'plan_metadata' not in data['pln']['propDoseCalc']
    assert 'beam_metadata' not in data['pln']['propDoseCalc']
    assert 'custom' not in manifest['planning']['plan']


@pytest.mark.parametrize('enabled', [False, True])
def test_opengl_configuration(monkeypatch, case, tmp_path, enabled):
    setup_workflow(monkeypatch, case, tmp_path)
    monkeypatch.setattr(workflow, 'ENABLE_OPENGL', enabled)
    workflow.main()
    root=tmp_path/'workflow'
    text=(root/'inputs/common.txt').read_text()
    assert f'b:Gr/Enable = "{enabled}"' in text
    assert f'b:Ts/UseQt = "{enabled}"' in text
    assert f'b:Ts/PauseBeforeQuit = "{enabled}"' in text
    assert ('s:Gr/PatientView/Type = "OpenGL"' in text) == enabled
    assert ('b:Gr/PatientView/IncludeAxes = "True"' in text) == enabled
    assert ('s:Gr/PatientView/AxesComponent = "Patient"' in text) == enabled
    assert ('d:Gr/PatientView/AxesSize = 100 mm' in text) == enabled
    assert json.loads((root/'manifest.json').read_text())['enable_opengl'] == enabled
    monkeypatch.setattr(workflow, 'ENABLE_OPENGL', not enabled)
    with pytest.raises(ValueError, match='different inputs/settings'):
        workflow.main()


@pytest.mark.parametrize('path',['projects/custom','runs/patient-dose','/tmp/patient-project'])
@pytest.mark.parametrize('stage',['prepare','collect','forward'])
def test_project_path_passed_without_prefix(monkeypatch,path,stage):
    calls=[]
    monkeypatch.setattr(workflow,stage,lambda project_dir: calls.append(project_dir))
    workflow.main([stage,'--project',path])
    assert calls==[path]


@pytest.mark.parametrize('stage',['collect','forward'])
def test_project_required_and_old_option_removed(stage):
    with pytest.raises(SystemExit):
        workflow.parse_arguments([stage])
    with pytest.raises(SystemExit):
        workflow.parse_arguments([stage,'--run-dir','runs/old'])
