import shutil
import numpy as np
import pydicom
import pytest
import SimpleITK as sitk
from minibeam import TOPASPhotonEngine
from minibeam.topas.manifest import load_manifest,sha256
from minibeam.geometry.patient import dicom_inputs


def test_original_dicom_bundle(case,dicom_series,tmp_path):
    ct,plan,cst,stf,_=case
    directory=dicom_series(ct.cube_hu)
    engine=TOPASPhotonEngine(plan,dicom_dir=str(directory))
    root=engine.prepare_jobs(ct,cst,stf)
    m=load_manifest(root)
    assert m['patient_component']=='TsDicomPatient'
    assert len(m['dicom']['files'])==ct.size[2]
    for source,target in zip(sorted(directory.glob('*.dcm')), sorted((root/'inputs/dicom').glob('*.dcm'))):
        assert sha256(source)==sha256(target)
    config=(root/'inputs/common.txt').read_text()
    assert 'Type = "TsDicomPatient"' in config
    assert 'DicomDirectory = "inputs/dicom"' in config
    assert 'NumberOfVoxels' not in config and 'VoxelSize' not in config
    assert not list(root.rglob('*.raw'))
    moved=tmp_path/'moved';shutil.copytree(root,moved)
    assert load_manifest(moved)['bundle_id']==m['bundle_id']
    ds=pydicom.dcmread(directory/'000.dcm');ds.SeriesDescription='changed';ds.save_as(directory/'000.dcm')
    with pytest.raises(ValueError,match='different'):
        engine.prepare_jobs(ct,cst,stf)


@pytest.mark.parametrize('change,match', [('duplicate','Duplicate'),('series','exactly one'),
    ('orientation','orientation'),('position','positions'),('spacing','spacing'),('pixels','HU values')])
def test_dicom_mismatch_rejected(case,dicom_series,change,match):
    ct=case[0];directory=dicom_series(ct.cube_hu)
    path=directory/'001.dcm';ds=pydicom.dcmread(path)
    if change=='duplicate':
        shutil.copyfile(directory/'000.dcm',directory/'duplicate.dcm')
    else:
        if change=='series':ds.SeriesInstanceUID=pydicom.uid.generate_uid()
        if change=='orientation':ds.ImageOrientationPatient=[0,1,0,1,0,0]
        if change=='position':ds.ImagePositionPatient[2]+=0.5
        if change=='spacing':ds.PixelSpacing=[7,7]
        if change=='pixels':ds.RescaleIntercept=-500
        ds.save_as(path)
    with pytest.raises(ValueError,match=match):dicom_inputs(directory,ct)


def test_dicom_required(case):
    ct,plan,cst,stf,_=case
    with pytest.raises(ValueError,match='dicom_dir'):
        TOPASPhotonEngine(plan).prepare_jobs(ct,cst,stf)


def test_anisotropic_offset_dicom(dicom_series):
    from pyRadPlan.ct import CT
    array=np.arange(3*5*7,dtype=np.float32).reshape(3,5,7)-30
    image=sitk.GetImageFromArray(array)
    image.SetSpacing((1.2,2.3,4.5));image.SetOrigin((11.,-72.,19.))
    ct=CT(cube_hu=image)
    directory=dicom_series(image)
    files,_=dicom_inputs(directory,ct)
    assert len(files)==3
