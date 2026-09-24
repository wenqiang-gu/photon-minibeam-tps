import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/photon-minibeam-mpl')
os.environ.setdefault('PYRADPLAN_XP_PREFER_GPU', 'false')
import numpy as np
import SimpleITK as sitk
import pytest
from pyRadPlan import PhotonPlan, generate_stf
from pyRadPlan.ct import CT
from pyRadPlan.cst import StructureSet, create_voi
from pyRadPlan.machines import PhotonLINAC
from minibeam import TOPASPhotonEngine

@pytest.fixture
def case(tmp_path):
    image = sitk.GetImageFromArray(np.zeros((12,20,20), dtype=np.float32))
    image.SetSpacing((3.,3.,3.)); image.SetOrigin((-28.5,-28.5,-16.5))
    ct = CT(cube_hu=image)
    mask = np.zeros((12,20,20),dtype=np.uint8); mask[5:8,8:12,7:10]=1
    cst = StructureSet(ct_image=ct, vois=[create_voi(name='TARGET', grid=ct.grid, mask=mask, voi_type='TARGET')])
    machine = PhotonLINAC(version=2,name='IdealMonoPhoton',energies=np.array([1.25]),sad=1000.,scd=500.)
    data = machine.model_dump(exclude_none=True,exclude_computed_fields=True)
    data['meta']={'radiation_mode':'photons'}
    plan = PhotonPlan(machine=data)
    plan.prop_stf={'generator':'photonSingleBixel','energy':1.25,'gantry_angles':[0.,90.], 'couch_angles':[0.,15.], 'bixel_width':5.,'add_margin':False}
    plan.prop_dose_calc={'engine':'TOPASPhoton','bundle_dir':str(tmp_path/'bundle'),'histories':10,'geometry_config':False}
    stf = generate_stf(ct,cst,plan)
    return ct,plan,cst,stf,TOPASPhotonEngine(plan,water=True)


@pytest.fixture
def dicom_series(tmp_path):
    """Create original CT fixtures; production code never synthesizes DICOM."""
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

    def write(image):
        root=tmp_path/'dicom';root.mkdir(exist_ok=True)
        array=sitk.GetArrayFromImage(image)
        study,series,frame=generate_uid(),generate_uid(),generate_uid()
        for z,pixels in enumerate(array):
            meta=FileMetaDataset();meta.TransferSyntaxUID=ExplicitVRLittleEndian
            meta.MediaStorageSOPClassUID=CTImageStorage;meta.MediaStorageSOPInstanceUID=generate_uid()
            ds=FileDataset(str(root/f'{z:03d}.dcm'),{},file_meta=meta,preamble=b'\0'*128)
            ds.SOPClassUID=CTImageStorage;ds.SOPInstanceUID=meta.MediaStorageSOPInstanceUID
            ds.Modality='CT';ds.StudyInstanceUID=study;ds.SeriesInstanceUID=series;ds.FrameOfReferenceUID=frame
            ds.PatientName='Synthetic';ds.PatientID='Synthetic';ds.PatientPosition='HFS'
            ds.SeriesNumber=1;ds.InstanceNumber=z+1
            ds.ImageOrientationPatient=[1.,0.,0.,0.,1.,0.]
            ds.ImagePositionPatient=list(image.TransformIndexToPhysicalPoint((0,0,z)))
            ds.PixelSpacing=list(image.GetSpacing()[:2][::-1]);ds.SliceThickness=image.GetSpacing()[2]
            ds.Rows=pixels.shape[0];ds.Columns=pixels.shape[1]
            ds.SamplesPerPixel=1;ds.PhotometricInterpretation='MONOCHROME2'
            ds.BitsAllocated=16;ds.BitsStored=16;ds.HighBit=15;ds.PixelRepresentation=1
            ds.RescaleSlope=1;ds.RescaleIntercept=-1000;ds.RescaleType='HU'
            ds.PixelData=(pixels+1000).astype('<i2').tobytes()
            ds.save_as(ds.filename,enforce_file_format=True)
        return root
    return write


@pytest.fixture(autouse=True)
def reference_hardware_settings(request, monkeypatch):
    """Hardware regressions use a frozen reference, never editable patient settings."""
    if request.module.__name__.split('.')[-1] not in {
            'test_hardware', 'test_collision_diagnostics', 'test_patient_report', 'test_workflows'}:
        return
    from pathlib import Path
    from minibeam.geometry.configuration import GeometryConfig
    reference=Path(__file__).parent/'fixtures/reference_geometry.toml'
    original=GeometryConfig.load.__func__
    monkeypatch.setattr(GeometryConfig,'load',classmethod(
        lambda cls,path=None: original(cls,reference if path is None else path)))
