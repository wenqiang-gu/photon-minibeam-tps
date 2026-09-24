import os
import pytest
from pyRadPlan import load_patient

@pytest.mark.patient
def test_native_patient_import():
    path=os.environ.get('PHOTON_TPS_DICOM')
    if not path: pytest.skip('Set PHOTON_TPS_DICOM to test local patient data')
    ct,cst=load_patient(path)
    assert tuple(ct.size)==(341,341,160)
    assert len(cst.vois)==51
