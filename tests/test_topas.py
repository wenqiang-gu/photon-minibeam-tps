"""Actual transport tests; run with PHOTON_TPS_TOPAS=/path/to/topas."""
import os
import json
from pathlib import Path
import numpy as np
import pytest
import SimpleITK as sitk
from pyRadPlan.ct import CT
from minibeam import TOPASPhotonEngine
import subprocess
import re
from minibeam.geometry.devices import GeometryInclude
from minibeam.topas.manifest import load_manifest


def execute_jobs(root, executable):
    outputs = []
    for job in load_manifest(root)["jobs"]:
        result = subprocess.run([executable, job["parameter_file"]], cwd=root,
                                check=True, capture_output=True, text=True)
        outputs.append(result.stdout)
    return outputs


@pytest.mark.topas
@pytest.mark.parametrize("heterogeneous", [False,True])
def test_actual_transport(case,heterogeneous,dicom_series):
    executable=os.environ.get("PHOTON_TPS_TOPAS")
    if not executable:pytest.skip("Set PHOTON_TPS_TOPAS for transport integration")
    ct,plan,cst,stf,_=case
    if heterogeneous:
        array=sitk.GetArrayFromImage(ct.cube_hu)
        array[:,:,:5]=-700;array[:,:,11:]=500
        image=sitk.GetImageFromArray(array);image.CopyInformation(ct.cube_hu)
        ct=CT(cube_hu=image)
        # Same spatial grid, different transport material.
        cst.ct_image=ct
    engine=TOPASPhotonEngine(plan,water=not heterogeneous,histories=3000,
                             dicom_dir=str(dicom_series(ct.cube_hu)) if heterogeneous else None,
                             dose_spacing_mm=(5.,7.,5.) if heterogeneous else None)
    if heterogeneous:
        engine.geometry=GeometryInclude('b:Ge/Patient/DumpImagingValues = "True"')
    root=engine.prepare_jobs(ct,cst,stf)
    outputs=execute_jobs(root,executable)
    if heterogeneous:
        expected=sitk.GetArrayFromImage(ct.cube_hu).ravel()
        for output in outputs:
            decoded=np.array([int(v) for v in re.findall(r'Pixel: \d+ has value: (-?\d+)',output)])
            np.testing.assert_array_equal(decoded,expected)
    dij=engine.collect_results(root)
    assert dij.physical_dose.flat[0].nnz>0
    for column in range(2):assert dij.physical_dose.flat[0][:,column].sum()>0
    w=np.array([1.e6,2.e6])
    result=engine.collect_forward(w,root)
    np.testing.assert_allclose(dij.physical_dose.flat[0]@w,sitk.GetArrayFromImage(result["physical_dose"]).ravel())
    # Retry only one job after moving aside its partial output.
    output=root/'results/bixel_000001.csv'
    output.rename(root/'partial.csv')
    job=load_manifest(root)['jobs'][0]
    subprocess.run([executable,job['parameter_file']],cwd=root,check=True,
                   stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    assert engine.collect_results(root).physical_dose.flat[0].nnz > 0



@pytest.mark.topas
@pytest.mark.patient
def test_actual_patient_beamlet(case,tmp_path):
    from pyRadPlan import load_patient, generate_stf
    from pyRadPlan.cst import create_voi
    executable=os.environ.get('PHOTON_TPS_TOPAS')
    directory=os.environ.get('PHOTON_TPS_DICOM')
    if not executable or not directory:
        pytest.skip('Set both PHOTON_TPS_TOPAS and PHOTON_TPS_DICOM for patient transport')
    ct,cst=load_patient(directory)
    # Numerical smoke test only; does not choose a target for the user workflow.
    selected=next(v.name for v in cst.vois if v.voi_type=='TARGET' and np.any(sitk.GetArrayViewFromImage(v.mask)))
    for index,voi in enumerate(cst.vois):
        if voi.voi_type=='TARGET' and voi.name!=selected:
            data=voi.model_dump(exclude_computed_fields=True)
            data.update(voi_type='OAR',overlap_priority=5)
            cst.vois[index]=create_voi(data)
    plan=case[1]
    plan.prop_stf.update(gantry_angles=[0.],couch_angles=[0.])
    stf=generate_stf(ct,cst,plan)
    engine=TOPASPhotonEngine(plan,histories=2000,dose_spacing_mm=(10.,10.,10.),dicom_dir=directory)
    root=engine.prepare_jobs(ct,cst,stf,bundle_dir=tmp_path/'patient-smoke')
    execute_jobs(root,executable)
    dij=engine.collect_results(root)
    assert dij.physical_dose.flat[0].shape[1]==1
    assert dij.physical_dose.flat[0].sum()>0


@pytest.mark.topas
def test_point_source_particles(case, tmp_path):
    """Score replayed point-source photons in vacuum to verify spectrum and square fluence."""
    executable = os.environ.get('PHOTON_TPS_TOPAS')
    if not executable:
        pytest.skip('Set PHOTON_TPS_TOPAS for emitted-particle integration')
    from minibeam.sources.empirical import EMPIRICAL_ENERGY_MEV, EMPIRICAL_SPECTRUM_WEIGHT
    from pyRadPlan import generate_stf
    ct, plan, cst, _, engine = case
    # Align with world Z so ASCII's omitted Z direction is reconstructed accurately.
    plan.prop_stf.update(gantry_angles=[90.], couch_angles=[90.])
    stf = generate_stf(ct, cst, plan)
    histories = 50000
    engine.histories = histories
    definition, _, _, assets, _ = engine._definition(ct, stf)
    import shutil
    for name, path in assets.items():
        target = tmp_path/name
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(path,target)
    text = definition['sources'][0] + f'''
s:Ge/World/Material = "Vacuum"
d:Ge/World/HLX = 2000 mm
d:Ge/World/HLY = 2000 mm
d:Ge/World/HLZ = 2000 mm
i:Ts/NumberOfThreads = 1
i:Ts/Seed = 7321
b:Ts/PauseBeforeQuit = "False"
b:Gr/Enable = "False"
i:So/Beam/NumberOfHistoriesInRun = {histories}
s:Ge/Film/Type = "TsBox"
s:Ge/Film/Parent = "Source"
s:Ge/Film/Material = "Vacuum"
d:Ge/Film/HLX = 1 mm
d:Ge/Film/HLY = 1 mm
d:Ge/Film/HLZ = 0.001 mm
d:Ge/Film/TransZ = 10 mm
s:Sc/Particles/Quantity = "PhaseSpace"
s:Sc/Particles/Surface = "Film/ZMinusSurface"
s:Sc/Particles/OutputType = "ASCII"
s:Sc/Particles/OutputFile = "particles"
s:Sc/Particles/IfOutputFileAlreadyExists = "Exit"
'''
    (tmp_path/'particles.txt').write_text(text)
    subprocess.run([executable,'particles.txt'],cwd=tmp_path,check=True,capture_output=True,text=True)
    particles=np.loadtxt(tmp_path/'particles.phsp')
    assert len(particles)==histories
    assert np.all(particles[:,6]==1) and np.all(particles[:,7]==22)
    assert np.all(particles[:,9]==1)
    energies=particles[:,5]
    indices=np.argmin(abs(energies[:,None]-EMPIRICAL_ENERGY_MEV),axis=1)
    np.testing.assert_allclose(energies,EMPIRICAL_ENERGY_MEV[indices],atol=1e-6)
    probabilities=EMPIRICAL_SPECTRUM_WEIGHT/EMPIRICAL_SPECTRUM_WEIGHT.sum()
    counts=np.bincount(indices,minlength=28)
    assert np.all(abs(counts-histories*probabilities)<6*np.sqrt(histories*probabilities*(1-probabilities))+2)
    src=definition['jobs'][0]['source']
    frame=np.asarray(src['local_to_world'])
    z=np.sqrt(np.maximum(0,1-np.sum(particles[:,3:5]**2,axis=1)))
    z=np.where(particles[:,8],-z,z)
    directions=np.column_stack((particles[:,3:5],z)) @ frame
    assert np.all(directions[:,2]>0)
    projected = directions[:,:2]/directions[:,2,None]*stf.beams[0].sad
    width = stf.beams[0].bixel_width
    assert np.all(projected >= -width/2-1e-5) and np.all(projected < width/2+1e-5)
    np.testing.assert_allclose(projected.mean(axis=0),0.,atol=.03)
    np.testing.assert_allclose(projected.var(axis=0),width**2/12,rtol=.03)
    # Back-project each direction from the film to the common source point.
    from minibeam.geometry.coordinates import patient_center
    origin=np.asarray(src['source_lps_mm'])-patient_center(ct)
    positions_mm=(particles[:,:3]*10-origin) @ frame
    np.testing.assert_allclose(positions_mm[:,:2],9.999*directions[:,:2]/directions[:,2,None],atol=1e-5)
