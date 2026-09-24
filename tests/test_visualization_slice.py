import numpy as np
import pytest
from minibeam.geometry.patient import axial_slice_for_isocenter
from minibeam.topas.manifest import load_manifest


@pytest.mark.parametrize('coordinate,expected', [(0,1),(1,2),(5.49,6),(5.5,7),(11,12),(-20,1),(30,12)])
def test_native_slice_index(case,coordinate,expected):
    ct=case[0]
    z=ct.origin[2]+coordinate*ct.grid.resolution_vector[2]
    record=axial_slice_for_isocenter(ct,[0,0,z])
    assert record['slice_index_1based']==expected
    assert record['slice_center_lps_mm']==pytest.approx(ct.origin[2]+(expected-1)*ct.grid.resolution_vector[2])


@pytest.mark.parametrize('enabled,water', [(True,False),(False,False),(True,True)])
def test_per_job_slice_display(case,dicom_series,enabled,water):
    ct,_,cst,stf,engine=case
    engine.water=water
    engine.enable_opengl=enabled
    if not water:engine.dicom_dir=str(dicom_series(ct.cube_hu))
    for index,beam in enumerate(stf.beams):
        iso=np.array(beam.iso_center,copy=True)
        iso[2]=ct.origin[2]+(2+index)*ct.grid.resolution_vector[2]
        beam.iso_center=iso
    root=engine.prepare_jobs(ct,cst,stf)
    manifest=load_manifest(root)
    common=(root/'inputs/common.txt').read_text()
    assert ('s:Ge/CTOutline/Type = "TsBox"' in common) == (enabled and not water)
    if enabled and not water:
        assert 'd:Ge/CTOutline/TransZ = 0 mm' in common
        assert 'd:Ge/CTOutline/HLZ = 18 mm' in common
    for index,job in enumerate(manifest['jobs']):
        text=(root/job['parameter_file']).read_text()
        if enabled and not water:
            assert f'iv:Ge/Patient/ShowSpecificSlicesZ = 1 {3+index}' in text
            assert job['visualization_slice']['slice_index_1based']==3+index
            assert 'ShowOnlyOutlineIfVoxelCountExceeds = 2147483647' in text
        else:
            assert 'ShowSpecificSlices' not in text
            assert 'visualization_slice' not in job
        assert 'RestrictVoxels' not in text
    assert manifest['ct_grid']==manifest['dose_grid']


def test_outline_matches_asymmetric_ct():
    import SimpleITK as sitk
    from pyRadPlan.ct import CT
    from minibeam.geometry.patient import patient_parameters
    from minibeam.geometry.coordinates import patient_center
    image=sitk.GetImageFromArray(np.zeros((7,9,13),dtype=np.float32))
    image.SetSpacing((1.2,2.3,3.4));image.SetOrigin((-44.,12.,73.))
    ct=CT(cube_hu=image)
    text=patient_parameters(ct,water=False,num_threads=1,world_half=1500,enable_opengl=True)
    assert 's:Ge/CTOutline/ParallelWorldName = "CTOutlineWorld"' in text
    assert 'b:Ge/CTOutline/IsParallel = "True"' in text
    assert 's:Ge/CTOutline/Color = "Aqua"' in text
    assert 's:Ge/CTOutline/DrawingStyle = "Wireframe"' in text
    for axis,n,spacing,origin,center in zip('XYZ',ct.size,ct.grid.resolution_vector,ct.origin,patient_center(ct)):
        half=n*spacing/2
        assert f'd:Ge/CTOutline/HL{axis} = {half:.12g} mm' in text
        assert f'd:Ge/CTOutline/Trans{axis} = 0 mm' in text
        assert f'd:Ge/CTOutline/Rot{axis} = 0 deg' in text
        assert center-half == pytest.approx(origin-spacing/2)
        assert center+half == pytest.approx(origin+(n-.5)*spacing)
    outline='\n'.join(line for line in text.splitlines() if '/CTOutline/' in line)
    assert 'Material' not in outline and 'Bins' not in outline
    assert 'Sc/' not in text


@pytest.mark.topas
def test_outline_topas_transport(case,dicom_series):
    """Construct the graphics-only parallel box and run transport without a GUI."""
    import os
    import subprocess
    executable=os.environ.get('PHOTON_TPS_TOPAS')
    if not executable:
        pytest.skip('Set PHOTON_TPS_TOPAS for OpenTOPAS outline validation')
    ct,_,cst,stf,engine=case
    engine.water=False
    engine.dicom_dir=str(dicom_series(ct.cube_hu))
    engine.enable_opengl=True
    root=engine.prepare_jobs(ct,cst,stf)
    for job in load_manifest(root)['jobs']:
        # Keep CTOutline and slice parameters intact; disable only the GUI for CI.
        wrapper=root/'batch-validation.txt'
        wrapper.write_text(f'includeFile = {job["parameter_file"]}\n'
                           'b:Gr/Enable = "False"\nb:Ts/UseQt = "False"\n'
                           'b:Ts/PauseBeforeQuit = "False"\n'
                           'b:Ge/CheckForOverlaps = "True"\n')
        process=subprocess.run([executable,str(wrapper)],cwd=root,capture_output=True,text=True,timeout=120)
        output=process.stdout+process.stderr
        assert process.returncode==0,output
        assert 'Creating parallel world: CTOutlineWorld' in output
        assert 'Checking overlaps for volume CTOutline:0 (G4Box) ... OK!' in output
        assert 'Overlap is detected' not in output
    assert engine.collect_results(root).physical_dose.flat[0].shape == (20*20*12,2)
