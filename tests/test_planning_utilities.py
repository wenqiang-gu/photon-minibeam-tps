"""Shared settings preserve native machine and engine conventions."""
import numpy as np
import pytest
from pyRadPlan.machines import PhotonLINAC
from minibeam.workflow.planning import photon_machine, topas_settings, positive_number, positive_integer, boolean
import patient_workflow as patient


@pytest.mark.parametrize('sad',[800.,1000.,1100.])
def test_machine_equivalence(sad):
    original=PhotonLINAC(version=2,name='IdealSpectrumPhoton',energies=np.array([6.]),sad=sad,scd=sad/2)
    expected=original.model_dump(exclude_none=True,exclude_computed_fields=True)
    expected['meta']={'radiation_mode':'photons'}
    actual=photon_machine(sad)
    assert actual.keys()==expected.keys()
    for key in actual:
        if isinstance(actual[key],np.ndarray):np.testing.assert_array_equal(actual[key],expected[key])
        else:assert actual[key]==expected[key]


@pytest.mark.parametrize('source',['point','phase_space'])
@pytest.mark.parametrize('execution',['separate','combined'])
@pytest.mark.parametrize('water',[False,True])
def test_settings_equivalence(source,execution,water):
    bases=['one','two']
    expected=dict(engine='TOPASPhoton',bundle_dir='example',histories=100,num_threads=4,
        beamlet_execution=execution,water=water,enable_opengl=False,
        source_config={'type':source},dose_spacing_mm=(2.,2.,2.))
    if source=='phase_space':expected['source_config'].update(file_bases=['one','two'], selection='field')
    actual=topas_settings(project_dir='example',source_type=source,phase_space_file_bases=bases,
        histories=100,num_threads=4,execution=execution,water=water,enable_opengl=False,dose_spacing_mm=(2.,2.,2.))
    assert actual==expected
    bases.append('three')
    assert actual==expected


@pytest.mark.parametrize('fn,value',[(positive_number,True),(positive_number,np.nan),
    (positive_number,0),(positive_integer,True),(positive_integer,1.5),(boolean,1)])
def test_validation(fn,value):
    with pytest.raises(ValueError):fn('setting',value)


@pytest.mark.parametrize('central',[False,True])
@pytest.mark.parametrize('execution',['separate','combined'])
def test_patient_is_always_dicom(monkeypatch,central,execution):
    monkeypatch.setattr(patient,'ONLY_CENTRAL_BEAMLET',central)
    monkeypatch.setattr(patient,'BEAMLET_EXECUTION',execution)
    plan=patient.patient.configure_plan(patient.planning_settings(), project_dir='example')
    assert not hasattr(patient,'WATER')
    assert plan.prop_dose_calc['water'] is False
    assert plan.prop_dose_calc['dicom_dir']==str(patient.DICOM_DIR)
    assert plan.prop_stf['generator']==('photonSingleBixel' if central else 'photonIMRT')


def test_water_generated_parameters_equivalent(monkeypatch,tmp_path):
    import water_phantom as water
    from test_water_workflow import small_water
    from minibeam.topas.manifest import load_manifest
    small_water(monkeypatch)
    water.prepare(tmp_path/'shared')
    # Previous inline settings construction, independent of the new helper.
    def inline_settings(**k):
        source={'type':k['source_type']}
        if k['source_type']=='phase_space':source['file_bases']=k['phase_space_file_bases']
        settings=dict(engine='TOPASPhoton',bundle_dir=str(k['project_dir']),water=k['water'],
            beamlet_execution=k['execution'],source_config=source,histories=k['histories'],
            num_threads=k['num_threads'],enable_opengl=k['enable_opengl'])
        if k['dose_spacing_mm'] is not None:settings['dose_spacing_mm']=k['dose_spacing_mm']
        return settings
    monkeypatch.setattr(water.water,'topas_settings',inline_settings)
    water.prepare(tmp_path/'inline')
    a,b=tmp_path/'shared',tmp_path/'inline'
    for name in ['inputs/common.txt','inputs/devices.txt','jobs/beam_000001.txt']:
        assert (a/name).read_text()==(b/name).read_text()
    assert load_manifest(a)['jobs']==load_manifest(b)['jobs']
