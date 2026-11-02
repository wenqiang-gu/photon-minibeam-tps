"""PreCheck changes startup validation only, not replay physics or normalization."""
from dataclasses import replace
import os
import subprocess
import numpy as np
import pytest
import patient_workflow as patient
import water_phantom as water
from minibeam import TOPASPhotonEngine
from minibeam.topas.manifest import load_manifest
from minibeam.workflow.planning import topas_settings
from projects import add_batches
from test_phase_space import make_iaea


@pytest.mark.parametrize('enabled',[False,True])
@pytest.mark.parametrize('script',[patient,water])
def test_workflow_settings(script,enabled,monkeypatch,tmp_path,capsys):
    monkeypatch.setattr(script,'TOPAS_PHASE_SPACE_PRECHECK',enabled)
    settings=script.planning_settings()
    assert settings.topas_phase_space_precheck is enabled
    if script is patient:
        plan=patient.patient.configure_plan(settings,project_dir=tmp_path)
    else:
        from test_water_workflow import small_water
        small_water(monkeypatch)
        settings=water.planning_settings()
        _,_,_,plan,_,_=water.water.build_planning(settings,tmp_path)
    assert plan.prop_dose_calc['phase_space_precheck'] is enabled
    assert f'PreCheck: {enabled}' in capsys.readouterr().out


@pytest.mark.parametrize('bad',[0,1,'False',None,np.bool_(False)])
def test_reject_nonboolean(case,bad):
    with pytest.raises(ValueError,match='phase_space_precheck'):
        TOPASPhotonEngine(case[1],phase_space_precheck=bad)
    with pytest.raises(ValueError,match='TOPAS_PHASE_SPACE_PRECHECK'):
        topas_settings(project_dir='unused',source_type='point',phase_space_file_bases=[],
            histories=10,num_threads=1,execution='combined',water=True,enable_opengl=False,
            dose_spacing_mm=None,phase_space_precheck=bad)


@pytest.mark.parametrize('source',['point','phase_space'])
@pytest.mark.parametrize('execution',['separate','combined'])
def test_generated_parameters_and_batch_inheritance(case,tmp_path,source,execution):
    ct,plan,cst,stf,_=case
    source_config={'type':source}
    if source=='phase_space':source_config['file_base']=str(make_iaea(tmp_path/'source'))
    manifests=[]
    for enabled in (True,False):
        engine=TOPASPhotonEngine(plan,water=True,source_config=source_config,
            phase_space_precheck=enabled,beamlet_execution=execution)
        root=engine.prepare_jobs(ct,cst,stf,bundle_dir=tmp_path/str(enabled))
        m=load_manifest(root);manifests.append(m)
        assert m['phase_space_precheck'] is enabled
        for job in m['jobs']:
            text=(root/job['parameter_file']).read_text()
            assert f'b:So/Beam/PhaseSpacePreCheck = "{enabled}"' in text
            assert 'b:So/Beam/PhaseSpaceIncludeEmptyHistories = "False"' in text
        add_batches.add(root,1)
        record=add_batches.inventory(root,m)[1]
        for job in record['jobs']:
            assert f'b:So/Beam/PhaseSpacePreCheck = "{enabled}"' in (root/job['parameter_file']).read_text()
    a,b=manifests
    assert a['request_id']!=b['request_id']
    assert a['normalization']==b['normalization']
    for ja,jb in zip(a['jobs'],b['jobs']):
        for key in ('histories','normalization_histories','source','members'):
            assert ja[key]==jb[key]
    for name in a['files']:
        if name.endswith(('.phsp','.header')):assert a['files'][name]==b['files'][name]


@pytest.mark.topas
def test_local_precheck_disabled(case,tmp_path):
    exe=os.environ.get('PHOTON_TPS_TOPAS')
    if not exe:pytest.skip('Set PHOTON_TPS_TOPAS for replay transport')
    ct,plan,cst,stf,_=case
    base=make_iaea(tmp_path/'particles')
    e=TOPASPhotonEngine(plan,water=True,phase_space_precheck=False,
        source_config={'type':'phase_space','file_base':str(base)},beamlet_execution='combined')
    root=e.prepare_jobs(ct,cst,stf)
    for job in load_manifest(root)['jobs']:
        r=subprocess.run([exe,job['parameter_file']],cwd=root,capture_output=True,text=True,timeout=120)
        assert r.returncode==0,r.stdout+r.stderr
        assert 'performing PreCheck' not in r.stdout
    assert e.collect_results(root).physical_dose.flat[0].nnz>0


def test_disabled_saved_collection_and_native_apis(case,tmp_path,monkeypatch):
    from minibeam.workflow.artifacts import save_planning_snapshot
    from minibeam.workflow.collection import collect_bundle
    from test_results import fake_results
    ct,plan,cst,stf,_=case
    plan.prop_dose_calc['phase_space_precheck']=False
    e=TOPASPhotonEngine(plan,water=True)
    root=e.prepare_jobs(ct,cst,stf)
    save_planning_snapshot(ct,cst,plan,stf,root,{})
    fake_results(root)
    def forbidden(*a,**kw):pytest.fail('Saved results invoked preparation')
    monkeypatch.setattr(TOPASPhotonEngine,'prepare_jobs',forbidden)
    assert e.collect_results(root).physical_dose.flat[0].nnz>0
    for stage in ('collect','forward'):
        collect_bundle(root,stage,weight_per_bixel=1.)
    assert (root/'derived/result.mat').is_file()
    assert (root/'derived/dose.mha').is_file()
