"""Batch invariants: fixed replay, saved-only append, averaging, portable submission."""
import ast
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
import scipy.sparse as sp
import SimpleITK as sitk
from scipy.io import loadmat
from projects import add_batches as batches
from minibeam.topas import results
from minibeam.topas.manifest import load_manifest
from minibeam.workflow.artifacts import save_planning_snapshot
from minibeam.workflow.collection import collect_bundle
from test_results import fake_results


def prepared(case):
    ct, plan, cst, stf, engine = case
    root = engine.prepare_jobs(ct, cst, stf)
    save_planning_snapshot(ct,cst,plan,stf,root,{})
    fake_results(root)
    return root


def batch_results(root, record, factor):
    manifest = load_manifest(root)
    for original, job in zip(manifest['jobs'],record['jobs']):
        lines = (root/original['output']).read_text().replace(original['scorer'],job['scorer']).splitlines()
        for i,line in enumerate(lines):
            if line.startswith('#'): continue
            row = line.split(',')
            for k in (3,4,6): row[k] = str(float(row[k])*factor)
            lines[i] = ','.join(row)
        (root/job['output']).write_text('\n'.join(lines)+'\n')


def test_append_and_average(case, tmp_path):
    root = prepared(case)
    before = {str(p.relative_to(root)): batches.sha256(p) for p in root.rglob('*') if p.is_file()}
    batches.add(root, 2, dry_run=True)
    assert not (root/'batches.json').exists()
    batches.add(root, 2)
    for name,h in before.items(): assert batches.sha256(root/name)==h
    manifest=load_manifest(root)
    records=batches.inventory(root,manifest)
    assert len(records)==3
    seeds=[j['seed'] for b in records for j in b['jobs']]
    assert len(set(seeds))==len(seeds)
    for b,factor in zip(records[1:],[3,5]): batch_results(root,b,factor)
    baseline=results.collect_results(root/'batches/batch_001', 10**8).physical_dose.flat[0].toarray()/3
    merged=results.collect_results(root,10**8).physical_dose.flat[0].toarray()
    np.testing.assert_allclose(merged,baseline*3)
    variance=sp.load_npz(root/'derived/variance_of_mean.npz').toarray()
    np.testing.assert_allclose(variance,baseline**2*4/3)
    forward=results.collect_forward([2,7],root)
    np.testing.assert_allclose(sitk.GetArrayFromImage(forward['physical_dose']).ravel(),merged@[2,7])
    assert 'physical_dose_std_error' not in forward
    collect_bundle(root,'collect')
    saved=loadmat(root/'derived/result.mat',simplify_cells=True)
    assert saved['dij']['physicalDose'].shape[1]==2
    assert json.loads(saved['batch_metadata_json'])['number_averaged']==3
    assert 'fixed saved replay' in (root/'derived/indexing.md').read_text()
    # Original inputs not needed, paths remain portable.
    moved=tmp_path/'relocated';shutil.move(root,moved)
    batches.add(moved,1)
    collect_bundle(moved/'batches/batch_001','collect')
    assert (moved/'batches/batch_001/derived/result.mat').exists()
    np.testing.assert_allclose(results.collect_results(moved,10**8).physical_dose.flat[0].toarray(), merged)


def test_missing_and_malformed_partial(case):
    root=prepared(case);batches.add(root,2)
    records=batches.inventory(root,load_manifest(root))
    batch_results(root,records[1],2)
    batch_results(root,records[2],4)
    (root/records[2]['jobs'][1]['output']).unlink()
    _,_,m=results.collection_context(root)
    assert m['batch_metadata']['included_batches']==['batch_000','batch_001']
    assert m['batch_metadata']['excluded_batches'][0]['batch_id']=='batch_002'
    (root/records[2]['jobs'][0]['output']).write_text('broken\n')
    with pytest.raises(ValueError): results.collect_results(root,10**8)


def test_seed_determinism_lock_integrity(case,tmp_path):
    root=prepared(case);other=tmp_path/'copy';shutil.copytree(root,other)
    batches.add(root,1);batches.add(root,2);batches.add(other,3)
    assert (root/'batches.json').read_bytes()==(other/'batches.json').read_bytes()
    (root/'.batch-add.lock').touch()
    with pytest.raises(ValueError,match='locked'):batches.add(root,1)
    (root/'.batch-add.lock').unlink()
    (root/'batches/batch_001/jobs/bixel_000001.txt').write_text('corrupt')
    with pytest.raises(ValueError,match='parameters differ'):batches.add(root,1)


@pytest.mark.parametrize('count',[0,-1,True,1.2])
def test_count_validation(case,count):
    root=prepared(case)
    with pytest.raises(ValueError,match='positive integer'): batches.add(root,count)


def test_initial_engine_and_native(case):
    ct,plan,cst,stf,engine=case
    engine.number_of_batches=3
    root=engine.prepare_jobs(ct,cst,stf)
    assert len(batches.inventory(root,load_manifest(root)))==3
    fake_results(root)
    for b in batches.inventory(root,load_manifest(root))[1:]:batch_results(root,b,2)
    dij=engine._calc_dose(ct,cst,stf)
    assert dij.physical_dose.flat[0].shape[1]==2


def test_cli_study_dry_run_and_slurm(case,tmp_path):
    root=prepared(case); study=tmp_path/'study';study.mkdir()
    shutil.move(root,study/'shift_000');root=study/'shift_000'
    (study/'study.json').write_text(json.dumps({'setups':[{'directory':'shift_000'}]}))
    script=Path(__file__).parents[1]/'projects/add_batches.py'
    p=subprocess.run([sys.executable,str(script),str(study),'--count','2','--dry-run'],capture_output=True,text=True)
    assert p.returncode==0,p.stderr
    assert not (root/'batches.json').exists()
    batches.main([str(study),'--count','2'])
    from projects.submit_topas import discover
    tasks=discover(study)
    assert len(tasks)==6 and len({(t['batch_id'],t['job_id']) for t in tasks})==6
    one=discover(root/'batches/batch_002')
    assert len(one)==2
    assert all(t['bundle']==str(root) for t in one)
    ast.parse(script.read_text(),feature_version=(3,8))


def test_saved_forward_batch_plot(case):
    root=prepared(case);batches.add(root,1)
    batch_results(root,batches.inventory(root,load_manifest(root))[1],2)
    collect_bundle(root,'forward',weight_per_bixel=2)
    metadata=json.loads((root/'derived/forward_metadata.json').read_text())
    assert metadata['batch_metadata']['number_averaged']==2
    assert metadata['dose_plot']['status']=='complete'
    assert metadata['dose_plot']['beams'][0]['batch_metadata']['number_averaged']==2

@pytest.mark.parametrize('source', ['point','phase_space'])
@pytest.mark.parametrize('execution', ['separate','combined'])
def test_modes_and_saved_source_removal(case,tmp_path,source,execution):
    from minibeam import TOPASPhotonEngine
    from test_phase_space import make_iaea
    ct,plan,cst,stf,_=case
    base=make_iaea(tmp_path/'source')
    config={'type':'point'} if source=='point' else {'type':'phase_space','file_base':str(base)}
    engine=TOPASPhotonEngine(plan, water=True, source_config=config, beamlet_execution=execution, number_of_batches=2)
    root=engine.prepare_jobs(ct,cst,stf)
    for suffix in ('.phsp','.header'):Path(str(base)+suffix).unlink()
    fake_results(root)
    record=batches.inventory(root,load_manifest(root))[1]
    batch_results(root,record,3)
    batches.add(root,1)
    m=engine.collect_results(root).physical_dose.flat[0]
    np.testing.assert_allclose(m.toarray()[0],[2e-10,4e-10])
    assert m.shape[1]==2
    jobs=load_manifest(root)['jobs']
    for job in jobs:
        assert 'PhaseSpaceMultipleUse = 1' in (root/job['parameter_file']).read_text()


@pytest.mark.parametrize('module_name', ['patient_workflow','water_phantom'])
def test_workflow_settings(module_name,monkeypatch,tmp_path):
    import importlib
    module=importlib.import_module(module_name)
    monkeypatch.setattr(module,'NUMBER_OF_BATCHES',4)
    monkeypatch.setattr(module,'RANDOM_SEED',9876)
    settings=module.planning_settings()
    assert settings.number_of_batches==4 and settings.random_seed==9876
    from minibeam.workflow.planning import topas_settings
    for bad in (True,0,-1,1.5):
        with pytest.raises(ValueError):
            topas_settings(project_dir=tmp_path,source_type='point',phase_space_file_bases=[],
                histories=10,num_threads=1,execution='combined',water=True,enable_opengl=False,
                dose_spacing_mm=None,number_of_batches=bad)


@pytest.mark.topas
@pytest.mark.parametrize('source',['point','phase_space'])
def test_local_repeated_replay(case,tmp_path,source):
    import os
    from minibeam import TOPASPhotonEngine
    from test_phase_space import make_iaea
    exe=os.environ.get('PHOTON_TPS_TOPAS')
    if not exe:pytest.skip('Set PHOTON_TPS_TOPAS')
    ct,plan,cst,stf,_=case
    base=make_iaea(tmp_path/'source',original=100,records=[(1,-1.,0.,0.,0.,0.,1.,1,0)])
    config={'type':'point'} if source=='point' else {'type':'phase_space','file_base':str(base)}
    engine=TOPASPhotonEngine(plan,water=True,histories=100,source_config=config,number_of_batches=2)
    root=engine.prepare_jobs(ct,cst,stf)
    records=batches.inventory(root,load_manifest(root))
    for record in records:
        for job in record['jobs']:
            p=subprocess.run([exe,job['parameter_file']],cwd=root,capture_output=True,text=True)
            assert p.returncode==0,p.stdout+p.stderr
    matrix=engine.collect_results(root).physical_dose.flat[0]
    assert matrix.shape[1]==2
    dose=engine.collect_forward([1,1],root)['physical_dose']
    np.testing.assert_allclose(matrix@[1,1],sitk.GetArrayFromImage(dose).ravel())


def test_batch_warnings_and_original_batch_selection(case,capsys):
    from test_scorer_warnings import inject
    root=prepared(case);batches.add(root,1)
    records=batches.inventory(root,load_manifest(root))
    batch_results(root,records[1],2)
    inject(root/records[1]['jobs'][0]['output'])
    capsys.readouterr()
    collect_bundle(root,'forward')
    output=capsys.readouterr().out
    assert output.count('Scorer warning for ')==1
    m=json.loads((root/'derived/forward_metadata.json').read_text())
    assert m['review_required']
    assert m['scorer_warnings'][0]['job_id'].startswith('batch_001/')
    assert len(m['dose_plot']['beams'][0]['scorer_warnings'])==1
    # Batch zero is addressable without collecting the other batch.
    collect_bundle(root/'batches/batch_000','collect')
    m=json.loads((root/'batches/batch_000/derived/collect_metadata.json').read_text())
    assert m['batch_metadata']['included_batches']==['batch_000']


def test_rollback_on_index_publication_failure(case,monkeypatch):
    root=prepared(case)
    original=batches.atomic_json
    def fail(path,value):
        if Path(path).name=='batches.json':raise OSError('synthetic publication failure')
        return original(path,value)
    monkeypatch.setattr(batches,'atomic_json',fail)
    with pytest.raises(OSError,match='publication'):batches.add(root,2)
    assert not (root/'batches.json').exists()
    assert not (root/'.batch-add.lock').exists()
    assert not list((root/'batches').iterdir())


def test_default_shell_and_override(case,tmp_path):
    root=prepared(case)
    shell=Path(__file__).parents[1]/'projects/add_batches.sh'
    for arguments,expected in [([], '1 existing + 3 additional = 4'),(['--count','2'],'1 existing + 2 additional = 3')]:
        result=subprocess.run(['bash',str(shell),str(root),'--dry-run',*arguments],capture_output=True,text=True)
        assert result.returncode==0,result.stderr
        assert expected in result.stdout
    assert not (root/'batches.json').exists()
