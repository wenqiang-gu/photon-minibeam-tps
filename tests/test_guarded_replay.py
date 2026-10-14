"""Exact replay and original-history statistics, including zero histories."""
import json
import os
from pathlib import Path
import subprocess
import numpy as np
import pytest
from minibeam.sources.iaea import TOPAS_DTYPE, finish_guarded_replay, write_topas_header
from minibeam.topas.scoring import score_rows
from projects.replay_audit import audit_setup, enforce


@pytest.mark.parametrize('scores,H', [([2.],10),([2.,4.],10),([2.,0.,7.],3),([0.,0.],680000000)])
def test_zero_history_variance(tmp_path,scores,H):
    from test_score_chunks import fixture
    path,job,grid,_,write=fixture(tmp_path,single=True)
    N=len(scores);S=sum(scores);sd=np.std(scores,ddof=1) if N>1 else 0
    job.update(histories=N,normalization_histories=H,replay_policy='nonempty_guard_v1')
    write([f'{S}, {S/N}, {N}, {sd}'])
    _,dose,var=list(score_rows(path,job,grid))[0]
    Q=sum(x*x for x in scores)
    assert dose==pytest.approx(S/H)
    assert var==pytest.approx((Q-S*S/H)/H/(H-1))


def tiny_setup(tmp_path, guarded=True):
    (tmp_path/'jobs').mkdir();(tmp_path/'inputs').mkdir()
    a=np.zeros(2,dtype=TOPAS_DTYPE);a['pdg']=22;a['weight']=.05;a['energy']=1;a[0]['new_history']=1
    base=tmp_path/'inputs/source';a.tofile(base.with_suffix('.phsp'))
    replay=finish_guarded_replay(base,100,1,2) if guarded else None
    if not guarded:write_topas_header(base.with_suffix('.header'),100,1,2)
    config='s:So/Beam/PhaseSpaceFileName = "inputs/source"\n'+f'b:So/Beam/PhaseSpaceIncludeEmptyHistories = "{not guarded}"\n'
    config+='b:So/Beam/PhaseSpacePreCheck = "True"\ni:So/Beam/PhaseSpaceMultipleUse = 1\ni:So/Beam/NumberOfHistoriesInRun = '+('1' if guarded else '100')+'\n'
    (tmp_path/'jobs/beam.txt').write_text(config)
    job=dict(job_id='beam',column_index=1,parameter_file='jobs/beam.txt',histories=1,normalization_histories=100)
    if guarded:job.update(replay_policy='nonempty_guard_v1',source=dict(replay=replay))
    return dict(jobs=[job])


def test_audit_trigger_and_guard(tmp_path):
    bad=tmp_path/'bad';bad.mkdir();m=tiny_setup(bad,False)
    assert audit_setup(bad,m)[0]['status']=='known_trigger'
    with pytest.raises(ValueError,match='Unsafe'):enforce(bad,m,write_report=False)
    assert not (tmp_path/'bad-replay-audit').exists()
    with pytest.raises(ValueError,match='Unsafe'):enforce(bad,m)
    assert (tmp_path/'bad-replay-audit/audit.json').exists()
    good=tmp_path/'good';good.mkdir();m=tiny_setup(good)
    assert enforce(good,m)[0]['status']=='guarded'
    a=np.fromfile(good/'inputs/source.phsp',TOPAS_DTYPE);a[-1]['new_history']=0;a.tofile(good/'inputs/source.phsp')
    with pytest.raises(ValueError,match='Unsafe'):enforce(good,m)


@pytest.mark.topas
@pytest.mark.parametrize('threads',[1,2,8])
@pytest.mark.parametrize('N,tail',[(1,1),(1,3),(11,1),(11,3),(10003,1),(10003,3)])
def test_exact_topas_replay(tmp_path,threads,N,tail):
    executable=os.environ.get('PHOTON_TPS_TOPAS')
    if not executable:pytest.skip('Set PHOTON_TPS_TOPAS for exact replay tests')
    count=3*(N-1)+tail
    a=np.zeros(count,dtype=TOPAS_DTYPE);a['energy']=1;a['weight']=.01+np.arange(count)%7*.01
    a['pdg']=22;a['new_history'][::3]=1;a['x']=np.arange(count)/10000
    a.tofile(tmp_path/'source.phsp');finish_guarded_replay(tmp_path/'source',max(N,680000000),N,count)
    (tmp_path/'run.txt').write_text(f'''i:Ts/NumberOfThreads = {threads}
i:Ts/Seed = 12345
b:Ts/PauseBeforeQuit = "False"
s:Ge/World/Material = "G4_Galactic"
d:Ge/World/HLX = 1 m
d:Ge/World/HLY = 1 m
d:Ge/World/HLZ = 1 m
s:Ge/Detector/Parent = "World"
s:Ge/Detector/Type = "TsBox"
s:Ge/Detector/Material = "G4_Galactic"
d:Ge/Detector/HLX = 100 mm
d:Ge/Detector/HLY = 100 mm
d:Ge/Detector/HLZ = 1 mm
d:Ge/Detector/TransZ = 10 mm
s:So/Beam/Type = "PhaseSpace"
s:So/Beam/Component = "World"
s:So/Beam/PhaseSpaceFileName = "source"
b:So/Beam/PhaseSpacePreCheck = "True"
b:So/Beam/PhaseSpaceIncludeEmptyHistories = "False"
i:So/Beam/PhaseSpaceMultipleUse = 1
i:So/Beam/NumberOfHistoriesInRun = {N}
s:Sc/Count/Quantity = "PhaseSpace"
s:Sc/Count/Surface = "Detector/ZPlusSurface"
s:Sc/Count/OutputType = "ASCII"
s:Sc/Count/OutputFile = "crossings"
s:Sc/Count/IfOutputFileAlreadyExists = "Overwrite"
b:Sc/Count/IncludeEventID = "True"
''')
    p=subprocess.run([executable,'run.txt'],cwd=tmp_path,text=True,capture_output=True,timeout=60)
    assert p.returncode==0,p.stdout+p.stderr
    b=np.loadtxt(tmp_path/'crossings.phsp',ndmin=2);ids=np.rint(b[:,0]*10000).astype(int)
    np.testing.assert_array_equal(np.sort(ids),np.arange(count))
    np.testing.assert_allclose(b[:,6],a[ids]['weight'],rtol=1e-6)
    events=b[np.argsort(ids),-1].astype(int);hist=np.arange(count)//3
    assert len(set(events))==N
    assert not np.any((events[1:]!=events[:-1])&(hist[1:]==hist[:-1]))


def test_known_legacy_trigger_blocked_at_entry_points(case,tmp_path):
    from minibeam import TOPASPhotonEngine
    from minibeam.topas import results
    from minibeam.topas.manifest import load_manifest, sha256, digest_json
    from projects import add_batches, submit_topas
    from test_phase_space import make_iaea
    ct,plan,cst,stf,_=case
    base=make_iaea(tmp_path/'original',records=[
        (1,-1.,0.,0.,0.,0.,.05,1,0),(1,1.,0.,0.,0.,0.,.05,0,0)])
    engine=TOPASPhotonEngine(plan,water=True,histories=10,
        source_config={'type':'phase_space','file_base':str(base)})
    root=engine.prepare_jobs(ct,cst,stf);m=load_manifest(root)
    # Reconstruct the old writer's valid, signed legacy representation.
    for file in root.glob('inputs/*.phsp'):
        a=np.fromfile(file,TOPAS_DTYPE);a[:-1].tofile(file)
        write_topas_header(file.with_suffix('.header'),10,1,2)
    for job in m['jobs']:
        job.pop('replay_policy');job['source'].pop('replay');job['histories']=10
        p=root/job['parameter_file'];s=p.read_text()
        s=s.replace('PhaseSpaceIncludeEmptyHistories = "False"','PhaseSpaceIncludeEmptyHistories = "True"')
        s=s.replace('NumberOfHistoriesInRun = 1\n','NumberOfHistoriesInRun = 10\n')
        p.write_text(s)
    m['schema_version']=3
    m['files']={name:sha256(root/name) for name in m['files']}
    m['bundle_id']=digest_json({k:v for k,v in m.items() if k!='bundle_id'})
    (root/'manifest.json').write_text(json.dumps(m))
    assert all(r['status']=='known_trigger' for r in audit_setup(root,m))
    before={str(p.relative_to(root)):sha256(p) for p in root.rglob('*') if p.is_file()}
    actions=[lambda:results.collect_results(root,10**8),lambda:results.collect_forward([1,1],root),
             lambda:add_batches.add(root,1,dry_run=True),lambda:submit_topas.discover(root,dry_run=True)]
    for action in actions:
        with pytest.raises(ValueError,match='Unsafe'):action()
    after={str(p.relative_to(root)):sha256(p) for p in root.rglob('*') if p.is_file()}
    assert before==after
