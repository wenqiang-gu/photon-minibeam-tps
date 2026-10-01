from pathlib import Path
import shutil
import numpy as np
import pytest
from test_phase_space import make_iaea, context
from minibeam.sources.iaea import IAEA_DTYPE, TOPAS_DTYPE, read_header, validated_parts
from minibeam.sources.phase_space import PhaseSpaceBeamletSource
from minibeam import TOPASPhotonEngine
from minibeam.topas.manifest import load_manifest


@pytest.fixture
def parts(tmp_path):
    a=make_iaea(tmp_path/'part1')
    # Same layout and history numbering, distinct particle energies.
    rows=np.fromfile(str(a)+'.phsp',dtype=IAEA_DTYPE).copy()
    rows['energy']*=1.1
    b=make_iaea(tmp_path/'part2',rows)
    return [a,b]


@pytest.mark.parametrize('histories,allocation,selected,nonempty',[
    (2,[2,0],2,1),(12,[12,0],4,2),(13,[12,1],4,2),(14,[12,2],6,3),(24,[12,12],8,4)])
def test_prefix_crossing(case,parts,histories,allocation,selected,nonempty):
    s=PhaseSpaceBeamletSource(file_bases=parts);s.prepare(case[3],histories)
    p=s.render(context(case));r=p.record
    assert [a['consumed_original_histories'] for a in r['original_files']]==allocation
    assert r['selected_particles']==selected
    assert r['nonempty_selected_histories']==nonempty
    assert r['empty_histories']==histories-nonempty
    assert r['available_original_histories']==24
    data=np.fromfile(p.assets[1][1],dtype=TOPAS_DTYPE)
    assert int(data['new_history'].sum())==nonempty
    np.testing.assert_allclose(data['weight'],.05)
    assert set(data['pdg']) >= {22,11}
    # Deterministic replay, including the boundary's new-history flags.
    old=data.tobytes();s.prepare(case[3],histories)
    assert Path(s.render(context(case)).assets[1][1]).read_bytes()==old


def test_global_ids_and_single_equivalence(case,parts):
    audits=[]
    ids=np.concatenate([ids for _,_,ids in validated_parts([read_header(p) for p in parts],audits)])
    assert ids.tolist()==[2,2,5,7,7,14,14,17,19,19]
    assert [a['trailing_empty_histories'] for a in audits]==[5,5]
    old=PhaseSpaceBeamletSource(parts[0]);new=PhaseSpaceBeamletSource(file_bases=[parts[0]])
    for s in [old,new]:s.prepare(case[3],12)
    assert old.render(context(case)).assets[1][1].read_bytes()==new.render(context(case)).assets[1][1].read_bytes()
    assert old.render(context(case)).record==new.render(context(case)).record


@pytest.mark.parametrize('failure',['duplicate_path','duplicate_content','plane','description','truncated','missing','malformed','budget','both'])
def test_rejections(case,parts,failure):
    a,b=parts
    if failure=='duplicate_path':parts=[a,a]
    if failure=='duplicate_content':shutil.copyfile(str(a)+'.phsp',str(b)+'.phsp')
    if failure=='plane':
        p=Path(str(b)+'.header');p.write_text(p.read_text().replace('27.21','28.21'))
    if failure=='description':
        p=Path(str(b)+'.header');p.write_text(p.read_text()+'$INITIAL_SOURCE_DESCRIPTION:\nDifferent source\n')
    if failure=='truncated':
        p=Path(str(b)+'.phsp');p.write_bytes(p.read_bytes()[:-1])
    if failure=='missing':Path(str(b)+'.phsp').unlink()
    if failure=='malformed':
        rows=np.fromfile(str(b)+'.phsp',dtype=IAEA_DTYPE);rows['weight'][0]=-1;rows.tofile(str(b)+'.phsp')
    with pytest.raises((ValueError,FileNotFoundError)):
        s=PhaseSpaceBeamletSource(a,file_bases=parts) if failure=='both' else PhaseSpaceBeamletSource(file_bases=parts)
        s.prepare(case[3],25 if failure=='budget' else 2)  # Validate even unused second file.


@pytest.mark.parametrize('mode',['separate','combined'])
def test_portable_collection(case,parts,tmp_path,mode):
    from test_results import fake_results
    from minibeam.workflow.artifacts import save_planning_snapshot
    from minibeam.workflow.collection import collect_bundle
    import SimpleITK as sitk
    ct,plan,cst,stf,_=case
    config={'type':'phase_space','file_bases':list(map(str,parts))}
    engine=TOPASPhotonEngine(plan,water=True,histories=24,source_config=config,beamlet_execution=mode)
    root=engine.prepare_jobs(ct,cst,stf);save_planning_snapshot(ct,cst,plan,stf,root,{})
    fake_results(root)
    m=load_manifest(root)
    assert all(j['source']['represented_original_histories']==24 for j in m['jobs'])
    assert len(list((root/'inputs').glob('phase_*.phsp')))==1
    moved=tmp_path/'moved';shutil.move(root,moved)
    for p in parts:
        for suffix in ['.header','.phsp']:Path(str(p)+suffix).unlink()
    matrix=engine.collect_results(moved).physical_dose.flat[0]
    image=engine.collect_forward([2.,3.],moved)['physical_dose']
    np.testing.assert_allclose(matrix@np.array([2.,3.]),sitk.GetArrayFromImage(image).ravel())
    collect_bundle(moved,'collect');collect_bundle(moved,'forward',weight_per_bixel=2.)
    with pytest.raises(ValueError):TOPASPhotonEngine(plan,source_config=dict(config,file_base='conflict'))


@pytest.mark.topas
def test_multiple_topas_replay(case,parts):
    import os,subprocess
    executable=os.environ.get('PHOTON_TPS_TOPAS')
    if not executable:pytest.skip('Set PHOTON_TPS_TOPAS')
    ct,plan,cst,stf,_=case
    engine=TOPASPhotonEngine(plan,water=True,histories=24,source_config={'type':'phase_space','file_bases':list(map(str,parts))})
    root=engine.prepare_jobs(ct,cst,stf)
    for job in load_manifest(root)['jobs']:
        p=subprocess.run([executable,job['parameter_file']],cwd=root,capture_output=True,text=True,timeout=120)
        assert p.returncode==0,p.stdout+p.stderr
        assert 'PreCheck showed no problems' in p.stdout
    assert engine.collect_results(root).physical_dose.flat[0].nnz>0

@pytest.mark.parametrize('values',[None,[], 'single-string', [None], [True], ['']])
def test_invalid_file_lists(values):
    with pytest.raises(ValueError):PhaseSpaceBeamletSource(file_bases=values)


def test_order_changes_fingerprint(case,parts,tmp_path):
    ct,plan,cst,stf,_=case
    ids=[]
    for i,order in enumerate([parts,list(reversed(parts))]):
        engine=TOPASPhotonEngine(plan,water=True,histories=14,source_config={'type':'phase_space','file_bases':list(map(str,order))})
        root=engine.prepare_jobs(ct,cst,stf,bundle_dir=tmp_path/f'order{i}')
        ids.append(load_manifest(root)['request_id'])
    assert ids[0]!=ids[1]
