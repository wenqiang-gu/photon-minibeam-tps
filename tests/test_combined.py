"""Disjoint source partitions and explicit aggregate column semantics."""
import json
import shutil
import os
from pathlib import Path
import numpy as np
import pytest
from pyRadPlan import generate_stf
from minibeam import TOPASPhotonEngine
from minibeam.sources.bixels import groups, contains
from minibeam.sources.iaea import TOPAS_DTYPE
from minibeam.topas.manifest import load_manifest, member_jobs
from minibeam.workflow.artifacts import save_artifacts
from minibeam.workflow.collection import collect_bundle
from test_results import fake_results


def two_squares(case):
    from minibeam.geometry.spatial import beam_basis
    stf=case[3].model_copy(deep=True)
    for beam in stf.beams:
        beam.rays.append(beam.rays[0].model_copy(deep=True))
        for ray, x in zip(beam.rays,[-2.5,2.5]):
            ray.ray_pos=np.array([x,0.,0.]) @ beam_basis(beam).T
            ray.ray_pos_bev=np.array([x,0.,0.])
    return stf


def replay(root,job):
    text=(root/job['parameter_file']).read_text()
    stem=text.split('PhaseSpaceFileName = "')[1].split('"')[0]
    return np.fromfile(root/(stem+'.phsp'),dtype=TOPAS_DTYPE)


def test_disjoint_boundaries_and_invalid_geometry(case):
    stf=two_squares(case)
    a,b=groups(stf)[:2]
    for x in [-5.,-1.,0.,1.,5.]:
        selected=sum(bool(contains(g[0]['selection'],x,0.)) for g in [a,b])
        assert selected == (0 if x==5 else 1)
    stf.beams[0].rays[1].ray_pos=stf.beams[0].rays[0].ray_pos.copy()
    with pytest.raises(ValueError,match='Overlapping'):groups(stf)
    stf=two_squares(case)
    stf.beams[0].rays[0].beamlets.append(stf.beams[0].rays[0].beamlets[0].model_copy())
    with pytest.raises(ValueError,match='exactly one'):groups(stf)


def test_point_job_budget_and_combined_normalization(case,tmp_path):
    ct,plan,cst,_,_=case; stf=two_squares(case)
    roots={}
    for mode in ['separate','combined']:
        engine=TOPASPhotonEngine(plan,water=True,histories=100,beamlet_execution=mode)
        root=engine.prepare_jobs(ct,cst,stf,bundle_dir=tmp_path/mode)
        engine.bundle_dir=str(root)
        roots[mode]=root
        m=load_manifest(root)
        assert len(m['jobs']) == (4 if mode=='separate' else 2)
        fake_results(root)
        dij=engine.collect_results(root)
        factor=2 if mode=='combined' else 1
        assert dij.physical_dose.flat[0][0,0]==pytest.approx(1e-10)
        assert dij.ray_num[0] == (-1 if mode=='combined' else 0)
        assert all(j['histories']==100 and j['normalization_histories']==100 for j in m['jobs'])
        if mode=='combined':
            assert m['normalization']=='independent_primary_photon'
            with pytest.raises(ValueError,match='equal exposure'):
                engine.calc_dose_forward(ct,cst,stf,w=[1,2,1,1])
            result=engine.calc_dose_forward(ct,cst,stf,w=[3,3,7,7])
            import SimpleITK as sitk
            np.testing.assert_allclose(sitk.GetArrayFromImage(result['physical_dose']).ravel(),dij.physical_dose.flat[0]@[3,7])
            expected_error=np.sqrt((3*2e-10)**2/100+(7*4e-10)**2/100)
            assert sitk.GetArrayFromImage(result['physical_dose_std_error']).ravel()[0]==pytest.approx(expected_error)
            save_artifacts(ct,cst,plan,stf,root,'prepare',{})
            collect_bundle(root,'collect')
            from scipy.io import loadmat
            exported=loadmat(root/'derived/result.mat',simplify_cells=True)
            assert exported['dij']['physicalDose'].shape[1]==2
            assert np.all(exported['dij']['rayNum']==0)
            assert len(exported['column_mapping'])==2
            assert '0 sentinels' in (root/'derived/indexing.md').read_text()
    separate=load_manifest(roots['separate']);combined=load_manifest(roots['combined'])
    for bi,job in enumerate(combined['jobs']):
        actual=replay(roots['combined'],job)
        counts=job['source']['photons_per_member']
        assert len(actual)==sum(counts)==100
        assert len(counts)==len(job['members'])
        for member,rows in zip(job['members'],np.split(actual,np.cumsum(counts)[:-1])):
            u,v=rows['u'].astype(float),rows['v'].astype(float)
            z=np.sqrt(1-u*u-v*v)
            assert contains(member['selection'],1000*u/z,1000*v/z).all()
    moved=tmp_path/'moved';shutil.move(roots['combined'],moved)
    assert load_manifest(moved)['bundle_id']==combined['bundle_id']


def test_combined_limit_before_generation(case):
    stf=two_squares(case)
    from minibeam.sources.beamlet import PointBeamletSource
    with pytest.raises(ValueError,match='integer limit'):
        PointBeamletSource().prepare(stf,2**31,execution='combined')


def test_phase_union_once_with_history_groups(case,tmp_path):
    from test_phase_space import make_iaea
    # Two records in history 2 fall on opposite sides; x=0 belongs only to the right square.
    base=make_iaea(tmp_path/'phase',records=[(1,-1.,-.1,0.,0.,0.,.05,2,0),
        (2,.2,0.,0.,0.,0.,.05,0,0),(3,-1.,.1,0.,0.,0.,.05,3,0)],original=12)
    ct,plan,cst,_,_=case;stf=two_squares(case)
    roots={}
    for mode in ['separate','combined']:
        engine=TOPASPhotonEngine(plan,water=True,histories=10,beamlet_execution=mode,
                                source_config={'type':'phase_space','file_base':str(base)})
        root=engine.prepare_jobs(ct,cst,stf,bundle_dir=tmp_path/mode);roots[mode]=root
        manifest=load_manifest(root)
        assert all(j['histories']==j['normalization_histories']==10 for j in manifest['jobs'])
        assert manifest['normalization']=='original_accelerator_history'
        if mode=='combined':
            a=replay(root,manifest['jobs'][0])
            assert len(a)==3 and a['new_history'].tolist()==[1,0,1]
            assert a['pdg'].tolist()==[22,11,-11]
            np.testing.assert_allclose(a['weight'],.05)
            assert manifest['jobs'][0]['source']['empty_histories']==8
            assert manifest['jobs'][0]['source']['particles_per_member_bixel']==[1,2]
            # Identical beam-local selections share the same replay pair across beams.
            assert len(list((root/'inputs').glob('phase_*.phsp')))==1
            fake_results(root)
            assert 'physical_dose_std_error' not in engine.collect_forward([1,1],root)
    separate=load_manifest(roots['separate'])
    joined=np.concatenate([replay(roots['separate'],j) for j in separate['jobs'][:2]])
    np.testing.assert_array_equal(np.sort(joined['energy']),np.sort(a['energy']))


@pytest.mark.topas
@pytest.mark.parametrize('phase_space',[False,True])
def test_actual_combined_transport(case,tmp_path,phase_space):
    executable=os.environ.get('PHOTON_TPS_TOPAS')
    if not executable:pytest.skip('Set PHOTON_TPS_TOPAS for local transport')
    from test_topas import execute_jobs
    from test_phase_space import make_iaea
    ct,plan,cst,_,_=case;stf=two_squares(case)
    stf.beams=stf.beams[:1]
    options={}
    count=10000
    if phase_space:
        count=1000
        records=[]
        for _ in range(count):
            records.extend([(1,-1.,-.1,0.,0.,0.,.05,1,0),(1,1.,.1,0.,0.,0.,.05,0,0)])
        base=make_iaea(tmp_path/'input',records=records,original=count)
        options['source_config']={'type':'phase_space','file_base':str(base)}
    summaries={}
    for mode in ['separate','combined']:
        engine=TOPASPhotonEngine(plan,water=True,histories=count,beamlet_execution=mode,dose_spacing_mm=(60.,60.,36.),**options)
        root=engine.prepare_jobs(ct,cst,stf,bundle_dir=tmp_path/mode)
        execute_jobs(root,executable)
        scores=list(engine.iter_scores(root))
        summaries[mode]=(sum(d.sum() for _,d,_ in scores),sum(v.sum() for _,_,v in scores))
    a,va=summaries['separate'];b,vb=summaries['combined']
    if not phase_space:
        a /= 2
        va /= 4
    assert a>0 and b>0
    # One scoring voxel avoids unrecorded spatial covariance in this comparison.
    assert abs(a-b) <= 6*np.sqrt(va+vb)
    if phase_space:
        # All fixture particles lie inside the two-square union. A single wide
        # square therefore replays the unfiltered fixture with the same histories.
        unfiltered = stf.model_copy(deep=True)
        beam = unfiltered.beams[0]
        beam.rays = beam.rays[:1]
        beam.rays[0].ray_pos = np.zeros(3)
        beam.rays[0].ray_pos_bev = np.zeros(3)
        beam.bixel_width = 10.
        engine = TOPASPhotonEngine(plan,water=True,histories=count,dose_spacing_mm=(60.,60.,36.),**options)
        root = engine.prepare_jobs(ct,cst,unfiltered,bundle_dir=tmp_path/'unfiltered')
        execute_jobs(root,executable)
        value = engine.collect_results(root).physical_dose.flat[0].sum()
        assert value == pytest.approx(b,rel=1e-12)



@pytest.mark.parametrize('mutation',['denominator','column','members','histories','normalization'])
def test_combined_manifest_rejects_invalid_mapping(case,tmp_path,mutation):
    from minibeam.topas.manifest import digest_json
    ct,plan,cst,_,_=case;stf=two_squares(case)
    engine=TOPASPhotonEngine(plan,water=True,beamlet_execution='combined')
    root=engine.prepare_jobs(ct,cst,stf)
    m=load_manifest(root)
    if mutation=='denominator':m['jobs'][0]['normalization_histories']+=1
    if mutation=='column':m['jobs'][0]['column_index']=99
    if mutation=='members':m['jobs'][0]['members'][1]['bixel_index']=1
    if mutation=='histories':m['jobs'][0]['histories']+=1
    if mutation=='normalization':m.update(normalization='photons_per_member_bixel',units='Gy/(photon per member bixel)',weight_units='photons per member bixel')
    m['bundle_id']=digest_json({k:v for k,v in m.items() if k!='bundle_id'})
    (root/'manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError):load_manifest(root)


def test_schema_two_remains_collectable(case):
    from minibeam.topas.manifest import digest_json
    ct,_,cst,stf,engine=case
    root=engine.prepare_jobs(ct,cst,stf)
    m=load_manifest(root);m['schema_version']=2
    m.pop('beamlet_execution');m.pop('column_mapping')
    for j in m['jobs']:
        for k in ('members','column_index','normalization_histories'):j.pop(k)
    m['bundle_id']=digest_json({k:v for k,v in m.items() if k!='bundle_id'})
    (root/'manifest.json').write_text(json.dumps(m))
    fake_results(root)
    assert engine.collect_results(root).physical_dose.flat[0].shape[1]==2


def test_native_combined_api_and_forward_artifacts(case,tmp_path):
    from pyRadPlan import calc_dose_influence, calc_dose_forward
    from scipy.io import loadmat
    from minibeam.topas.manifest import sha256
    ct,plan,cst,_,_=case;stf=two_squares(case)
    plan.prop_dose_calc.update(beamlet_execution='combined',water=True)
    engine=TOPASPhotonEngine(plan)
    root=engine.prepare_jobs(ct,cst,stf)
    save_artifacts(ct,cst,plan,stf,root,'prepare',{})
    before=sha256(root/'derived/steering.mat')
    fake_results(root)
    dij=calc_dose_influence(ct,cst,stf,plan)
    assert dij.physical_dose.flat[0].shape==(4800,2)
    result=calc_dose_forward(ct,cst,stf,plan,weights=[1.,1.,2.,2.])
    assert 'physical_dose' in result
    collect_bundle(root,'forward',weight_per_bixel=[1.,2.])
    assert sha256(root/'derived/steering.mat')==before
    report=(root/'derived/indexing.md').read_text()
    assert 'separately completed collect stage' in report and '0 sentinels' in report
    m=json.loads((root/'derived/forward_metadata.json').read_text())
    assert m['weight_units']=='primary photons' and m['weights']==[1.,2.]
    plan.prop_dose_calc['beamlet_execution']='separate'
    with pytest.raises(ValueError,match='execution mode'):
        calc_dose_influence(ct,cst,stf,plan)


def test_unsegmented_scorer_validation(tmp_path):
    from minibeam.topas.scoring import score_rows
    grid=dict(dimensions=[1,1,1],resolution=dict(x=60,y=60,z=36))
    job=dict(scorer='Dose_test',histories=20,normalization_histories=10)
    text='# TOPAS Version: 4.2.p3\n# Results for scorer: Dose_test\n# Scored in component: Patient\n# DoseToMedium ( Gy ) : Sum Mean Histories_with_Scorer_Active Standard_Deviation\n2,0.1,20,0.2\n'
    path=tmp_path/'score.csv';path.write_text(text)
    assert list(score_rows(path,job,grid))==[(0,.2,.008000000000000002)]
    path.write_text(text+'2,0.1,20,0.2\n')
    with pytest.raises(ValueError,match='Duplicate'):list(score_rows(path,job,grid))
    path.write_text(text.replace('component: Patient','component: Other'))
    with pytest.raises(ValueError,match='component'):list(score_rows(path,job,grid))


def test_legacy_combined_normalization_remains_collectable(case,tmp_path):
    from minibeam.topas.manifest import digest_json
    ct,plan,cst,_,_=case
    root=TOPASPhotonEngine(plan,water=True,histories=20,beamlet_execution='combined').prepare_jobs(
        ct,cst,two_squares(case),bundle_dir=tmp_path/'legacy')
    m=load_manifest(root)
    m.pop('history_budget')
    m.update(normalization='photons_per_member_bixel',units='Gy/(photon per member bixel)',
             weight_units='photons per member bixel')
    for job in m['jobs']:
        job['normalization_histories']=10
    m['bundle_id']=digest_json({k:v for k,v in m.items() if k!='bundle_id'})
    (root/'manifest.json').write_text(json.dumps(m))
    assert load_manifest(root)['normalization']=='photons_per_member_bixel'


def test_point_budget_not_divisible_by_members(case,tmp_path):
    ct,plan,cst,_,_=case
    stf=two_squares(case)
    engine=TOPASPhotonEngine(plan,water=True,histories=3,beamlet_execution='combined')
    root=engine.prepare_jobs(ct,cst,stf,bundle_dir=tmp_path/'three')
    manifest=load_manifest(root)
    for job in manifest['jobs']:
        assert len(replay(root,job))==job['histories']==job['normalization_histories']==3
        assert sum(job['source']['photons_per_member'])==3
