"""Outside-field replay is the disjoint downstream complement of field replay."""
from conftest import physical_replay_records
import numpy as np
import pytest
from minibeam.sources.phase_space import PhaseSpaceBeamletSource
from minibeam.sources.iaea import TOPAS_DTYPE
from test_all_forward import source_files
from test_phase_space import context, make_iaea

@pytest.mark.parametrize('histories',[10,12,16])
def test_partition_multifile(case,tmp_path,histories):
    bases=source_files(tmp_path); arrays={}
    for mode in ['field','outside_field','all_forward']:
        src=PhaseSpaceBeamletSource(file_bases=bases,selection=mode)
        src.prepare(case[3],histories,execution='combined')
        assert len(src.selections)==1
        state=next(iter(src.selections.values()));record=state['record']
        arrays[mode]=physical_replay_records(src.render(context(case)).assets[1][1],dtype=TOPAS_DTYPE)
        assert record['represented_original_histories']==histories
        assert record['excluded_backward_or_tangent_particles']==2
        if mode=='outside_field':
            assert record['particles_per_member_bixel']==[]
            assert record['rejected_inside_field_particles']==1
            assert record['retained_outside_field_particles']==len(arrays[mode])
            assert record['nonempty_selected_histories']==(3 if histories==16 else 2)
            assert arrays[mode]['new_history'].all()
    # Compare all particle properties except reconstructed history-start flags.
    fields=[n for n in TOPAS_DTYPE.names if n!='new_history']
    def particles(a):return sorted(tuple(r[n].item() for n in fields) for r in a)
    assert particles(arrays['field'])+particles(arrays['outside_field'])==sorted(particles(arrays['all_forward']))


def test_boundaries_and_empty_members(case,tmp_path):
    base=make_iaea(tmp_path/'boundary',original=10,records=[
        (1,-1.,-.25,0.,0.,0.,.05,1,0), # included lower boundary
        (1,-1.,.25,0.,0.,0.,.05,1,0),  # excluded upper boundary
        (1,-1.,0.,.25,0.,0.,.05,1,0)])
    src=PhaseSpaceBeamletSource(base,selection='outside_field')
    src.prepare(case[3],10,execution='combined')
    assert next(iter(src.selections.values()))['particles']==2
    missing=PhaseSpaceBeamletSource('/does-not-exist',selection='outside_field')
    with pytest.raises(ValueError,match='requires combined'):missing.prepare(case[3],10)
    only=make_iaea(tmp_path/'only',original=10,records=[(1,-1.,0.,0.,0.,0.,.05,1,0)])
    with pytest.raises(ValueError,match='no particles'):
        PhaseSpaceBeamletSource(only,selection='outside_field').prepare(case[3],10,execution='combined')


def test_irregular_union_allows_empty_excluded_bixel(case,tmp_path):
    from copy import deepcopy
    from minibeam.geometry.spatial import beam_basis
    stf=deepcopy(case[3])
    for beam in stf.beams:
        extra=deepcopy(beam.rays[0]);extra.ray_pos=extra.ray_pos+10*beam_basis(beam)[:,0]
        beam.rays.append(extra)
    # x=5 mm lies in the gap. No particles intersect the bixel centered at 10 mm.
    base=make_iaea(tmp_path/'gap',original=10,records=[(1,-1.,.5,0.,0.,0.,.05,1,0)])
    src=PhaseSpaceBeamletSource(base,selection='outside_field')
    src.prepare(stf,10,execution='combined')
    assert len(src.selections)==1
    assert next(iter(src.selections.values()))['particles']==1


def test_outside_batch_average(case,tmp_path):
    from minibeam import TOPASPhotonEngine
    from minibeam.topas import results
    from minibeam.topas.manifest import load_manifest
    from projects import add_batches as batches
    from test_results import fake_results
    from test_batches import batch_results
    ct,plan,cst,stf,_=case
    base=source_files(tmp_path)[0]
    e=TOPASPhotonEngine(plan,water=True,histories=10,beamlet_execution='combined',
        source_config={'type':'phase_space','file_base':str(base),'selection':'outside_field'})
    root=e.prepare_jobs(ct,cst,stf);fake_results(root)
    original=results.collect_results(root,10**8).physical_dose.flat[0].toarray()
    batches.add(root,1)
    batch_results(root,batches.inventory(root,load_manifest(root))[1],3)
    merged=results.collect_results(root,10**8).physical_dose.flat[0].toarray()
    np.testing.assert_allclose(merged,2*original)


@pytest.mark.topas
def test_three_way_transport_additivity(case,tmp_path):
    import os, subprocess
    from minibeam import TOPASPhotonEngine
    from minibeam.topas.manifest import load_manifest
    executable=os.environ.get('PHOTON_TPS_TOPAS')
    if not executable:pytest.skip('Set PHOTON_TPS_TOPAS for transport comparison')
    count=5000
    records=[r for _ in range(count) for r in
             [(1,-2,0,0,0,0,.05,1,0),(1,2,.5,0,0,0,.05,0,0)]]
    base=make_iaea(tmp_path/'source',records,original=2*count)
    ct,plan,cst,stf,_=case
    stf=stf.model_copy(deep=True);stf.beams=stf.beams[:1]
    totals={}
    for mode in ('field','outside_field','all_forward'):
        engine=TOPASPhotonEngine(plan,water=True,histories=2*count,num_threads=2,
            beamlet_execution='combined',bundle_dir=str(tmp_path/mode),
            source_config={'type':'phase_space','file_base':str(base),'selection':mode})
        root=engine.prepare_jobs(ct,cst,stf)
        job=load_manifest(root)['jobs'][0]
        assert job['histories']==count and job['normalization_histories']==2*count
        p=subprocess.run([executable,job['parameter_file']],cwd=root,capture_output=True,text=True,timeout=120)
        assert p.returncode==0,p.stdout+p.stderr
        totals[mode]=float(engine.collect_results(root).physical_dose.flat[0].sum())
    assert min(totals.values())>0
    assert totals['all_forward']==pytest.approx(totals['field']+totals['outside_field'],rel=.15)
