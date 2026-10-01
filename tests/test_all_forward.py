"""Unrestricted replay keeps source histories, not just projected bixel members."""
import json
import shutil
import os
import numpy as np
import pytest
from scipy.io import loadmat
from minibeam import TOPASPhotonEngine
from minibeam.sources.phase_space import PhaseSpaceBeamletSource
from minibeam.sources.iaea import TOPAS_DTYPE
from minibeam.topas.manifest import load_manifest
from minibeam.workflow.artifacts import save_planning_snapshot
from minibeam.workflow.collection import collect_bundle
from test_phase_space import make_iaea, context
from test_combined import replay
from test_results import fake_results


def source_files(tmp_path):
    # History 2 contains central + peripheral particles; history 5 is entirely
    # peripheral. History 7 is backward/tangent only. All trailing histories empty.
    first=make_iaea(tmp_path/'first',original=12,records=[
        (1,-1.,0.,0.,0.,0.,.05,2,0),
        (2,.2,1.,0.,0.,0.,.07,0,0),
        (3,-1.,2.,0.,0.,0.,.09,3,0),
        (-1,-1.,0.,0.,0.,0.,.05,2,0),
        (1,1.,0.,0.,1.,0.,.05,0,0)])
    second=make_iaea(tmp_path/'second',original=8,records=[
        (1,-2.,3.,0.,0.,0.,.03,2,0)])
    return [first,second]


@pytest.mark.parametrize('histories,particles,reached',[ (10,3,2),(12,3,2),(16,4,3)])
def test_prefix_properties_and_sharing(case,tmp_path,histories,particles,reached):
    src=PhaseSpaceBeamletSource(file_bases=source_files(tmp_path),selection='all_forward')
    src.prepare(case[3],histories,execution='combined')
    assert len(src.selections)==1  # Different beam/couch angles share local records.
    state=next(iter(src.selections.values()));r=state['record']
    assert r['selected_particles']==particles
    assert r['nonempty_selected_histories']==reached
    assert r['empty_histories']==histories-reached
    assert r['excluded_backward_or_tangent_particles']==2
    assert r['particles_per_member_bixel']==[]  # No fictitious bixel partition.
    f=src.render(context(case))
    data=np.fromfile(f.assets[1][1],dtype=TOPAS_DTYPE)
    np.testing.assert_array_equal(data['pdg'][:3],[22,11,-11])
    np.testing.assert_allclose(data['weight'][:3],[.05,.07,.09])
    np.testing.assert_array_equal(data['new_history'][:3],[True,False,True])
    for i in range(2):
        frame=src.render(context(case,i)).record
        from minibeam.geometry.spatial import beam_basis
        b=case[3].beams[i]
        np.testing.assert_allclose(frame['phase_space_plane_lps_mm'],b.iso_center+b.source_point+272.1*beam_basis(b)[:,2])


def test_field_unchanged_and_early_rejection(case,tmp_path):
    base=source_files(tmp_path)[0]
    for kwargs in ({},{'selection':'field'}):
        s=PhaseSpaceBeamletSource(base,**kwargs);s.prepare(case[3],10,execution='combined')
        assert next(iter(s.selections.values()))['record']['selected_particles']==1
    for bad in ('unknown',True,None):
        with pytest.raises(ValueError,match='selection'):PhaseSpaceBeamletSource(base,selection=bad)
    src=PhaseSpaceBeamletSource('/missing',selection='all_forward')
    with pytest.raises(ValueError,match='requires combined'):src.prepare(case[3],10)
    with pytest.raises(ValueError,match='requires combined'):
        TOPASPhotonEngine(case[1],source_model=src)


def test_saved_whole_source_collection(case,tmp_path):
    ct,plan,cst,stf,_=case
    base=source_files(tmp_path)[0]
    engine=TOPASPhotonEngine(plan,water=True,histories=10,beamlet_execution='combined',
        source_config={'type':'phase_space','file_base':str(base),'selection':'all_forward'})
    root=engine.prepare_jobs(ct,cst,stf)
    save_planning_snapshot(ct,cst,plan,stf,root,{})
    m=load_manifest(root)
    assert len(m['column_mapping'])==2
    assert len(list((root/'inputs').glob('*.phsp')))==1
    assert len(replay(root,m['jobs'][0]))==3
    fake_results(root)
    moved=tmp_path/'moved';shutil.move(root,moved)
    for p in tmp_path.glob('first.*'):p.unlink()
    collect_bundle(moved,'collect')
    saved=loadmat(moved/'derived/result.mat',simplify_cells=True)
    assert saved['phase_space_selection']=='all_forward'
    assert saved['dij']['physicalDose'].shape[1]==2
    np.testing.assert_allclose(saved['dij']['physicalDose'].toarray()[0], [1e-10,2e-10], rtol=1e-12)
    assert 'whole-source beam dose' in (moved/'derived/indexing.md').read_text()
    collect_bundle(moved,'forward',weight_per_bixel=2.)
    meta=json.loads((moved/'derived/forward_metadata.json').read_text())
    import SimpleITK as sitk
    image=sitk.ReadImage(str(moved/'derived/dose_scoring_grid.mha'))
    assert sitk.GetArrayFromImage(image).flat[0] == pytest.approx(6e-10)
    assert meta['phase_space_selection']=='all_forward'
    assert meta['dose_plot']['status']=='complete'
    assert all(b['phase_space_selection']=='all_forward' for b in meta['dose_plot']['beams'])
    assert not (moved/'derived/dose_std_error_dose_grid.mha').exists()


@pytest.mark.topas
def test_outside_field_transport(case,tmp_path):
    executable=os.environ.get('PHOTON_TPS_TOPAS')
    if not executable:pytest.skip('Set PHOTON_TPS_TOPAS')
    from test_topas import execute_jobs
    ct,plan,cst,stf,_=case
    # 4 mm off-axis misses the 5 mm bixel but intersects the synthetic water.
    records=[]
    for _ in range(100):
        records.extend([(1,-2.,0.,0.,0.,0.,1.,1,0),(1,2.,.4,0.,0.,0.,1.,0,0)])
    base=make_iaea(tmp_path/'transport',records=records,original=120)
    for mode,count in [('field',100),('all_forward',200)]:
        engine=TOPASPhotonEngine(plan,water=True,histories=120,beamlet_execution='combined',
            source_config={'type':'phase_space','file_base':str(base),'selection':mode})
        root=engine.prepare_jobs(ct,cst,stf,bundle_dir=tmp_path/mode)
        m=load_manifest(root)
        assert all(j['source']['selected_particles']==count for j in m['jobs'])
        execute_jobs(root,executable)
        dij=engine.collect_results(root)
        assert dij.physical_dose.flat[0].sum()>0


@pytest.mark.parametrize('which', ['patient', 'water'])
def test_top_level_propagation(which,monkeypatch,tmp_path):
    import patient_workflow as patient
    import water_phantom as water
    module = patient if which == 'patient' else water
    monkeypatch.setattr(module, 'SOURCE_TYPE', 'phase_space')
    monkeypatch.setattr(module, 'PHASE_SPACE_SELECTION', 'all_forward')
    if which == 'patient':
        monkeypatch.setattr(module, 'BEAMLET_EXECUTION', 'combined')
        settings=module.planning_settings()
        plan=module.patient.configure_plan(settings,project_dir=tmp_path)
    else:
        settings=module.planning_settings()
        plan=module.water.build_planning(settings,tmp_path)[3]
    assert plan.prop_dose_calc['source_config']['selection']=='all_forward'


def test_point_ignores_phase_selection():
    from minibeam.workflow.planning import topas_settings
    opts=topas_settings(project_dir='unused',source_type='point',phase_space_file_bases=[],
        histories=10,num_threads=1,execution='separate',water=True,enable_opengl=False,
        dose_spacing_mm=None,phase_space_selection='inactive-invalid')
    assert opts['source_config']=={'type':'point'}
