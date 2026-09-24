import json
import numpy as np
import pytest
import SimpleITK as sitk
from pyRadPlan.geometry import lps
from minibeam.workflow.beam_dose_plot import beam_display_grid, project_beam, BeamDosePlots
from minibeam.workflow.dose_plot import snapshot_images
from minibeam.topas import results
from minibeam.workflow.artifacts import save_artifacts
from minibeam.workflow.collection import collect_bundle
from test_results import fake_results
from test_dose_plot import fixture


def parameters(gantry=0.,couch=0.):
    rotation=lps.get_beam_rotation_matrix(gantry,couch)
    return dict(gantry_angle=gantry,couch_angle=couch,source_point=(rotation@np.array([0.,-1000.,0.])).tolist(),
                iso_center=[-4.,26.,34.])


@pytest.mark.parametrize('gantry',[0.,45.,135.,225.,315.])
@pytest.mark.parametrize('couch',[0.,20.])
def test_frames_and_projection_bounds(gantry,couch):
    saved,m=fixture();ct,masks,_=snapshot_images(saved,m)
    p=parameters(gantry,couch)
    ref,center,basis=beam_display_grid(ct,p)
    np.testing.assert_allclose(basis.T@basis,np.eye(3),atol=1e-12)
    np.testing.assert_allclose(basis[:,2],-np.array(p['source_point'])/1000,atol=1e-12)
    np.testing.assert_allclose(ref.GetOrigin(),np.array(p['iso_center'])+basis@center)
    view=project_beam(ct,masks,sitk.Cast(ct,sitk.sitkFloat64)*0,p)
    assert view['record']['depth_slab']['target_names']==['TARGET']
    assert view['ct'].shape==view['dose'].shape
    assert not view['dose'].any()


def test_zero_angle_matches_target_slab_and_full_ct():
    saved,m=fixture();ct,masks,_=snapshot_images(saved,m)
    ct.SetSpacing((1.,1.,1.))
    for mask in masks.values():mask.CopyInformation(ct)
    a=np.zeros(ct.GetSize()[::-1]);a[0,0,0]=100;a[1,2,1]=4
    d=sitk.GetImageFromArray(a);d.CopyInformation(ct)
    p=parameters();p['iso_center']=list(ct.TransformIndexToPhysicalPoint((3,2,1)))
    v=project_beam(ct,masks,d,p)
    np.testing.assert_array_equal(v['dose'],a[:,2,:])
    np.testing.assert_array_equal(v['ct'],sitk.GetArrayFromImage(ct)[:,2,:])
    assert v['record']['depth_slab']['slice_indices_zero_based_inclusive']==[2,2]


@pytest.mark.parametrize('mode',['separate','combined'])
def test_streaming_beams_sum_and_read_once(case,monkeypatch,mode):
    from minibeam import TOPASPhotonEngine
    from test_combined import two_squares
    ct,plan,cst,_,_=case;stf=two_squares(case)
    engine=TOPASPhotonEngine(plan,water=True,beamlet_execution=mode)
    root=engine.prepare_jobs(ct,cst,stf);m=fake_results(root)
    weights=np.arange(1,len(m['jobs'])+1,dtype=float)
    calls=[];original=results.score_rows
    def counted(*args,**kwargs):
        calls.append(args[1]['job_id'])
        yield from original(*args,**kwargs)
    monkeypatch.setattr(results,'score_rows',counted)
    beams={}
    result=results.collect_forward(weights,root,on_beam=lambda b,d:beams.update({b:sitk.GetArrayFromImage(d)}))
    assert calls==[j['job_id'] for j in m['jobs']]
    np.testing.assert_allclose(sum(beams.values()),sitk.GetArrayFromImage(result['physical_dose']))
    assert len(beams)==2


def test_plot_failures_and_zero_weights(case,monkeypatch):
    import minibeam.workflow.beam_dose_plot as plotting
    ct,plan,cst,stf,engine=case
    root=engine.prepare_jobs(ct,cst,stf);fake_results(root)
    save_artifacts(ct,cst,plan,stf,root,'prepare',{})
    original=plotting.project_beam
    def fail_first(ct,masks,dose,p):
        if p['gantry_angle']==0.:raise ValueError('first beam failed')
        return original(ct,masks,dose,p)
    monkeypatch.setattr(plotting,'project_beam',fail_first)
    collect_bundle(root,'forward',weight_per_bixel=0.)
    meta=json.loads((root/'derived/forward_metadata.json').read_text())
    records=meta['dose_plot']['beams']
    assert meta['dose_plot']['status']=='partial'
    assert records[0]['status']=='failed' and records[1]['status']=='complete'
    assert (root/records[1]['path']).exists()
    assert records[1]['bixel_centers']  # Zero exposure still shows selected centers.
    assert (root/'derived/dose.mha').exists()
    assert not (root/'derived/dose_projections.png').exists()


def test_shared_limits_and_warning_attribution(case):
    from minibeam.workflow.collection import planning_snapshot
    from minibeam.topas.manifest import load_manifest
    ct,plan,cst,stf,engine=case
    root=engine.prepare_jobs(ct,cst,stf);m=load_manifest(root)
    save_artifacts(ct,cst,plan,stf,root,'prepare',{})
    snapshot,_=planning_snapshot(root,m)
    warnings=[dict(job_id=m['jobs'][0]['job_id'])]
    plots=BeamDosePlots(snapshot,m,[1.,2.],warnings)
    for i in [1,2]:
        d=sitk.Cast(ct.cube_hu,sitk.sitkFloat64)*0+i
        plots.consume(i,d)
    record=plots.save(root/'derived')
    a,b=record['beams']
    assert a['dose_color_limits_gy']==b['dose_color_limits_gy']==[0.,2.]
    assert len(a['scorer_warnings'])==1 and b['scorer_warnings']==[]

@pytest.mark.parametrize('gantry,couch', [(0,0),(90,0),(45,0),(135,20),(225,30),(315,10)])
@pytest.mark.parametrize('mode', ['separate','combined'])
def test_saved_bixel_centers(gantry,couch,mode,tmp_path):
    from scipy.io import savemat, loadmat
    from minibeam.workflow.beam_dose_plot import bixel_centers
    saved,m=fixture();ct,_,_=snapshot_images(saved,m)
    p=parameters(gantry,couch)
    _,_,basis=beam_display_grid(ct,p)
    uv=np.array([[0.,0.],[-5.,10.],[15.,-20.]])
    offsets=uv@basis[:,:2].T
    rays=np.empty((1,3),dtype=[('rayPos',object)])
    for i,r in enumerate(offsets):rays[0,i]['rayPos']=r
    steering=dict(isoCenter=p['iso_center'],ray=rays)
    path=tmp_path/'steering.mat';savemat(path,dict(stf=steering))
    members=[dict(beam_index=1,ray_index=i+1,beamlet_index=1,bixel_index=i+1) for i in range(3)]
    # Another beamlet on the last ray shares its dot.
    members.append(dict(beam_index=1,ray_index=3,beamlet_index=2,bixel_index=4))
    if mode=='combined':
        jobs=[dict(job_id='beam_001',source={},members=members)]
    else:
        jobs=[dict(job_id=f'bixel_{i+1}',source={},**v) for i,v in enumerate(members)]
    for snapshot in [loadmat(path),loadmat(path,simplify_cells=True)]:
        centers=bixel_centers(snapshot,dict(jobs=jobs),1,basis)
        np.testing.assert_allclose([c['uv_mm'] for c in centers],uv,atol=1e-12)
        np.testing.assert_allclose([c['center_lps_mm'] for c in centers],offsets+p['iso_center'])
        assert len(centers)==3 and len(centers[-1]['bixels'])==2
        assert [b['bixel_index'] for c in centers for b in c['bixels']]==[1,2,3,4]
