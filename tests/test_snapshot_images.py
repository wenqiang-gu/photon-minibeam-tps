import json
import numpy as np
import SimpleITK as sitk
import pytest
from minibeam.workflow.artifacts import snapshot_images


def fixture():
    shape=(4,5,3)  # MATLAB Y,X,Z
    cube=np.arange(np.prod(shape)).reshape(shape)
    cst=np.empty((1,6),dtype=object)
    cst[0]=[0,'TARGET','TARGET',np.array([1+2+4*(3+5*1)]),{},[]]
    grid=dict(dimensions=[5,4,3],origin=[-10.,20.,30.],resolution=dict(x=2.,y=3.,z=4.),direction=np.eye(3).tolist())
    manifest=dict(ct_grid=grid,planning={'plan':{'target_names':['TARGET']}},
                  jobs=[{'iso_center_lps_mm':[-4.,26.,34.]}],weight_units='primary photons')
    return dict(ct={'cubeHU':cube},cst=cst),manifest


@pytest.mark.parametrize('oblique', [False, True])
def test_snapshot_coordinates_and_roi_order(oblique):
    saved, manifest = fixture()
    if oblique:
        angle = np.pi / 6
        manifest['ct_grid']['direction'] = [[np.cos(angle),-np.sin(angle),0],
                                            [np.sin(angle),np.cos(angle),0], [0,0,1]]
    ct, masks, missing = snapshot_images(saved, manifest)
    assert not missing
    assert masks['TARGET'][3,2,1] == 1
    assert masks['TARGET'].GetDirection() == ct.GetDirection()
    expected = np.asarray(ct.GetOrigin()) + np.asarray(ct.GetDirection()).reshape(3,3) @ (np.array([3,2,1])*ct.GetSpacing())
    np.testing.assert_allclose(masks['TARGET'].TransformIndexToPhysicalPoint((3,2,1)), expected)
    np.testing.assert_array_equal(sitk.GetArrayFromImage(ct), saved['ct']['cubeHU'].transpose(2,0,1))


def test_missing_and_invalid_roi():
    saved, manifest = fixture()
    manifest['planning']['plan']['target_names'] = ['absent']
    assert snapshot_images(saved, manifest)[2] == ['absent']
    manifest['planning']['plan']['target_names'] = ['TARGET']
    saved['cst'][0,3] = np.array([0])
    with pytest.raises(ValueError, match='Invalid saved ROI'):
        snapshot_images(saved, manifest)


def test_plot_failure_preserves_forward(case,monkeypatch):
    from test_results import fake_results
    from minibeam.workflow.artifacts import save_planning_snapshot
    from minibeam.workflow.collection import collect_bundle
    import minibeam.workflow.beam_dose_plot as plot
    ct,plan,cst,stf,engine=case
    root=engine.prepare_jobs(ct,cst,stf)
    save_planning_snapshot(ct,cst,plan,stf,root,{})
    fake_results(root)
    def fail(*a,**k):raise RuntimeError('test plot failure')
    monkeypatch.setattr(plot,'BeamDosePlots',fail)
    collect_bundle(root,'forward')
    assert (root/'derived/dose.mha').exists()
    meta=json.loads((root/'derived/forward_metadata.json').read_text())
    assert meta['status']=='complete' and meta['dose_plot']['status']=='failed'
    assert 'test plot failure' in (root/'derived/indexing.md').read_text()


