import json
import numpy as np
import SimpleITK as sitk
import pytest
from minibeam.workflow.dose_plot import plane_views, snapshot_images, display_images, save_dose_projections


def fixture():
    shape=(4,5,3)  # MATLAB Y,X,Z
    cube=np.arange(np.prod(shape)).reshape(shape)
    cst=np.empty((1,6),dtype=object)
    cst[0]=[0,'TARGET','TARGET',np.array([1+2+4*(3+5*1)]),{},[]]
    grid=dict(dimensions=[5,4,3],origin=[-10.,20.,30.],resolution=dict(x=2.,y=3.,z=4.),direction=np.eye(3).tolist())
    manifest=dict(ct_grid=grid,planning={'plan':{'target_names':['TARGET']}},
                  jobs=[{'iso_center_lps_mm':[-4.,26.,34.]}],weight_units='primary photons')
    return dict(ct={'cubeHU':cube},cst=cst),manifest


def test_axes_indices_and_physical_coordinates():
    a=np.arange(3*4*5).reshape(3,4,5)
    views=plane_views(a,[3,2,1])
    for axis, (projection,reference), expected in zip([0,1,2],views,[a[1],a[:,2],a[:,:,3]]):
        np.testing.assert_array_equal(projection,a.max(axis=axis))
        np.testing.assert_array_equal(reference,expected)
    saved,m=fixture()
    ct,masks,missing=snapshot_images(saved,m)
    assert not missing
    assert masks['TARGET'][3,2,1]==1
    assert masks['TARGET'].TransformIndexToPhysicalPoint((3,2,1))==(-4.,26.,34.)
    np.testing.assert_array_equal(sitk.GetArrayFromImage(ct),saved['ct']['cubeHU'].transpose(2,0,1))


@pytest.mark.parametrize('oblique',[False,True])
def test_display_resampling(oblique):
    saved,m=fixture()
    if oblique:
        angle=np.pi/6
        m['ct_grid']['direction']=[[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]]
    ct,masks,_=snapshot_images(saved,m)
    dose=sitk.Cast(ct,sitk.sitkFloat64)*0
    a,d,regions=display_images(ct,dose,masks)
    np.testing.assert_allclose(a.GetDirection(),np.eye(3).ravel())
    assert a.GetSize()==d.GetSize()==regions['TARGET'].GetSize()
    assert set(np.unique(sitk.GetArrayFromImage(regions['TARGET']))) <= {0,1}
    assert sitk.GetArrayFromImage(regions['TARGET']).sum()>0
    if not oblique:
        np.testing.assert_array_equal(sitk.GetArrayFromImage(a),sitk.GetArrayFromImage(ct))


def test_zero_missing_targets_and_shared_limits(tmp_path):
    saved,m=fixture()
    ct,_,_=snapshot_images(saved,m)
    m['planning']['plan']['target_names']=['absent']
    out=tmp_path/'dose_projections.png'
    record=save_dose_projections(out,saved,m,sitk.Cast(ct,sitk.sitkFloat64)*0,[1.],[])
    assert record['missing_targets']==['absent']
    assert record['dose_color_limits_gy']==[0.,1.]
    assert out.read_bytes().startswith(b'\x89PNG')


def test_plot_failure_preserves_forward(case,monkeypatch):
    from test_results import fake_results
    from minibeam.workflow.artifacts import save_artifacts
    from minibeam.workflow.collection import collect_bundle
    import minibeam.workflow.beam_dose_plot as plot
    ct,plan,cst,stf,engine=case
    root=engine.prepare_jobs(ct,cst,stf)
    save_artifacts(ct,cst,plan,stf,root,'prepare',{})
    fake_results(root)
    def fail(*a,**k):raise RuntimeError('test plot failure')
    monkeypatch.setattr(plot,'BeamDosePlots',fail)
    collect_bundle(root,'forward')
    assert (root/'derived/dose.mha').exists()
    meta=json.loads((root/'derived/forward_metadata.json').read_text())
    assert meta['status']=='complete' and meta['dose_plot']['status']=='failed'
    assert 'test plot failure' in (root/'derived/indexing.md').read_text()


def test_target_slab_limits_only_xz():
    from minibeam.workflow.dose_plot import target_y_slab, dose_views
    values=np.zeros((3,6,5));values[0,0,0]=100
    values[1,2,2]=4;values[1,4,3]=6
    ref=sitk.GetImageFromArray(values);ref.SetOrigin((0.,10.,0.));ref.SetSpacing((1.,2.,1.))
    mask=np.zeros_like(values,dtype=np.uint8);mask[0,2,0]=1;mask[0,4,0]=1
    roi=sitk.GetImageFromArray(mask);roi.CopyInformation(ref)
    slab=target_y_slab({'target':roi},ref)
    assert slab['slice_indices_zero_based_inclusive']==[2,4]
    assert slab['y_voxel_centers_lps_mm']==[14.,18.]
    assert slab['y_voxel_boundaries_lps_mm']==[13.,19.]
    before=values.copy();original=plane_views(values,[0,0,0]);views=dose_views(values,[0,0,0],slab)
    assert views[1][0][0,0]==0
    assert views[1][0][1,2]==4 and views[1][0][1,3]==6  # slab, not ROI masking
    for k in (0,2):np.testing.assert_array_equal(views[k][0],original[k][0])
    for k in range(3):np.testing.assert_array_equal(views[k][1],original[k][1])
    np.testing.assert_array_equal(values,before)


def test_single_multiple_empty_and_oblique_slab():
    from minibeam.workflow.dose_plot import target_y_slab
    saved,m=fixture();ct,masks,_=snapshot_images(saved,m)
    assert target_y_slab(masks,ct)['slice_indices_zero_based_inclusive']==[2,2]
    extra=sitk.Image(masks['TARGET']);extra[0,0,0]=1
    slab=target_y_slab({**masks,'other':extra},ct)
    assert slab['slice_indices_zero_based_inclusive']==[0,2]
    assert slab['target_names']==['TARGET','other']
    assert target_y_slab({},ct)['fallback']
    assert target_y_slab({'empty':sitk.Image(ct.GetSize(),sitk.sitkUInt8)},ct)['fallback']
    angle=np.pi/6
    direction=[np.cos(angle),-np.sin(angle),0,np.sin(angle),np.cos(angle),0,0,0,1]
    ct.SetDirection(direction)
    for mask in masks.values():mask.SetDirection(direction)
    ref,dose,regions=display_images(ct,sitk.Cast(ct,sitk.sitkFloat64)*0,masks)
    slab=target_y_slab(regions,ref)
    rows=np.flatnonzero(sitk.GetArrayFromImage(regions['TARGET']).any(axis=(0,2)))
    np.testing.assert_allclose(slab['y_voxel_centers_lps_mm'],
                              ref.GetOrigin()[1]+rows[[0,-1]]*ref.GetSpacing()[1])
