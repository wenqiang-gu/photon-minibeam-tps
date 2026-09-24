import numpy as np
import scipy.sparse as sp
from scipy.io import loadmat
from scipy.spatial.transform import Rotation
import SimpleITK as sitk
from pyRadPlan.dij import Dij
from minibeam.geometry.coordinates import source_frame, topas_angles, scoring_grid
from pyRadPlan import generate_stf
from scipy.io import savemat
import pytest


@pytest.mark.parametrize("gantry,couch", [(0,0),(90,0),(180,0),(270,0),(37,21),(215,-33)])
def test_source_rotations(case, gantry, couch):
    ct, plan, cst, _, engine = case
    plan.prop_stf.update(gantry_angles=[gantry], couch_angles=[couch])
    stf = generate_stf(ct,cst,plan)
    definition, _, _, _, center = engine._definition(ct, stf)
    src = definition["jobs"][0]["source"]
    world = Rotation.from_euler("xyz", src["topas_euler_deg"], degrees=True).as_matrix().T
    direction = np.array(src["aim_lps_mm"]) - src["source_lps_mm"]
    np.testing.assert_allclose(world[:,2], direction/np.linalg.norm(direction), atol=1e-12)
    np.testing.assert_allclose(src["aim_lps_mm"], stf.beams[0].iso_center)
    np.testing.assert_allclose(world.T @ world, np.eye(3), atol=1e-12)


def test_coarse_grid_same_outer_edges(case):
    ct = case[0]
    grid = scoring_grid(ct, (7., 8., 5.))
    np.testing.assert_allclose(grid.origin-grid.resolution_vector/2, ct.grid.origin-ct.grid.resolution_vector/2)
    np.testing.assert_allclose(np.array(grid.dimensions)*grid.resolution_vector,
                               np.array(ct.size)*ct.grid.resolution_vector)


def test_matrad_roundtrip_no_native_mutation(case, tmp_path):
    ct, plan, cst, stf, _ = case
    n = ct.grid.num_voxels
    matrix = sp.csc_matrix(np.arange(n*2).reshape(n,2) * 1e-10)
    original = matrix.copy()
    dij = Dij(ct_grid=ct.grid, dose_grid=ct.grid, physical_dose=matrix,
              num_of_beams=2, beam_num=np.array([0,1]), ray_num=np.array([0,0]), bixel_num=np.array([0,0]))
    for filename in ["first.mat", "second.mat"]:
        exported = dict(ct=ct.to_matrad(), cst=cst.to_matrad(), pln=plan.to_matrad(), stf=stf.to_matrad(), dij=dij.to_matrad())
        for key in ("beamNum","rayNum","bixelNum"):
            exported['dij'][key] = exported['dij'][key] + 1
        savemat(tmp_path/filename, exported)
        data = loadmat(tmp_path/filename, simplify_cells=True)
        for column in range(2):
            expected = original[:,column].toarray().reshape(ct.size[::-1]).transpose(1,2,0).ravel(order='F')
            np.testing.assert_array_equal(data['dij']['physicalDose'][:,column].toarray().ravel(),expected)
        np.testing.assert_array_equal(data['dij']['beamNum'],[1,2])
        mask = sitk.GetArrayFromImage(cst.vois[0].mask).transpose(1,2,0)
        np.testing.assert_array_equal(np.asarray(data['cst'][3]).ravel(), np.flatnonzero(mask.ravel(order='F'))+1)
    assert (dij.physical_dose.flat[0] != original).nnz == 0
    np.testing.assert_array_equal(dij.beam_num,[0,1])


def test_native_target_covering(case):
    ct,plan,cst,_,_=case
    plan.prop_stf.update(generator='photonIMRT',bixel_width=3.)
    stf=generate_stf(ct,cst,plan)
    assert stf.total_number_of_bixels > 2
    np.testing.assert_allclose(stf.beams[0].iso_center,cst.target_center_of_mass())
    assert stf.beams[0].rays[0].beamlets[0].energy == 1.25


def test_native_ct_cube_alignment(case,tmp_path):
    from pyRadPlan.ct import CT
    image=case[0].cube_hu
    array=np.arange(np.prod(image.GetSize()),dtype=np.float32).reshape(image.GetSize()[::-1])
    marked=sitk.GetImageFromArray(array);marked.CopyInformation(image)
    ct=CT(cube_hu=marked)
    savemat(tmp_path/'ct.mat',{'ct':ct.to_matrad()})
    exported=loadmat(tmp_path/'ct.mat',simplify_cells=True)['ct']
    np.testing.assert_array_equal(exported['cubeHU'],array.transpose(1,2,0))
    np.testing.assert_array_equal(exported['cubeDim'],exported['cubeHU'].shape)
