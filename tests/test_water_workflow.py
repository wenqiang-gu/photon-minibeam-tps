"""Full-field water steering and saved-project workflow regressions."""
import json
import os
import shutil
from importlib.resources import files

import numpy as np
import pytest
import SimpleITK as sitk
from scipy.io import loadmat
from pyRadPlan import generate_stf
from pyRadPlan.cst import create_voi
from minibeam.geometry.spatial import beam_basis
from minibeam.square_field import StfGeneratorPhotonSquareField
from minibeam.topas.manifest import load_manifest
import water_phantom as water
from test_results import fake_results


def small_water(monkeypatch):
    monkeypatch.setattr(water, 'PHANTOM_SIZE_MM', 20.)
    monkeypatch.setattr(water, 'PHANTOM_SPACING_MM', 5.)
    monkeypatch.setattr(water, 'SOURCE_TYPE', 'point')
    monkeypatch.setattr(water, 'ENABLE_OPENGL', False)
    monkeypatch.setattr(water, 'PHASE_SPACE_SELECTION', 'field')
    monkeypatch.setattr(water, 'HISTORIES_PER_JOB', 100)
    monkeypatch.setattr(water, 'TOPAS_THREADS_PER_JOB', 1)


def test_native_water_geometry(tmp_path):
    _, ct, cst, plan, stf, metadata = water.water.build_planning(water.planning_settings(), tmp_path/'new')
    assert ct.size == (100,100,100)
    np.testing.assert_allclose(ct.cube_hu.TransformContinuousIndexToPhysicalPoint((-.5,)*3), [-100,0,-100])
    np.testing.assert_allclose(ct.cube_hu.TransformContinuousIndexToPhysicalPoint((99.5,)*3), [100,200,100])
    beam = stf.beams[0]
    np.testing.assert_allclose(beam.source_point + beam.iso_center, [0,-1000,0])
    assert stf.total_number_of_bixels == 400
    assert plan.prop_dose_calc['beamlet_execution'] == 'combined'
    positions = np.array([r.ray_pos_bev for r in beam.rays])
    np.testing.assert_allclose(np.unique(positions[:,0]), np.arange(-47.5,50,5))
    assert metadata['water_measurement']['field_width_mm'] == beam.geometry.field.width_mm
    # A tiny off-center target must not change the field's rays.
    mask = np.zeros((100,100,100),np.uint8); mask[20,30,40] = 1
    cst.vois[1] = create_voi(name='WATER',grid=ct.grid,mask=mask,voi_type='TARGET')
    again = generate_stf(ct,cst,plan)
    np.testing.assert_array_equal([r.ray_pos for r in again.beams[0].rays], [r.ray_pos for r in beam.rays])
    assert not (tmp_path/'new').exists()


@pytest.mark.parametrize('angle,couch', [(0,0),(45,0),(135,20)])
def test_native_rotations(monkeypatch,tmp_path,angle,couch):
    small_water(monkeypatch)
    _,ct,cst,plan,_,_=water.water.build_planning(water.planning_settings(), tmp_path/'r')
    plan.prop_stf.update(gantry_angles=[angle],couch_angles=[couch])
    beam=generate_stf(ct,cst,plan).beams[0]
    local=np.array([r.ray_pos @ beam_basis(beam) for r in beam.rays])
    assert np.allclose(local[:,2],0)
    np.testing.assert_allclose(np.max(np.abs(local[:,:2]),axis=0),[47.5,47.5])
    for ray in beam.rays:
        np.testing.assert_allclose(ray.target_point,2*ray.ray_pos-beam.source_point)


@pytest.mark.parametrize('width,bixel,count',[(100,5,400),(15,5,9),(5,5,1)])
def test_tiling(width,bixel,count):
    generator=StfGeneratorPhotonSquareField();generator.field_width_mm=width
    p=generator._generate_ray_positions_in_isocenter_plane({'bixel_width':bixel})
    assert p.shape==(3,count)
    assert p[0].min()-bixel/2 == -width/2
    assert p[2].max()+bixel/2 == width/2


@pytest.mark.parametrize('width', [None,True,-1,np.inf,12])
def test_invalid_tiling(width):
    generator=StfGeneratorPhotonSquareField();generator.field_width_mm=width
    with pytest.raises(ValueError):
        generator._generate_ray_positions_in_isocenter_plane({'bixel_width':5})


def test_saved_water(monkeypatch,tmp_path):
    small_water(monkeypatch)
    root=tmp_path/'water'
    water.prepare(root)
    manifest=fake_results(root)
    assert len(manifest['jobs'])==1
    assert manifest['beamlet_execution']=='combined'
    assert 'i:Ts/NumberOfThreads = 1' in (root/'inputs/common.txt').read_text()
    moved=tmp_path/'relocated';shutil.move(root,moved)
    def forbidden(*a,**kw): pytest.fail('Saved stage invoked planning')
    monkeypatch.setattr(water.water,'build_planning',forbidden)
    monkeypatch.setattr(water,'SOURCE_TYPE','invalid')
    monkeypatch.setattr(water,'FORWARD_WEIGHT',3.)
    water.collect(moved);water.forward(moved)
    d=loadmat(moved/'derived/result.mat',simplify_cells=True)['dij']['physicalDose']
    assert d.shape==(64,1)
    dose=sitk.GetArrayFromImage(sitk.ReadImage(str(moved/'derived/dose.mha')))
    np.testing.assert_allclose(np.sort(dose.ravel()),np.sort(np.asarray(d@np.array([3.])).ravel()))
    metadata=json.loads((moved/'derived/forward_metadata.json').read_text())
    assert metadata['dose_plot']['status']=='complete'
    record=metadata['dose_plot']['beams'][0]
    assert record['reference_slice_depth_mm']==2.5
    assert record['depth_slab']['target_names']==['WATER']
    assert (moved/'derived/indexing.md').exists()


def test_cli_and_config(monkeypatch,tmp_path):
    assert water.parse_arguments(['inspect']).project_dir=='projects/water-phantom'
    for stage in ('prepare','collect','forward'):
        with pytest.raises(SystemExit):water.parse_arguments([stage])
        assert water.parse_arguments([stage,'--project','projects/custom']).project_dir == 'projects/custom'
    assert files('minibeam.geometry').joinpath('water_config.toml').is_file()
    small_water(monkeypatch)
    monkeypatch.setattr(water,'TOPAS_THREADS_PER_JOB',True)
    with pytest.raises(ValueError,match='positive integer'):water.water.build_planning(water.planning_settings(), tmp_path)


@pytest.mark.topas
@pytest.mark.parametrize('phase',[False,True])
def test_water_transport(monkeypatch,tmp_path,phase):
    executable=os.environ.get('PHOTON_TPS_TOPAS')
    if not executable:pytest.skip('Set PHOTON_TPS_TOPAS for transport')
    from test_topas import execute_jobs
    from test_phase_space import make_iaea
    small_water(monkeypatch)
    monkeypatch.setattr(water,'PHANTOM_SIZE_MM',100.)
    monkeypatch.setattr(water,'PHANTOM_SPACING_MM',10.)
    monkeypatch.setattr(water,'HISTORIES_PER_JOB',2000)
    if phase:
        # Multi-particle histories and trailing empties survive full-field replay.
        records = []
        for x in np.arange(-47.5, 50, 5):
            for y in np.arange(-47.5, 50, 5):
                records.extend([(1,-2.,x/10,y/10,0.,0.,.05,1,0),
                                (1,2.,x/10,y/10,0.,0.,.05,0,0)])
        base=make_iaea(tmp_path/'source', records=records,original=2000)
        monkeypatch.setattr(water,'SOURCE_TYPE','phase_space')
        monkeypatch.setattr(water,'PHASE_SPACE_FILE_BASES',[str(base)])
    water.prepare(tmp_path/'water')
    output=execute_jobs(tmp_path/'water',executable)
    assert not any('Overlap is detected' in x for x in output)
    water.collect(tmp_path/'water')
    matrix=loadmat(tmp_path/'water/derived/result.mat',simplify_cells=True)['dij']['physicalDose']
    assert matrix.shape==(1000,1) and matrix.sum()>0


def test_tiled_phase_union_matches_direct_square(monkeypatch,tmp_path):
    from test_phase_space import make_iaea
    from test_combined import replay
    from minibeam import TOPASPhotonEngine
    small_water(monkeypatch)
    records=[(1,-2.,x/10,y/10,0.,0.,.05,1,0)
             for x in np.arange(-47.5,50,5) for y in np.arange(-47.5,50,5)]
    # Outer left edge included; outer right excluded; internal shared edge once.
    records += [(2,-1.,-5.,0.,0.,0.,.05,2,0),
                (3,.5,0.,0.,0.,0.,.05,0,0),
                (1,-1.,5.,0.,0.,0.,.05,3,0)]
    base=make_iaea(tmp_path/'source',records=records,original=1000)
    monkeypatch.setattr(water,'SOURCE_TYPE','phase_space')
    monkeypatch.setattr(water,'PHASE_SPACE_FILE_BASES',[str(base)])
    monkeypatch.setattr(water,'HISTORIES_PER_JOB',1000)
    _,ct,cst,plan,stf,_=water.water.build_planning(water.planning_settings(), tmp_path/'tiled')
    engine=TOPASPhotonEngine(plan)
    tiled=engine.prepare_jobs(ct,cst,stf)
    plan.prop_stf['bixel_width']=100.
    direct=generate_stf(ct,cst,plan)
    root=engine.prepare_jobs(ct,cst,direct,bundle_dir=tmp_path/'direct')
    m=load_manifest(tiled);n=load_manifest(root)
    a,b=replay(tiled,m['jobs'][0]),replay(root,n['jobs'][0])
    np.testing.assert_array_equal(a,b)
    assert len(a)==402
    source=m['jobs'][0]['source']
    assert source['empty_histories']==599
    assert source['selected_species_counts']==dict(photons=400,electrons=1,positrons=1)
    assert sum(source['particles_per_member_bixel'])==402
