import numpy as np
import pytest
from pyRadPlan import generate_stf
from minibeam import TOPASPhotonEngine
from minibeam.topas.manifest import load_manifest
from test_results import fake_results


def test_planning_snapshot_and_reuse(case):
    ct,plan,cst,stf,engine=case
    root=engine.prepare_jobs(ct,cst,stf)
    manifest=load_manifest(root)
    info=manifest['planning']
    assert info['plan']['target_names']==['TARGET']
    assert 'custom' not in info['plan']
    for index,beam in enumerate(stf.beams):
        entry=info['beams'][index]
        assert entry['beam_index']==index+1
        assert entry['parameters']['gantry_angle']==beam.gantry_angle
        assert entry['parameters']['couch_angle']==beam.couch_angle
        np.testing.assert_allclose(entry['parameters']['iso_center'],beam.iso_center)
        assert entry['num_bixels']==beam.total_number_of_bixels
        assert entry['job_ids']==[j['job_id'] for j in manifest['jobs'] if j['beam_index']==index+1]
        assert 'custom' not in entry
    assert engine.prepare_jobs(ct,cst,stf)==root


def test_job_parameters_and_matrix_column_order(case,tmp_path):
    ct,plan,cst,_,_=case
    plan.prop_stf.update(generator='photonIMRT',gantry_angles=[37.,215.],couch_angles=[21.,-33.],
                         iso_center=np.array([[1.,2.,3.]]))
    stf=generate_stf(ct,cst,plan)
    # Exercise local beamlet indexing separately from ray and global bixel indices.
    # Square source supports one beamlet per ray; local labels are still checked.
    engine=TOPASPhotonEngine(plan,water=True)
    root=engine.prepare_jobs(ct,cst,stf)
    manifest=load_manifest(root)
    assert len(manifest['jobs'])>len(stf.beams)
    expected=[]
    for bi,beam in enumerate(stf.beams,1):
        for ri,ray in enumerate(beam.rays,1):
            for li,bixel in enumerate(ray.beamlets,1):
                expected.append((bi,ri,li,beam,ray,bixel))
    assert len(expected)==len(manifest['jobs'])
    for index,(job,(bi,ri,li,beam,ray,bixel)) in enumerate(zip(manifest['jobs'],expected),1):
        assert job['job_id']==f'bixel_{index:06d}'
        assert (job['bixel_index'],job['beam_index'],job['ray_index'],job['beamlet_index'])==(index,bi,ri,li)
        assert job['parameter_file']==f'jobs/bixel_{index:06d}.txt'
        assert job['output']==f'results/bixel_{index:06d}.csv'
        assert job['gantry_angle_deg']==beam.gantry_angle
        assert job['couch_angle_deg']==beam.couch_angle
        np.testing.assert_allclose(job['iso_center_lps_mm'],beam.iso_center)
        assert job['sad_mm']==beam.sad
        assert job['bixel_width_mm']==beam.bixel_width
        assert job['source']['energy_model']=='discrete_spectrum'
        np.testing.assert_allclose(job['source']['source_lps_mm'],beam.iso_center+beam.source_point)
        np.testing.assert_allclose(job['source']['aim_lps_mm'],beam.iso_center+ray.ray_pos)
        assert job['seed']==engine.seed+index-1 and job['histories']==engine.histories
    second=engine.prepare_jobs(ct,cst,stf,bundle_dir=tmp_path/'repeat')
    assert load_manifest(second)['bundle_id']==manifest['bundle_id']
    fake_results(root)
    dij=engine.collect_results(root)
    for job in manifest['jobs']:
        column=job['bixel_index']-1
        assert dij.physical_dose.flat[0][0,column]==pytest.approx(job['bixel_index']*1e-10,abs=1e-20)
        assert dij.beam_num[column]==job['beam_index']-1
        assert dij.ray_num[column]==job['ray_index']-1
        assert dij.bixel_num[column]==job['beamlet_index']-1


@pytest.mark.parametrize('option',['plan_metadata','beam_metadata'])
def test_removed_annotation_options_rejected(case,option):
    with pytest.raises(ValueError,match='Unknown'):
        TOPASPhotonEngine(case[1],**{option:{}})
