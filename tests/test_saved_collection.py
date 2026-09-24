import json
import shutil
import numpy as np
import pytest
from scipy.io import loadmat
from minibeam.topas.manifest import sha256
from minibeam.workflow.collection import collect_bundle, planning_snapshot
from test_results import fake_results
from test_workflows import setup_workflow
import patient_workflow as workflow


@pytest.mark.parametrize('coarse', [False, True])
def test_saved_collection_independent_and_relocated(case, monkeypatch, tmp_path, coarse):
    setup_workflow(monkeypatch, case, tmp_path)
    if coarse:
        monkeypatch.setattr(workflow, 'DOSE_SPACING_MM', (6.,6.,6.))
    root = tmp_path/'workflow'
    workflow.main(['prepare','--project',str(root)])
    fake_results(root)
    moved = tmp_path/'relocated'
    shutil.move(root, moved)
    before = sha256(moved/'derived/steering.mat')
    def forbidden(*a,**kw):
        pytest.fail('Collection touched preparation inputs')
    monkeypatch.setattr(workflow,'load_patient', forbidden)
    monkeypatch.setattr(workflow,'generate_stf', forbidden)
    monkeypatch.setattr(workflow,'resolve_geometry', forbidden)
    monkeypatch.setattr(workflow.TOPASPhotonEngine,'prepare_jobs', forbidden)
    monkeypatch.setattr(workflow,'ENABLE_COLLIMATOR', 'invalid inactive value')
    monkeypatch.setattr(workflow,'SOURCE_TYPE', 'invalid inactive source')
    for stage in ('collect','forward'):
        workflow.main([stage,'--project',str(moved)])
        report=(moved/'derived/indexing.md').read_text()
        assert 'NOT the global' in report and '[Z,Y,X]' in report
        assert ('grids differ' if coarse else 'grids match') in report
        assert f'Stage: `{stage}`' in report
        assert sha256(moved/'derived/steering.mat') == before
        if stage == 'forward':
            metadata=json.loads((moved/'derived/forward_metadata.json').read_text())
            assert metadata['dose_plot']['status']=='complete'
            assert all((moved / r['path']).is_file() for r in metadata['dose_plot']['beams'])

    d=loadmat(moved/'derived/result.mat',simplify_cells=True)['dij']
    assert d['physicalDose'].shape[1] == 2
    np.testing.assert_array_equal(d['beamNum'], [1,2])
    np.testing.assert_array_equal(d['bixelNum'], [1,1])
    (moved/'derived/steering.mat').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='hash'):
        collect_bundle(moved,'collect')


def test_legacy_snapshot_and_mismatch(case, monkeypatch, tmp_path):
    setup_workflow(monkeypatch,case,tmp_path)
    workflow.main()
    root=tmp_path/'workflow'
    m=fake_results(root)
    (root/'derived/planning_snapshot.json').unlink()
    _, note=planning_snapshot(root,m)
    assert 'unhashed' in note
    m['planning']['beams'][0]['parameters']['gantry_angle'] += 5
    with pytest.raises(ValueError,match='gantryAngle'):
        planning_snapshot(root,m)


def test_collect_requires_directory():
    with pytest.raises(SystemExit):
        workflow.parse_arguments(['collect'])


def test_native_rejects_changed_ray_without_preparation(case,monkeypatch):
    ct,plan,cst,stf,engine=case
    root=engine.prepare_jobs(ct,cst,stf)
    fake_results(root)
    def forbidden(*a,**kw): pytest.fail('Source preparation called')
    monkeypatch.setattr(engine,'prepare_jobs',forbidden)
    engine._ready(ct,cst,stf)
    changed=stf.model_copy(deep=True)
    changed.beams[0].rays[0].ray_pos[0]+=1
    with pytest.raises(ValueError,match='aiming'):
        engine._ready(ct,cst,changed)
