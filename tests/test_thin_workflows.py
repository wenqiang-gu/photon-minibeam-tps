"""Root scripts expose settings and dispatch; package stages own planning."""
from dataclasses import fields
import ast
import inspect
import numpy as np
import pytest
import patient_workflow as patient
import water_phantom as water


@pytest.mark.parametrize('script',[patient,water])
def test_settings_snapshot_is_explicit_and_independent(script):
    snapshot=script.planning_settings()
    for field in fields(snapshot):
        expected=getattr(script,field.name.upper())
        actual=getattr(snapshot,field.name)
        if isinstance(expected,np.ndarray):np.testing.assert_array_equal(actual,expected)
        else:assert actual==expected
    snapshot.phase_space_file_bases.append('independent')
    assert 'independent' not in script.PHASE_SPACE_FILE_BASES
    # No native planning construction remains at the root.
    tree=ast.parse(inspect.getsource(script))
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
                   and n.func.id in {'PhotonPlan','generate_stf'} for n in ast.walk(tree))


@pytest.mark.parametrize('script',[patient,water])
@pytest.mark.parametrize('stage',['collect','forward'])
def test_saved_stages_never_package_planning_settings(script,stage,monkeypatch):
    monkeypatch.setattr(script,'planning_settings',lambda: pytest.fail('Saved stage built planning settings'))
    seen=[]
    monkeypatch.setattr(script,'collect_saved_run',lambda *a,**kw:seen.append((a,kw)))
    script.main([stage,'--project','runs/existing'])
    assert seen[0][0]==('runs/existing',stage)
    assert seen[0][1]==({} if stage=='collect' else {'weight_per_bixel':getattr(script,
        'FORWARD_WEIGHT_PER_BIXEL' if script is patient else 'FORWARD_WEIGHT')})


@pytest.mark.parametrize('script',[patient,water])
@pytest.mark.parametrize('stage',['inspect','prepare'])
def test_planning_stages_pass_settings(script,stage,monkeypatch):
    module=script.patient if script is patient else script.water
    seen=[]
    monkeypatch.setattr(module,stage,lambda settings,path:seen.append((settings,path)))
    script.main([stage,'--project','projects/chosen'])
    assert seen[0][1]=='projects/chosen'
    assert seen[0][0].histories_per_job==script.HISTORIES_PER_JOB
