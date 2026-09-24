import json
import numpy as np
import pytest
from scipy.io import loadmat
from minibeam.geometry.configuration import GeometryConfig
from minibeam.geometry.assembly import TreatmentHead
from minibeam.geometry.patient_report import patient_scene
from minibeam.steering import enrich_stf
from minibeam.workflow.geometry import resolve_geometry
from minibeam.workflow.study import effective_shift_fractions
from minibeam.topas.manifest import load_manifest


@pytest.mark.parametrize('enabled',[True,False])
@pytest.mark.parametrize('toml_enabled',[True,False])
def test_switch_overrides_toml_and_loads_once(tmp_path,monkeypatch,enabled,toml_enabled):
    data=GeometryConfig.load().data
    data['aperture']['enabled']=toml_enabled
    path=tmp_path/'head.toml';path.write_text(GeometryConfig.from_data(data).text)
    before=path.read_bytes()
    original=GeometryConfig.load
    calls=[]
    def load(path=None):
        calls.append(path)
        return original(path)
    monkeypatch.setattr(GeometryConfig,'load',load)
    geometry=resolve_geometry(config_path=path,enable_collimator=enabled)
    assert calls==[path]
    assert geometry.aperture.enabled is enabled
    assert geometry.mlc.model_dump()==data['mlc']
    assert geometry.jaws.model_dump()==data['jaws']
    assert path.read_bytes()==before


def test_disabled_assembly_conflict(tmp_path):
    data=GeometryConfig.load().data;data['assembly']['enabled']=False
    path=tmp_path/'head.toml';path.write_text(GeometryConfig.from_data(data).text)
    with pytest.raises(ValueError,match='assembly.enabled=false'):
        resolve_geometry(config_path=path,enable_collimator=True)
    assert not resolve_geometry(config_path=path,enable_collimator=False).assembly.enabled


@pytest.mark.parametrize('value',[None,1,0,'false',np.bool_(True)])
def test_switch_requires_boolean(value):
    with pytest.raises(ValueError,match='ENABLE_COLLIMATOR'):
        effective_shift_fractions(enable_collimator=value,values=[])


def test_disabled_workflow_ignores_overrides_and_preserves_hardware(case,tmp_path,monkeypatch,capsys):
    import patient_workflow as workflow
    from test_workflows import setup_workflow
    from test_results import fake_results
    setup_workflow(monkeypatch,case,tmp_path)
    monkeypatch.setattr(workflow,'ENABLE_COLLIMATOR',False)
    # Deliberately invalid overrides must be ignored while disabled.
    monkeypatch.setattr(workflow,'SLIT_ENTRANCE_WIDTH_MM',object())
    monkeypatch.setattr(workflow,'NOMINAL_ENTRANCE_CTC_MM',None)
    requests=[]
    original=GeometryConfig.load
    calls=[]
    def load(path=None):
        calls.append(path)
        return original(path)
    monkeypatch.setattr(GeometryConfig,'load',load)
    for index,fractions in enumerate([[],[0.,0.25,0.5,0.75],{'invalid':'inactive'}]):
        monkeypatch.setattr(workflow,'COLLIMATOR_SHIFT_FRACTIONS',fractions)
        assert workflow.parse_arguments(['prepare']).project_dir=='projects/patient'
        root=tmp_path/f'without-{index}'
        calls.clear()
        workflow.main(['prepare','--project',str(root)])
        assert calls==[None], 'The writer must use attached geometry rather than reread TOML'
        m=load_manifest(root);requests.append(m['request_id'])
        assert not (root/'study.json').exists()
        assert not list(root.glob('shift_*'))
        for b in m['beam_geometry']:
            assert b['geometry']['aperture']['enabled'] is False
            assert b['resolved']['aperture'] is None
            assert {e['name'] for e in b['resolved']['envelopes']}=={'mlc','jaws'}
        for job in m['jobs']:
            text=(root/job['parameter_file']).read_text()
            assert 'Ge/Collimator/' not in text and 'Ge/CollimatorAir/' not in text and 'Ge/Blade_' not in text
            assert 'Ge/MLCPositive/' in text and 'Ge/LowerJawPositive/' in text
        saved=loadmat(root/'derived/steering.mat',simplify_cells=True,variable_names=['beam_geometry'])
        assert all(not b['geometry']['aperture']['enabled'] for b in saved['beam_geometry'])
        fake_results(root)
        for stage in ['collect','forward']:
            calls.clear()
            workflow.main([stage,'--project',str(root)])
            assert calls==[]
        assert (root/'derived/result.mat').is_file()
        assert (root/'derived/dose.mha').is_file()
    assert len(set(requests))==1
    assert 'overrides are inactive' in capsys.readouterr().out
    monkeypatch.setattr(workflow,'ENABLE_COLLIMATOR',True)
    monkeypatch.setattr(workflow,'COLLIMATOR_SHIFT_FRACTIONS',[])
    workflow.main(['collect','--project',str(root)])


def test_report_scene_has_only_remaining_hardware(case):
    ct,_,_,native,_=case
    head=TreatmentHead(None)
    off=enrich_stf(native,resolve_geometry(config_path=None,enable_collimator=False))
    on=enrich_stf(native,resolve_geometry(config_path=None,enable_collimator=True))
    before=patient_scene(ct,on,head)
    after=patient_scene(ct,off,head)
    for a,b in zip(before['beams'],after['beams']):
        expected={name:points for name,points in a['solids'] if name.startswith(('MLC','LowerJaw'))}
        assert {name for name,_ in b['solids']}==set(expected)
        for name,points in b['solids']:np.testing.assert_array_equal(points,expected[name])
        np.testing.assert_array_equal(a['source'],b['source'])
        np.testing.assert_array_equal(a['iso'],b['iso'])


def test_reenable_restores_layout_and_disabled_rejects_study_root(tmp_path,monkeypatch):
    import patient_workflow as workflow
    fractions=[0.,0.25,0.5,0.75]
    monkeypatch.setattr(workflow,'COLLIMATOR_SHIFT_FRACTIONS',fractions)
    monkeypatch.setattr(workflow,'ENABLE_COLLIMATOR',False)
    assert workflow.parse_arguments(['prepare']).project_dir=='projects/patient'
    (tmp_path/'study.json').write_text('{}')
    with pytest.raises(SystemExit,match='study directory'):
        workflow.main(['prepare','--project',str(tmp_path)])
    monkeypatch.setattr(workflow,'ENABLE_COLLIMATOR',True)
    assert workflow.parse_arguments(['prepare']).project_dir=='projects/patient-slit-study'
    np.testing.assert_array_equal(effective_shift_fractions(enable_collimator=True,values=fractions),fractions)
    assert workflow.COLLIMATOR_SHIFT_FRACTIONS is fractions
