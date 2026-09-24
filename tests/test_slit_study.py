import json
import numpy as np
import pytest
from minibeam.topas.manifest import load_manifest


@pytest.mark.parametrize('only_central',[False,True])
@pytest.mark.parametrize('fractions',[[0.0],[0.0,0.25,0.5,0.75]])
def test_shift_setup_workflow(case,tmp_path,monkeypatch,only_central,fractions):
    import patient_workflow as study
    from test_results import fake_results
    ct,plan,cst,_,_=case
    monkeypatch.setattr(study,'GANTRY_ANGLES',[45.,135.,225.,315.])
    monkeypatch.setattr(study,'BEAMLET_EXECUTION','separate')
    monkeypatch.setattr(study,'COLLIMATOR_SHIFT_FRACTIONS',fractions)
    monkeypatch.setattr(study,'ONLY_CENTRAL_BEAMLET',only_central)
    monkeypatch.setattr(study,'load_patient',lambda _: (ct,cst))
    monkeypatch.setattr(study,'read_roi_metadata',lambda *a: {'omitted_rois':[],'original_dicom_rois':[]})
    monkeypatch.setattr(study,'TARGET','TARGET')
    monkeypatch.setattr(study,'SOURCE_TYPE','point')
    monkeypatch.setattr(study,'HISTORIES_PER_JOB',10)
    monkeypatch.setattr(study,'WATER',True)
    original=study.generate_stf
    calls=[]
    def once(*args):
        calls.append(True)
        return original(*args)
    monkeypatch.setattr(study,'generate_stf',once)
    root=tmp_path/'study'
    study.main(['inspect','--project',str(root)])
    assert not root.exists()
    calls.clear()
    study.main(['prepare','--project',str(root)])
    assert len(calls)==1
    index=json.loads((root/'study.json').read_text())
    assert len(index['setups'])==len(fractions)
    assert index['settings']['target_covering_bixels']==(not only_central)
    assert index['settings']['shifts']==study.COLLIMATOR_SHIFT_FRACTIONS
    assert [entry['directory'] for entry in index['setups']]==[f'shift_{round(f*100):03d}' for f in fractions]
    assert [entry['shift_fraction'] for entry in index['setups']]==study.COLLIMATOR_SHIFT_FRACTIONS
    assert [entry['lateral_shift_mm'] for entry in index['setups']]==[f*6 for f in fractions]
    requests=[]
    associations=[]
    for entry,shift in zip(index['setups'],[f*6 for f in fractions]):
        assert entry['preparation_status']=='prepared'
        path=root/entry['directory'];m=load_manifest(path)
        assert (len(m['jobs'])>4) if not only_central else (len(m['jobs'])==4)
        associations.append([(j['beam_index'],j['ray_index'],j['beamlet_index'],j['source']) for j in m['jobs']])
        assert {j['gantry_angle_deg'] for j in m['jobs']}=={45.,135.,225.,315.}
        assert [j['bixel_index'] for j in m['jobs']]==list(range(1,len(m['jobs'])+1))
        assert all(b['geometry']['aperture']['lateral_shift_mm']==shift for b in m['beam_geometry'])
        assert entry['beam_geometry']==m['beam_geometry']
        requests.append(m['request_id'])
        fake_results(path)
    assert len(set(requests))==len(fractions)
    assert all(a==associations[0] for a in associations)
    for stage in ['collect','forward']:
        study.main([stage,'--project',str(root)])
    for entry in index['setups']:
        assert (root/entry['directory']/'derived/result.mat').exists()
        assert (root/entry['directory']/'derived/dose.mha').exists()
    before=(root/'study.json').read_bytes()
    monkeypatch.setattr(study,'ONLY_CENTRAL_BEAMLET',not only_central)
    study.main(['collect','--project',str(root)])
    after=json.loads((root/'study.json').read_text())
    assert after['settings']==json.loads(before)['settings']
    assert all(e['last_stage_status']=='complete' for e in after['setups'])


@pytest.mark.parametrize('fractions,default',[([], 'projects/patient'),([0.0], 'projects/patient-slit-study')])
def test_array_cli_defaults(monkeypatch,fractions,default):
    import patient_workflow as workflow
    monkeypatch.setattr(workflow,'COLLIMATOR_SHIFT_FRACTIONS',fractions)
    assert workflow.parse_arguments(['prepare']).project_dir == default
    assert workflow.parse_arguments(['collect','--project','runs/custom']).project_dir == 'runs/custom'
    with pytest.raises(SystemExit):
        workflow.parse_arguments(['prepare','--study'])


@pytest.mark.parametrize('fractions,marker', [([], 'study.json'), ([0.0], 'manifest.json')])
def test_wrong_directory_mode_explained(tmp_path,monkeypatch,fractions,marker):
    import patient_workflow as workflow
    monkeypatch.setattr(workflow,'COLLIMATOR_SHIFT_FRACTIONS',fractions)
    (tmp_path/marker).write_text('{}')
    with pytest.raises(SystemExit,match='directory'):
        workflow.main(['prepare','--project',str(tmp_path)])


@pytest.mark.parametrize('fractions',[0.0,None,[[0.0]],[-0.1],[1.0],[float('nan')],[float('inf')],[0.0,0.001]])
def test_invalid_arrays(monkeypatch,fractions):
    import patient_workflow as workflow
    monkeypatch.setattr(workflow,'COLLIMATOR_SHIFT_FRACTIONS',fractions)
    with pytest.raises(ValueError):
        workflow.main(['prepare'])


@pytest.mark.parametrize('fractions',[[],[0.0]])
def test_geometry_config_and_empty_array(case,tmp_path,monkeypatch,fractions):
    import patient_workflow as workflow
    from test_workflows import setup_workflow
    from minibeam.geometry.configuration import GeometryConfig
    from test_results import fake_results
    setup_workflow(monkeypatch,case,tmp_path)
    monkeypatch.setattr(workflow,'COLLIMATOR_SHIFT_FRACTIONS',fractions)
    data=GeometryConfig.load().data
    data['field']['width_mm']=80.0
    data['aperture'].update(slit_entrance_width_mm=1.0,slit_entrance_ctc_mm=5.0,lateral_shift_mm=1.25)
    config=tmp_path/'custom.toml'
    config.write_text(GeometryConfig.from_data(data).text)
    monkeypatch.setattr(workflow,'GEOMETRY_CONFIG',str(config))
    if not fractions:
        # Unused slit overrides must not affect the configured single run.
        monkeypatch.setattr(workflow,'SLIT_ENTRANCE_WIDTH_MM',999.0)
        monkeypatch.setattr(workflow,'NOMINAL_ENTRANCE_CTC_MM',1.0)
    root=tmp_path/'run'
    workflow.main(['prepare','--project',str(root)])
    run=root/'shift_000' if fractions else root
    manifest=load_manifest(run)
    assert (root/'study.json').exists()==bool(fractions)
    for record in manifest['beam_geometry']:
        geometry=record['geometry']
        assert geometry['field']['width_mm']==80.0
        assert geometry['aperture']['lateral_shift_mm']==(0.0 if fractions else 1.25)
        assert geometry['aperture']['slit_entrance_width_mm']==(2.0 if fractions else 1.0)
    fake_results(run)
    for stage in ['collect','forward']:
        workflow.main([stage,'--project',str(root)])
    assert (run/'derived/result.mat').exists()
    assert (run/'derived/dose.mha').exists()


@pytest.mark.parametrize('fractions',[[],[0.0]])
@pytest.mark.parametrize('target',[None,'TARGET'])
def test_inspect_lists_rois_and_optional_estimates(case,tmp_path,monkeypatch,capsys,fractions,target):
    import patient_workflow as workflow
    from test_workflows import setup_workflow
    setup_workflow(monkeypatch,case,tmp_path,target=target)
    monkeypatch.setattr(workflow,'COLLIMATOR_SHIFT_FRACTIONS',fractions)
    monkeypatch.setattr(workflow,'read_roi_metadata',lambda *a: {'omitted_rois':[], 'original_dicom_rois':[{'number':38,'name':'TARGET'}]})
    root=tmp_path/'inspect'
    workflow.main(['inspect','--project',str(root)])
    output=capsys.readouterr().out
    assert '38  TARGET' in output
    assert ('CSV estimate' in output)==(target is not None)
    assert not root.exists()


@pytest.fixture(autouse=True)
def explicit_collimator_setting(monkeypatch):
    import patient_workflow
    monkeypatch.setattr(patient_workflow,'ENABLE_COLLIMATOR',True)
