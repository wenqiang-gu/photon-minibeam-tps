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
    monkeypatch.setattr(patient_workflow,'COLLIMATOR_ROTATION_DEG',[])

@pytest.mark.parametrize('rotations,fractions,count', [([],[],0),([0.],[],1),([],[0.],1),([0.,45.,90.],[0.,.25,.5,.75],12)])
def test_rotation_shift_product(rotations,fractions,count):
    from minibeam.workflow.study import setup_settings
    from minibeam.geometry.models import BeamGeometry
    geometry=BeamGeometry.from_config().replace(aperture={'rotation_z_deg':12.,'lateral_shift_mm':1.25})
    setups=setup_settings(slit_width_mm=2.,nominal_ctc_mm=6.,collimator_shift_fractions=fractions,
                          collimator_rotations=rotations,geometry=geometry)
    assert len(setups)==count
    for name,fraction,angle,config in setups:
        assert config.aperture.rotation_z_deg==(12. if angle is None else 12.+angle)
        assert config.aperture.lateral_shift_mm==(1.25 if fraction is None else fraction*6)
    if count==12:
        assert [s[0] for s in setups]==[f'rotation_{int(r):03d}_shift_{int(f*100):03d}' for r in rotations for f in fractions]


@pytest.mark.parametrize('values', [None,0.,[[0.]],['90'],[True],[0.,False],[float('nan')],[float('inf')],[0.,0.],[-0.,0.]])
def test_invalid_rotation_arrays(values):
    from minibeam.workflow.study import effective_rotations
    with pytest.raises(ValueError):effective_rotations(enable_collimator=True,values=values)
    assert not effective_rotations(enable_collimator=False,values=values).size


def test_rotation_names():
    from minibeam.workflow.study import rotation_name
    assert [rotation_name(a) for a in [-45,22.5,90]]==['rotation_m045','rotation_022p5','rotation_090']


@pytest.mark.parametrize('mode',['separate','combined'])
def test_rotation_study_saved_collection(case,tmp_path,monkeypatch,mode):
    import shutil
    import patient_workflow as workflow
    from test_workflows import setup_workflow
    from test_results import fake_results
    from scipy.io import loadmat
    setup_workflow(monkeypatch,case,tmp_path)
    monkeypatch.setattr(workflow,'read_roi_metadata',lambda *a: {'omitted_rois':[],'original_dicom_rois':[]})
    monkeypatch.setattr(workflow,'COLLIMATOR_ROTATION_DEG',[0.,45.,90.])
    monkeypatch.setattr(workflow,'COLLIMATOR_SHIFT_FRACTIONS',[0.,.25,.5,.75])
    monkeypatch.setattr(workflow,'BEAMLET_EXECUTION',mode)
    calls=[];original=workflow.generate_stf
    def once(*a):calls.append(1);return original(*a)
    monkeypatch.setattr(workflow,'generate_stf',once)
    root=tmp_path/'study';workflow.prepare(root)
    assert len(calls)==1
    index=json.loads((root/'study.json').read_text());assert len(index['setups'])==12
    identities=[]
    for entry in index['setups']:
        assert entry['rotation_convention']=='toml_baseline_plus_offset_v1'
        assert entry['baseline_rotation_z_deg']==-90.
        assert entry['additional_rotation_deg']==entry['rotation_deg']
        assert entry['resolved_rotation_z_deg']==entry['baseline_rotation_z_deg']+entry['additional_rotation_deg']
        run=root/entry['directory'];m=load_manifest(run);fake_results(run)
        identities.append(m['request_id'])
        for beam in m['beam_geometry']:
            ap=beam['geometry']['aperture']
            assert ap['rotation_z_deg']==entry['rotation_deg']-90
            assert ap['lateral_shift_mm']==entry['lateral_shift_mm']
        saved=loadmat(run/'derived/steering.mat',simplify_cells=True)
        assert saved['beam_geometry'][0]['geometry']['aperture']['rotation_z_deg']==entry['rotation_deg']-90
    assert len(set(identities))==12
    moved=tmp_path/'moved';shutil.move(root,moved)
    monkeypatch.setattr(workflow,'COLLIMATOR_ROTATION_DEG','invalid current settings')
    monkeypatch.setattr(workflow,'load_patient',lambda *a:pytest.fail('saved collection loaded patient'))
    workflow.collect(moved);workflow.forward(moved)
    for entry in index['setups']:
        assert (moved/entry['directory']/'derived/result.mat').exists()
        assert (moved/entry['directory']/'derived/dose.mha').exists()

@pytest.mark.parametrize('enabled,rotations,fractions,expected', [
    (True,[],[],'projects/patient'),(True,[0.],[],'projects/patient-slit-study'),
    (True,[],[0.],'projects/patient-slit-study'),(False,'ignored','ignored','projects/patient')])
def test_rotation_directory_defaults(monkeypatch,enabled,rotations,fractions,expected):
    import patient_workflow as workflow
    monkeypatch.setattr(workflow,'ENABLE_COLLIMATOR',enabled)
    monkeypatch.setattr(workflow,'COLLIMATOR_ROTATION_DEG',rotations)
    monkeypatch.setattr(workflow,'COLLIMATOR_SHIFT_FRACTIONS',fractions)
    assert workflow.parse_arguments(['prepare']).project_dir==expected


def test_rotation_only_retains_toml(case,tmp_path,monkeypatch):
    import patient_workflow as workflow
    from test_workflows import setup_workflow
    from minibeam.geometry.configuration import GeometryConfig
    setup_workflow(monkeypatch,case,tmp_path)
    monkeypatch.setattr(workflow,'read_roi_metadata',lambda *a: {'omitted_rois':[],'original_dicom_rois':[]})
    monkeypatch.setattr(workflow,'COLLIMATOR_ROTATION_DEG',[22.5])
    monkeypatch.setattr(workflow,'SLIT_ENTRANCE_WIDTH_MM',None)
    monkeypatch.setattr(workflow,'NOMINAL_ENTRANCE_CTC_MM',None)
    root=tmp_path/'rotation-only';workflow.prepare(root)
    index=json.loads((root/'study.json').read_text());entry=index['setups'][0]
    assert entry['directory']=='rotation_022p5'
    assert entry['shift_fraction'] is None and not entry['shift_override_applied']
    m=load_manifest(root/entry['directory']);ap=m['beam_geometry'][0]['geometry']['aperture']
    base=GeometryConfig.load().data['aperture']
    for key in ['slit_entrance_width_mm','slit_entrance_ctc_mm','lateral_shift_mm']:assert ap[key]==base[key]
    assert ap['rotation_z_deg']==-67.5
    monkeypatch.setattr(workflow,'COLLIMATOR_ROTATION_DEG',[45.])
    with pytest.raises(SystemExit,match='settings changed'):workflow.prepare(root)


@pytest.mark.parametrize('baseline',[-90.,0.,12.])
@pytest.mark.parametrize('shift',[0.,.25,.5,.75])
def test_empty_equals_zero_rotation_with_tilts(baseline,shift):
    from minibeam.workflow.study import setup_settings
    from minibeam.geometry.models import BeamGeometry
    geo=BeamGeometry.from_config().replace(aperture={'rotation_z_deg':baseline,'rotation_x_deg':2.,'rotation_y_deg':3.})
    args=dict(slit_width_mm=2.,nominal_ctc_mm=6.,collimator_shift_fractions=[shift],geometry=geo)
    implicit=setup_settings(**args,collimator_rotations=[])[0]
    explicit=setup_settings(**args,collimator_rotations=[0.])[0]
    assert implicit[3]==explicit[3]
    assert implicit[0].startswith('shift_')
    assert explicit[0].startswith('rotation_000_shift_')
    setups=setup_settings(**args,collimator_rotations=[0.,45.,90.])
    assert [s[3].aperture.rotation_z_deg for s in setups]==[baseline,baseline+45,baseline+90]
    assert geo.aperture.rotation_z_deg==baseline
