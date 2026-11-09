"""Recording-plane placement, portable batches and real TOPAS phase-space output."""
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

import water_phantom as water
from minibeam import TOPASPhotonEngine
from minibeam.geometry.assembly import TreatmentHead
from minibeam.geometry.coordinates import patient_center
from minibeam.geometry.spatial import corners, placement_rotation
from minibeam.topas.manifest import load_manifest
from minibeam.topas.phase_space_recording import resolve_recorders, validate_recorders
from projects import add_batches as batches
from test_water_workflow import small_water


@pytest.fixture
def settings(monkeypatch):
    small_water(monkeypatch)
    monkeypatch.setattr(water, 'FIELD_SIZE_MM', 20.)
    monkeypatch.setattr(water, 'NUMBER_OF_BATCHES', 1)
    monkeypatch.setattr(water, 'ENABLE_COLLIMATOR', True)
    monkeypatch.setattr(water, 'PHASE_SPACE_RECORDERS', [])
    return water.planning_settings()


def planning(settings, tmp_path):
    return water.water.build_planning(settings, tmp_path/'run')


def test_geometry_locations(settings, tmp_path):
    _, ct, _, _, stf, _ = planning(settings, tmp_path)
    beam=stf.beams[0]; head=TreatmentHead(None)
    config=[dict(name=loc, location=loc) for loc in ('mlc_exit','jaw_exit','collimator_exit')]
    records=resolve_recorders(config, beam, (head,), patient_center(ct))
    envelopes=head.resolved(beam)[2]
    for r,e in zip(records,envelopes):
        assert r['resolved_source_distance_mm']==pytest.approx(corners(e)[:,2].max()+beam.sad+.01)
        np.testing.assert_allclose(r['surface_center_lps_mm'],[0,r['surface_position_beam_mm'][2],0],atol=1e-12)
    assert records[-1]['resolved_source_distance_mm']==pytest.approx(757.01)


def test_tilted_exit_and_changed_ssd(settings,tmp_path):
    settings=replace(settings,ssd_mm=1100.)
    _,ct,_,_,stf,_=planning(settings,tmp_path)
    beam=stf.beams[0]
    beam.geometry=beam.geometry.replace(aperture={'rotation_x_deg':12.,'rotation_y_deg':5.,'thickness_mm':80.})
    head=TreatmentHead(None)
    e=head.resolved(beam)[2][-1]
    r=resolve_recorders([dict(name='exit',location='collimator_exit',offset_mm=.2)],beam,(head,),patient_center(ct))[0]
    assert r['resolved_source_distance_mm']==pytest.approx(corners(e)[:,2].max()+1100+.2)
    assert r['surface_position_beam_mm'][2]==pytest.approx(r['resolved_source_distance_mm']-1100)
    # Film remains perpendicular to beam even when the device is tilted.
    np.testing.assert_allclose(placement_rotation(*r['rotation_deg'])[:,2],r['normal_lps'],atol=1e-12)


@pytest.mark.parametrize('bad',[
    None, {}, [dict(name='../bad',location='jaw_exit')],
    [dict(name='a',location='unknown')],
    [dict(name='a',location='source_distance')],
    [dict(name='a',location='source_distance',source_distance_mm=500,offset_mm=0)],
    [dict(name='a',location='jaw_exit',source_distance_mm=500)],
    [dict(name='a',location='jaw_exit',offset_mm=-1)],
    [dict(name='a',location='source_distance',source_distance_mm=float('nan'))],
    [dict(name='a',location='source_distance',source_distance_mm=True)],
    [dict(name='a',location='jaw_exit',size_mm=[0,10])],
    [dict(name='a',location='jaw_exit',size_mm=[10])],
    [dict(name='a',location='jaw_exit')]*2,
])
def test_invalid_settings(bad):
    with pytest.raises(ValueError):validate_recorders(bad)


def test_disabled_and_coincident(settings,tmp_path):
    settings=replace(settings,enable_collimator=False)
    _,ct,_,_,stf,_=planning(settings,tmp_path)
    args=(stf.beams[0],(TreatmentHead(None),),patient_center(ct))
    with pytest.raises(ValueError,match='requires an enabled'):
        resolve_recorders([dict(name='exit',location='collimator_exit')],*args)
    explicit=dict(name='reference',location='source_distance',source_distance_mm=757.01)
    assert resolve_recorders([explicit],*args)[0]['resolved_source_distance_mm']==757.01
    with pytest.raises(ValueError,match='Coincident'):
        resolve_recorders([explicit,dict(explicit,name='other')],*args)


def test_inspect_and_batch_outputs(settings,tmp_path,capsys):
    config=[dict(name='jaw',location='jaw_exit'),
            dict(name='exit',location='collimator_exit')]
    settings=replace(settings,phase_space_recorders=config,number_of_batches=2)
    root=tmp_path/'run'
    water.water.inspect(settings,root)
    assert 'Phase-space recorder' in capsys.readouterr().out
    assert not root.exists()
    water.water.prepare(settings,root)
    manifest=load_manifest(root)
    records=batches.inventory(root,manifest)
    assert len(records)==2
    bases=[]
    for batch in records:
        job=batch['jobs'][0]
        text=(root/job['parameter_file']).read_text()
        assert 'Quantity = "DoseToMedium"' in text
        assert text.count('Quantity = "PhaseSpace"')==2
        assert text.count('IsParallel = "True"')==2
        for recorder in job['phase_space_recorders']:
            assert f's:Sc/{recorder["scorer"]}/OutputFile = "{recorder["output_base"]}"' in text
            assert recorder['normalization_histories']==100
            assert recorder['transported_histories']==100
            assert 'b:Sc/' + recorder['scorer'] + '/OutputToConsole = "False"' in text
            bases.append(recorder['output_base'])
    assert len(set(bases))==4
    batches.add(root,1)
    assert len(batches.inventory(root,manifest))==3
    # A fresh process using only portable stdlib batch code can validate/add too.
    subprocess.run([os.sys.executable,'-m','projects.add_batches',str(root),'--count','1','--dry-run'],check=True,capture_output=True,text=True)
    changed=replace(settings,phase_space_recorders=[dict(name='changed',location='jaw_exit')])
    with pytest.raises(ValueError,match='different inputs/settings'):
        water.water.prepare(changed,root)
    fake=records[1]['jobs'][0].copy()
    fake['phase_space_recorders']=[dict(r) for r in fake['phase_space_recorders']]
    fake['phase_space_recorders'][0]['output_base']='results/wrong'
    # Integrity checks must detect a modified auxiliary output path.
    record=json.loads((root/'batches/batch_001/batch.json').read_text())
    record['jobs'][0]=fake
    record['batch_fingerprint']=batches.digest({k:v for k,v in record.items() if k!='batch_fingerprint'})
    (root/'batches/batch_001/batch.json').write_text(json.dumps(record))
    index=json.loads((root/'batches.json').read_text())
    index['batches'][0]['batch_fingerprint']=record['batch_fingerprint']
    index['index_id']=batches.digest({k:v for k,v in index.items() if k!='index_id'})
    (root/'batches.json').write_text(json.dumps(index))
    with pytest.raises(ValueError,match='Invalid batch phase-space recorder metadata'):
        batches.inventory(root,manifest)


def test_disabled_recorder_and_world_bounds(settings,tmp_path):
    _,ct,cst,plan,stf,_=planning(settings,tmp_path)
    engine=TOPASPhotonEngine(plan)
    definition=engine._definition(ct,stf)[0]
    assert 'phase_space_recorders' not in definition['jobs'][0]
    assert 'PhspFilm_' not in definition['sources'][0]
    engine=TOPASPhotonEngine(plan,phase_space_recorders=[dict(name='far',location='source_distance',source_distance_mm=5000.,size_mm=[4000.,300.])])
    definition=engine._definition(ct,stf)[0]
    assert definition['world_half_mm']>4000


def test_replay_normalization(settings,tmp_path):
    from test_phase_space import make_iaea
    base=make_iaea(tmp_path/'source')
    settings=replace(settings,source_type='phase_space',phase_space_file_bases=[str(base)],
        histories_per_job=12,phase_space_selection='all_forward',
        phase_space_recorders=[dict(name='exit',location='collimator_exit')])
    _,ct,_,plan,stf,_=planning(settings,tmp_path)
    job=TOPASPhotonEngine(plan)._definition(ct,stf)[0]['jobs'][0]
    r=job['phase_space_recorders'][0]
    assert r['normalization_histories']==12
    assert r['transported_histories']==job['histories']<12
    assert r['upstream_omitted_histories']==12-job['histories']
    assert r['source_normalization']=='original_accelerator_history'


@pytest.mark.topas
@pytest.mark.parametrize('threads', [1, 2])
def test_topas_binary_recorders(settings,tmp_path,threads):
    executable=os.environ.get('PHOTON_TPS_TOPAS')
    if not executable:pytest.skip('Set PHOTON_TPS_TOPAS for real transport')
    # Two material-free planes after the jaws, in front of water.
    settings=replace(settings,enable_collimator=False,histories_per_job=1000,number_of_batches=2,
        topas_threads_per_job=threads,
        phase_space_recorders=[dict(name='upstream',location='source_distance',source_distance_mm=800.),
                               dict(name='downstream',location='source_distance',source_distance_mm=900.),
                               dict(name='tiny',location='source_distance',source_distance_mm=950.,size_mm=[.002,.002])])
    root=tmp_path/'transport';water.water.prepare(settings,root)
    manifest=load_manifest(root)
    for batch in batches.inventory(root,manifest):
        job=batch['jobs'][0]
        result=subprocess.run([executable,job['parameter_file']],cwd=root,capture_output=True,text=True,timeout=60)
        assert result.returncode==0,(result.stdout+result.stderr)[-6000:]
        assert (root/job['output']).is_file()
        for recorder in job['phase_space_recorders']:
            base=root/recorder['output_base']
            header=Path(str(base)+'.header').read_text()
            assert 'Number of Original Histories: 1000' in header
            assert Path(str(base)+'.phsp').stat().st_size>0
            assert 'Binary' in header
            assert 'Overlap is detected' not in result.stdout+result.stderr
            from minibeam.sources.iaea import TOPAS_DTYPE
            data=np.fromfile(str(base)+'.phsp',dtype=TOPAS_DTYPE)
            physical=data[data['weight']>0]
            assert int(physical['new_history'].sum()-data['weight'][data['weight']<0].sum())==1000
            if recorder['name']=='tiny':
                assert len(physical)==0
                continue
            assert len(physical)>0
            positions=np.column_stack([physical[a] for a in ('x','y','z')])*10  # TOPAS outputs cm.
            np.testing.assert_allclose(positions[:,1],recorder['surface_center_topas_mm'][1],atol=1e-3)
            assert np.all(physical['v']>0)  # Water beam points along World +Y.
            assert np.all(physical['weight']==1)
        existing={str(p):batches.sha256(p) for p in (root/Path(job['output']).parent).glob('*')}
        retry=subprocess.run([executable,job['parameter_file']],cwd=root,capture_output=True,text=True,timeout=60)
        assert retry.returncode!=0
        assert all(batches.sha256(Path(p))==sha for p,sha in existing.items())
