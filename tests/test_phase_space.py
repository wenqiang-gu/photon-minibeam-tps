import json
from pathlib import Path
import shutil
import numpy as np
import pytest
from minibeam import TOPASPhotonEngine
from minibeam.sources.iaea import IAEA_DTYPE,TOPAS_DTYPE,read_header,validated_chunks
from minibeam.sources.phase_space import PhaseSpaceBeamletSource
from minibeam.topas.contracts import JobContext
from minibeam.geometry.coordinates import patient_center
from minibeam.geometry.spatial import beam_basis
from minibeam.topas.manifest import load_manifest


def make_iaea(base, records=None, original=12):
    if records is None:
        records=[(1,-1.,0.,0.,0.,0.,.05,2,0), (2,.2,.02,0.,0.,0.,.05,0,0),
                 (1,-2.,.3,0.,0.,0.,.05,3,0), (3,-1.,-.1,0.,0.,0.,.05,2,0),
                 (1,1.,0.,0.,0.,0.,.05,0,0)]
    a=np.array(records,dtype=IAEA_DTYPE);a.tofile(str(base)+'.phsp')
    counts=[int((np.abs(a['code'])==code).sum()) for code in [1,2,3]]
    Path(str(base)+'.header').write_text(f'''$FILE_TYPE:
0
$CHECKSUM:
{a.nbytes}
$RECORD_CONTENTS:
1
1
0
1
1
1
1
0
2
1
2
$RECORD_CONSTANT:
27.21
$RECORD_LENGTH:
33
$BYTE_ORDER:
1234
$ORIG_HISTORIES:
{original}
$PARTICLES:
{len(a)}
$PHOTONS:
{counts[0]}
$ELECTRONS:
{counts[1]}
$POSITRONS:
{counts[2]}
''')
    return base


@pytest.fixture
def phase_base(tmp_path):
    return make_iaea(tmp_path/'sample')


def context(case,index=0):
    ct,_,_,stf,_=case;beam=stf.beams[index];ray=beam.rays[0]
    return JobContext(beam,ray,ray.beamlets[0],patient_center(ct),index+1,index+1,1,1)


def test_streaming_layout_and_empty_histories(phase_base):
    info=read_header(phase_base);audit={}
    chunks=list(validated_chunks(info,audit,chunk_size=1))
    assert [int(ids[0]) for _,ids in chunks]==[2,2,5,7,7]
    assert audit['trailing_empty_histories']==5
    assert audit['nonempty_histories']==3
    assert audit['species_counts']=={'photons':3,'electrons':1,'positrons':1}
    assert info['plane_distance_mm']==272.1


@pytest.mark.parametrize('field,value', [('weight',-1),('energy',float('nan')),('x',float('inf')),
    ('u',2),('code',4),('increment',-1),('increment',0)])
def test_invalid_records(phase_base,field,value):
    p=Path(str(phase_base)+'.phsp');a=np.fromfile(p,dtype=IAEA_DTYPE);a[0][field]=value;a.tofile(p)
    with pytest.raises(ValueError):list(validated_chunks(read_header(phase_base),{}))


def test_unsupported_and_truncated(phase_base):
    p=Path(str(phase_base)+'.header');text=p.read_text();p.write_text(text.replace('1234','4321'))
    with pytest.raises(ValueError,match='Unsupported'):read_header(phase_base)
    p.write_text(text)
    with Path(str(phase_base)+'.phsp').open('ab') as f:f.write(b'!')
    with pytest.raises(ValueError,match='size'):read_header(phase_base)


def test_selection_grouping_assets_and_energy(case,phase_base):
    source=PhaseSpaceBeamletSource(phase_base);source.prepare(case[3],10)
    assert len(source.selections)==1
    part=source.render(context(case));record=part.record
    assert record['selected_particles']==4 and record['nonempty_selected_histories']==2
    assert record['empty_histories']==8 and record['excluded_particles']==1
    particles=np.fromfile(part.assets[1][1],dtype=TOPAS_DTYPE)
    assert particles['new_history'].tolist()==[1,0,1,0]
    assert particles['pdg'].tolist()==[22,11,-11,22]
    np.testing.assert_allclose(particles['weight'],.05)
    assert 'BeamEnergy' not in part.text and 'BeamAngular' not in part.text
    assert 'PhaseSpaceMultipleUse = 1' in part.text
    assert source.render(context(case,1)).assets==part.assets
    for i in [0,1]:
        c=context(case,i);r=source.render(c).record
        expected=c.beam.iso_center+c.beam.source_point+beam_basis(c.beam)[:,2]*272.1
        np.testing.assert_allclose(r['phase_space_plane_lps_mm'],expected)
        np.testing.assert_allclose(r['local_to_world'],beam_basis(c.beam))


def test_prefix_empty_selection_and_no_recycling(case,phase_base):
    source=PhaseSpaceBeamletSource(phase_base)
    with pytest.raises(ValueError,match='exceed'):source.prepare(case[3],13)
    source.prepare(case[3],2)
    part=source.render(context(case))
    assert part.record['selected_particles']==2 and part.record['empty_histories']==1
    case[3].beams[0].rays[0].ray_pos=beam_basis(case[3].beams[0])@np.array([100.,0,0])
    with pytest.raises(ValueError,match='selects no particles'):source.prepare(case[3],10)


def test_square_partition_boundaries_and_overlap(case,tmp_path):
    # Boundary at x=2.5 mm belongs only to the tile centered at 5 mm.
    base=make_iaea(tmp_path/'edges',[(1,-1,x,0,0,0,.05,1,0) for x in [-.25,0,.249,.25,.749]],original=10)
    stf=case[3];stf.beams=stf.beams[:1];b=stf.beams[0]
    other=b.rays[0].model_copy(deep=True);other.ray_pos=beam_basis(b)@np.array([5.,0,0]);b.rays.append(other)
    source=PhaseSpaceBeamletSource(base);source.prepare(stf,10)
    counts=[s['particles'] for s in source.selections.values()]
    assert counts==[3,2] and sum(counts)==5
    other.ray_pos=beam_basis(b)@np.array([4.,0,0])
    with pytest.raises(ValueError,match='Overlapping'):source.prepare(stf,10)


def test_native_collection_units_and_forward(case,phase_base,tmp_path):
    from test_results import fake_results
    from pyRadPlan import calc_dose_forward,calc_dose_influence
    import SimpleITK as sitk
    ct,plan,cst,stf,_=case
    plan.prop_dose_calc.update(source_config={'type':'phase_space','file_base':str(phase_base)},water=True)
    engine=TOPASPhotonEngine(plan)
    root=engine.prepare_jobs(ct,cst,stf)
    manifest=load_manifest(root)
    assert manifest['units']=='Gy/original accelerator history'
    assert len(list((root/'inputs').glob('phase_*.phsp')))==1
    assert manifest['jobs'][0]['source']['represented_original_histories']==10
    fake_results(root)
    dij=calc_dose_influence(ct,cst,stf,plan)
    forward=calc_dose_forward(ct,cst,stf,plan,weights=[2,3])
    np.testing.assert_allclose(dij.physical_dose.flat[0]@[2,3],sitk.GetArrayFromImage(forward['physical_dose']).ravel())
    assert 'physical_dose_std_error' not in forward
    moved=tmp_path/'relocated';shutil.copytree(root,moved)
    assert engine.collect_results(moved).physical_dose.flat[0].shape==(4800,2)
    assert plan.to_matrad()['propDoseCalc']['source_config']['type']=='phase_space'
    with pytest.raises(ValueError,match='not both'):
        TOPASPhotonEngine(plan,source_model=source_dummy())


def source_dummy():
    from minibeam.sources.beamlet import PointBeamletSource
    return PointBeamletSource()


@pytest.mark.topas
def test_topas_replay(case,phase_base):
    import os,subprocess
    executable=os.environ.get('PHOTON_TPS_TOPAS')
    if not executable:pytest.skip('Set PHOTON_TPS_TOPAS for phase-space transport')
    ct,plan,cst,stf,_=case
    engine=TOPASPhotonEngine(plan,water=True,source_config={'type':'phase_space','file_base':str(phase_base)})
    root=engine.prepare_jobs(ct,cst,stf)
    for job in load_manifest(root)['jobs']:
        process=subprocess.run([executable,job['parameter_file']],cwd=root,capture_output=True,text=True,timeout=120)
        assert process.returncode==0,process.stdout+process.stderr
        assert 'PreCheck showed no problems' in process.stdout
    assert engine.collect_results(root).physical_dose.flat[0].nnz>0


def test_trajectory_projection_and_history_reconstruction(case,tmp_path,monkeypatch):
    import minibeam.sources.phase_space as module
    # History 1 starts with an excluded photon; its selected electron must become
    # a new TOPAS event. The forward tilted photon lands outside the square.
    base=make_iaea(tmp_path/'directions',[(1,-1,1,0,0,0,.05,1,0),
        (2,.2,0,0,0,0,.05,0,9), (1,-1,0,0,.1,0,.05,1,0),
        (-1,-1,0,0,0,0,.05,1,0), (3,-1,0,0,0,0,.05,1,0)],original=10)
    original=module.validated_chunks
    monkeypatch.setattr(module,'validated_chunks',lambda info,audit:original(info,audit,chunk_size=1))
    source=PhaseSpaceBeamletSource(base);source.prepare(case[3],10)
    part=source.render(context(case));a=np.fromfile(part.assets[1][1],dtype=TOPAS_DTYPE)
    assert a['pdg'].tolist()==[11,-11] and a['new_history'].tolist()==[1,1]
    assert part.record['empty_histories']==8
    assert part.record['excluded_backward_or_tangent_particles']==1
    assert part.record['excluded_particles']==3
    case[3].beams[0].rays[0].beamlets[0].energy=99.
    assert source.render(context(case)).text==part.text


def test_report_plane_and_transforms(case,phase_base):
    from pyRadPlan import generate_stf
    from minibeam.geometry.patient_report import patient_scene
    ct,plan,cst,_,_=case
    plan.prop_stf.update(gantry_angles=[0.,45.,180.,315.],couch_angles=[0.,25.,0.,-15.])
    stf=generate_stf(ct,cst,plan)
    data=(ct,plan,cst,stf,None);source=PhaseSpaceBeamletSource(phase_base);source.prepare(stf,10)
    records=[]
    for i,beam in enumerate(stf.beams):
        record=source.render(context(data,i)).record
        origin=np.asarray(record['phase_space_plane_lps_mm'])
        focal=beam.iso_center+beam.source_point
        np.testing.assert_allclose(np.linalg.norm(origin-focal),272.1)
        np.testing.assert_allclose((origin-focal)/272.1,(beam.iso_center-focal)/beam.sad,atol=1e-12)
        records.append(dict(record,beam_index=i+1))
    scene=patient_scene(ct,stf,None,records)
    for item,record in zip(scene['beams'],records):
        np.testing.assert_allclose(item['phase_plane'],record['phase_space_plane_corners_lps_mm'])
        np.testing.assert_allclose((item['phase_plane']-record['phase_space_plane_lps_mm'])@beam_basis(item['beam'])[:,2],0,atol=1e-10)


@pytest.mark.topas
def test_exhaustive_partition_matches_unfiltered_replay(case,tmp_path):
    import os,subprocess
    executable=os.environ.get('PHOTON_TPS_TOPAS')
    if not executable:pytest.skip('Set PHOTON_TPS_TOPAS for partition transport')
    count=5000
    records=[r for _ in range(count) for r in [(1,-2,0,0,0,0,.05,1,0),(1,2,.5,0,0,0,.05,0,0)]]
    base=make_iaea(tmp_path/'partition',records,original=count)
    ct,plan,cst,stf,_=case;stf.beams=stf.beams[:1];beam=stf.beams[0]
    other=beam.rays[0].model_copy(deep=True);other.ray_pos=beam_basis(beam)@np.array([5.,0,0]);beam.rays.append(other)
    source=PhaseSpaceBeamletSource(base);source.prepare(stf,count)
    # The disjoint replay pairs retain every original particle exactly once.
    arrays=[np.fromfile(source.render(JobContext(beam,ray,ray.beamlets[0],patient_center(ct),i+1,1,i+1,1)).assets[1][1],dtype=TOPAS_DTYPE)
            for i,ray in enumerate(beam.rays)]
    assert sum(len(a) for a in arrays)==len(records)
    assert all(int(a['new_history'].sum())==count for a in arrays)
    totals=[]
    for name in ('partition','unfiltered'):
        if name=='unfiltered':
            beam.rays=beam.rays[:1];beam.bixel_width=10.
            beam.rays[0].ray_pos=beam_basis(beam)@np.array([2.5,0,0])
        engine=TOPASPhotonEngine(plan,water=True,histories=count,bundle_dir=str(tmp_path/name),
            source_config={'type':'phase_space','file_base':str(base)})
        root=engine.prepare_jobs(ct,cst,stf)
        for job in load_manifest(root)['jobs']:
            p=subprocess.run([executable,job['parameter_file']],cwd=root,capture_output=True,text=True,timeout=120)
            assert p.returncode==0,p.stdout+p.stderr
        matrix=engine.collect_results(root).physical_dose.flat[0]
        totals.append(float(matrix.sum()))
    # Independently transported streams have Monte Carlo fluctuations. Check
    # integrated dose as well as the exact particle partition above.
    assert totals[0]>0 and totals[0]==pytest.approx(totals[1],rel=.15)


@pytest.mark.parametrize('fault', ['missing','truncated','duplicate','histories','configuration'])
def test_phase_space_result_validation(case,phase_base,fault):
    from test_results import fake_results
    from minibeam import ResultsPendingError
    ct,plan,cst,stf,_=case
    engine=TOPASPhotonEngine(plan,water=True,source_config={'type':'phase_space','file_base':str(phase_base)})
    root=engine.prepare_jobs(ct,cst,stf);manifest=fake_results(root)
    path=root/manifest['jobs'][0]['output'];lines=path.read_text().splitlines()
    if fault=='missing':path.unlink()
    else:
        if fault=='truncated':lines.pop()
        if fault=='duplicate':lines.append(lines[-1])
        if fault=='histories':lines[-1]=lines[-1].replace(', 10,',', 4,') # selected particles are NOT histories
        if fault=='configuration':lines[1]='# Results for scorer: Dose_from_another_configuration'
        path.write_text('\n'.join(lines)+'\n')
    with pytest.raises((ValueError,ResultsPendingError)):engine.collect_results(root)


def test_multiple_beamlets_rejected(case,phase_base):
    ray=case[3].beams[0].rays[0];ray.beamlets.append(ray.beamlets[0].model_copy(deep=True))
    with pytest.raises(ValueError,match='exactly one'):
        PhaseSpaceBeamletSource(phase_base).prepare(case[3],10)


@pytest.mark.patient
@pytest.mark.topas
def test_elekta_patient_replay(case,tmp_path):
    import os,subprocess
    from pyRadPlan import load_patient,generate_stf
    from minibeam.workflow.patient import select_target
    executable=os.environ.get('PHOTON_TPS_TOPAS')
    directory=os.environ.get('PHOTON_TPS_DICOM')
    base=os.environ.get('PHOTON_TPS_PHASE_SPACE')
    if not all((executable,directory,base)):
        pytest.skip('Set TOPAS, DICOM and PHOTON_TPS_PHASE_SPACE (basename) for Elekta patient replay')
    ct,cst=load_patient(directory);select_target(cst,'PTV2017fw')
    plan=case[1];plan.prop_stf.update(gantry_angles=[45.],couch_angles=[0.])
    stf=generate_stf(ct,cst,plan)
    engine=TOPASPhotonEngine(plan,geometry_config=False,dicom_dir=directory,
        dose_spacing_mm=(10.,10.,10.),histories=1_000_000,
        source_config={'type':'phase_space','file_base':base})
    root=engine.prepare_jobs(ct,cst,stf)
    job=load_manifest(root)['jobs'][0]
    p=subprocess.run([executable,job['parameter_file']],cwd=root,capture_output=True,text=True,timeout=180)
    assert p.returncode==0,p.stdout+p.stderr
    assert 'Overlap is detected' not in p.stdout+p.stderr
    assert engine.collect_results(root).physical_dose.flat[0].nnz>0
    assert 'physical_dose_std_error' not in engine.collect_forward([1],root)


def test_coarse_forward_withholds_uncertainty(case,phase_base):
    from test_results import fake_results
    from pyRadPlan import calc_dose_forward
    ct,plan,cst,stf,_=case
    plan.prop_dose_calc.update(water=True,dose_spacing_mm=(6.,6.,6.),
        source_config={'type':'phase_space','file_base':str(phase_base)})
    root=TOPASPhotonEngine(plan).prepare_jobs(ct,cst,stf);fake_results(root)
    result=calc_dose_forward(ct,cst,stf,plan,weights=[1.,1.])
    assert result['physical_dose'].GetSize()==ct.cube_hu.GetSize()
    assert result['physical_dose_dose_grid'].GetSize()!=(ct.cube_hu.GetSize())
    assert not any('std_error' in key for key in result)
