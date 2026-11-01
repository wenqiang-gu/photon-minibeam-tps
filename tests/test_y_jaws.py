"""Native Y-shaped jaw geometry and fixed drawing-reference semantics."""
import copy
import os
from pathlib import Path
import subprocess
import numpy as np
import pytest
from scipy.spatial import ConvexHull
from scipy.optimize import linprog
from minibeam.geometry.configuration import GeometryConfig
from minibeam.geometry.models import BeamGeometry
from minibeam.geometry.assembly import TreatmentHead, resolve
from minibeam.geometry.diagnostics import material_solids
from minibeam.geometry.spatial import placement_rotation, beam_basis


def geometry(**jaw):
    return BeamGeometry.from_config().replace(jaws=jaw, aperture={'enabled':False})


@pytest.mark.parametrize('config', [None, Path('minibeam/geometry/water_config.toml')])
def test_packaged_dimensions(config):
    g, _, envelopes, text = resolve(GeometryConfig.load(config), 1000)
    s = g.jaw_shape
    assert s['physical_width_mm'] == pytest.approx(192.905)
    assert s['nominal_physical_length_mm'] == pytest.approx(164.675)
    assert s['thin_axial_source_mm'] == [432,467]
    assert s['total_axial_source_mm'] == [432,509]
    assert len(s['pieces']) == 8
    assert 'TsCAD' not in text and '.stl' not in text
    assert text.count('s:Ge/LowerJaw') == 8*3  # type, parent, material
    np.testing.assert_allclose(envelopes[1]['half_size'], [96.4525,188.2,38.5])


@pytest.mark.parametrize('sad', [900.,1000.,1100.])
@pytest.mark.parametrize('field', [50.,100.,200.])
def test_hardware_not_resized_by_treatment_distance(sad, field):
    g, *_ = resolve(geometry().replace(field={'height_mm':field}).as_config(),sad)
    s = g.jaw_shape
    assert s['physical_width_mm'] == pytest.approx(192.905)
    assert s['nominal_physical_length_mm'] == pytest.approx(164.675)
    assert s['rear_distance_from_axis_mm'] == pytest.approx(field/2*470.5/sad + 164.675)
    tip = g.jaw_round_tip
    assert tip.tangent_y == pytest.approx(field/2/sad*(tip.tangent_z+sad))


@pytest.mark.parametrize('thin', [20.,35.,77.])
@pytest.mark.parametrize('rounded', [False,True])
def test_partition_volume_and_no_material_overlap(thin,rounded):
    g, *_ = resolve(geometry(thin_region_thickness_mm=thin,round_tip_enabled=rounded).as_config(),1000)
    s = g.jaw_shape
    # Integrate the SAME faceted rounded front; other regions have linear sides.
    polys = [p for p in s['pieces'] if p['name'].startswith('LowerJawPositive')]
    base = next(p for p in polys if p['name'].endswith('Base'))
    total = sum(ConvexHull(p['vertices_beam_mm']).volume for p in polys)
    expected = ConvexHull(base['vertices_beam_mm']).volume
    if thin < 77:
        front = next(p for p in polys if p['name'].endswith('Front'))
        expected += ConvexHull(front['vertices_beam_mm']).volume
        a = np.asarray(s['physical_outline_points_mm'])
        expected += (77-thin)*sum((a[i,1]+a[i+1,1])*(a[i+1,0]-a[i,0]) for i in range(1,len(a)-1))
    assert total == pytest.approx(expected)
    # Convex half-space intersection: touching is permitted, positive inradius is not.
    for i, p in enumerate(s['pieces']):
        for q in s['pieces'][i+1:]:
            eq = np.vstack([ConvexHull(p['vertices_beam_mm']).equations, ConvexHull(q['vertices_beam_mm']).equations])
            r = linprog([0,0,0,-1], A_ub=np.c_[eq[:,:3], np.ones(len(eq))], b_ub=-eq[:,3],
                        bounds=[(None,None)]*3+[(0,None)], method='highs')
            assert r.status==2 or (r.success and r.x[3] < 1e-7)
    pos=np.concatenate([p['vertices_beam_mm'] for p in s['pieces'] if 'Positive' in p['name']])
    neg=np.concatenate([p['vertices_beam_mm'] for p in s['pieces'] if 'Negative' in p['name']])*[1,-1,1]
    assert np.max(np.min(np.linalg.norm(pos[:,None]-neg[None,:],axis=2),axis=1)) < 1e-9


def test_front_and_base_share_interface_and_full_width():
    g,*_=resolve(geometry().as_config(),1000)
    p={p['name']:p for p in g.jaw_shape['pieces']}
    a=np.array(p['LowerJawPositiveBase']['vertices_beam_mm'])
    b=np.array(p['LowerJawPositiveFront']['vertices_beam_mm'])
    interface=-533. # 467 from source
    assert min(a[a[:,2]==interface,1]) == pytest.approx(min(b[b[:,2]==interface,1]))
    for x in (a,b):
        assert np.ptp(x[:,0]) == pytest.approx(g.jaw_shape['physical_width_mm'])


def test_outline_edits_and_immutable_snapshot():
    old=geometry()
    points=[list(p) for p in old.jaws.outline_points_mm]
    points[-1][0]=400.
    new=old.replace(jaws={'outline_points_mm':points,'thin_region_thickness_mm':20.})
    points[0][1]=1 # no alias into the immutable geometry
    g,*_=resolve(new.as_config(),1000)
    assert g.jaw_shape['nominal_physical_length_mm']==pytest.approx(400*.4705)
    assert g.jaw_shape['physical_width_mm']==pytest.approx(192.905)
    assert old.jaws.outline_points_mm[-1][0]==350
    assert new.as_config().sha256!=old.as_config().sha256
    assert BeamGeometry.model_validate(new.model_dump())==new
    assert new.model_copy(deep=True)==new
    g,*_=resolve(old.replace(jaws={'outline_reference_distance_mm':2000.}).as_config(),1000)
    assert g.jaw_shape['physical_width_mm']==pytest.approx(192.905/2)


@pytest.mark.parametrize('change', [
    {'thin_region_thickness_mm':0.}, {'thin_region_thickness_mm':78.}, {'thin_region_thickness_mm':True},
    {'thickness_mm':True}, {'outline_reference_distance_mm':0.}, {'outline_reference_distance_mm':float('inf')},
    {'outline_reference_distance_mm':True}, {'outer_width_mm':230.},
    {'outline_points_mm':[[0,205],[30,205],[210,30],[210,30]]},
    {'outline_points_mm':[[1,205],[30,205],[210,30],[350,30]]},
    {'outline_points_mm':[[0,205],[30,200],[210,30],[350,30]]},
    {'outline_points_mm':[[0,205],[30,205],[210,30],[350,40]]},
    {'outline_points_mm':[[False,205],[30,205],[210,30],[350,30]]},
    {'outline_points_mm':[[0,205],[30,205],[210,30],[350,float('nan')]]},
])
def test_invalid_settings(change):
    with pytest.raises(ValueError): geometry(**change)
    data=GeometryConfig.load().data;data['jaws'].update(change)
    with pytest.raises(ValueError): GeometryConfig.from_data(data).validate()


def test_incompatible_front_disabled_and_containment():
    points=[[0,205],[.01,205],[210,30],[350,30]]
    bad=geometry(outline_points_mm=points)
    with pytest.raises(ValueError,match='front band'): resolve(bad.as_config(),1000)
    g,_,envelopes,text=resolve(bad.replace(jaws={'enabled':False}).as_config(),1000)
    assert not hasattr(g,'jaw_shape') and 'Ge/LowerJaw' not in text
    assert not any(e['name']=='jaws' for e in envelopes)


def test_native_parameters_match_collision_and_metadata(case):
    from minibeam.steering import enrich_stf, matrad_steering
    from minibeam.geometry.patient_report import patient_scene
    stf=enrich_stf(case[3],geometry())
    head=TreatmentHead(geometry().as_config())
    exported=matrad_steering(stf)['beam_geometry']
    scene=patient_scene(case[0],stf,head)
    for i,beam in enumerate(stf.beams):
        g,_,_,text=head.resolved(beam)
        actual=dict(material_solids(head,beam))
        world=dict(scene['beams'][i]['solids'])
        for p in g.jaw_shape['pieces']:
            polygon=np.asarray(p['polygon']);h=p['extrusion_half_mm']
            local=np.concatenate([np.c_[polygon,np.full(len(polygon),z)] for z in (-h,h)])
            r=placement_rotation(-90,-90,0) if p['plane']=='yz' else np.eye(3)
            transformed=local@r.T+[0,0,p['center_z_mm']]
            np.testing.assert_allclose(transformed,actual[p['name']],atol=1e-9)
            np.testing.assert_allclose(world[p['name']],actual[p['name']]@beam_basis(beam).T+beam.iso_center)
        saved=exported[i]['resolved']['mlc_jaws']['jaw_shape']
        assert saved['physical_width_mm']==pytest.approx(192.905)
        assert exported[i]['geometry']['jaws']['model']=='y_shaped'
    # Explicit legacy model remains byte-for-byte equivalent when selector is omitted.
    legacy=GeometryConfig.load(Path('tests/fixtures/uniform_geometry.toml')).data
    a=resolve(GeometryConfig.from_data(legacy),1000)[3]
    legacy['jaws'].pop('model',None)
    assert resolve(GeometryConfig.from_data(legacy),1000)[3]==a


@pytest.mark.topas
def test_native_jaw_transport(case):
    exe=os.environ.get('PHOTON_TPS_TOPAS')
    if not exe:pytest.skip('Set PHOTON_TPS_TOPAS for native jaw transport')
    from minibeam import TOPASPhotonEngine
    from minibeam.topas.manifest import load_manifest
    ct,plan,cst,stf,_=case
    engine=TOPASPhotonEngine(plan,water=True,devices=(TreatmentHead(geometry().as_config()),),histories=200)
    root=engine.prepare_jobs(ct,cst,stf)
    for job in load_manifest(root)['jobs']:
        p=subprocess.run([exe,job['parameter_file']],cwd=root,capture_output=True,text=True)
        assert p.returncode==0,p.stdout+p.stderr
        assert 'Overlap is detected' not in p.stdout
    assert engine.collect_results(root).physical_dose.flat[0].nnz>0


def test_saved_outline_comparison_rejects_changes():
    from minibeam.workflow.collection import _same_geometry_snapshot
    expected=geometry().model_dump()
    actual=copy.deepcopy(expected)
    actual['jaws']['outline_points_mm']=np.asarray(actual['jaws']['outline_points_mm'])
    assert _same_geometry_snapshot(actual,expected)
    actual['jaws']['outline_points_mm'][-1,0]+=1
    assert not _same_geometry_snapshot(actual,expected)


@pytest.mark.topas
def test_transport_across_native_piece_interfaces(tmp_path):
    """Start photons in the thin base, then cross base/front/shoulder/spine faces."""
    exe=os.environ.get('PHOTON_TPS_TOPAS')
    if not exe:pytest.skip('Set PHOTON_TPS_TOPAS for native-piece interface transport')
    g,_,_,solids=resolve(geometry().replace(mlc={'enabled':False}).as_config(),1000)
    text='''
s:Ge/World/Material = "Vacuum"
d:Ge/World/HLX = 1000 mm
d:Ge/World/HLY = 1000 mm
d:Ge/World/HLZ = 1000 mm
s:Ge/TreatmentHead/Type = "Group"
s:Ge/TreatmentHead/Parent = "World"
b:Ge/CheckForOverlaps = "True"
b:Ge/QuitIfOverlapDetected = "True"
i:Ge/CheckForOverlapsResolution = 10000
s:Ge/Probe/Type = "Group"
s:Ge/Probe/Parent = "World"
d:Ge/Probe/TransX = 0 mm
d:Ge/Probe/TransY = 100 mm
d:Ge/Probe/TransZ = -534 mm
s:So/Beam/Type = "Beam"
s:So/Beam/Component = "Probe"
s:So/Beam/BeamParticle = "gamma"
d:So/Beam/BeamEnergy = 6 MeV
u:So/Beam/BeamEnergySpread = 0
s:So/Beam/BeamPositionDistribution = "Flat"
s:So/Beam/BeamPositionCutoffShape = "Rectangle"
d:So/Beam/BeamPositionCutoffX = 96 mm
d:So/Beam/BeamPositionCutoffY = 88 mm
s:So/Beam/BeamAngularDistribution = "None"
i:So/Beam/NumberOfHistoriesInRun = 3000
i:Ts/Seed = 12345
i:Ts/NumberOfThreads = 1
i:Ts/ShowHistoryCountAtInterval = 100000
b:Ts/PauseBeforeQuit = "False"
b:Gr/Enable = "False"
'''+solids
    names=[p['name'] for p in g.jaw_shape['pieces'] if 'Positive' in p['name']]
    for name in names:
        text+=f'''s:Sc/{name}/Quantity = "EnergyDeposit"
s:Sc/{name}/Component = "{name}"
s:Sc/{name}/OutputType = "CSV"
s:Sc/{name}/OutputFile = "{name}"
s:Sc/{name}/IfOutputFileAlreadyExists = "Exit"
sv:Sc/{name}/Report = 1 "Sum"
'''
    (tmp_path/'interfaces.txt').write_text(text)
    p=subprocess.run([exe,'interfaces.txt'],cwd=tmp_path,capture_output=True,text=True)
    (tmp_path/'transport.log').write_text(p.stdout+p.stderr)
    assert p.returncode==0,p.stdout+p.stderr
    assert 'Overlap is detected' not in p.stdout+p.stderr
    assert 'stuck' not in (p.stdout+p.stderr).lower()
    for name in names:
        assert np.loadtxt(tmp_path/f'{name}.csv',comments='#').sum()>0
