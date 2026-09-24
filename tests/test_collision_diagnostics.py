import json
import numpy as np
import pytest
from minibeam import TOPASPhotonEngine
from minibeam.geometry.assembly import TreatmentHead, GeometryCollisionError, resolve
from minibeam.geometry.configuration import GeometryConfig
from minibeam.geometry.diagnostics import (box_vertices, intersection, beam_diagnostics,
                                           material_solids, crop_investigation)
from minibeam.geometry.spatial import beam_basis


def candidate_config():
    text = GeometryConfig.load().text
    for before, after in [('450.0','425.0'),('550.0','510.0'),('707.0','607.0')]:
        text = text.replace('source_to_center_mm = '+before,'source_to_center_mm = '+after)
    return GeometryConfig(text.replace('width_mm = 156.0','width_mm = 158.0'))


def test_candidate_layout():
    config = candidate_config(); config.validate()
    g,ap,_,_ = resolve(config,1000)
    assert g.mlc.source_to_center == 425
    assert g.lower_jaw.source_to_center == 510
    assert ap.collimator_center_z == -393
    assert ap.collimator_width == 158
    assert g.mlc.exit_z < g.lower_jaw.entrance_z
    assert g.lower_jaw.exit_z < ap.collimator_center_z-ap.collimator_thickness/2
    assert ap.exit_edge_gap > 0


def test_convex_intersection_and_touching():
    vertices = box_vertices([0,0,0],[1,2,3])
    hit = intersection(vertices, [-.5,-.5,-.5],[.5,.5,.5])
    assert hit['intersection_inradius_mm'] == pytest.approx(.5)
    assert np.max(np.abs(hit['witness_lps_mm'])) < 1e-7
    assert intersection(vertices,[1,-1,-1],[3,1,1]) is None
    assert intersection(vertices,[1.01,-1,-1],[3,1,1]) is None
    # Neither box contains the other's vertices; an edge/face intersection still exists.
    assert intersection(box_vertices([0,0,0],[4,.2,.2]),[-.2,-4,-.2],[.2,4,.2])


def test_opening_is_not_material(case):
    _,_,_,stf,_ = case
    head = TreatmentHead(candidate_config())
    solids = material_solids(head,stf.beams[0])
    # The central slit is open at its entrance, through its entire thickness.
    for name, vertices in solids:
        assert intersection(vertices,[-.1,-.1,-440],[.1,.1,-345]) is None, name
    assert len(solids) == 4+4+18


def test_failed_prepare_keeps_diagnostics_separate(case,tmp_path):
    ct,plan,cst,stf,_ = case
    head = TreatmentHead(candidate_config())
    for beam in stf.beams:
        beam.iso_center = beam.iso_center + beam_basis(beam)[:,2]*393
    path=tmp_path/'candidate.toml';path.write_text(head.config.text)
    engine=TOPASPhotonEngine(plan,water=True,geometry_config=str(path))
    existing=tmp_path/'existing';existing.mkdir();(existing/'keep.txt').write_text('unchanged')
    with pytest.raises(GeometryCollisionError,match='Patient diagnostics'):
        engine.prepare_jobs(ct,cst,stf,bundle_dir=existing)
    assert list(existing.iterdir()) == [existing/'keep.txt']
    folders=list(tmp_path.glob('existing-diagnostics-*'));assert len(folders)==1
    report=folders[0]
    data=json.loads((report/'collisions.json').read_text())
    assert len(data['beams']) == 2
    assert (report/'geometry.pdf').read_bytes().startswith(b'%PDF')
    assert not (report/'manifest.json').exists()
    assert not (report/'jobs').exists()
    assert data['crop_investigation']['applied'] is False
    assert data['crop_investigation']['roi_count'] == len(cst.vois)
    assert all(any(x['intersects'] for x in b['material_solids']) for b in data['beams'])


def test_clear_hardware_and_crop_no_mutation(case):
    ct,_,cst,stf,_ = case
    head=TreatmentHead(candidate_config())
    before=np.asarray(ct.cube_hu).copy()
    for i,beam in enumerate(stf.beams,1):
        record=beam_diagnostics(ct,beam,head,i)
        assert not any(x['intersects'] for x in record['envelopes']+record['material_solids'])
    crop=crop_investigation(ct,cst,stf,head)
    assert crop['required_bounds_xyz_inclusive_0based']==[[0,0,0],[19,19,11]]
    np.testing.assert_array_equal(np.asarray(ct.cube_hu),before)


@pytest.mark.patient
def test_patient_candidate_remains_blocked():
    import os
    from pathlib import Path
    from pyRadPlan import load_patient,generate_stf
    import patient_workflow as workflow
    from minibeam.workflow.patient import select_target
    directory=os.environ.get('PHOTON_TPS_DICOM')
    if not directory:
        pytest.skip('Set PHOTON_TPS_DICOM for patient collision investigation')
    ct,cst=load_patient(directory)
    select_target(cst,'PTV2017fw')
    plan=workflow.configure_plan(project_dir=Path('unused'))
    plan.prop_stf.update(gantry_angles=[45.,135.,225.,315.],couch_angles=[0.]*4)
    stf=generate_stf(ct,cst,plan)
    head=TreatmentHead(candidate_config())
    for i,beam in enumerate(stf.beams,1):
        assert beam.sad == 1000
        record=beam_diagnostics(ct,beam,head,i)
        collisions=[item['component'] for item in record['material_solids'] if item['intersects']]
        assert 'FrameYNegative' in collisions and 'FrameYPositive' in collisions
        assert any(name.startswith('Blade_') for name in collisions)
        assert not any(name.startswith(('MLC','LowerJaw')) for name in collisions)
    crop=crop_investigation(ct,cst,stf,head)
    assert crop['roi_count']==51
    assert {'CouchSurface','CouchInterior'} <= {roi['name'] for roi in crop['rois']}
    assert not crop['clears_material_solids']


def test_patient_cli_explains_collision_without_traceback(case,tmp_path,monkeypatch):
    from argparse import Namespace
    import patient_workflow as workflow
    ct,_,cst,_,_=case
    monkeypatch.setattr(workflow,'COLLIMATOR_SHIFT_FRACTIONS',[])
    monkeypatch.setattr(workflow,'parse_arguments',lambda argv=None: Namespace(stage='prepare',project_dir=str(tmp_path/'patient')))
    monkeypatch.setattr(workflow,'load_patient',lambda _: (ct,cst))
    monkeypatch.setattr(workflow,'read_roi_metadata',lambda *_: {'omitted_rois': []})
    monkeypatch.setattr(workflow,'TARGET','TARGET')
    def collision(*args,**kwargs):
        raise GeometryCollisionError('See patient diagnostics: geometry.pdf')
    monkeypatch.setattr(workflow.TOPASPhotonEngine,'prepare_jobs',collision)
    with pytest.raises(SystemExit,match='Geometry preparation stopped:.*patient diagnostics') as raised:
        workflow.main()
    assert raised.value.__suppress_context__
