"""Agility-inspired curved Y jaws and legacy flat-jaw compatibility."""
import os
from pathlib import Path
import subprocess
import numpy as np
import pytest
from minibeam.geometry.configuration import GeometryConfig
from minibeam.geometry.models import BeamGeometry
from minibeam.geometry.assembly import resolve, TreatmentHead, serializable
from minibeam.geometry.mlc_jaws import _rounded_bank_polygon, _render_rounded_bank_solid
from minibeam.geometry.spatial import placement_rotation
from minibeam.geometry.diagnostics import material_solids


def test_published_shape_at_preserved_position():
    g,_,_,text=resolve(GeometryConfig.load(),1000.)
    jaw,tip=g.lower_jaw,g.jaw_round_tip
    assert jaw.entrance_z+1000==431.5
    assert jaw.exit_z+1000==508.5
    assert tip.circle_center_z+1000==466.5
    assert tip.radius==135
    slope=.05
    assert (tip.circle_center_y-slope*(tip.circle_center_z+1000))/np.hypot(1,slope)==pytest.approx(135)
    assert tip.tangent_y==pytest.approx(slope*(tip.tangent_z+1000))
    assert 's:Ge/LowerJawPositive/Type = "G4ExtrudedSolid"' in text
    assert 'circle_center_y' in serializable(tip)


@pytest.mark.parametrize('width,sad',[(20,1000),(100,1100),(200,1000)])
def test_field_tangency(width,sad):
    cfg=BeamGeometry.from_config().replace(field={'height_mm':width})
    g,*_=resolve(cfg.as_config(),sad);t=g.jaw_round_tip;slope=width/2/sad
    assert (t.circle_center_y-slope*(t.circle_center_z+sad))/np.hypot(1,slope)==pytest.approx(t.radius)


@pytest.mark.parametrize('positive',[False,True])
def test_writer_matches_collision_vertices(case,positive):
    head=TreatmentHead(GeometryConfig.load());g,*_=head.resolved(case[3].beams[1])
    polygon=np.asarray(_rounded_bank_polygon(g,positive,axis='Y'))
    local=np.concatenate([np.column_stack((polygon,np.full(len(polygon),z))) for z in (-200,200)])
    world=local@placement_rotation(-90,-90,0).T+[0,0,g.lower_jaw.center_z]
    name='LowerJaw'+('Positive' if positive else 'Negative')
    actual=dict(material_solids(head,case[3].beams[1]))[name]
    # Ignore vertex enumeration; compare the complete physical vertex sets.
    assert np.max(np.min(np.linalg.norm(world[:,None]-actual[None,:],axis=2),axis=1))<1e-10
    assert len(polygon)==67


def test_legacy_and_disabled():
    data=GeometryConfig.load().data
    for key in ('round_tip_enabled','tip_radius_mm','tip_center_z_offset_mm'):data['jaws'].pop(key)
    snapshot=BeamGeometry.model_validate(data)
    g,_,_,text=resolve(snapshot.as_config(),1000)
    assert g.jaw_round_tip is None
    assert 's:Ge/LowerJawPositive/Type = "G4GTrap"' in text
    g,_,_,text=resolve(snapshot.replace(jaws={'enabled':False}).as_config(),1000)
    assert g.lower_jaw is None and 'Ge/LowerJaw' not in text


@pytest.mark.parametrize('change',[{'tip_radius_mm':10.},{'tip_center_z_offset_mm':200.},
                                  {'tip_radius_mm':10000.}])
def test_invalid_curvature(change):
    with pytest.raises(ValueError):resolve(BeamGeometry.from_config().replace(jaws=change).as_config(),1000)


def test_mlc_unchanged():
    legacy=BeamGeometry.from_config().replace(jaws={'round_tip_enabled':False}).as_config()
    old,*_=resolve(legacy,1000)
    new,*_=resolve(GeometryConfig.load(),1000)
    for positive in (False,True):
        assert _render_rounded_bank_solid(old,positive)==_render_rounded_bank_solid(new,positive)


@pytest.mark.topas
def test_curved_jaw_transport(case):
    exe=os.environ.get('PHOTON_TPS_TOPAS')
    if not exe:pytest.skip('Set PHOTON_TPS_TOPAS for curved-jaw transport')
    from minibeam import TOPASPhotonEngine
    cfg=BeamGeometry.from_config().replace(aperture={'enabled':False})
    ct,plan,cst,stf,_=case
    engine=TOPASPhotonEngine(plan,water=True,devices=(TreatmentHead(cfg.as_config()),),histories=1000)
    root=engine.prepare_jobs(ct,cst,stf)
    from minibeam.topas.manifest import load_manifest
    for job in load_manifest(root)['jobs']:
        result=subprocess.run([exe,job['parameter_file']],cwd=root,capture_output=True,text=True)
        assert result.returncode==0,result.stdout+result.stderr
    assert engine.collect_results(root).physical_dose.flat[0].nnz>0


def test_snapshots_and_fingerprints(case):
    from minibeam.steering import enrich_stf, matrad_steering
    from minibeam.geometry.patient_report import patient_scene
    from minibeam.geometry.spatial import beam_basis
    from minibeam.geometry.coordinates import patient_center
    geometry=BeamGeometry.from_config()
    stf=enrich_stf(case[3],geometry)
    records=matrad_steering(stf)['beam_geometry']
    assert len(records)==2
    for record in records:
        assert record['resolved']['mlc_jaws']['jaw_round_tip']['radius']==135
        assert record['geometry']['jaws']['round_tip_enabled'] is True
    assert geometry.as_config().sha256!=geometry.replace(jaws={'tip_radius_mm':136.}).as_config().sha256
    head=TreatmentHead(geometry.as_config())
    scene=patient_scene(case[0],stf,head)
    for beam,item in zip(stf.beams,scene['beams']):
        for (name,local),(world_name,world) in zip(material_solids(head,beam),item['solids']):
            assert name==world_name
            np.testing.assert_allclose(world,local@beam_basis(beam).T+beam.iso_center)
