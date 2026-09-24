import numpy as np
import pytest
from pyRadPlan import generate_stf
from minibeam.sources.energy import EmpiricalSpectrum, MonoenergeticEnergy
from minibeam.sources.beamlet import PointBeamletSource
from minibeam.sources.empirical import EMPIRICAL_ENERGY_MEV, EMPIRICAL_SPECTRUM_WEIGHT


def test_supplied_spectrum_and_rendering():
    expected_weights = [
        .032047941,.139322874,.128271317,.10410132,.087722697,.074515256,
        .061865442,.051816519,.044850118,.038812754,.033491917,.029783373,
        .025720295,.022598543,.019733182,.017636211,.015519179,.014236603,
        .012188171,.010440358,.009278173,.007876917,.006674784,.005250251,
        .003518869,.001849387,.000746224,.00013132]
    np.testing.assert_allclose(EMPIRICAL_ENERGY_MEV, .13 + .25*np.arange(28), atol=1e-15)
    np.testing.assert_array_equal(EMPIRICAL_SPECTRUM_WEIGHT, expected_weights)
    fragment = EmpiricalSpectrum().render(None)
    np.testing.assert_array_equal(fragment.record['spectrum_weights'], expected_weights)
    expected = np.array(expected_weights)/sum(expected_weights)
    np.testing.assert_allclose(fragment.record['spectrum_probabilities'], expected)
    emitted = np.fromstring(fragment.text.splitlines()[2].split('= 28 ')[1], sep=' ')
    np.testing.assert_allclose(emitted, expected, rtol=1e-15)
    assert emitted.sum() == pytest.approx(1.)
    assert 'BeamEnergy =' not in fragment.text


@pytest.mark.parametrize('energies,weights', [([],[]),([1],[0]),([1],[-1]),
    ([0],[1]),([2,1],[1,1]),([1,1],[1,1]),([1],[1,2]),([np.nan],[1]),
    ([1],[np.inf]),([[1]],[[1]])])
def test_invalid_spectrum(energies, weights):
    with pytest.raises(ValueError, match='Spectrum requires'):
        EmpiricalSpectrum(energies, weights).render(None)


@pytest.mark.parametrize('gantry,couch', [(0,0),(90,0),(180,0),(270,0),(37,21),(215,-33)])
def test_point_squares_central_and_off_axis(case, gantry, couch):
    ct, plan, cst, _, engine = case
    plan.prop_stf.update(generator='photonIMRT', bixel_width=3., gantry_angles=[gantry], couch_angles=[couch])
    stf = generate_stf(ct,cst,plan)
    definition = engine._definition(ct,stf)[0]
    for job, text in zip(definition['jobs'], definition['sources']):
        src=job['source']; beam=stf.beams[job['beam_index']-1]
        source=beam.iso_center+beam.source_point
        aim=beam.iso_center+beam.rays[job['ray_index']-1].ray_pos
        distance=np.linalg.norm(aim-source)
        np.testing.assert_allclose(src['source_lps_mm'],source)
        from minibeam.geometry.spatial import beam_basis
        np.testing.assert_allclose(src['local_to_world'],beam_basis(beam),atol=1e-12)
        np.testing.assert_allclose(src['aim_lps_mm'],aim)
        assert src['selections'][0]['width_mm']==beam.bixel_width
        assert 'Type = "PhaseSpace"' in text
        assert 'BeamAngularDistribution' not in text
    assert len(definition['jobs'])>1


def test_spectrum_independent_of_native_energy(case):
    ct, _, _, stf, engine=case
    before=engine._definition(ct,stf)[0]['sources']
    for beam in stf.beams:
        for ray in beam.rays:
            for bixel in ray.beamlets:bixel.energy=9.
    assert engine._definition(ct,stf)[0]['sources']==before
    engine.source_model=PointBeamletSource(energy_model=MonoenergeticEnergy())
    assert engine._definition(ct,stf)[0]['jobs'][0]['source']['energy_mev']==9


def test_workflow_nominal_energy(tmp_path):
    import patient_workflow as workflow
    plan=workflow.configure_plan(project_dir=tmp_path)
    assert 'energy' not in plan.prop_stf
    assert not hasattr(workflow,'ENERGY_MEV')


def test_spectrum_change_invalidates_existing_bundle(case):
    ct, _, cst, stf, engine=case
    root=engine.prepare_jobs(ct,cst,stf)
    engine.source_model.energy_model.weights[0] *= 2
    with pytest.raises(ValueError, match='different inputs/settings'):
        engine.prepare_jobs(ct,cst,stf,bundle_dir=root)
