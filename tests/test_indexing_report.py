"""Report examples describe saved MATLAB layouts and actual run parameters."""
from copy import deepcopy
import pytest
from minibeam.workflow.collection import indexing_report


@pytest.mark.parametrize('stage', ['collect','forward'])
@pytest.mark.parametrize('phase_space', [False,True])
@pytest.mark.parametrize('coarse', [False,True])
@pytest.mark.parametrize('multiple', [False,True])
def test_report_examples(stage, phase_space, coarse, multiple):
    grid=dict(dimensions=[12,12,8],origin=[-5.,2.,7.],resolution=dict(x=2.,y=2.,z=3.),direction=[[1,0,0],[0,1,0],[0,0,1]])
    jobs=[dict(beam_index=1,ray_index=1,beamlet_index=1,bixel_index=1,job_id='bixel_000001',gantry_angle_deg=45.,couch_angle_deg=10.)]
    if multiple:
        jobs.append(dict(jobs[0],beamlet_index=2,bixel_index=2,job_id='bixel_000002'))
    jobs.append(dict(jobs[0],beam_index=2,bixel_index=len(jobs)+1,job_id=f'bixel_{len(jobs)+1:06d}',gantry_angle_deg=135.))
    m=dict(bundle_id='example',ct_grid=grid,dose_grid=deepcopy(grid),jobs=jobs,
           units='Gy/original accelerator history' if phase_space else 'Gy/primary photon',
           weight_units='original accelerator histories' if phase_space else 'primary photons',
           forward_uncertainty='unavailable: shared phase-space histories' if phase_space else 'independent columns',
           planning={'plan':{'target_names':["PTV's target"]}})
    if coarse: m['dose_grid']['dimensions']=[6,6,4]
    report=indexing_report(m,stage)
    for snippet in ("planning = load('steering.mat');", "result = load('result.mat');", 'dij.physicalDose{1}',
                    'result.stf(b)', 'beam.ray(r)', 'ray.energy(k)', 'result.beam_geometry{b}',
                    'job.bixel_index == j', 'job.beamlet_index == k', 'full(D(:,j))', 'full(D * weights)',
                    'planning.cst{roiRow,4}{1}', 'mask(voxelIndices) = true;', 'pln.propOpt', 'pln.propSeq',
                    'original_dicom_rois', 'zero-based structure numbering', '[Z,Y,X]', 'not empty placeholders'):
        assert snippet in report
    assert "roiName = 'PTV''s target'" in report
    assert f'doseDims = {[6,6,4] if coarse else [12,12,8]}'.replace(' ', '') in report.replace(' ', '')
    assert m['units'] in report and m['weight_units'] in report
    assert ('grids differ' if coarse else 'grids match') in report
    assert ('separately completed collect stage' in report) == (stage=='forward')
    assert '| 1 | 45 / 10 | '+('1–2' if multiple else '1')+' |' in report
    assert f'| 2 | 135 / 10 | {len(jobs)} |' in report
    assert 'results.mat' not in report
