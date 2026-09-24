import json
import numpy as np
import pytest
import SimpleITK as sitk
from scipy.io import loadmat
from minibeam.topas.scoring import score_rows, _UNSCORED_WARNING, _UNSCORED_DETAILS
from minibeam.workflow.artifacts import save_artifacts
from minibeam.workflow.collection import collect_bundle
from test_results import fake_results

BLOCK = "\n".join((_UNSCORED_WARNING, *_UNSCORED_DETAILS,
    "# Total number of steps not scored for this reason: 1",
    "# Total amount of energy not scored for this reason: 4.15305e-08 MeV"))


def inject(path, block=BLOCK):
    text = path.read_text()
    path.write_text(block + "\n" + text)


def test_both_stages_preserve_dose_with_multiple_warnings(case, capsys):
    ct,plan,cst,stf,engine = case
    root = engine.prepare_jobs(ct,cst,stf)
    m = fake_results(root)
    save_artifacts(ct,cst,plan,stf,root,'prepare',{})
    # Include an explicit zero-dose bin.
    for job in m['jobs']:
        path=root/job['output']
        lines=path.read_text().splitlines()
        row=next(i for i,line in enumerate(lines) if not line.startswith('#'))
        lines[row]=f"0,0,0,0,0,{job['histories']},0"
        path.write_text("\n".join(lines)+"\n")
    expected=engine.collect_results(root).physical_dose.flat[0].toarray()
    for job in m['jobs']:
        inject(root/job['output'])
    for stage in ('collect','forward'):
        capsys.readouterr()
        collect_bundle(root,stage,weight_per_bixel=2.)
        output=capsys.readouterr().out
        assert output.count('Scorer warning for ')==len(m['jobs'])
        assert 'completed with scorer warnings; validity requires user review' in output
        metadata=json.loads((root/f'derived/{stage}_metadata.json').read_text())
        assert metadata['status']=='complete' and metadata['review_required']
        assert len(metadata['scorer_warnings'])==len(m['jobs'])
        assert metadata['scorer_warnings'][0]['unscored_energy_mev']==4.15305e-8
        assert 'Statistical uncertainty does not account for missing energy' in (root/'derived/indexing.md').read_text()
    dose=sitk.GetArrayFromImage(sitk.ReadImage(str(root/'derived/dose.mha')))
    np.testing.assert_allclose(dose.ravel(),expected @ np.full(len(m['jobs']),2.))
    # Native collection shares exactly the same reader and unchanged values.
    np.testing.assert_array_equal(engine.collect_results(root).physical_dose.flat[0].toarray(),expected)


@pytest.mark.parametrize('block', [
    BLOCK.replace('4.15305e-08','-1'), BLOCK.replace('4.15305e-08','1e999'),
    BLOCK.replace('reason: 1','reason: 1.5'), BLOCK.rsplit('\n',1)[0],
    BLOCK+'\n'+BLOCK, BLOCK+'\n# Warning: unknown warning',
    BLOCK+'\n# Filtered by: something',
    BLOCK.replace(_UNSCORED_WARNING,'# Warning: different failure'),
    BLOCK.replace(_UNSCORED_DETAILS[0],'# missing description'),
])
def test_invalid_warning_blocks_rejected(case,block):
    ct,_,cst,stf,engine=case
    root=engine.prepare_jobs(ct,cst,stf); m=fake_results(root)
    job=m['jobs'][0]; path=root/job['output']; inject(path,block)
    with pytest.raises(ValueError):
        list(score_rows(path,job,m['dose_grid']))


@pytest.mark.parametrize('mutation',['missing','duplicate','histories'])
def test_warning_does_not_bypass_integrity(case,mutation):
    ct,_,cst,stf,engine=case
    root=engine.prepare_jobs(ct,cst,stf);m=fake_results(root)
    job=m['jobs'][0];path=root/job['output'];inject(path)
    lines=path.read_text().splitlines()
    if mutation=='missing':lines.pop()
    elif mutation=='duplicate':lines.append(lines[-1])
    else:lines[-1]=lines[-1].replace(', 10,',', 9,')
    path.write_text('\n'.join(lines)+'\n')
    with pytest.raises(ValueError):
        list(score_rows(path,job,m['dose_grid']))
