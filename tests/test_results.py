import json
import shutil
import numpy as np
import SimpleITK as sitk
import pytest
from minibeam import ResultsPendingError, TOPASPhotonEngine
from minibeam.topas.manifest import load_manifest
from minibeam.topas.scoring import score_rows
from pyRadPlan import calc_dose_forward, calc_dose_influence
from pyRadPlan.dose.engines import get_engine


def fake_results(root):
    manifest = load_manifest(root)
    nx, ny, nz = manifest["dose_grid"]["dimensions"]
    for job in manifest["jobs"]:
        path = root/job["output"]
        with path.open("w") as f:
            f.write(f'# TOPAS Version: 4.2.p3\n# Results for scorer: {job["scorer"]}\n')
            for axis,n in zip("XYZ",[nx,ny,nz]):
                f.write(f'# {axis} in {n} bins of {manifest["dose_grid"]["resolution"][axis.lower()]} mm\n')
            f.write('# DoseToMedium ( Gy ) : Sum Mean Histories_with_Scorer_Active Standard_Deviation\n')
            for z in range(nz):
                for y in range(ny):
                    for x in range(nx):
                        # Distinct XYZ and bixel patterns reveal permutations.
                        dose = (1+x+10*y+100*z)*job["bixel_index"]*1e-10
                        f.write(f'{x}, {y}, {z}, {dose*job["histories"]:.16g}, {dose:.16g}, {job["histories"]}, {dose*2:.16g}\n')
    return manifest


def test_pending_and_tamper(case):
    ct, plan, cst, stf, engine = case
    root = engine.prepare_jobs(ct,cst,stf)
    with pytest.raises(ResultsPendingError):
        engine.collect_results(root)
    fake_results(root)
    (root/"inputs/devices.txt").write_text("changed geometry")
    with pytest.raises(ValueError,match="modified"):
        engine.collect_results(root)


def test_forward_matrix_normalization_relocation(case, tmp_path):
    ct,plan,cst,stf,engine = case
    root = engine.prepare_jobs(ct,cst,stf)
    fake_results(root)
    relocated = tmp_path/"moved bundle"
    shutil.copytree(root,relocated)
    dij = engine.collect_results(relocated)
    result = engine.collect_forward([3.,7.],relocated)
    np.testing.assert_allclose(dij.physical_dose.flat[0] @ [3.,7.], sitk.GetArrayFromImage(result["physical_dose"]).ravel())
    base = 1e-10
    assert dij.physical_dose.flat[0][0,0] == base
    error = sitk.GetArrayFromImage(result["physical_dose_std_error"]).ravel()[0]
    assert error == pytest.approx(np.sqrt((3*2*base)**2/10 + (7*4*base)**2/10))
    plan.prop_dose_calc.update(water=True)
    first = calc_dose_influence(ct,cst,stf,plan)
    second = calc_dose_forward(ct,cst,stf,plan,weights=np.array([3.,7.]))
    assert plan.prop_dose_calc["engine"] == "TOPASPhoton"
    np.testing.assert_allclose(sitk.GetArrayFromImage(second["physical_dose"]).ravel(),first.physical_dose.flat[0]@[3.,7.])


@pytest.mark.parametrize("mutation,match", [("truncated","Incomplete"),("duplicate","Duplicate"),
                                           ("history","history"),("unit","units"),("scorer","identity")])
def test_bad_results(case,mutation,match):
    ct,_,cst,stf,engine = case
    root = engine.prepare_jobs(ct,cst,stf)
    manifest = fake_results(root)
    job = manifest["jobs"][0]
    path = root/job["output"]
    lines = path.read_text().splitlines()
    if mutation == "truncated": lines.pop()
    if mutation == "duplicate": lines.append(lines[-1])
    if mutation == "history": lines[-1] = lines[-1].replace(", 10,", ", 9,")
    if mutation == "unit": lines = [line.replace("( Gy )", "( MeV )") for line in lines]
    if mutation == "scorer": lines[1] = "# Results for scorer: other"
    path.write_text("\n".join(lines)+"\n")
    with pytest.raises(ValueError,match=match):
        list(score_rows(path,job,manifest["dose_grid"]))


def test_settings_invalidate_and_budget(case):
    ct,plan,cst,stf,engine=case
    root=engine.prepare_jobs(ct,cst,stf)
    assert engine.prepare_jobs(ct,cst,stf)==root
    other=TOPASPhotonEngine(plan,water=True,seed=90)
    with pytest.raises(ValueError,match="different"):
        other.prepare_jobs(ct,cst,stf)
    fake_results(root)
    engine.max_matrix_bytes=1
    with pytest.raises(MemoryError):engine.collect_results(root)


def test_particle_dispatch_preserved(case):
    from pyRadPlan.dose.engines import get_available_engines
    from pyRadPlan.dose.engines._topasmc import ParticleTOPASMCEngine
    assert "TOPAS" in get_available_engines("protons")
    from pyRadPlan import IonPlan
    engine=get_engine(IonPlan(prop_dose_calc={"engine":"TOPAS"}))
    assert isinstance(engine,ParticleTOPASMCEngine)


def test_invalid_weights(case):
    ct,_,cst,stf,engine=case
    root=engine.prepare_jobs(ct,cst,stf)
    for weights in [[-1,1],[np.nan,1],[1]]:
        with pytest.raises(ValueError,match="Weights"):
            engine.collect_forward(weights,root)


def test_cluster_bundle_layout(case):
    ct,_,cst,stf,engine=case
    root=engine.prepare_jobs(ct,cst,stf)
    manifest=load_manifest(root)
    assert manifest['schema_version']==3
    assert {p.name for p in root.iterdir()}=={'inputs','jobs','results','derived','manifest.json'}
    assert not list(root.rglob('*.py'))
    for job in manifest['jobs']:
        assert manifest['request_id'] in job['scorer']
        assert 'receipt' not in job
        assert 'includeFile = inputs/devices.txt' in (root/job['parameter_file']).read_text()
    fake_results(root)
    assert engine.collect_results(root).physical_dose.flat[0].nnz>0


def test_wrong_configuration_results_rejected(case,tmp_path):
    ct,plan,cst,stf,engine=case
    root=engine.prepare_jobs(ct,cst,stf)
    other=TOPASPhotonEngine(plan,water=True,seed=54321)
    other_root=other.prepare_jobs(ct,cst,stf,bundle_dir=tmp_path/'other')
    fake_results(other_root)
    shutil.copytree(other_root/'results',root/'results',dirs_exist_ok=True)
    with pytest.raises(ValueError,match='identity'):
        engine.collect_results(root)


def test_old_bundle_untouched(case,tmp_path):
    ct,_,cst,stf,engine=case
    root=tmp_path/'legacy';root.mkdir()
    text=json.dumps({'schema_version':1})
    (root/'manifest.json').write_text(text)
    with pytest.raises(ValueError,match='new directory'):
        engine.prepare_jobs(ct,cst,stf,bundle_dir=root)
    assert (root/'manifest.json').read_text()==text


def test_native_dose_never_prepares_or_executes(case):
    ct,plan,cst,stf,engine=case
    from pathlib import Path
    with pytest.raises(ResultsPendingError,match='Prepare'):
        engine.calc_dose_forward(ct,cst,stf,[1.,1.])
    assert not Path(engine.bundle_dir).exists()
    for option in ['executable','workers']:
        with pytest.raises(ValueError,match='Unknown'):
            TOPASPhotonEngine(plan,**{option:1})
