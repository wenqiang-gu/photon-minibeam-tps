"""Saved MHA contract and CLI integration; no original transport inputs needed."""
import json
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from tools.analysis_io import read_grid
from tools.analysis_numerics import mc_pdd, mc_profile
from tools.dose_analysis import main


def save_water(root, *, factor=1., exposure=2., spacing=(1., 2., 1.5), warning=False):
    derived = root / 'derived'
    derived.mkdir(parents=True, exist_ok=True)
    # Native NumPy [Z, Y, X], intentionally unequal transverse counts/spacings.
    z, y, x = np.indices((15, 6, 13))
    dose = factor * exposure * (2 + (x % 2)*8 + z + y*.5)
    image = sitk.GetImageFromArray(dose.astype(float))
    surface = [10., 20., 30.]
    image.SetSpacing(spacing)
    image.SetOrigin((4., 21., 19.5))
    sitk.WriteImage(image, str(derived/'dose_scoring_grid.mha'))
    convention = dict(normalization='original_accelerator_history', units='Gy/original accelerator history',
                      weight_units='original accelerator histories')
    manifest = dict(bundle_id='fixture', **convention, jobs=[dict(job_id='beam_000001')],
        dose_grid=dict(dimensions=list(image.GetSize()), resolution=dict(zip('xyz', spacing)),
                       origin=list(image.GetOrigin()), direction=np.eye(3).tolist()),
        provenance=dict(synthetic='water', water_measurement=dict(surface_center_lps_mm=surface,
                       field_width_mm=100., ssd_mm=1000.)),
        planning=dict(beams=[dict(parameters=dict(gantry_angle=0., couch_angle=0., iso_center=surface,
                                                 source_point=[0., -1000., 0.]))]))
    (root/'manifest.json').write_text(json.dumps(manifest))
    metadata = dict(stage='forward', bundle_id='fixture', **convention, weights=[exposure],
                    scoring_grid_dose='dose_scoring_grid.mha', review_required=warning,
                    scorer_warnings=[{'text':'unscored step'}] if warning else [],
                    batch_metadata=dict(number_of_batches=3, uncertainty='conditional on fixed replay'))
    (derived/'forward_metadata.json').write_text(json.dumps(metadata))
    return derived/'dose_scoring_grid.mha', dose


def change_json(path, update):
    value=json.loads(path.read_text()); update(value); path.write_text(json.dumps(value))


def test_axes_exposure_batch_and_shifts(tmp_path):
    path, dose=save_water(tmp_path)
    g=read_grid(path)
    np.testing.assert_array_equal(g.dose, dose.transpose(2,0,1))
    np.testing.assert_allclose(g.x, np.arange(13)-6)
    np.testing.assert_allclose(g.y, np.arange(15)*1.5-10.5)
    np.testing.assert_allclose(g.z, np.arange(6)*2+1)
    assert g.exposure==2 and g.sd is None
    assert mc_pdd(g, 3, 3).spread is None
    curve, actual=mc_profile(g, 'x', 4.2, 3, 3)
    assert actual==5 and curve.spread is None
    shifted=g.shift_xy(2,-3)
    assert shifted.dose is g.dose
    np.testing.assert_allclose(shifted.x,g.x+2)
    assert shifted.provenance==g.provenance


@pytest.mark.parametrize('mutation,match',[
    (lambda m:m.update(weights=[0]),'positive common'),
    (lambda m:m.update(weights=[-1]),'positive common'),
    (lambda m:m.update(weights=[1,2]),'positive common'),
    (lambda m:m.update(units='cGy'),'units'),
    (lambda m:m.update(bundle_id='wrong'),'identities'),
])
def test_invalid_metadata(tmp_path, mutation, match):
    path,_=save_water(tmp_path)
    change_json(path.parent/'forward_metadata.json', mutation)
    with pytest.raises(ValueError,match=match): read_grid(path)


def test_geometry_and_missing_metadata(tmp_path):
    path,_=save_water(tmp_path)
    image=sitk.ReadImage(str(path)); image.SetOrigin((0,0,0)); sitk.WriteImage(image,str(path))
    with pytest.raises(ValueError,match='grid differs'): read_grid(path)
    path,_=save_water(tmp_path)
    image=sitk.ReadImage(str(path)); direction=[-1.,0,0,0,-1,0,0,0,1]
    image.SetDirection(direction); sitk.WriteImage(image,str(path))
    change_json(tmp_path/'manifest.json',lambda m:m['dose_grid'].update(direction=np.reshape(direction,(3,3)).tolist()))
    with pytest.raises(ValueError,match='orientation'): read_grid(path)
    path,_=save_water(tmp_path)
    change_json(tmp_path/'manifest.json',lambda m:m['planning']['beams'][0]['parameters'].update(gantry_angle=45))
    with pytest.raises(ValueError,match='gantry'): read_grid(path)
    (path.parent/'forward_metadata.json').unlink()
    with pytest.raises(FileNotFoundError): read_grid(path)


def test_finer_spacing_and_relocation(tmp_path):
    path,_=save_water(tmp_path/'original',spacing=(.5,.5,.5))
    (tmp_path/'original').rename(tmp_path/'moved')
    g=read_grid(tmp_path/'moved/derived/dose_scoring_grid.mha')
    np.testing.assert_allclose(np.diff(g.z),.5)
    assert g.sd is None


def fixture_config(tmp_path):
    op,_=save_water(tmp_path/'open',warning=True)
    sl,_=save_water(tmp_path/'slit',factor=.3)
    coords=np.arange(-6.,7.)
    maps=[]
    for depth in [1.,5.,11.]:
        values=20+5*(np.indices((13,13))[0]%2)+np.ones((13,13))*depth
        matrix=np.empty((14,14));matrix[0,0]=0;matrix[0,1:]=coords;matrix[1:,0]=coords;matrix[1:,1:]=values
        file=tmp_path/f'map{depth}.csv';np.savetxt(file,matrix,delimiter=',')
        maps.append('{file='+json.dumps(str(file))+', depth_mm='+str(depth)+'}')
    agl=tmp_path/'agl.csv';agl.write_text('Depth (mm),PDD\n0,10\n1,20\n5,30\n11,20\n')
    config=tmp_path/'config.toml'
    config.write_text(f'''version=1
[defaults]
reference_depth_mm=5.0
normalization_roi_fraction=1.0
x_roi_half_width_mm=3.0
y_roi_half_width_mm=3.0
depths_mm=[1.,5.,11.]
output_root={json.dumps(str(tmp_path/'out'))}
[fields."10"]
mc_open={json.dumps(str(op))}
mc_collimator={json.dumps(str(sl))}
agl={json.dumps(str(agl))}
open_maps=[{','.join(maps)}]
slit_maps=[{','.join(maps)}]
slit_map_x_offset_mm=0.0
slit_map_y_offset_mm=0.0
'''.replace('[1.,5.,11.]','[1.0,5.0,11.0]'))
    return config


@pytest.mark.parametrize('command,extra',[
    ('pdd',[]), ('pdd',['--mc-only']),
    ('lateral',[]), ('lateral',['--mc-only','--combine-depths','--mc-shift-x-mm','1']),
    ('mc-fields-ratio',[]),('data-fields-ratio',[]),('output-factor',[]),
    ('pvdr-depth',[]),('pvdr-depth',['--fields','10']),
])
def test_commands(tmp_path, command, extra):
    config=fixture_config(tmp_path)
    assert main([command,'--config',str(config),*extra])==0
    assert list((tmp_path/'out').rglob('*.png'))
    assert list((tmp_path/'out').rglob('*.csv'))
    assert main(['report','--output-dir',str(tmp_path/'out')])==0
    if command=='mc-fields-ratio':
        import csv
        records=list(csv.DictReader(next((tmp_path/'out').rglob('ratios.csv')).open()))
        np.testing.assert_allclose([float(r['value']) for r in records],.3)
    if command=='output-factor':
        meta=json.loads(next((tmp_path/'out').rglob('metadata.json')).read_text())
        assert all(row['output_factor']==1 for row in meta['results']['output_factors'])


def test_measurement_only_without_mc_files(tmp_path):
    config=fixture_config(tmp_path)
    import shutil
    shutil.rmtree(tmp_path/'open');shutil.rmtree(tmp_path/'slit')
    assert main(['data-fields-ratio','--config',str(config)])==0


def test_normalization_species_units_and_no_second_division(tmp_path):
    path,dose=save_water(tmp_path,exposure=7.)
    for name in ('manifest.json','derived/forward_metadata.json'):
        change_json(tmp_path/name,lambda m:m.update(normalization='independent_primary_photon',
                    units='Gy/primary photon', weight_units='primary photons'))
    grid=read_grid(path)
    assert grid.normalization=='independent_primary_photon'
    np.testing.assert_allclose(grid.dose, dose.transpose(2,0,1))
    np.testing.assert_allclose(grid.dose/grid.exposure, dose.transpose(2,0,1)/7)
    assert grid.sd is None


def test_incompatible_normalizations_for_ratio(tmp_path,capsys):
    config=fixture_config(tmp_path)
    for name in ('manifest.json','derived/forward_metadata.json'):
        change_json(tmp_path/'slit'/name,lambda m:m.update(normalization='independent_primary_photon',
                    units='Gy/primary photon', weight_units='primary photons'))
    assert main(['mc-fields-ratio','--config',str(config)])==1
    assert 'units differ' in capsys.readouterr().err


def test_warnings_report_and_batch_provenance(tmp_path,capsys):
    config=fixture_config(tmp_path)
    assert main(['pdd','--config',str(config),'--mc-only'])==0
    assert 'validity requires user review' in capsys.readouterr().out
    report=(tmp_path/'out/report.md').read_text()
    assert 'unscored step' in report and 'fixed replay' in report
    assert 'number_of_batches' in report
    metadata=json.loads(next((tmp_path/'out').rglob('metadata.json')).read_text())
    assert metadata['saved_dose']['open']['exposure_weight']==2
    assert 'voxel covariance' in metadata['uncertainty']


def test_config_and_cli_reject_removed_swap(tmp_path):
    from tools.analysis_config import load_config
    from tools.dose_analysis import parser
    config=fixture_config(tmp_path)
    config.write_text(config.read_text().replace('[defaults]','[defaults]\nswap_mc_xy=true'))
    with pytest.raises(ValueError,match='unknown key'):load_config(config)
    with pytest.raises(SystemExit):parser().parse_args(['lateral','--swap-mc-xy'])


def test_measurement_centering(tmp_path):
    from tools.analysis_io import read_plane
    file=tmp_path/'measurement.csv'
    np.savetxt(file,[[0,100,102,104],[98,1,2,3],[102,4,5,6]],delimiter=',')
    p=read_plane(file,5,slit=True)
    np.testing.assert_array_equal(p.x,[-2,0,2]);np.testing.assert_array_equal(p.y,[-2,2])
    p=read_plane(file,5,slit=True,offset_x=100,offset_y=98)
    np.testing.assert_array_equal(p.x,[0,2,4]);np.testing.assert_array_equal(p.y,[0,4])


def test_individual_batch_uses_setup_manifest(tmp_path):
    path,_=save_water(tmp_path)
    batch=tmp_path/'batches/batch_001/derived';batch.mkdir(parents=True)
    import shutil
    for name in ('forward_metadata.json','dose_scoring_grid.mha'):
        shutil.copyfile(path.parent/name,batch/name)
    assert read_grid(batch/'dose_scoring_grid.mha').exposure==2


def test_report_needs_no_original_inputs(tmp_path):
    config=fixture_config(tmp_path)
    assert main(['lateral','--config',str(config)])==0
    import shutil
    shutil.rmtree(tmp_path/'open');shutil.rmtree(tmp_path/'slit')
    config.unlink()
    assert main(['report','--output-dir',str(tmp_path/'out')])==0


def test_optional_historical_curves(tmp_path):
    config=fixture_config(tmp_path)
    config.write_text(config.read_text()+'''\n[defaults.output_factor_reference]
attribution="Fixture reference at 100 mm"
values=[[10,1.0]]
[defaults.point_source_pvdr_reference]
attribution="Fixture historical simulation"
values=[[5.5,4.0],[20.0,3.0]]
''')
    assert main(['pvdr-depth','--config',str(config),'--fields','10'])==0
    meta=json.loads((tmp_path/'out/pvdr-depth/collimator/metadata.json').read_text())
    assert meta['results']['point_source_reference']['attribution']=='Fixture historical simulation'
    assert main(['output-factor','--config',str(config)])==0
    meta=json.loads((tmp_path/'out/output-factor/open/metadata.json').read_text())
    assert not meta['results']['reference']['included'] # reference not applicable at 5 mm
