"""Collect completed portable runs independently of editable planning settings."""
import json
from pathlib import Path
from contextlib import contextmanager
import tempfile
from importlib.metadata import version
import numpy as np
import SimpleITK as sitk
from scipy.io import loadmat, savemat
from pyRadPlan.core import Grid
from ..topas import manifest as io, results


@contextmanager
def atomic_path(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.', suffix=path.suffix, dir=path.parent)
    import os
    os.close(fd)
    temporary = Path(name)
    try:
        yield temporary
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path, value):
    with atomic_path(path) as temp:
        temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def _items(value):
    return value if isinstance(value, list) else [value]


def _matlab_empty_cells(value):
    # scipy reads an empty MATLAB struct as None inside an object cell.
    if isinstance(value, dict):
        for key in value:
            value[key] = _matlab_empty_cells(value[key])
    elif isinstance(value, np.ndarray):
        if value.dtype.names:
            for key in value.dtype.names:
                _matlab_empty_cells(value[key])
        elif value.dtype == object:
            for index in np.ndindex(value.shape):
                value[index] = _matlab_empty_cells(value[index])
    return np.empty((0, 0)) if value is None else value


def planning_snapshot(root, manifest):
    path = root/'derived/steering.mat'
    receipt = root/'derived/planning_snapshot.json'
    integrity = 'Legacy planning snapshot is unhashed; structural checks cannot establish original content integrity.'
    if receipt.exists():
        record = json.loads(receipt.read_text())
        if record['bundle_id'] != manifest['bundle_id'] or record['sha256'] != io.sha256(path):
            raise ValueError('Planning snapshot hash or bundle identity mismatch')
        integrity = 'Planning snapshot SHA-256 and bundle association verified.'
    data = loadmat(path, simplify_cells=True)
    if not {'ct','cst','pln','stf'}.issubset(data):
        raise ValueError('Incomplete planning snapshot')
    ct = data['ct']; grid = manifest['ct_grid']
    if any(g['dimensions'][0] != g['dimensions'][1] for g in (grid, manifest['dose_grid'])):
        raise ValueError('Native matRad export requires square transverse grids with pyRadPlan 0.5.0')
    for key, expected in [('size',grid['dimensions']), ('origin',grid['origin']), ('direction',grid['direction'])]:
        if not np.allclose(np.asarray(ct[key]).ravel(), np.asarray(expected).ravel()):
            raise ValueError(f'Planning snapshot CT {key} mismatch')
    if not np.allclose([ct['resolution'][a] for a in 'xyz'], [grid['resolution'][a] for a in 'xyz']):
        raise ValueError('Planning snapshot CT spacing mismatch')
    if not isinstance(data['pln'], dict) or np.asarray(data['cst']).size == 0:
        raise ValueError('Missing planning structures or plan')
    cst = np.asarray(data['cst'], dtype=object)
    if cst.ndim == 1:
        cst = cst.reshape(1, -1)
    if cst.shape[1] != 6:
        raise ValueError('Malformed structure snapshot')
    for structure in cst:
        indices = np.asarray(structure[3]).ravel()
        if indices.size and (not np.isfinite(indices.astype(float)).all() or np.any(indices < 1) or np.any(indices > np.prod(grid['dimensions']))):
            raise ValueError('Structure snapshot voxel indices outside CT grid')
    saved_plan = manifest['planning']['plan']
    for key, native in [('radiationMode','radiation_mode'), ('numOfFractions','num_of_fractions'), ('prescribedDose','prescribed_dose')]:
        if native in saved_plan and data['pln'].get(key) != saved_plan[native]:
            raise ValueError(f'Planning snapshot {key} mismatch')
    targets = sorted(str(row[1]) for row in cst if row[2] == 'TARGET')
    if targets != sorted(saved_plan.get('target_names', targets)):
        raise ValueError('Planning snapshot target associations mismatch')
    for key, expected in saved_plan.get('steering_settings', {}).items():
        actual = data['pln'].get('propStf', {}).get(key)
        matches = actual == expected if isinstance(expected, str) else actual is not None and np.allclose(np.asarray(actual).ravel(), np.asarray(expected).ravel())
        if not matches:
            raise ValueError(f'Planning snapshot steering setting {key} mismatch')
    beams = _items(data['stf'])
    saved = manifest['planning']['beams']
    if len(beams) != len(saved):
        raise ValueError('Planning snapshot beam count mismatch')
    for beam, record in zip(beams, saved):
        p = record['parameters']
        for key, native in [('gantryAngle','gantry_angle'),('couchAngle','couch_angle'),('SAD','sad'),('bixelWidth','bixel_width'),('isoCenter','iso_center'),('sourcePoint','source_point')]:
            if not np.allclose(beam[key], p[native]):
                raise ValueError(f'Planning snapshot beam {key} mismatch')
        if int(beam['totalNumOfBixels']) != record['num_bixels'] or int(beam['numOfRays']) != record['num_rays']:
            raise ValueError('Planning snapshot steering counts mismatch')
    for job in io.member_jobs(manifest):
        beam = beams[job['beam_index']-1]
        rays = _items(beam['ray'])
        if len(rays) != int(beam['numOfRays']) or job['ray_index'] > len(rays):
            raise ValueError('Planning snapshot ray association mismatch')
        ray = rays[job['ray_index']-1]
        counts = np.atleast_1d(beam['numOfBixelsPerRay'])
        if job['beamlet_index'] > counts[job['ray_index']-1]:
            raise ValueError('Planning snapshot bixel association mismatch')
        source = job['source']
        if 'aim_lps_mm' in source and not np.allclose(source['aim_lps_mm'], np.asarray(beam['isoCenter'])+ray['rayPos']):
            raise ValueError('Planning snapshot ray aiming point mismatch')
        if 'selection' in source and not np.allclose(source['selection']['center_xy_mm'], np.asarray(ray['rayPos_bev'])[[0,2]] * [1,-1]):
            raise ValueError('Planning snapshot phase-space square mismatch')
    if manifest.get('beam_geometry'):

        geometry = _items(data.get('beam_geometry', []))
        if len(geometry) != len(saved) or any(g['configuration_sha256'] != s['configuration_sha256'] or g['beam_index'] != s['beam_index'] or g['geometry'] != s['geometry'] for g,s in zip(geometry,manifest['beam_geometry'])):
            raise ValueError('Planning snapshot beam geometry mismatch')
    del data
    # Preserve native MATLAB cell/struct shapes rather than reserializing squeezed data.
    return _matlab_empty_cells({k:v for k,v in loadmat(path).items() if not k.startswith('__')}), integrity


def indexing_report(manifest, stage, image=None):
    if manifest.get('beamlet_execution') == 'combined':
        from .combined_report import combined_report
        return combined_report(manifest, stage, image)
    nx,ny,nz = manifest['dose_grid']['dimensions']
    jobs = manifest['jobs']
    targets = manifest.get('planning', {}).get('plan', {}).get('target_names', [])
    roi_name = (targets[0] if targets else 'YOUR_ROI_NAME').replace("'", "''")
    lines = ['# Dose indexing', '', f"Run: `{manifest['bundle_id']}`. Stage: `{stage}`.",
        f"Normalization: {manifest['units']}; weights: {manifest['weight_units']}.", '',
        '## Files and MATLAB loading', '',
        '| File / variable | Meaning |', '|---|---|',
        '| `steering.mat` | Prepared planning snapshot; contains no calculated dose. |',
        '| `result.mat` | The saved planning variables plus collected `dij`. The filename is singular. |',
        '| `ct` | CT voxel values (`cubeHU{1}`), dimensions and physical coordinates. |',
        '| `cst` | Structure cell table: Native structure number, name, type, voxel-index scenarios, parameters and objectives. |',
        '| `pln` | Native planning settings and saved source/engine configuration. |',
        '| `stf` | Native beam struct array; each beam contains a `ray` struct array and per-ray beamlet fields. |',
        '| `beam_geometry` | Cell array of resolved hardware records, associated by 1-based `beam_index`. |',
        '| `dij` | Sparse dose influence matrix, grids and column labels. |', '',
        'Run the examples from this `derived` directory. Keep planning and result variables separate:', '',
        '```matlab', "planning = load('steering.mat');", 'stf = planning.stf;',
        'beam = stf(1);', 'ray = beam.ray(1);', 'hardware = planning.beam_geometry{1};',
        'assert(hardware.beam_index == 1);', 'gantry = beam.gantryAngle;  % degrees',
        'couch = beam.couchAngle;    % degrees', 'isoLPS = beam.isoCenter;    % mm', '```', '',
        '`SAD` and `bixelWidth` are millimeters. `sourcePoint` and `rayPos` are relative to the isocenter; add `isoCenter` for absolute LPS positions. `_bev` fields use the native beam coordinate frame.',
        'Only selected beamlets appear in steering, TOPAS jobs and dose columns. Unselected beamlets are absent, not empty placeholders.',
        'Unused `pln.propOpt` and `pln.propSeq` settings may be empty because optimization and sequencing are not implemented. These empty fields do not indicate missing dose columns.', '',
        '## Dose matrix and voxel rows', '',
        ('This forward stage does not create result.mat. The following dose-matrix examples require a separately completed collect stage.' if stage == 'forward' else 'This collect stage writes result.mat. The following examples use its calculated dose matrix.'), '',
        '```matlab', "result = load('result.mat');", 'dij = result.dij;',
        'D = dij.physicalDose{1};', 'assert(issparse(D));',
        f'assert(isequal(size(D), [{nx*ny*nz}, {len(jobs)}]));', '```', '',
        '`{1}` selects the single nominal scenario; it does not select a beam. Keep `D` sparse; convert only the dose vector you want to display to a full array.',
        f'Matrix rows: {nx*ny*nz:,}; MATLAB dose cube dimensions `[Y,X,Z] = [{ny},{nx},{nz}]`.',
        'MATLAB uses 1-based, column-major voxel indexing (Y varies fastest).', '',
        '```matlab', f'doseDims = [{ny},{nx},{nz}];  % [Y,X,Z]',
        'iy = 1; ix = 1; iz = 1;  % choose valid 1-based voxel coordinates',
        'row = sub2ind(doseDims, iy, ix, iz);',
        'j = 1;  % choose a global beamlet column',
        'beamletCube = reshape(full(D(:,j)), doseDims);',
        f'weights = ones(size(D,2),1);  % {manifest["weight_units"]} per beamlet',
        'doseCube = reshape(full(D * weights), doseDims);  % Gy', '```', '',
        f'Each matrix coefficient is in {manifest["units"]}. Weights follow saved column order and are not MU; no implicit MU calibration is applied.', '',
        '## Beamlet columns and TOPAS jobs', '',
        f'Matrix columns / exposure weights: {len(jobs)} beamlets in manifest job order.', '',
        '| Beam (1-based) | Gantry / couch (deg) | Columns | Job IDs |', '|---|---|---|---|']
    for bi in dict.fromkeys(j['beam_index'] for j in jobs):
        group = [j for j in jobs if j['beam_index']==bi]
        columns = str(group[0]['bixel_index']) if len(group)==1 else f"{group[0]['bixel_index']}–{group[-1]['bixel_index']}"
        ids = f"{group[0]['job_id']} … {group[-1]['job_id']}"
        lines.append(f"| {bi} | {group[0]['gantry_angle_deg']:g} / {group[0]['couch_angle_deg']:g} | {columns} | {ids} |")
    lines += ['', '`beamNum` identifies the beam in `stf` and `beam_geometry`. `rayNum` is local to that beam; `bixelNum` is local to that ray. All exported labels are 1-based. `bixelNum` is NOT the global matrix-column number; `manifest.jobs[].bixel_index` is.',
        'In native Python objects the labels are zero-based; the exported MATLAB labels below are already 1-based. Do not add one again.', '',
        '```matlab', 'j = 1;', 'b = dij.beamNum(j);', 'r = dij.rayNum(j);',
        'k = dij.bixelNum(j);  % beamlet within ray r, not column j',
        'beam = result.stf(b);', 'ray = beam.ray(r);',
        'assert(k <= beam.numOfBixelsPerRay(r));',
        'nominalEnergy = ray.energy(k);  % native label, not the source spectrum',
        'hardware = result.beam_geometry{b};', 'assert(hardware.beam_index == b);',
        "manifest = jsondecode(fileread('../manifest.json'));", 'job = manifest.jobs(j);',
        'assert(job.bixel_index == j && job.beam_index == b);',
        'assert(job.ray_index == r && job.beamlet_index == k);',
        'disp(job.job_id);', 'disp(job.parameter_file);  % relative to bundle root', '```', '',
        'Beamlet fields such as `energy` are arrays within each native ray, rather than a separate MATLAB beamlet struct array. One ray may contain multiple beamlets. With one beamlet per ray, all `bixelNum` values are 1 even when there are many columns.',
        'Each shift has independent numbering starting at 1.', '',
        '## Structures and CT voxels', '',
        'The following reconstructs an ROI on the CT grid, using the first (nominal) scenario:', '',
        '```matlab', f"roiName = '{roi_name}';  % edit to an existing ROI name",
        'roiRow = find(strcmp(planning.cst(:,2), roiName));',
        "assert(isscalar(roiRow), 'ROI name must identify exactly one structure');",
        'ctCube = planning.ct.cubeHU{1};', 'mask = false(size(ctCube));',
        'voxelIndices = planning.cst{roiRow,4}{1};', 'mask(voxelIndices) = true;',
        'nativeStructureNumber = planning.cst{roiRow,1};  % native numbering, not voxel indices', '```', '',
        'An empty ROI has an empty index list and an all-false mask. The first cst column retains pyRadPlan 0.5.0 zero-based structure numbering; it is not the original DICOM ROI number. Original DICOM ROI numbers and names are preserved in metadata.json (original_dicom_rois). Structure numbers, 1-based MATLAB table rows and 1-based voxel indices are distinct.',
        'Structure voxel indices use the CT cube in MATLAB [Y,X,Z] order. Do not use CT indices directly on a different scoring grid: resample/map the mask to that grid before extracting dose.', '',
        '## Forward dose, MHA and uncertainty', '',
        '`variance_of_mean.npz` retains native rows: `x + nx*(y + ny*z)` for zero-based voxel coordinates; its rows are not MATLAB-permuted. Its columns follow the same job order.',
        f"Forward uncertainty: {manifest.get('forward_uncertainty', 'not specified')}.", '',
        f"Structure (`cst`) indices reference the CT grid {manifest['ct_grid']['dimensions']} (XYZ). CT and scoring grids {'match' if manifest['ct_grid']==manifest['dose_grid'] else 'differ'}.",
        'Forward weights follow the same column order. Forward dose sums weighted beamlet contributions into one cube; it does not retain separate columns.',
        'Forward writes dose.mha, not a new result.mat. The MATLAB example applies after collect.', '']
    grid = manifest['ct_grid']
    if image is not None:
        lines += [f'MHA dimensions (XYZ): {image.GetSize()}; origin (mm): {image.GetOrigin()}; spacing (mm): {image.GetSpacing()}; direction: {image.GetDirection()}.']
    else:
        lines += [f"Forward MHA uses the CT grid: origin {grid['origin']}, spacing {grid['resolution']}, direction {grid['direction']}."]
    lines += ['Physical coordinates are DICOM LPS: Left, Posterior, Superior. SimpleITK NumPy arrays use zero-based `[Z,Y,X]` indexing.', '']
    return '\n'.join(lines)


def scorer_warning_report(diagnostics):
    """Keep scoring caveats beside both matrix and forward-dose indexing."""
    if not diagnostics:
        return "\n## Scorer warnings\n\nNo scorer warnings were reported.\n"
    lines = ["", "## Scorer warnings", "",
             "**Completed with scorer warnings; validity requires user review.**",
             "Statistical uncertainty does not account for missing energy. No lost-dose percentage can be inferred from these values.", ""]
    for warning in diagnostics:
        lines += [f"### {warning['job_id']}", "",
                  f"CSV: `{warning['csv_path']}`",
                  warning['warning_text'],
                  f"Unscored steps: {warning['unscored_steps']}; total unscored energy: {warning['unscored_energy_mev']:.9g} MeV.",
                  f"Simulated histories: {warning['histories']:,}; source normalization: `{warning['normalization']}`.", ""]
    return "\n".join(lines)


def collect_bundle(root, stage, *, weight_per_bixel=1.0):
    root = Path(root).resolve()
    manifest = io.load_manifest(root)
    exported, integrity = planning_snapshot(root, manifest)
    derived = root/'derived'
    if manifest.get('column_mapping'):
        exported['column_mapping'] = manifest['column_mapping']
    exported['beamlet_execution'] = manifest.get('beamlet_execution','separate')
    image = None
    diagnostics = []
    current = io.implementation_hashes()
    old = manifest.get('implementation_sha256', {})
    status = dict(stage=stage, bundle_id=manifest['bundle_id'], planning_snapshot_integrity=integrity,
        implementation_differences=sorted(k for k in old.keys() | current.keys() if old.get(k)!=current.get(k)),
        normalization=manifest['normalization'], units=manifest['units'], weight_units=manifest['weight_units'],
        preparation_versions=manifest.get('versions', {}),
        collection_versions={name: version(name) for name in manifest.get('versions', {})})
    if stage == 'collect':
        dij = results.collect_results(root, 4*1024**3, manifest=manifest, diagnostics=diagnostics)
        exported['dij'] = dij.to_matrad()
        # Upstream optional scenario containers can contain None cells.
        for value in exported['dij'].values():
            if isinstance(value, np.ndarray) and value.dtype == object:
                for index in np.ndindex(value.shape):
                    if value[index] is None:
                        value[index] = np.empty(0)
        for key in ('beamNum','rayNum','bixelNum'):
            exported['dij'][key] += 1
        with atomic_path(derived/'result.mat') as temp:
            savemat(temp, exported, long_field_names=True)
        status['matrix_shape'] = list(dij.physical_dose.flat[0].shape)
    elif stage == 'forward':
        weights = np.full(len(manifest['jobs']), weight_per_bixel, dtype=float) if np.isscalar(weight_per_bixel) else np.asarray(weight_per_bixel, dtype=float)
        plots = None
        try:
            from .beam_dose_plot import BeamDosePlots
            plots = BeamDosePlots(exported, manifest, weights, diagnostics)
        except Exception as error:
            status['dose_plot'] = dict(status='failed', error=str(error))
            print(f'Beam plotting unavailable: {error}')
        result = results.collect_forward(weights, root, manifest=manifest, diagnostics=diagnostics,
                                         on_beam=plots.consume if plots is not None else None)
        image = result['physical_dose']
        if 'physical_dose_std_error' in result:
            with atomic_path(derived/'dose_std_error_dose_grid.mha') as temp:
                sitk.WriteImage(result['physical_dose_std_error'], str(temp))
        if manifest['ct_grid'] != manifest['dose_grid']:
            grid = Grid.model_validate(manifest['ct_grid'])
            image = sitk.Resample(image, list(grid.dimensions), sitk.Transform(), sitk.sitkLinear,
                tuple(grid.origin), tuple(grid.resolution_vector), tuple(grid.direction_vector), 0., sitk.sitkFloat64)
        with atomic_path(derived/'dose.mha') as temp:
            sitk.WriteImage(image, str(temp))
        status['weights'] = weights.tolist()
        status['forward_uncertainty'] = manifest.get('forward_uncertainty')
        # Beam callbacks retain only 2D views; render after the common scale is known.
        if plots is not None:
            try:
                status['dose_plot'] = plots.save(derived)
            except Exception as error:
                status['dose_plot'] = dict(status='failed', error=str(error))
                print(f'Dose saved, but beam plotting failed: {error}')

    else:
        raise ValueError('Collection stage must be collect or forward')
    with atomic_path(derived/'indexing.md') as temp:
        report = indexing_report(manifest, stage, image) + scorer_warning_report(diagnostics)
        if stage == 'forward':
            from .beam_dose_plot import beam_plot_report
            report += beam_plot_report(status['dose_plot'])
        temp.write_text(report)
    status['scorer_warnings'] = diagnostics
    status['review_required'] = bool(diagnostics)
    status['status'] = 'complete'
    write_json(derived/f'{stage}_metadata.json', status)
    outcome = 'completed with scorer warnings; validity requires user review' if diagnostics else 'complete'
    print(f'{root.name}: {stage} {outcome}; indexing report: {derived / "indexing.md"}')


def collect_saved_run(root, stage, *, weight_per_bixel=1.0):
    root = Path(root).resolve()
    study_path = root/'study.json'
    if study_path.exists() and (root/'manifest.json').exists():
        raise ValueError('Ambiguous run: both study.json and manifest.json exist')
    if not study_path.exists():
        return collect_bundle(root, stage, weight_per_bixel=weight_per_bixel)
    index = json.loads(study_path.read_text())
    entries = index['setups']
    paths = [io.bundle_path(root, e['directory']) for e in entries]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError('Empty or duplicate study directories')
    failures = []
    for entry, path in zip(entries, paths):
        entry['last_stage'] = stage
        try:
            collect_bundle(path, stage, weight_per_bixel=weight_per_bixel)
            entry['last_stage_status'] = 'complete'
            entry.pop('error',None)
        except (ValueError, OSError, RuntimeError) as error:
            entry['last_stage_status'] = 'failed'
            entry['error'] = str(error)
            failures.append(entry['directory'])
            print(f"{entry['directory']}: {error}")
        write_json(study_path, index)
    if failures:
        raise ValueError('Incomplete setups: '+', '.join(failures))
