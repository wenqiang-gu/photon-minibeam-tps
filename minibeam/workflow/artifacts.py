"""Preparation artifacts and saved MATLAB image reconstruction."""
import json
import numpy as np
import SimpleITK as sitk
from scipy.io import savemat
from ..steering import matrad_steering


def save_planning_snapshot(ct, cst, plan, stf, root, metadata):
    """Write only the prepared native planning snapshot and its integrity record."""
    derived = root / 'derived'
    derived.mkdir(exist_ok=True)
    exported = {'ct': ct.to_matrad(), 'cst': cst.to_matrad(), 'pln': plan.to_matrad(), **matrad_steering(stf)}
    manifest = json.loads((root / 'manifest.json').read_text())
    # Native callers may have been enriched inside the adapter. Export exactly
    # the resolved geometry recorded at preparation, not a new TOML read.
    def matlab(value):
        if value is None: return np.empty(0)
        if isinstance(value, dict): return {k: matlab(v) for k,v in value.items()}
        if isinstance(value, list): return [matlab(v) for v in value]
        return value
    if manifest.get('crop_metadata'):
        exported['crop_metadata'] = matlab(manifest['crop_metadata'])
    if manifest.get('column_mapping'):
        exported['column_mapping'] = manifest['column_mapping']
    exported['beamlet_execution'] = manifest.get('beamlet_execution','separate')
    from ..topas.manifest import phase_space_selection
    exported['phase_space_selection'] = phase_space_selection(manifest)
    if manifest.get('beam_geometry'):
        exported['beam_geometry'] = matlab(manifest['beam_geometry'])
    metadata = dict(metadata, units=manifest['units'], weight_units=manifest['weight_units'],
        source_normalization=manifest.get('normalization', 'independent_primary_photon'),
        forward_uncertainty=manifest.get('forward_uncertainty', 'independent columns'),
        normalization='TOPAS Sum / job.normalization_histories (active histories for legacy bundles)',
        lps_to_topas_translation_mm=manifest['lps_to_topas_translation_mm'])
    from .collection import atomic_path, write_json
    from ..topas.manifest import sha256
    destination = derived / 'steering.mat'
    with atomic_path(destination) as temp:
        savemat(temp, exported, long_field_names=True)
    write_json(derived/'planning_snapshot.json', dict(bundle_id=manifest['bundle_id'], sha256=sha256(destination)))
    write_json(derived/'metadata.json', metadata)
    print(f'Run directory: {root}; artifacts: {derived}')


def prepare_project(root, ct, cst, plan, stf, metadata):
    """Prepare portable inputs, persist native planning, and return the project path."""
    from .. import TOPASPhotonEngine
    root = TOPASPhotonEngine(plan).prepare_jobs(ct, cst, stf, bundle_dir=root, provenance=metadata)
    save_planning_snapshot(ct, cst, plan, stf, root, metadata)
    return root


def _uncell(value):
    # plotting accepts both squeezed loadmat data and the preserved export cells.
    while isinstance(value, np.ndarray) and value.size == 1 and value.dtype == object:
        value = value.flat[0]
    return value


def snapshot_images(snapshot, manifest):
    """Reconstruct CT and selected masks from MATLAB [Y,X,Z] snapshots."""
    grid = manifest['ct_grid']
    nx, ny, nz = grid['dimensions']
    shape = (ny, nx, nz)

    def image(cube):
        result = sitk.GetImageFromArray(cube.transpose(2, 0, 1))
        result.SetOrigin(tuple(grid['origin']))
        result.SetSpacing(tuple(grid['resolution'][a] for a in 'xyz'))
        result.SetDirection(tuple(np.asarray(grid['direction']).ravel()))
        return result

    ct = image(np.asarray(_uncell(snapshot['ct']['cubeHU'])).reshape(shape))
    rows = np.asarray(snapshot['cst'], dtype=object).reshape(-1, 6)
    targets = manifest['planning']['plan'].get('target_names', [])
    if isinstance(targets, str):
        targets = [targets]
    masks, missing = {}, []
    for name in targets:
        matches = [r for r in rows if r[1] == name]
        if len(matches) != 1:
            missing.append(name)
            continue
        indices = np.asarray(_uncell(matches[0][3]), dtype=float).ravel()
        if not indices.size:
            missing.append(name)
            continue
        if not np.isfinite(indices).all() or np.any(indices != np.floor(indices)) or np.any((indices < 1) | (indices > np.prod(shape))):
            raise ValueError(f'Invalid saved ROI indices: {name}')
        flat = np.zeros(np.prod(shape), dtype=np.uint8)
        flat[indices.astype(np.int64)-1] = 1
        masks[name] = image(flat.reshape(shape, order='F'))
    if not targets:
        missing.append('(no saved selected target)')
    return ct, masks, missing
