"""Native MATLAB snapshots and local dose artifacts shared by patient workflows."""
import json
import numpy as np
import SimpleITK as sitk
from scipy.io import savemat
from ..steering import matrad_steering


def save_artifacts(ct, cst, plan, stf, root, stage, metadata, *, dij=None, result=None, weights=None):
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
    if manifest.get('column_mapping'):
        exported['column_mapping'] = manifest['column_mapping']
    exported['beamlet_execution'] = manifest.get('beamlet_execution','separate')
    if manifest.get('beam_geometry'):
        exported['beam_geometry'] = matlab(manifest['beam_geometry'])
    metadata = dict(metadata, units=manifest['units'], weight_units=manifest['weight_units'],
        source_normalization=manifest.get('normalization', 'independent_primary_photon'),
        forward_uncertainty=manifest.get('forward_uncertainty', 'independent columns'),
        normalization='TOPAS Sum / job.normalization_histories (active histories for legacy bundles)',
        lps_to_topas_translation_mm=manifest['lps_to_topas_translation_mm'])
    if stage == 'collect':
        exported['dij'] = dij.to_matrad()
        # pyRadPlan 0.5.0 leaves these exported labels zero based.
        for key in ('beamNum', 'rayNum', 'bixelNum'):
            exported['dij'][key] = exported['dij'][key] + 1
    elif stage == 'forward':
        sitk.WriteImage(result['physical_dose'], str(derived / 'dose.mha'))
        metadata['weights'] = np.asarray(weights).tolist()
        if manifest['normalization'] == 'independent_primary_photon':
            metadata['weights_primary_photons'] = metadata['weights']
    from .collection import atomic_path, write_json
    from ..topas.manifest import sha256
    destination = derived / ('result.mat' if stage == 'collect' else 'steering.mat')
    if stage != 'forward':
        with atomic_path(destination) as temp:
            savemat(temp, exported, long_field_names=True)
    if stage == 'prepare':
        write_json(derived/'planning_snapshot.json', dict(bundle_id=manifest['bundle_id'], sha256=sha256(destination)))
    (derived / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    print(f'Run directory: {root}; artifacts: {derived}')
