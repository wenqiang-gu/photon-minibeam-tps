"""Assemble native Dij or streaming forward dose from validated TOPAS scores."""
from pathlib import Path
import time
import numpy as np
import scipy.sparse as sp
import SimpleITK as sitk
from pyRadPlan.core import Grid
from pyRadPlan.dij import Dij
from . import manifest as bundle_io
from .scoring import score_chunks
from .contracts import ResultsPendingError

def _image(vector, grid):
    image = sitk.GetImageFromArray(np.asarray(vector).reshape(grid.dimensions[::-1]))
    image.SetOrigin(tuple(grid.origin))
    image.SetSpacing(tuple(grid.resolution_vector))
    image.SetDirection(tuple(grid.direction_vector))
    return image


def collection_context(path, *, manifest=None, diagnostics=None):
    """Freeze completed batches once. Invalid partial results are never ignored."""
    from ..batches import locate, inventory
    output_root = Path(path).resolve()
    root, selected = locate(output_root)
    if manifest is not None and '_transport_batches' in manifest:
        return root, output_root, manifest
    manifest = dict(bundle_io.load_manifest(root) if manifest is None else manifest)
    batches = inventory(root, manifest)
    if selected is not None:
        batches = [b for b in batches if b['batch_id'] == selected]
        if not batches: raise ValueError('Batch is not in the saved index')
    included, excluded = [], []
    for batch in batches:
        present = [bundle_io.bundle_path(root, j['output']).is_file() for j in batch['jobs']]
        if all(present):
            included.append(batch)
        else:
            missing = [j['job_id'] for j, exists in zip(batch['jobs'], present) if not exists]
            excluded.append(dict(batch_id=batch['batch_id'], missing_jobs=missing))
            print(f"Incomplete {batch['batch_id']}: excluded ({len(missing)} missing jobs)")
            # Validate files that do exist even though this batch cannot contribute.
            print(f"Validating existing files in excluded {batch['batch_id']} (not averaged)")
            started = time.monotonic()
            checked_count = 0
            for job, exists in zip(batch['jobs'], present):
                if exists:
                    checked = dict(job, job_id=batch['batch_id']+'/'+job['job_id'])
                    checked_count += 1
                    _read_score(root, checked, manifest['dose_grid'], manifest['normalization'],
                                diagnostics, checked_count, sum(present), started)
    if not included:
        raise ResultsPendingError('No fully completed transport batches; run all jobs in at least one batch')
    manifest['_transport_batches'] = included
    repeated = len(included) > 1
    metadata = dict(included_batches=[b['batch_id'] for b in included], excluded_batches=excluded,
        number_averaged=len(included), method='arithmetic mean of equal-history normalized columns',
        source_reuse='fixed saved replay particles; no reduction of source-sampling uncertainty',
        variance_convention=('between-batch variance / number of batches; conditional transport uncertainty'
                             if repeated else 'legacy single-batch scorer variance'),
        fingerprints=[dict(batch_id=b['batch_id'], fingerprint=b.get('batch_fingerprint',manifest['bundle_id'])) for b in included],
        jobs=[dict(batch_id=b['batch_id'], job_id=j['job_id'], seed=j['seed'],
                   histories=j['histories'], output=j['output']) for b in included for j in b['jobs']])
    manifest['batch_metadata'] = metadata
    if repeated:
        manifest['forward_uncertainty'] = 'unavailable: batch averaging; combined covariance not implemented'
    print('Averaging completed batches: '+', '.join(metadata['included_batches']))
    return root, output_root, manifest


def _read_score(root, job, grid, normalization, diagnostics, number, total, started_all):
    output = bundle_io.bundle_path(root, job['output'])
    nvox = int(np.prod(grid['dimensions']))
    started = last_update = time.monotonic()
    print(f"Reading {number}/{total}: {job['job_id']} — {output} "
          f"({output.stat().st_size / 1024**2:.1f} MiB)", flush=True)
    dose, variance = np.zeros(nvox), np.zeros(nvox)
    processed = 0
    for row, value, var in score_chunks(output, job, grid, diagnostics=diagnostics, normalization=normalization):
        dose[row], variance[row] = value, var
        processed += len(row)
        now = time.monotonic()
        if now-last_update >= 5:
            print(f"  {job['job_id']}: {processed:,}/{nvox:,} bins ({processed/nvox:.0%}); {now-started:.1f} s", flush=True)
            last_update = now
    print(f"Validated {number}/{total}: {job['job_id']} in {time.monotonic()-started:.1f} s; "
          f"total elapsed {time.monotonic()-started_all:.1f} s", flush=True)
    return dose, variance


def iter_scores(bundle_dir, *, manifest=None, diagnostics=None):
    root, _, manifest = collection_context(bundle_dir, manifest=manifest, diagnostics=diagnostics)
    batches = manifest['_transport_batches']
    total = len(batches)*len(manifest['jobs'])
    started = time.monotonic()
    number = 0
    for column, original in enumerate(manifest['jobs']):
        mean = m2 = variance = None
        for n, batch in enumerate(batches, 1):
            job = batch['jobs'][column]
            checked = dict(job, job_id=batch['batch_id']+'/'+job['job_id']) if len(batches)>1 or batch['batch_id']!='batch_000' else job
            number += 1
            dose, var = _read_score(root, checked, manifest['dose_grid'], manifest['normalization'],
                                    diagnostics, number, total, started)
            if n == 1:
                mean, variance = dose, var
                if len(batches)>1: m2 = np.zeros_like(mean)
            else:
                delta = dose - mean
                mean += delta/n
                m2 += delta*(dose-mean)
        if len(batches)>1:
            variance = m2/(len(batches)*(len(batches)-1))
        yield original, mean, variance

def collect_results(bundle_dir, max_matrix_bytes, *, manifest=None, diagnostics=None) -> Dij:
    root, output_root, manifest = collection_context(bundle_dir, manifest=manifest, diagnostics=diagnostics)
    columns, variances = [], []
    total_bytes = 0
    for _, dose, variance in iter_scores(root, manifest=manifest, diagnostics=diagnostics):
        column, var_column = sp.csc_matrix(dose[:, None]), sp.csc_matrix(variance[:, None])
        total_bytes += sum(a.nbytes for m in (column, var_column) for a in (m.data, m.indices, m.indptr))
        if 2 * total_bytes > max_matrix_bytes:
            raise MemoryError("Matrix assembly exceeds max_matrix_bytes; use streaming forward dose or a coarser grid")
        columns.append(column)
        variances.append(var_column)
    matrix = sp.hstack(columns, format="csc")
    variance_matrix = sp.hstack(variances, format="csc")
    derived = output_root / "derived"
    derived.mkdir(exist_ok=True)
    from ..workflow.collection import atomic_path
    with atomic_path(derived / "variance_of_mean.npz") as temp:
        sp.save_npz(temp, variance_matrix)
    # Keep uncertainty separate: upstream Dij's variance multiplication uses
    # w rather than w^2. Our forward path below propagates independent errors.
    jobs = manifest["jobs"]
    return Dij(dose_grid=Grid.model_validate(manifest["dose_grid"]),
               ct_grid=Grid.model_validate(manifest["ct_grid"]), physical_dose=matrix,
               num_of_beams=len({j["beam_index"] for j in jobs}),
               beam_num=np.array([j["beam_index"] - 1 for j in jobs]),
               ray_num=np.array([j["ray_index"] - 1 for j in jobs]),
               bixel_num=np.array([j["beamlet_index"] - 1 for j in jobs]))

def collect_forward(weights, bundle_dir, *, manifest=None, diagnostics=None, on_beam=None):
    # Reject invalid weights before inspecting simulation completion.
    from ..batches import locate
    base, _ = locate(bundle_dir)
    manifest = bundle_io.load_manifest(base) if manifest is None else manifest
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (len(manifest["jobs"]),) or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError(f"Weights must be one finite nonnegative exposure per saved dose column in {manifest.get('weight_units','primary photons')}")
    root, output_root, manifest = collection_context(bundle_dir, manifest=manifest, diagnostics=diagnostics)
    grid = Grid.model_validate(manifest["dose_grid"])
    dose_sum = np.zeros(grid.num_voxels)
    variance_sum = np.zeros(grid.num_voxels)
    current_beam = None
    beam_sum = np.zeros(grid.num_voxels) if on_beam is not None else None
    for w, (job, dose, variance) in zip(weights, iter_scores(root, manifest=manifest, diagnostics=diagnostics)):
        if on_beam is not None:
            if current_beam is not None and current_beam != job['beam_index']:
                on_beam(current_beam, _image(beam_sum, grid))
                beam_sum.fill(0.)
            current_beam = job['beam_index']
            beam_sum += w * dose
        dose_sum += w * dose
        variance_sum += w * w * variance
    if on_beam is not None and current_beam is not None:
        on_beam(current_beam, _image(beam_sum, grid))
    result = {"physical_dose": _image(dose_sum, grid)}
    if manifest.get('normalization') != 'original_accelerator_history' and len(manifest['_transport_batches']) == 1:
        result['physical_dose_std_error'] = _image(np.sqrt(variance_sum), grid)
    return result
