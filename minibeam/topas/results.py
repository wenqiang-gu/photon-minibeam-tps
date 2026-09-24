"""Assemble native Dij or streaming forward dose from validated TOPAS scores."""
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import SimpleITK as sitk
from pyRadPlan.core import Grid
from pyRadPlan.dij import Dij
from . import manifest as bundle_io
from .scoring import score_rows
from .contracts import ResultsPendingError

def _image(vector, grid):
    image = sitk.GetImageFromArray(np.asarray(vector).reshape(grid.dimensions[::-1]))
    image.SetOrigin(tuple(grid.origin))
    image.SetSpacing(tuple(grid.resolution_vector))
    image.SetDirection(tuple(grid.direction_vector))
    return image


def iter_scores(bundle_dir, *, manifest=None, diagnostics=None):
    root = Path(bundle_dir)
    manifest = bundle_io.load_manifest(root) if manifest is None else manifest
    nvox = int(np.prod(manifest["dose_grid"]["dimensions"]))
    for job in manifest["jobs"]:
        output = bundle_io.bundle_path(root, job["output"])
        if not output.is_file():
            raise ResultsPendingError(f"Missing result for {job['job_id']}; run TOPAS on {job['parameter_file']} from {root}")
        dose = np.zeros(nvox)
        variance = np.zeros(nvox)
        for row, value, var in score_rows(output, job, manifest["dose_grid"], diagnostics=diagnostics, normalization=manifest["normalization"]):
            dose[row], variance[row] = value, var
        yield job, dose, variance

def collect_results(bundle_dir, max_matrix_bytes, *, manifest=None, diagnostics=None) -> Dij:
    root = Path(bundle_dir)
    manifest = bundle_io.load_manifest(root) if manifest is None else manifest
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
    derived = root / "derived"
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
    root = Path(bundle_dir)
    manifest = bundle_io.load_manifest(root) if manifest is None else manifest
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (len(manifest["jobs"]),) or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError(f"Weights must be one finite nonnegative exposure per saved dose column in {manifest.get('weight_units','primary photons')}")
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
    if manifest.get('normalization') != 'original_accelerator_history':
        result['physical_dose_std_error'] = _image(np.sqrt(variance_sum), grid)
    return result
