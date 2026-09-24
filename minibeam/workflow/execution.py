"""Execute a local workflow stage; TOPAS itself runs externally."""
import numpy as np
from pyRadPlan import calc_dose_influence, calc_dose_forward
from minibeam import TOPASPhotonEngine
from .artifacts import save_artifacts


def run_stage(stage, root, ct, cst, plan, stf, metadata, *, weight_per_bixel=1.0):
    """Validate once per stage and save native planning/dose artifacts."""
    dij = result = weights = None
    if stage == 'prepare':
        root = TOPASPhotonEngine(plan).prepare_jobs(ct, cst, stf, provenance=metadata)
    elif stage == 'collect':
        dij = calc_dose_influence(ct, cst, stf, plan)
    elif stage == 'forward':
        weights = np.full(stf.total_number_of_bixels, weight_per_bixel, dtype=float)
        result = calc_dose_forward(ct, cst, stf, plan, weights=weights)
    else:
        raise ValueError(f'Unknown simulation stage: {stage}')
    save_artifacts(ct, cst, plan, stf, root, stage, metadata,
                   dij=dij, result=result, weights=weights)
