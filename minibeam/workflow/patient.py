"""ROI bookkeeping and explicit target selection for the patient example."""
from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk
from pyRadPlan.cst import create_voi


def read_roi_metadata(directory, cst):
    """Read original ROI identifiers without importing or rasterizing contours."""
    rois = []
    for path in sorted(Path(directory).rglob("*")):
        if not path.is_file():
            continue
        try:
            dataset = pydicom.dcmread(path, stop_before_pixels=True)
        except pydicom.errors.InvalidDicomError:
            continue
        if getattr(dataset, "Modality", None) == "RTSTRUCT":
            rois.extend({"name": str(roi.ROIName), "number": int(roi.ROINumber)}
                        for roi in dataset.StructureSetROISequence)
    imported_names = {voi.name for voi in cst.vois}
    return {
        "original_dicom_rois": rois,
        "omitted_rois": [roi for roi in rois if roi["name"] not in imported_names],
    }


def select_target(cst, target_name):
    """Make one nonempty ROI the sole TARGET in the native StructureSet, in place.

    Other imported TARGET structures become OARs so they cannot enlarge the
    steering field or shift the default isocenter. Masks and names are retained.
    """
    matches = [voi for voi in cst.vois if voi.name == target_name]
    if len(matches) != 1 or not np.any(sitk.GetArrayViewFromImage(matches[0].mask)):
        raise ValueError("Selected target must exist uniquely and be nonempty")
    for index, voi in enumerate(cst.vois):
        if voi.name == target_name:
            kind = "TARGET"
        elif voi.voi_type == "TARGET":
            kind = "OAR"
        else:
            continue
        if kind != voi.voi_type:
            data = voi.model_dump(exclude_computed_fields=True)
            data.update(voi_type=kind, overlap_priority=0 if kind == "TARGET" else 5)
            cst.vois[index] = create_voi(data)
