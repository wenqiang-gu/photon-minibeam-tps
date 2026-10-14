"""Readers for saved MHA dose, Sphinx CSV maps, and AGL PDD curves."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import SimpleITK as sitk


@dataclass
class Grid:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray  # Historical numerical-helper name for depth, not physical MHA Z.
    dose: np.ndarray  # [analysis X, analysis Y, depth]
    sd: np.ndarray | None = None
    exposure: float = 1.0
    unit: str = "Gy"
    path: Path | None = None
    normalization: str = ""
    provenance: dict = field(default_factory=dict)

    def shift_xy(self, x_mm: float, y_mm: float) -> Grid:
        """Translate coordinate arrays without changing or resampling dose samples."""
        if not np.isfinite(x_mm) or not np.isfinite(y_mm):
            raise ValueError("MC shifts must be finite")
        return Grid(self.x + x_mm, self.y + y_mm, self.z, self.dose,
                    self.sd, self.exposure, self.unit, self.path, self.normalization, self.provenance)


@dataclass
class Plane:
    x: np.ndarray
    y: np.ndarray
    dose: np.ndarray  # [Y, X]
    depth: float
    path: Path | None = None
    centering: dict = field(default_factory=dict)


def read_grid(path: Path) -> Grid:
    """Read saved, exposure-weighted Gy; never divide by simulation histories.

    Internal array axes are [analysis X, analysis Y, depth] = [LPS X, Z, Y].
    Identity-direction water images only: arbitrary patient/beam resampling is
    intentionally outside this measurement-analysis reader.
    """
    path = Path(path).expanduser().resolve()
    meta_path = path.parent / "forward_metadata.json"
    meta = json.loads(meta_path.read_text())
    # Individual transport batches share the setup manifest, not a new plan.
    setup = path.parent.parent
    manifest_path = setup / "manifest.json"
    if not manifest_path.is_file() and setup.name.startswith("batch_") and setup.parent.name == "batches":
        manifest_path = setup.parent.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if meta.get("stage") != "forward" or not manifest.get("bundle_id") or meta.get("bundle_id") != manifest["bundle_id"]:
        raise ValueError("MHA forward metadata and manifest have inconsistent run identities")
    if path.name != meta.get("scoring_grid_dose", "dose_scoring_grid.mha"):
        raise ValueError("Use the saved dose_scoring_grid.mha, not a resampled dose image")
    convention = manifest.get("normalization")
    supported = {
        "original_accelerator_history": ("Gy/original accelerator history", "original accelerator histories"),
        "independent_primary_photon": ("Gy/primary photon", "primary photons"),
    }
    if convention not in supported:
        raise ValueError("Unsupported saved source normalization")
    units, weight_units = supported[convention]
    for key, expected in (("normalization", convention), ("units", units), ("weight_units", weight_units)):
        if meta.get(key) != expected or manifest.get(key) != expected:
            raise ValueError(f"Inconsistent saved {key}; cannot normalize MHA dose")
    raw_weights = meta.get("weights", [])
    if not isinstance(raw_weights, list) or any(isinstance(w, bool) or not isinstance(w, (int, float)) for w in raw_weights):
        raise ValueError("Exposure weights must be numeric, not Boolean")
    weights = np.asarray(raw_weights, dtype=float)
    if (weights.ndim != 1 or weights.size != len(manifest.get("jobs", [])) or not weights.size
            or not np.isfinite(weights).all() or (weights <= 0).any() or not np.all(weights == weights[0])):
        raise ValueError("Analysis requires saved positive common exposure weights, one per dose column")
    water = manifest.get("provenance", {}).get("water_measurement")
    if not water or manifest.get("provenance", {}).get("synthetic") != "water":
        raise ValueError("Analysis supports saved single-beam water projects only")
    beams = manifest.get("planning", {}).get("beams", [])
    if len(beams) != 1:
        raise ValueError("Analysis supports one beam only")
    beam = beams[0]["parameters"]
    if not np.allclose([beam["gantry_angle"], beam["couch_angle"]], 0, atol=1e-9, rtol=0):
        raise ValueError("Water beam must travel along physical +Y (zero gantry/couch)")
    surface = np.asarray(water["surface_center_lps_mm"], dtype=float)
    if surface.shape != (3,) or not np.isfinite(surface).all():
        raise ValueError("Invalid saved entrance surface")
    if not np.allclose(beam["iso_center"], surface, rtol=0, atol=1e-6):
        raise ValueError("Water isocenter must be the entrance center")
    source = np.asarray(beam["source_point"], dtype=float)
    if not np.allclose(source, [0, -float(water["ssd_mm"]), 0], rtol=0, atol=1e-6):
        raise ValueError("Saved water source direction/SSD is inconsistent")
    image = sitk.ReadImage(str(path))
    grid = manifest["dose_grid"]
    if image.GetDimension() != 3 or image.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError("Expected a scalar 3D MHA dose")
    if (list(image.GetSize()) != grid["dimensions"]
            or not np.allclose(image.GetSpacing(), [grid["resolution"][a] for a in "xyz"], rtol=0, atol=1e-6)
            or not np.allclose(image.GetOrigin(), grid["origin"], rtol=0, atol=1e-6)
            or not np.allclose(image.GetDirection(), np.asarray(grid["direction"]).ravel(), rtol=0, atol=1e-9)):
        raise ValueError("MHA grid differs from saved dose grid")
    if not np.allclose(image.GetDirection(), np.eye(3).ravel(), rtol=0, atol=1e-9):
        raise ValueError("Unsupported MHA orientation: axis-aligned identity direction required")
    dose = sitk.GetArrayFromImage(image).transpose(2, 0, 1)
    if not np.isfinite(dose).all() or (dose < 0).any():
        raise ValueError("MHA dose must be finite and nonnegative")
    coords = [image.GetOrigin()[i] + np.arange(image.GetSize()[i]) * image.GetSpacing()[i] - surface[i]
              for i in range(3)]
    provenance = dict(input_files=[str(path), str(meta_path), str(manifest_path)],
        bundle_id=manifest["bundle_id"], normalization=convention, coefficient_units=units,
        dose_units="Gy", exposure_weight=float(weights[0]), weight_units=weight_units,
        field_width_mm=water["field_width_mm"], ssd_mm=water["ssd_mm"],
        surface_center_lps_mm=surface.tolist(), coordinate_mapping="X=LPS X; Y=LPS Z; depth=LPS Y, relative to entrance center",
        scoring_grid=grid, batch_metadata=meta.get("batch_metadata", {}),
        scorer_warnings=meta.get("scorer_warnings", []),
        review_required=bool(meta.get("review_required") or meta.get("scorer_warnings")),
        phase_space_selection=meta.get("phase_space_selection"),
        uncertainty="No MC uncertainty inferred from MHA. Batch variation is conditional on fixed source replay; voxel covariance unavailable.")
    return Grid(coords[0], coords[2], coords[1], dose, exposure=float(weights[0]),
                path=path, normalization=convention, provenance=provenance)


def read_plane(path: Path, depth: float, *, slit: bool = False,
               offset_x: float | None = None, offset_y: float | None = None) -> Plane:
    matrix = np.genfromtxt(path, delimiter=",", encoding="utf-8-sig")
    if matrix.ndim != 2 or min(matrix.shape) < 3:
        raise ValueError(f"not a coordinate dose matrix: {path}")
    x, y, values = matrix[0, 1:], matrix[1:, 0], matrix[1:, 1:]
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError(f"nonfinite map coordinates: {path}")
    xi, yi = np.argsort(x), np.argsort(y)
    x, y, values = x[xi], y[yi], values[np.ix_(yi, xi)]
    if np.any(np.diff(x) <= 0) or np.any(np.diff(y) <= 0):
        raise ValueError(f"duplicate map coordinates: {path}")
    if np.isinf(values).any() or not np.isfinite(values).any():
        raise ValueError(f"invalid map doses: {path}")
    centering = {}
    if slit:
        cx, cy = float((x[0] + x[-1]) / 2), float((y[0] + y[-1]) / 2)
        ox = cx if offset_x is None else offset_x
        oy = cy if offset_y is None else offset_y
        centering = dict(center_x_mm=cx, center_y_mm=cy, offset_x_mm=ox, offset_y_mm=oy,
                         x_mode="automatic" if offset_x is None else "configured",
                         y_mode="automatic" if offset_y is None else "configured")
        x, y = x - ox, y - oy
    return Plane(x, y, values, depth, path, centering)


def read_agl(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    metadata, depths, values = {}, [], []
    started = False
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.reader(stream):
            if not row or not any(cell.strip() for cell in row):
                continue
            if not started:
                if len(row) >= 2 and row[0].strip() == "Depth (mm)" and row[1].strip() == "PDD":
                    started = True
                elif len(row) >= 2:
                    metadata[row[0].strip()] = row[1].strip()
                continue
            if len(row) < 2:
                raise ValueError(f"incomplete AGL data row in {path}")
            depths.append(float(row[0]))
            values.append(float(row[1]))
    x, v = np.asarray(depths), np.asarray(values)
    if not started or len(x) < 2 or not np.isfinite(x).all() or not np.isfinite(v).all() or (v < 0).any():
        raise ValueError(f"invalid AGL PDD: {path}")
    order = np.argsort(x)
    x, v = x[order], v[order]
    if (np.diff(x) <= 0).any():
        raise ValueError(f"duplicate AGL depths: {path}")
    return x, v, metadata
