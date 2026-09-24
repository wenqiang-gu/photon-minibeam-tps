"""Patient-centered TOPAS coordinates and explicit source frame rotations."""
import warnings
import numpy as np
from scipy.spatial.transform import Rotation
from pyRadPlan.core import Grid


def grid_dict(grid):
    return {"dimensions": list(map(int, grid.dimensions)),
            "resolution": {k: float(v) for k, v in grid.resolution.items()},
            "origin": np.asarray(grid.origin).tolist(),
            "direction": np.asarray(grid.direction).tolist()}


def patient_center(ct):
    if ct.num_of_ct_scen != 1 or not np.allclose(ct.direction, np.eye(3).ravel()):
        raise ValueError("Only a single axial LPS CT is supported")
    return np.asarray(ct.origin) + (np.asarray(ct.size) - 1) * ct.grid.resolution_vector / 2


def scoring_grid(ct, requested_spacing=None):
    if requested_spacing is None:
        return ct.grid
    spacing = np.asarray(requested_spacing, dtype=float)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError("dose_spacing_mm must contain three positive finite values")
    extent = np.asarray(ct.size) * ct.grid.resolution_vector
    bins = np.maximum(1, np.ceil(extent / spacing).astype(int))
    actual = extent / bins
    origin = patient_center(ct) - extent / 2 + actual / 2
    return Grid(dimensions=tuple(map(int, bins)), resolution=dict(zip("xyz", actual)), origin=origin)


def source_frame(direction, preferred_x):
    z = np.asarray(direction, dtype=float)
    z = z / np.linalg.norm(z)
    x = np.asarray(preferred_x, dtype=float)
    x = x - np.dot(x, z) * z
    if np.linalg.norm(x) < 1e-10:
        raise ValueError("Source x axis is parallel to the beam direction")
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


def topas_angles(local_to_world):
    # TOPAS calls rotateX, rotateY, rotateZ and applies the INVERSE matrix
    # to source positions/directions (TsVGenerator::TransformPrimaryForComponent).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # Euler gimbal lock has a valid canonical solution
        return Rotation.from_matrix(np.asarray(local_to_world).T).as_euler("xyz", degrees=True)


