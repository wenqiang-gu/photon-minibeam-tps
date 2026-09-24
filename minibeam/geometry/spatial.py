"""Bounds and conservative collision checks in physical TOPAS coordinates."""
from itertools import product
import numpy as np
from scipy.spatial.transform import Rotation
from pyRadPlan.geometry import lps
from .coordinates import source_frame


def placement_rotation(x=0.,y=0.,z=0.):
    # TOPAS placement angles are passive; invert its rotateX/Y/Z matrix.
    return Rotation.from_euler('xyz',[x,y,z],degrees=True).as_matrix().T


def beam_basis(beam):
    return source_frame(-np.asarray(beam.source_point),
        lps.get_beam_rotation_matrix(beam.gantry_angle,beam.couch_angle)[:,0])


def corners(envelope):
    points=np.array(list(product((-1,1),repeat=3)))*envelope['half_size']
    return points@np.asarray(envelope['rotation']).T+envelope['center']


def overlaps_box(center, rotation, half_size, ct_half_size):
    """Separating-axis test: rotated device envelope versus centered CT box."""
    rotation=np.asarray(rotation); half_size=np.asarray(half_size)
    axes=[*np.eye(3),*rotation.T]
    axes.extend(np.cross(a,b) for a in np.eye(3) for b in rotation.T)
    for axis in axes:
        if np.linalg.norm(axis)<1e-10:continue
        gap=abs(np.dot(center,axis))
        radius=np.dot(ct_half_size,np.abs(axis))+np.dot(half_size,np.abs(rotation.T@axis))
        if gap >= radius-1e-7:return False
    return True
