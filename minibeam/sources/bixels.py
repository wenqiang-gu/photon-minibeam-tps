"""Disjoint native square bixels on a beam's common isocenter plane."""
import numpy as np
from ..geometry.spatial import beam_basis


def square(beam, ray):
    local = np.asarray(ray.ray_pos) @ beam_basis(beam)
    if not np.isfinite(local).all() or abs(local[2]) > 1e-6:
        raise ValueError('Bixel centers must lie on the native isocenter plane')
    width, sad = float(beam.bixel_width), float(beam.sad)
    if not np.isfinite([width, sad]).all() or min(width, sad) <= 0:
        raise ValueError('Bixel width and SAD must be positive')
    if not np.isclose(np.linalg.norm(beam.source_point), sad, rtol=0, atol=1e-6):
        raise ValueError('Square source requires source-to-isocenter distance equal to SAD')
    center = np.round(local[:2],9)
    center[center == 0] = 0.0  # canonicalize negative zero for portable asset sharing
    return dict(sad_mm=round(sad,9), center_xy_mm=center.tolist(), width_mm=round(width,9))


def contains(selection, x, y):
    center = selection['center_xy_mm']; half = selection['width_mm']/2
    return (x >= center[0]-half) & (x < center[0]+half) & (y >= center[1]-half) & (y < center[1]+half)


def groups(stf, execution='separate'):
    if execution not in {'separate','combined'}:
        raise ValueError('beamlet_execution must be separate or combined')
    output = []; index = 0
    for bi, beam in enumerate(stf.beams,1):
        members = []
        for ri, ray in enumerate(beam.rays,1):
            if len(ray.beamlets) != 1 or getattr(ray.beamlets[0], 'is_field_based', False):
                raise ValueError('Square sources require exactly one non-field beamlet per ray')
            selection = square(beam, ray)
            for other in members:
                previous = other['selection']
                delta = np.abs(np.asarray(previous['center_xy_mm'])-selection['center_xy_mm'])
                if np.all(delta < (previous['width_mm']+selection['width_mm'])/2):
                    raise ValueError('Overlapping bixel selection regions')
            index += 1
            members.append(dict(bixel_index=index,beam_index=bi,ray_index=ri,beamlet_index=1,selection=selection))
        if not members:
            raise ValueError('Beam has no selected bixels')
        output.extend([members] if execution == 'combined' else [[m] for m in members])
    return output
