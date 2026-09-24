"""Match native dose requests to saved geometry without source preparation."""
import numpy as np
from ..geometry.coordinates import grid_dict, scoring_grid


def validate_request(manifest, ct, stf, dose_spacing_mm=None):
    if grid_dict(ct.grid) != manifest['ct_grid'] or grid_dict(scoring_grid(ct, dose_spacing_mm)) != manifest['dose_grid']:
        raise ValueError('Saved bundle grid differs from native dose request')
    if len(stf.beams) != len(manifest['planning']['beams']):
        raise ValueError('Saved bundle beam count differs from native dose request')
    from ..steering import has_geometry, validate_minibeam_stf
    if has_geometry(stf):
        stf = validate_minibeam_stf(stf)
    from .manifest import member_jobs
    jobs = iter(member_jobs(manifest))
    for bi, (beam, record) in enumerate(zip(stf.beams, manifest['planning']['beams']), 1):
        parameters = record['parameters']
        for key in ('gantry_angle','couch_angle','sad','bixel_width','iso_center','source_point'):
            if not np.allclose(getattr(beam,key), parameters[key]):
                raise ValueError(f'Saved beam {bi} {key} differs from native dose request')
        if hasattr(beam,'geometry'):
            if beam.geometry.model_dump(mode='json') != parameters.get('geometry'):
                raise ValueError('Attached geometry differs from saved bundle')
        for ri, ray in enumerate(beam.rays,1):
            for li, _ in enumerate(ray.beamlets,1):
                job = next(jobs, None)
                if job is None or (job['beam_index'],job['ray_index'],job['beamlet_index']) != (bi,ri,li):
                    raise ValueError('Saved beamlet associations differ from native dose request')
                source = job['source']
                if 'aim_lps_mm' in source and not np.allclose(source['aim_lps_mm'], beam.iso_center+ray.ray_pos):
                    raise ValueError('Saved ray aiming point differs from native dose request')
                if 'selection' in source:
                    from ..geometry.spatial import beam_basis
                    position = np.asarray(ray.ray_pos) @ beam_basis(beam)
                    if not np.allclose(source['selection']['center_xy_mm'], position[:2]):
                        raise ValueError('Saved phase-space selection differs from native dose request')
    if next(jobs, None) is not None:
        raise ValueError('Saved bundle has additional beamlets')
