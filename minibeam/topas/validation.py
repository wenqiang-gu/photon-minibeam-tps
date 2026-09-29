"""Match native dose requests to saved geometry without source preparation."""
import numpy as np
from ..geometry.coordinates import grid_dict, scoring_grid


def validate_request(manifest, ct, stf, dose_spacing_mm=None, ct_crop_voxels=None, cst=None, enforce_protection=True):
    from ..geometry.cropping import resolve_grids
    saved_crop = manifest.get('crop_metadata', {})
    if saved_crop.get('protection_policy') == 'unified_target_protected_v1':
        if enforce_protection != saved_crop['enforce_protection']:
            raise ValueError('Saved crop protection differs from native dose request')
        transport, dose_grid, _ = resolve_grids(ct, ct_crop_voxels, dose_spacing_mm, cst, enforce_protection)
    else:
        # Legacy collection consumes its saved policy; never reinterpret its
        # old material-only bypass as unified permission to clip ROIs.
        from pyRadPlan.core import Grid
        transport_grid = manifest.get('transport_grid', manifest['ct_grid'])
        if ct_crop_voxels is not None:
            if {a:list(ct_crop_voxels[a]) for a in 'xyz'} != saved_crop.get('retained_ranges'):
                raise ValueError('Saved legacy crop differs from native dose request')
        elif saved_crop.get('applied'):
            raise ValueError('Saved legacy crop differs from native dose request')
        from types import SimpleNamespace
        transport = SimpleNamespace(grid=Grid.model_validate(transport_grid))
        # Grid comparison only; source inputs and historical safety decisions
        # are not recomputed when consuming completed legacy results.
        transport.size = transport.grid.dimensions
        transport.origin = transport.grid.origin
        transport.num_of_ct_scen = 1
        transport.direction = transport.grid.direction_vector
        dose_grid = scoring_grid(transport, dose_spacing_mm)
    if grid_dict(transport.grid) != manifest.get('transport_grid', manifest['ct_grid']):
        raise ValueError('Saved transport crop differs from native dose request')
    if grid_dict(ct.grid) != manifest['ct_grid'] or grid_dict(dose_grid) != manifest['dose_grid']:
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
