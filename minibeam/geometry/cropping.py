"""Explicit CT transport crops; native planning images are never modified."""
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_fill_holes, binary_propagation
from pyRadPlan.ct import CT
from .coordinates import grid_dict, scoring_grid


class CropValidationError(ValueError):
    """An anatomical conflict with a complete numerical audit for previews."""
    def __init__(self, message, audit):
        super().__init__(message)
        self.audit = audit


def resolve_grids(ct, crop=None, spacing=None, cst=None, enforce_protection=True, *, diagnostic_masks=None, **obsolete):
    if obsolete:
        raise ValueError('Obsolete crop options; use enforce_protection (unified ROI/material checks, targets always protected)')
    if type(enforce_protection) is not bool:
        raise ValueError("enforce_ct_crop_protection must be a Boolean")
    audit = dict(protection_policy='unified_target_protected_v1', roi_clipping=[], conflicts=[],
                 material_flagged_voxels=0, enforce_protection=enforce_protection, protection_bypassed=False,
                 protection_counts=dict(clipped_roi=0, target_conflict=0, hu_above_threshold=0, enclosed_low_hu=0, material_union=0))
    size = np.asarray(ct.size, dtype=int)
    lo, hi = np.zeros(3, dtype=int), size.copy()
    if crop is not None:
        if not isinstance(crop, dict) or set(crop) != set('xyz'):
            raise ValueError('ct_crop_voxels requires x/y/z zero-based, stop-exclusive ranges')
        for i, axis in enumerate('xyz'):
            pair = crop[axis]
            if not isinstance(pair, (tuple, list)) or len(pair) != 2 or any(
                    isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) for v in pair):
                raise ValueError('Crop bounds must be integer pairs, not Boolean values')
            lo[i], hi[i] = pair
        if np.any(lo < 0) or np.any(hi > size) or np.any(lo >= hi):
            raise ValueError('Crop bounds must be nonempty and within original CT')
    applied = bool(np.any(lo) or np.any(hi != size))
    if applied:
        hu = sitk.GetArrayFromImage(ct.cube_hu)
        if not np.isfinite(hu).all():
            raise ValueError("Cannot classify external air in a nonfinite CT")
        # Conservative external-air rule: low-HU, boundary-connected, outside
        # filled tissue/support silhouettes in every anatomical plane.
        solid = hu > -950
        protected = solid.copy()
        for axis in range(3):
            view = np.moveaxis(solid, axis, 0)
            filled = np.stack([binary_fill_holes(s) for s in view])
            protected |= np.moveaxis(filled, 0, axis)
        air = ~protected
        seeds = np.zeros_like(air)
        for axis in range(3):
            for edge in (0, -1):
                sl = [slice(None)] * 3; sl[axis] = edge
                seeds[tuple(sl)] = air[tuple(sl)]
        external = binary_propagation(seeds, mask=air)
        excluded = np.ones_like(air)
        excluded[lo[2]:hi[2], lo[1]:hi[1], lo[0]:hi[0]] = False
        target_conflict = np.zeros_like(excluded)
        roi_conflict = np.zeros_like(excluded)
        if cst is not None:
            for voi in cst.vois:
                mask = sitk.GetArrayViewFromImage(voi.mask) > 0
                total = int(mask.sum())
                removed = int(np.count_nonzero(excluded & mask))
                is_target = voi.voi_type == 'TARGET'
                blocking = bool(removed and (enforce_protection or is_target))
                roi_conflict |= excluded & mask
                if is_target:
                    target_conflict |= excluded & mask
                record = dict(name=voi.name, total_voxels=total, removed_voxels=removed,
                              removed_fraction=removed/total if total else 0., is_target=is_target, blocking=blocking,
                              disposition='blocking' if blocking else ('bypassed' if removed else 'retained'),
                              clipping_extent='complete' if removed and removed == total else ('partial' if removed else 'none'),
                              complete_dose_coverage=removed == 0)
                audit['roi_clipping'].append(record)
                if blocking:
                    indices = np.argwhere(mask)[:, ::-1]
                    bounds = {axis: (int(start), int(stop)) for axis, start, stop in
                              zip('xyz', indices.min(axis=0), indices.max(axis=0) + 1)}
                    audit['conflicts'].append(
                        f"Crop removes structure voxels: {voi.name} ({removed:,} voxels). "
                        f"This structure requires ranges containing {bounds} (zero-based, stop-exclusive).")
        flagged_hu = excluded & solid
        flagged_air = excluded & ~solid & ~external
        unauthorized = int(flagged_hu.sum() + flagged_air.sum())
        audit['material_flagged_voxels'] = unauthorized
        audit['protection_bypassed'] = bool(not enforce_protection and (unauthorized or any(r['disposition']=='bypassed' for r in audit['roi_clipping'])))
        audit['protection_counts'] = dict(clipped_roi=int(roi_conflict.sum()), target_conflict=int(target_conflict.sum()),
            hu_above_threshold=int(flagged_hu.sum()), enclosed_low_hu=int(flagged_air.sum()),
            material_union=unauthorized)
        if diagnostic_masks is not None:
            diagnostic_masks.update(excluded=excluded, clipped_roi=roi_conflict, target_conflict=target_conflict,
                hu_above_threshold=flagged_hu, enclosed_low_hu=flagged_air)
        if unauthorized and enforce_protection:
            audit['conflicts'].append(f'Crop removes suspected anatomy/support or non-external air in excluded region ({unauthorized:,} voxels)')
        if audit['conflicts']:
            raise CropValidationError('\n'.join(audit['conflicts']) +
                '\nExpand the crop or set CT_CROP_VOXELS = None to retain the full CT.', audit)
        image = sitk.RegionOfInterest(ct.cube_hu, (hi-lo).tolist(), lo.tolist())
        transport = CT(cube_hu=image)
    else:
        transport = ct
    dose = scoring_grid(transport, spacing)
    metadata = dict(applied=applied, indexing='XYZ, zero-based, stop-exclusive',
        retained_ranges={a: [int(l), int(h)] for a,l,h in zip('xyz',lo,hi)},
        transport_grid=grid_dict(transport.grid),
        requested_dose_spacing_mm=list(spacing) if spacing is not None else [],
        actual_dose_spacing_mm=dose.resolution_vector.tolist(),
        safety_rule='selected targets always protected; enforce_protection controls non-target ROI and material checks',
        **audit)
    return transport, dose, metadata


def coverage_image(scored_image, reference):
    """Physical scoring extent sampled at reference voxel centers (0/1)."""
    ones = sitk.Image(scored_image.GetSize(), sitk.sitkUInt8) + 1
    ones.CopyInformation(scored_image)
    return sitk.Resample(ones, reference, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)


def map_structure_mask(mask, dose_grid):
    """Return a nearest-neighbor dose-grid mask and original-mask coverage."""
    reference = sitk.Image(list(map(int, dose_grid.dimensions)), sitk.sitkUInt8)
    reference.SetOrigin(tuple(dose_grid.origin)); reference.SetSpacing(tuple(dose_grid.resolution_vector))
    reference.SetDirection(tuple(dose_grid.direction_vector))
    mapped = sitk.Resample(mask, reference, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    inside = sitk.GetArrayFromImage(coverage_image(reference, mask)) > 0
    original = sitk.GetArrayFromImage(mask) > 0
    total = int(original.sum()); covered = int((original & inside).sum())
    return mapped, dict(original_voxels=total, covered_original_voxels=covered,
                        complete=total == covered)


def validate_saved_grids(manifest):
    """Check redundant crop/grid records without reopening patient inputs."""
    crop = manifest.get('crop_metadata')
    if crop is None:
        return
    policy = crop.get('protection_policy')
    if policy not in (None, 'unified_target_protected_v1'):
        raise ValueError('Unsupported saved crop protection policy')
    if policy and 'enforce_protection' not in crop:
        raise ValueError('Missing saved unified crop enforcement flag')
    flag = 'enforce_protection' if policy else 'enforce_material_check'
    if type(crop.get(flag, True)) is not bool:
        raise ValueError('Invalid saved crop material enforcement flag')
    from pyRadPlan.core import Grid
    original = Grid.model_validate(manifest['ct_grid'])
    transport = Grid.model_validate(manifest['transport_grid'])
    dose = Grid.model_validate(manifest['dose_grid'])
    if crop['transport_grid'] != manifest['transport_grid'] or crop['indexing'] != 'XYZ, zero-based, stop-exclusive':
        raise ValueError('Inconsistent saved crop metadata')
    bounds = np.asarray([crop['retained_ranges'][a] for a in 'xyz'])
    if bounds.shape != (3,2) or bounds.dtype.kind not in 'iu' or np.any(bounds[:,0]<0) or np.any(bounds[:,1]>original.dimensions) or np.any(bounds[:,1]<=bounds[:,0]):
        raise ValueError('Invalid saved crop bounds')
    lo,hi=bounds.T
    if not np.array_equal(transport.dimensions,hi-lo) or not np.allclose(transport.origin,original.origin+original.resolution_vector*lo):
        raise ValueError('Saved transport crop placement mismatch')
    if not np.allclose(transport.resolution_vector,original.resolution_vector) or not np.allclose(transport.direction,original.direction):
        raise ValueError('Saved transport crop changed CT spacing/orientation')
    expected_applied=bool(np.any(lo) or np.any(hi!=original.dimensions))
    if crop['applied'] != expected_applied:
        raise ValueError('Saved crop applied flag mismatch')
    for grid in (transport,dose):
        if not np.allclose(grid.direction,np.eye(3)):
            raise ValueError('Cropped transport requires axial identity direction')
    if not np.allclose(dose.origin-dose.resolution_vector/2,transport.origin-transport.resolution_vector/2) or not np.allclose(np.array(dose.dimensions)*dose.resolution_vector,np.array(transport.dimensions)*transport.resolution_vector):
        raise ValueError('Saved dose boundaries differ from transport crop')
    if not np.allclose(crop['actual_dose_spacing_mm'],dose.resolution_vector):
        raise ValueError('Saved dose spacing metadata mismatch')
