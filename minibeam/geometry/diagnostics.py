"""Read-only geometry investigation; never crop CT or relax overlap checks."""
from pathlib import Path
import json
import tempfile
import numpy as np
import SimpleITK as sitk
from scipy.spatial import ConvexHull
from scipy.optimize import linprog
from .spatial import corners, beam_basis, placement_rotation
from .coordinates import patient_center
from .mlc_jaws import _rounded_mlc_polygon


def box_vertices(center, half):
    return corners({'center': center, 'half_size': half, 'rotation': np.eye(3)})


def extrude(polygon, axis, half):
    """Extrude an X/Z or Y/Z polygon along the other transverse axis."""
    result = []
    for side in (-half, half):
        for transverse, z in polygon:
            result.append([transverse, side, z] if axis == 0 else [side, transverse, z])
    return np.asarray(result)


def material_solids(head, beam):
    """Convex polyhedra matching the current bank, frame and blade definitions.

    Rounded banks use the same 64-segment polygon as the TOPAS writer.
    Frame pieces partition the brass outside its daughter air cavity.
    """
    g, ap, _, _ = head.resolved(beam)
    solids = []
    for axis, stage, name in [(0, g.mlc, 'MLC'), (1, g.lower_jaw, 'LowerJaw')]:
        if stage is None:
            continue
        outer = (g.field_aperture_outer_width if axis == 0 else g.field_aperture_outer_height)/2
        half = (g.field_aperture_outer_height if axis == 0 else g.field_aperture_outer_width)/2
        for positive in (False, True):
            sign = 1 if positive else -1
            if axis == 0 and g.mlc_round_tip_enabled:
                polygon = np.asarray(_rounded_mlc_polygon(g, positive)) + [0, stage.center_z]
            else:
                polygon = [[sign*stage.entrance_opening/2, stage.entrance_z],
                           [sign*outer, stage.entrance_z], [sign*outer, stage.exit_z],
                           [sign*stage.exit_opening/2, stage.exit_z]]
            solids.append((name + ('Positive' if positive else 'Negative'), extrude(polygon, axis, half)))
    if ap is not None:
        if head.config_for(beam).data['aperture']['type'] != 'slits':
            raise ValueError('Detailed diagnostics currently support the slit aperture only')
        x, y, z = ap.collimator_width/2, ap.collimator_height/2, ap.collimator_thickness/2
        ax, ay = ap.collimator_air_width/2, ap.collimator_air_height/2
        rotation = placement_rotation(ap.collimator_rotation_x, ap.collimator_rotation_y)
        local = []
        for sign, label in [(-1, 'Negative'), (1, 'Positive')]:
            local.append(('FrameX'+label, box_vertices([sign*(x+ax)/2, 0, 0], [(x-ax)/2,y,z])))
            local.append(('FrameY'+label, box_vertices([0, sign*(y+ay)/2, 0], [ax,(y-ay)/2,z])))
        for blade in ap.collimator_blades:
            w = blade.projected_width_x/2
            polygon = [[blade.entrance_center_x-w,-z], [blade.entrance_center_x+w,-z],
                       [blade.exit_center_x+w,z], [blade.exit_center_x-w,z]]
            local.append((f'Blade_{blade.number:02d}', extrude(polygon, 0, ay)))
        solids += [(name, points@rotation.T + [ap.lateral_shift_mm,0,ap.collimator_center_z]) for name, points in local]
    return solids


def intersection(vertices, lower, upper):
    """Find a point strictly inside both a convex solid and an axis-aligned box.

    The optimized radius is the largest inscribed ball of their intersection,
    not a separation distance or overlap depth. Touching faces are not overlaps.
    """
    equations = ConvexHull(vertices).equations
    normals = np.vstack([equations[:,:3], np.eye(3), -np.eye(3)])
    limits = np.r_[-equations[:,3], upper, -np.asarray(lower)]
    answer = linprog([0,0,0,-1], A_ub=np.column_stack([normals, np.linalg.norm(normals,axis=1)]),
                     b_ub=limits, bounds=[(None,None)]*3+[(0,None)], method='highs')
    if answer.status == 2:
        return None
    if not answer.success:
        raise RuntimeError(f'Collision solver failed: {answer.message}')
    if answer.x[3] <= 1e-6:
        return None
    return {'witness_lps_mm': answer.x[:3].tolist(), 'intersection_inradius_mm': float(answer.x[3])}


def beam_diagnostics(ct, beam, head, index):
    half = np.asarray(ct.size)*ct.grid.resolution_vector/2
    center = patient_center(ct)
    basis = beam_basis(beam)
    _, _, envelopes, _ = head.resolved(beam)
    records = []
    for envelope in envelopes:
        points = corners(envelope)@basis.T + beam.iso_center
        hit = intersection(points, center-half, center+half)
        records.append({'component': envelope['name'], 'intersects': hit is not None,
                        'bounds_lps_mm': [points.min(axis=0).tolist(), points.max(axis=0).tolist()],
                        **(hit or {})})
    solids = []
    for name, vertices in material_solids(head, beam):
        points = vertices@basis.T + beam.iso_center
        hit = intersection(points, center-half, center+half)
        solids.append({'component': name, 'intersects': hit is not None,
                       'bounds_lps_mm': [points.min(axis=0).tolist(), points.max(axis=0).tolist()], **(hit or {})})
    return {'beam_index': index, 'gantry_angle_deg': float(beam.gantry_angle),
            'couch_angle_deg': float(beam.couch_angle), 'sad_mm': float(beam.sad),
            'iso_center_lps_mm': np.asarray(beam.iso_center).tolist(),
            'envelopes': records, 'material_solids': solids}


def mask_bounds(mask):
    coordinates = np.array(np.where(mask))
    if coordinates.size == 0:
        return None
    return [coordinates.min(axis=1)[::-1].tolist(), coordinates.max(axis=1)[::-1].tolist()]


def crop_investigation(ct, cst, stf, head):
    """A necessary crop bound retaining every ROI and every HU > -950 voxel.

    This deliberately does not classify low-HU voxels as safe to delete. If even
    this lower bound collides, an air-only rectangular crop cannot solve it
    under this retention rule. No crop is applied or proposed automatically.
    """
    hu = sitk.GetArrayViewFromImage(ct.cube_hu)
    retain = hu > -950
    roi_records = []
    for voi in cst.vois:
        mask = sitk.GetArrayViewFromImage(voi.mask) > 0
        retain |= mask
        roi_records.append({'name': voi.name, 'bounds_xyz_inclusive_0based': mask_bounds(mask)})
    bounds = mask_bounds(retain)
    result = {'applied': False, 'retention_rule': 'all ROI voxels and HU > -950; necessary bound only, not an approved crop',
              'roi_count': len(roi_records), 'rois': roi_records, 'required_bounds_xyz_inclusive_0based': bounds}
    if bounds is not None:
        lo, hi = np.asarray(bounds)
        spacing = ct.grid.resolution_vector
        lower = np.asarray(ct.origin)+(lo-.5)*spacing
        upper = np.asarray(ct.origin)+(hi+.5)*spacing
        collisions = []
        for i, beam in enumerate(stf.beams,1):
            basis = beam_basis(beam)
            for name, vertices in material_solids(head, beam):
                hit = intersection(vertices@basis.T + beam.iso_center, lower, upper)
                if hit:
                    collisions.append({'beam_index': i, 'component': name, **hit})
        result.update(bounds_lps_mm=[lower.tolist(),upper.tolist()], material_collisions=collisions,
                      clears_material_solids=not collisions)
    return result


def write_collision_diagnostics(ct, cst, stf, head, run_dir, *, patient_report=False):
    """Create a separate diagnostic directory; never modify a run bundle."""
    from .report import write_geometry_report
    root = Path(run_dir).resolve()
    root.parent.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix=root.name+'-diagnostics-', dir=root.parent))
    records = [beam_diagnostics(ct, beam, head, i) for i,beam in enumerate(stf.beams,1)]
    investigation = crop_investigation(ct,cst,stf,head)
    from ..steering import beam_geometry_records
    data = {'status': 'blocked', 'beam_geometry': beam_geometry_records(stf),
            'configuration_sha256': [head.config_for(b).sha256 for b in stf.beams],
            'coordinate_system': 'DICOM LPS mm', 'beams': records, 'crop_investigation': investigation,
            'notes': ['Intersections are with the CT transport box, not necessarily patient tissue.',
                      'The slit enclosing box is itself a TOPAS volume, including its air cavity.',
                      'Witness points lie inside intersecting volumes; radii are not penetration depths.',
                      'No CT, ROI, steering, or dose-grid changes were made.']}
    for i, beam in enumerate(stf.beams, 1):
        (output/f'geometry-beam-{i:03d}.toml').write_text(head.config_for(beam).text)
    (output/'collisions.json').write_text(json.dumps(data,indent=2)+'\n')
    lines = ['Patient geometry preparation blocked', '', *data['notes'], '']
    for beam in records:
        hits = [x for x in beam['material_solids'] if x['intersects']]
        lines.append(f"Beam {beam['beam_index']}: gantry {beam['gantry_angle_deg']:g}, couch {beam['couch_angle_deg']:g} deg")
        for hit in hits:
            lines.append(f"  {hit['component']}: intersection witness LPS mm {np.round(hit['witness_lps_mm'],3).tolist()}")
        if not hits:
            lines.append('  No material collision found; see enclosing-volume intersections in JSON.')
    lines += ['', f"ROIs retained in crop investigation: {investigation['roi_count']}",
              f"Necessary crop bound clears material solids: {investigation.get('clears_material_solids')}",
              'The crop bound is not an approved crop. No DICOM was modified.']
    (output/'collisions.txt').write_text('\n'.join(lines)+'\n')
    if patient_report:
        from .patient_report import write_patient_geometry_report
        write_patient_geometry_report(ct,cst,stf,head,output/'geometry.pdf',
            target_names=tuple(v.name for v in cst.vois if v.voi_type == 'TARGET'),diagnostics=records)
    else:
        write_geometry_report(ct,stf,head,output/'geometry.pdf',diagnostics=records)
    return output
