"""Read static nominal MLCX positions at the planning reference plane.

These readers do not construct physical leaf solids or apply IEC transforms.
"""
from dataclasses import dataclass
from pathlib import Path
import csv
import hashlib
import math


@dataclass(frozen=True)
class LeafPair:
    number: int
    lower_mm: float
    upper_mm: float
    bank_x1_mm: float
    bank_x2_mm: float


@dataclass(frozen=True)
class MLCShape:
    pairs: tuple[LeafPair, ...]
    source: str
    sha256: str
    beam_number: int | None = None
    control_point_index: int | None = None


def _shape(pairs, path, source, **metadata):
    if not pairs or any(not all(math.isfinite(v) for v in (p.lower_mm, p.upper_mm, p.bank_x1_mm, p.bank_x2_mm)) or p.lower_mm >= p.upper_mm or p.bank_x1_mm > p.bank_x2_mm for p in pairs):
        raise ValueError('Invalid leaf boundaries or crossed leaf positions')
    for a, b in zip(pairs, pairs[1:]):
        if not math.isclose(a.upper_mm, b.lower_mm, abs_tol=1e-6):
            raise ValueError('Leaf boundaries must be contiguous and ordered')
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return MLCShape(tuple(pairs), source, h.hexdigest(), **metadata)


def read_csv_shape(path):
    """Read example-format CSV: offsets in mm, bank positions in cm."""
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    required = {'leaf_pair', 'offset_mm', 'field_size_cm', 'bank_x1_cm', 'bank_x2_cm'}
    if not rows or not required.issubset(rows[0]):
        raise ValueError('CSV requires leaf_pair, offset_mm, field_size_cm, bank_x1_cm and bank_x2_cm')
    try:
        parsed = sorted((int(r['leaf_pair']), float(r['offset_mm']), float(r['field_size_cm']), float(r['bank_x1_cm']), float(r['bank_x2_cm'])) for r in rows)
    except (TypeError, ValueError, KeyError) as e:
        raise ValueError('Invalid MLC CSV row') from e
    if [r[0] for r in parsed] != list(range(1, len(parsed) + 1)) or len(parsed) < 2:
        raise ValueError('CSV must contain at least two consecutively numbered leaf pairs starting at 1')
    offsets = [r[1] for r in parsed]
    width = offsets[1] - offsets[0]
    if not math.isfinite(width) or width <= 0 or any(not math.isclose(b-a, width, abs_tol=1e-6) for a,b in zip(offsets, offsets[1:])):
        raise ValueError('CSV leaf offsets must be evenly spaced and increasing')
    pairs = []
    for number, center, field, x1, x2 in parsed:
        if not math.isfinite(field) or not math.isclose(field, x2-x1, abs_tol=1e-6):
            raise ValueError(f'Leaf pair {number}: field size does not match bank positions')
        pairs.append(LeafPair(number, center-width/2, center+width/2, x1*10, x2*10))
    return _shape(pairs, path, 'csv')


def read_rtplan_shape(path, *, beam_number, control_point_index):
    """Read MLCX geometry and inherited static control-point positions."""
    import pydicom
    if type(beam_number) is not int or type(control_point_index) is not int or control_point_index < 0:
        raise ValueError('Beam number and nonnegative control point index must be integers')
    ds = pydicom.dcmread(path)
    beams = [b for b in getattr(ds, 'BeamSequence', []) if int(b.BeamNumber) == beam_number]
    if len(beams) != 1:
        raise ValueError('RTPLAN beam number must identify exactly one beam')
    beam = beams[0]
    devices = [d for d in getattr(beam, 'BeamLimitingDeviceSequence', []) if d.RTBeamLimitingDeviceType == 'MLCX']
    if len(devices) != 1:
        raise ValueError('RTPLAN beam must contain exactly one MLCX device')
    device = devices[0]
    n = int(device.NumberOfLeafJawPairs)
    boundaries = [float(x) for x in getattr(device, 'LeafPositionBoundaries', [])]
    if n < 1 or len(boundaries) != n+1:
        raise ValueError('Invalid MLCX leaf boundaries')
    cps = getattr(beam, 'ControlPointSequence', [])
    if control_point_index >= len(cps):
        raise ValueError('RTPLAN control point index out of range')
    positions = None
    for cp in cps[:control_point_index+1]:
        items = [d for d in getattr(cp, 'BeamLimitingDevicePositionSequence', []) if d.RTBeamLimitingDeviceType == 'MLCX']
        if len(items) > 1:
            raise ValueError('Ambiguous MLCX positions in control point')
        if items:
            positions = [float(x) for x in items[0].LeafJawPositions]
    if positions is None or len(positions) != 2*n:
        raise ValueError('Missing or inconsistent inherited MLCX leaf positions')
    pairs = [LeafPair(i+1, boundaries[i], boundaries[i+1], positions[i], positions[n+i]) for i in range(n)]
    return _shape(pairs, path, 'rtplan', beam_number=beam_number, control_point_index=control_point_index)


def resolve_leaves(c, sad):
    """Rigid extruded leaves; side widths projected at the body midpoint.

    No tongue/groove, side focusing or transmission calibration is implied.
    Symmetric separation enforces a physical transverse gap without changing shapes.
    """
    import numpy as np
    minimum_gap = c.get('minimum_leaf_gap_mm', 0.)
    if isinstance(minimum_gap, bool) or not isinstance(minimum_gap, (int, float)) or not math.isfinite(minimum_gap) or minimum_gap < 0:
        raise ValueError('minimum_leaf_gap_mm must be finite and nonnegative (not Boolean)')
    path = Path(c['shape_file']).expanduser()
    shape = (read_csv_shape(path) if c['shape_source'] == 'csv' else
             read_rtplan_shape(path, beam_number=c['shape_beam_number'], control_point_index=c['shape_control_point_index']))
    d, t, radius = c['source_to_center_mm'], c['thickness_mm'], c['tip_radius_mm']
    zc = c['tip_center_z_offset_mm']
    length = c['leaf_length_mm']
    z = np.unique(np.r_[np.linspace(-t/2, t/2, 65), np.clip(zc,-t/2,t/2)])
    if radius <= max(abs(z-zc)):
        raise ValueError('MLC tip radius does not cover leaf thickness')
    arc = np.sqrt(radius**2-(z-zc)**2)
    leaves=[]
    pair_gaps=[]
    for pair in shape.pairs:
        profiles=[]
        for side, edge in [(-1,pair.bank_x1_mm),(1,pair.bank_x2_mm)]:
            slope=edge/sad
            if c['round_tip_enabled']:
                center=slope*(d+zc)+side*radius*math.sqrt(1+slope*slope)
                inner=center-side*arc
                apex=center-side*radius
            else:
                inner=slope*(d+z); apex=float(inner[len(inner)//2])
            profiles.append((inner,apex))
        # The rounded profiles share their axial curvature center: the minimum
        # transverse gap occurs there (or at the nearest end face). Flat profiles
        # have a linear separation with its minimum at an end face. All these
        # locations are included in z, also used by the generated polygon solids.
        original_gap=float(np.min(profiles[1][0]-profiles[0][0]))
        correction=max(0.,minimum_gap-original_gap)/2
        pair_gaps.append(dict(pair=pair.number, original_gap_mm=original_gap,
                              resolved_gap_mm=original_gap+2*correction,
                              negative_leaf_translation_mm=-correction,
                              positive_leaf_translation_mm=correction))
        for index,side in enumerate((-1,1)):
            inner,apex=profiles[index]; inner=inner+side*correction
            outer=apex+side*(length+correction)
            if np.any(side*(outer-inner)<=0): raise ValueError('Leaf length cannot contain tip')
            polygon=[(float(inner[0]),float(z[0])),(outer,float(z[0])),(outer,float(z[-1])),(float(inner[-1]),float(z[-1])), *[(float(x),float(zz)) for x,zz in zip(inner[-2:0:-1],z[-2:0:-1])]]
            if side<0: polygon.reverse()
            lo,hi=pair.lower_mm*d/sad,pair.upper_mm*d/sad
            if max(abs(lo),abs(hi))>c['outer_height_mm']/2+1e-6:
                raise ValueError('Leaf bank exceeds configured MLC transverse height')
            leaves.append(dict(name=f'MLCLeaf{pair.number:03d}'+('Positive' if side>0 else 'Negative'),
                pair=pair.number,bank=side,requested_position_mm=(pair.bank_x1_mm if side<0 else pair.bank_x2_mm),
                adjustment_mm=side*correction, polygon=polygon, y_center=(lo+hi)/2,y_half=(hi-lo)/2,center_z=d-sad))
    return dict(source=shape.source,sha256=shape.sha256,beam_number=shape.beam_number,
                control_point_index=shape.control_point_index,leaves=leaves,
                minimum_leaf_gap_mm=float(minimum_gap), pair_gaps=pair_gaps,
                adjusted_pair_count=sum(p['positive_leaf_translation_mm']>0 for p in pair_gaps),
                largest_leaf_adjustment_mm=max(p['positive_leaf_translation_mm'] for p in pair_gaps),
                gap_convention='minimum beam-X separation at common axial depth; physical MLC bank mm')


def render_leaves(record, material):
    lines=[]
    for leaf in record['leaves']:
        n=leaf['name']; polygon=leaf['polygon']
        lines.extend([f's:Ge/{n}/Type = "G4ExtrudedSolid"', f's:Ge/{n}/Parent = "TreatmentHead"',
            f's:Ge/{n}/Material = "{material}"',
            f'dv:Ge/{n}/Polygons = {2*len(polygon)} '+ ' '.join(f'{v:.12g}' for p in polygon for v in p)+' mm',
            f'd:Ge/{n}/HLZ = {leaf["y_half"]:.12g} mm',f'dv:Ge/{n}/Off1 = 2 0 0 mm',f'dv:Ge/{n}/Off2 = 2 0 0 mm',
            f'u:Ge/{n}/Scale1 = 1',f'u:Ge/{n}/Scale2 = 1',f'd:Ge/{n}/RotX = -90 deg',
            f'd:Ge/{n}/TransY = {leaf["y_center"]:.12g} mm',f'd:Ge/{n}/TransZ = {leaf["center_z"]:.12g} mm'])
    return '\n'.join(lines)


def leaf_gap_summary(record):
    """Shared text for preparation and geometry reports, from resolved solids."""
    return [f"MLC minimum physical tip gap: {record['minimum_leaf_gap_mm']:g} mm at bank",
            f"Adjusted pairs: {record['adjusted_pair_count']}/{len(record['pair_gaps'])}; largest per-leaf shift: {record['largest_leaf_adjustment_mm']:.6g} mm"]
