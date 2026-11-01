"""Native, convex-piece approximation of the Agility sculpted diaphragm.

The outline is a drawing at a FIXED reference distance, not a treatment field.
Its thin-region thickness and rear endpoint are modeling assumptions.
"""
import math
import numpy as np


def validate_outline(c):
    points = c['outline_points_mm']
    if not isinstance(points, (tuple, list)) or len(points) < 4:
        raise ValueError('Jaw outline requires at least four [distance, half-width] points')
    for point in points:
        if not isinstance(point, (tuple, list)) or len(point) != 2 or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in point):
            raise ValueError('Jaw outline points must be finite numeric pairs, not Booleans')
    a = np.asarray(points)
    if a[0, 0] != 0 or np.any(np.diff(a[:, 0]) <= 0) or np.any(a[:, 1] <= 0):
        raise ValueError('Jaw outline must start at zero with increasing distances and positive half-widths')
    if np.any(np.diff(a[:, 1]) > 0) or a[0, 1] != a[1, 1] or a[-1, 1] != a[-2, 1]:
        raise ValueError('Jaw outline requires non-increasing half-widths, a constant front band and rear spine')
    if c['thin_region_thickness_mm'] > c['thickness_mm']:
        raise ValueError('Jaw thin-region thickness cannot exceed total thickness')


def _piece(name, polygon, plane, half, center_z):
    polygon = np.asarray(polygon, dtype=float)
    if plane == 'yz':
        vertices = [[x, y, z + center_z] for x in (-half, half) for y, z in polygon]
    else:
        vertices = [[x, y, z + center_z] for z in (-half, half) for x, y in polygon]
    return dict(name=name, plane=plane, polygon=polygon.tolist(), extrusion_half_mm=float(half),
                center_z_mm=float(center_z), vertices_beam_mm=vertices)


def resolve_y_jaws(g, c, field, sad):
    """Partition each jaw into an upstream base and downstream convex pieces.

    All axial cuts interpolate the SAME 64-segment tip polygon. No overlapping
    unions or internal air daughters are needed. Polygon Z is body-relative;
    saved vertices are in the fixed beam frame (isocenter Z=0).
    """
    validate_outline(c)
    stage, tip = g.lower_jaw, g.jaw_round_tip
    scale = stage.source_to_center / c['outline_reference_distance_mm']
    outline = np.asarray(c['outline_points_mm'], dtype=float) * scale
    half_width = outline[0, 1]
    nominal_edge = field['height_mm'] / 2 * stage.source_to_center / sad
    rear = nominal_edge + outline[-1, 0]
    front_back = nominal_edge + outline[1, 0]
    t, thin = stage.thickness, c['thin_region_thickness_mm']
    levels = np.linspace(-t/2, t/2, 65)
    if tip is not None:
        inner = tip.circle_center_y - np.sqrt(tip.radius**2 - (levels + stage.center_z - tip.circle_center_z)**2)
    else:
        inner = np.linspace(stage.entrance_opening/2, stage.exit_opening/2, 65)
    if max(inner) >= front_back - 1e-8:
        raise ValueError('Y-shaped jaw front band does not contain the rounded/focused front; enlarge its outline distance')
    if rear - max(inner) <= field['side_margin_mm']:
        raise ValueError('Y-shaped jaw rear endpoint violates the physical side margin')
    split = -t/2 + thin
    # Insertion preserves the original piecewise-linear surface exactly.
    levels_cut = np.unique(np.r_[levels, split])
    inner_cut = np.interp(levels_cut, levels, inner)

    def front_polygon(lo, hi, back):
        take = (levels_cut >= lo) & (levels_cut <= hi)
        z, y = levels_cut[take], inner_cut[take]
        return [(y[0], lo), (back, lo), (back, hi), (y[-1], hi),
                *list(zip(y[1:-1], z[1:-1]))[::-1]]

    pieces = []
    for sign, label in ((-1, 'Negative'), (1, 'Positive')):
        def add_yz(label_part, polygon):
            polygon = [(sign*y, z) for y, z in polygon]
            if sign < 0: polygon.reverse()
            pieces.append(_piece('LowerJaw'+label+label_part, polygon, 'yz', half_width, stage.center_z))
        add_yz('Base', front_polygon(-t/2, split, rear))
        if thin < t:
            add_yz('Front', front_polygon(split, t/2, front_back))
            for i in range(1, len(outline)-1):
                (d0, w0), (d1, w1) = outline[i:i+2]
                y0, y1 = nominal_edge+d0, nominal_edge+d1
                polygon = [(-w0, sign*y0), (w0, sign*y0), (w1, sign*y1), (-w1, sign*y1)]
                if sign < 0: polygon.reverse()
                pieces.append(_piece(f'LowerJaw{label}Rear_{i:02d}', polygon, 'xy', (t-thin)/2,
                                     stage.center_z + thin/2))
    vertices = np.concatenate([p['vertices_beam_mm'] for p in pieces])
    g.jaw_outer_width = float(2*half_width)
    g.jaw_outer_height = float(2*rear)  # resolved pair envelope, not an input
    return dict(model='y_shaped', outline_reference_distance_mm=c['outline_reference_distance_mm'],
                outline_points_mm=c['outline_points_mm'], physical_outline_points_mm=outline.tolist(),
                physical_width_mm=float(2*half_width), nominal_physical_length_mm=float(outline[-1, 0]),
                nominal_edge_beam_y_mm=float(nominal_edge), rear_distance_from_axis_mm=float(rear),
                thin_region_thickness_mm=thin, total_thickness_mm=t,
                thin_axial_source_mm=[stage.source_to_center-t/2, stage.source_to_center-t/2+thin],
                total_axial_source_mm=[stage.source_to_center-t/2, stage.source_to_center+t/2],
                bounds_beam_mm=[vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()],
                approximation='Page 70 outline; assumed thin thickness and endpoint; fixed body-plane magnification; no leaf-controller constraints',
                pieces=pieces)


def render_y_jaws(shape, material):
    lines = ['# Y-shaped jaw: fixed drawing reference, physical upstream thin base']
    for p in shape['pieces']:
        name = p['name']
        rot = (-90, -90, 0) if p['plane'] == 'yz' else (0, 0, 0)
        coords = ' '.join(f'{v:.12f}' for xy in p['polygon'] for v in xy)
        lines += [f's:Ge/{name}/Type = "G4ExtrudedSolid"',
                  f's:Ge/{name}/Parent = "TreatmentHead"', f's:Ge/{name}/Material = "{material}"',
                  f'dv:Ge/{name}/Polygons = {2*len(p["polygon"])} {coords} mm',
                  f'd:Ge/{name}/HLZ = {p["extrusion_half_mm"]:.12f} mm',
                  f'dv:Ge/{name}/Off1 = 2 0 0 mm', f'u:Ge/{name}/Scale1 = 1',
                  f'dv:Ge/{name}/Off2 = 2 0 0 mm', f'u:Ge/{name}/Scale2 = 1',
                  *[f'd:Ge/{name}/Rot{axis} = {v} deg' for axis, v in zip('XYZ', rot)],
                  f'd:Ge/{name}/TransX = 0 mm', f'd:Ge/{name}/TransY = 0 mm',
                  f'd:Ge/{name}/TransZ = {p["center_z_mm"]:.12f} mm']
    return '\n'.join(lines)+'\n'


def jaw_shape_summary(shape):
    return [f"Y-shaped jaw (approximation): width {shape['physical_width_mm']:.3f} mm; nominal length per jaw {shape['nominal_physical_length_mm']:.3f} mm",
            f"Fixed outline reference: {shape['outline_reference_distance_mm']:g} mm; independent of SAD/SSD",
            f"Physical thin/total thickness: {shape['thin_region_thickness_mm']:g}/{shape['total_thickness_mm']:g} mm",
            'Thin base source distances: ' + ' / '.join(f'{v:g}' for v in shape['thin_axial_source_mm']) + ' mm',
            'Front band spans full width; MLC controller constraints not implemented.']


def write_jaw_detail_page(pdf, shape, sad, beam_index):
    """Beam-local detail from the SAME convex vertices as transport/collisions."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from scipy.spatial import ConvexHull
    fig, axes = plt.subplots(1, 3, figsize=(13, 7), layout='constrained')
    fig.suptitle(f'Y-shaped diaphragm approximation | Beam {beam_index}\n'
                 'Gray: upstream thin base; amber: additional downstream material')
    # Section through the rear spine of the positive jaw.
    section_y = shape['nominal_edge_beam_y_mm'] + np.mean(
        np.asarray(shape['physical_outline_points_mm'])[-2:, 0])
    for piece in shape['pieces']:
        vertices = np.asarray(piece['vertices_beam_mm'])
        color = '#b9c9d8' if piece['name'].endswith('Base') else '#bd8549'
        for ax, dims in zip(axes[:2], ([0, 1], [1, 2])):
            points = vertices[:, dims].copy()
            if dims[1] == 2: points[:, 1] += sad
            ax.add_patch(Polygon(points[ConvexHull(points).vertices], fc=color, ec='#544331', alpha=.6, lw=.5))
        # Intersect convex-piece edges with the requested transverse plane.
        section = []
        for face in ConvexHull(vertices).simplices:
            for i, j in zip(face, np.roll(face, 1)):
                a, b = vertices[i], vertices[j]
                if (a[1]-section_y)*(b[1]-section_y) < 0:
                    section.append(a+(b-a)*(section_y-a[1])/(b[1]-a[1]))
                if abs(a[1]-section_y) < 1e-9: section.append(a)
        if section:
            points = np.unique(np.asarray(section)[:, [0, 2]], axis=0)
            points[:, 1] += sad
            if len(points) >= 3:
                axes[2].add_patch(Polygon(points[ConvexHull(points).vertices], fc=color, ec='#544331', lw=.7))
    labels = [('X (mm)', 'Y (mm)', 'Beam-eye projection'),
              ('Y (mm)', 'Distance from source (mm)', 'Longitudinal projection'),
              ('X (mm)', 'Distance from source (mm)', f'Rear-spine section at Y={section_y:.2f} mm')]
    for ax, (x, y, title) in zip(axes, labels):
        ax.autoscale_view(); ax.margins(.1); ax.set_aspect('equal')
        ax.set(xlabel=x, ylabel=y, title=title); ax.grid(alpha=.2)
    # Source is above downstream material in longitudinal panels.
    for ax in axes[1:]: ax.invert_yaxis()
    fig.supxlabel('\n'.join(jaw_shape_summary(shape)) + '\nApproximate drawing dimensions; not a verified EVO hardware model.', fontsize=9)
    pdf.savefig(fig); plt.close(fig)
