"""Adapted geometry mathematics from vroc-photon-minibeam/TOPAS/generate.py.

Only device shapes are retained; source, phantom and dose workflow stay native.
"""
from __future__ import annotations
from dataclasses import dataclass
from types import SimpleNamespace as Geometry
import math

GENERATED_HEADER = "# Focused slit geometry adapted from vroc-photon-minibeam.\n"

@dataclass(frozen=True)
class CollimatorBlade:
    number: int
    entrance_center_x: float
    exit_center_x: float
    center_x: float
    angle_deg: float
    projected_width_x: float

def _build_collimator_blade_layout(
    *,
    slit_count: int,
    slit_width: float,
    blade_thickness: float,
    entrance_distance: float,
    exit_distance: float,
    air_width: float,
) -> tuple[
    tuple[CollimatorBlade, ...],
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
    float,
    float,
]:
    """Build symmetric, parallel-sided blades clipped at the two z faces."""
    if blade_thickness >= 2.0 * entrance_distance:
        raise ValueError(
            "blade_thickness_mm is too large relative to the source-to-"
            "collimator-entrance distance"
        )

    positive_blades: list[tuple[float, float, float, float]] = []
    next_inner_edge = slit_width / 2.0
    coefficient = blade_thickness / (2.0 * entrance_distance)
    quadratic_factor = 1.0 - coefficient * coefficient
    blades_per_side = (slit_count + 1) // 2

    for _ in range(blades_per_side):
        # The blade centerline points to the source. Its entrance-plane
        # projection is t/cos(theta), where theta depends on that centerline.
        discriminant = (
            coefficient * coefficient * next_inner_edge * next_inner_edge
            + quadratic_factor * blade_thickness * blade_thickness / 4.0
        )
        entrance_center = (
            next_inner_edge + math.sqrt(discriminant)
        ) / quadratic_factor
        angle = math.atan2(entrance_center, entrance_distance)
        projected_width = blade_thickness / math.cos(angle)
        exit_center = entrance_center * exit_distance / entrance_distance
        positive_blades.append(
            (entrance_center, exit_center, angle, projected_width)
        )
        next_inner_edge = (
            entrance_center + projected_width / 2.0 + slit_width
        )

    unsigned_layout = [
        (-entrance, -exit_position, -angle, projected)
        for entrance, exit_position, angle, projected
        in reversed(positive_blades)
    ] + positive_blades
    blades = tuple(
        CollimatorBlade(
            number=number,
            entrance_center_x=entrance,
            exit_center_x=exit_position,
            center_x=(entrance + exit_position) / 2.0,
            angle_deg=math.degrees(angle),
            projected_width_x=projected,
        )
        for number, (entrance, exit_position, angle, projected)
        in enumerate(unsigned_layout, start=1)
    )

    entrance_widths: list[float] = []
    exit_widths: list[float] = []
    entrance_centers: list[float] = []
    exit_centers: list[float] = []
    for left, right in zip(blades, blades[1:]):
        entrance_left = (
            left.entrance_center_x + left.projected_width_x / 2.0
        )
        entrance_right = (
            right.entrance_center_x - right.projected_width_x / 2.0
        )
        exit_left = left.exit_center_x + left.projected_width_x / 2.0
        exit_right = right.exit_center_x - right.projected_width_x / 2.0
        entrance_width = entrance_right - entrance_left
        exit_width = exit_right - exit_left
        if entrance_width <= 0 or exit_width <= 0:
            raise ValueError("collimator blades overlap or close an air slit")
        entrance_widths.append(entrance_width)
        exit_widths.append(exit_width)
        entrance_centers.append((entrance_left + entrance_right) / 2.0)
        exit_centers.append((exit_left + exit_right) / 2.0)

    if len(entrance_widths) != slit_count:
        raise ValueError("collimator blade layout does not produce slit_count gaps")
    if any(
        not math.isclose(value, slit_width, abs_tol=1e-9)
        for value in entrance_widths
    ):
        raise ValueError(
            "rotated blade layout does not preserve the requested entrance slit width"
        )

    entrance_ctcs = tuple(
        right - left
        for left, right in zip(entrance_centers, entrance_centers[1:])
    )
    exit_ctcs = tuple(
        right - left
        for left, right in zip(exit_centers, exit_centers[1:])
    )
    cavity_half_width = air_width / 2.0
    outer_blade = blades[-1]
    entrance_edge_gap = cavity_half_width - (
        outer_blade.entrance_center_x
        + outer_blade.projected_width_x / 2.0
    )
    exit_edge_gap = cavity_half_width - (
        outer_blade.exit_center_x
        + outer_blade.projected_width_x / 2.0
    )
    if entrance_edge_gap <= 0:
        raise ValueError(
            "collimator blade fan intersects the air-cavity wall at the entrance"
        )
    if exit_edge_gap <= 0:
        raise ValueError(
            "collimator blade fan intersects the air-cavity wall at the exit"
        )

    return (
        blades,
        tuple(entrance_centers),
        tuple(exit_centers),
        tuple(entrance_widths),
        tuple(exit_widths),
        entrance_ctcs,
        exit_ctcs,
        entrance_edge_gap,
        exit_edge_gap,
    )

def render_collimator(g: Geometry) -> str:
    half_x = g.collimator_width / 2.0
    half_y = g.collimator_height / 2.0
    half_z = g.collimator_thickness / 2.0
    air_half_x = g.collimator_air_width / 2.0
    air_half_y = g.collimator_air_height / 2.0
    lines = [
        GENERATED_HEADER.rstrip(),
        "# The brass frame contains a full-thickness air cavity.",
        f"# {len(g.collimator_blades)} parallel-sided brass blades are clipped flush with the",
        "# entrance and exit planes and focused toward the point source at",
        "# z = %.6f mm." % g.source_z,
        "# RotX and RotY rigidly rotate this complete pre-focused assembly",
        "# about its geometric center; all daughters inherit the transform.",
        "",
        's:Ge/Collimator/Type     = "TsBox"',
        's:Ge/Collimator/Parent   = "World"',
        f's:Ge/Collimator/Material = "{g.material}"',
        f"d:Ge/Collimator/HLX = {half_x:.6f} mm",
        f"d:Ge/Collimator/HLY = {half_y:.6f} mm",
        f"d:Ge/Collimator/HLZ = {half_z:.6f} mm",
        f"d:Ge/Collimator/TransX = {g.lateral_shift_mm:.6f} mm",
        "d:Ge/Collimator/TransY = 0 mm",
        f"d:Ge/Collimator/TransZ = {g.collimator_center_z:.6f} mm",
        f"d:Ge/Collimator/RotX = {g.collimator_rotation_x:.6f} deg",
        f"d:Ge/Collimator/RotY = {g.collimator_rotation_y:.6f} deg",
        "",
        's:Ge/CollimatorAir/Type     = "TsBox"',
        's:Ge/CollimatorAir/Parent   = "Collimator"',
        's:Ge/CollimatorAir/Material = "G4_AIR"',
        f"d:Ge/CollimatorAir/HLX = {air_half_x:.6f} mm",
        f"d:Ge/CollimatorAir/HLY = {air_half_y:.6f} mm",
        f"d:Ge/CollimatorAir/HLZ = {half_z:.6f} mm",
        "d:Ge/CollimatorAir/TransX = 0 mm",
        "d:Ge/CollimatorAir/TransY = 0 mm",
        "d:Ge/CollimatorAir/TransZ = 0 mm",
        "",
    ]
    for blade in g.collimator_blades:
        half_projected_width = blade.projected_width_x / 2.0
        theta = abs(blade.angle_deg)
        phi = 0.0 if blade.angle_deg >= 0 else 180.0
        lines.extend([
            (
                f"# Blade {blade.number:02d}: entrance center "
                f"{blade.entrance_center_x:.6f} mm, exit center "
                f"{blade.exit_center_x:.6f} mm"
            ),
            f's:Ge/Blade_{blade.number:02d}/Type     = "G4GTrap"',
            f's:Ge/Blade_{blade.number:02d}/Parent   = "CollimatorAir"',
            f's:Ge/Blade_{blade.number:02d}/Material = "{g.material}"',
            f"d:Ge/Blade_{blade.number:02d}/HLZ  = {half_z:.6f} mm",
            f"d:Ge/Blade_{blade.number:02d}/HLY1 = {air_half_y:.6f} mm",
            (
                f"d:Ge/Blade_{blade.number:02d}/HLX1 = "
                f"{half_projected_width:.6f} mm"
            ),
            (
                f"d:Ge/Blade_{blade.number:02d}/HLX2 = "
                f"{half_projected_width:.6f} mm"
            ),
            f"d:Ge/Blade_{blade.number:02d}/HLY2 = {air_half_y:.6f} mm",
            (
                f"d:Ge/Blade_{blade.number:02d}/HLX3 = "
                f"{half_projected_width:.6f} mm"
            ),
            (
                f"d:Ge/Blade_{blade.number:02d}/HLX4 = "
                f"{half_projected_width:.6f} mm"
            ),
            f"d:Ge/Blade_{blade.number:02d}/Theta = {theta:.9f} deg",
            f"d:Ge/Blade_{blade.number:02d}/Phi   = {phi:.0f} deg",
            f"d:Ge/Blade_{blade.number:02d}/Alp1  = 0 deg",
            f"d:Ge/Blade_{blade.number:02d}/Alp2  = 0 deg",
            (
                f"d:Ge/Blade_{blade.number:02d}/TransX = "
                f"{blade.center_x:.6f} mm"
            ),
            f"d:Ge/Blade_{blade.number:02d}/TransY = 0 mm",
            f"d:Ge/Blade_{blade.number:02d}/TransZ = 0 mm",
            "",
        ])
    return "\n".join(lines).rstrip() + "\n"


class SlitAperture:
    """Focused brass blades; replaceable aperture resolver/renderer/report geometry."""
    config_keys = {'enabled','type','material','source_to_center_mm','thickness_mm','width_mm','height_mm',
                   'rotation_x_deg','rotation_y_deg','brass_frame_thickness_x_mm','brass_frame_thickness_y_mm',
                   'slit_count','slit_entrance_width_mm','slit_entrance_ctc_mm','blade_thickness_mm','lateral_shift_mm'}

    @staticmethod
    def validate_config(config):
        n=config['slit_count']
        if type(n) is not int or n%2 != 1 or n>10001:
            raise ValueError('slit_count must be a positive odd integer <= 10001')

    @staticmethod
    def resolve(c, sad):
        from types import SimpleNamespace
        if not math.isclose(c['slit_entrance_width_mm']+c['blade_thickness_mm'],c['slit_entrance_ctc_mm'],abs_tol=1e-9):
            raise ValueError('Slit width plus blade thickness must equal nominal entrance CTC')
        width,height,thickness=c['width_mm'],c['height_mm'],c['thickness_mm']
        air_width=width-2*c['brass_frame_thickness_x_mm']
        air_height=height-2*c['brass_frame_thickness_y_mm']
        if min(air_width,air_height)<=0: raise ValueError('Slit frame leaves no air cavity')
        distance=c['source_to_center_mm']
        if distance-thickness/2<=0: raise ValueError('Aperture must be downstream of source')
        layout=_build_collimator_blade_layout(slit_count=c['slit_count'],slit_width=c['slit_entrance_width_mm'],
            blade_thickness=c['blade_thickness_mm'],entrance_distance=distance-thickness/2,
            exit_distance=distance+thickness/2,air_width=air_width)
        g=SimpleNamespace(source_z=-sad,material=c['material'],collimator_width=width,collimator_height=height,
            collimator_thickness=thickness,collimator_air_width=air_width,collimator_air_height=air_height,
            lateral_shift_mm=c.get('lateral_shift_mm',0.0),nominal_entrance_ctc_mm=c['slit_entrance_ctc_mm'],
            collimator_center_z=distance-sad,collimator_rotation_x=c['rotation_x_deg'],
            collimator_rotation_y=c['rotation_y_deg'],slit_count=c['slit_count'])
        keys=['collimator_blades','entrance_slit_centers','exit_slit_centers','entrance_slit_widths',
              'exit_slit_widths','entrance_slit_ctcs','exit_slit_ctcs','entrance_edge_gap','exit_edge_gap']
        for key,value in zip(keys,layout):setattr(g,key,value)
        return g

    @staticmethod
    def render(g, parent):
        return render_collimator(g).replace('Parent   = "World"',f'Parent   = "{parent}"')

    @staticmethod
    def envelope(g):
        from ..spatial import placement_rotation
        return {'name':'SlitCollimator', 'center':[g.lateral_shift_mm,0,g.collimator_center_z],
                'half_size':[g.collimator_width/2,g.collimator_height/2,g.collimator_thickness/2],
                'rotation':placement_rotation(g.collimator_rotation_x,g.collimator_rotation_y).tolist()}

    @staticmethod
    def drawing(g):
        # Beam-local polygons shared by the geometry report's projections.
        from ..spatial import placement_rotation
        import numpy as np
        rotation=placement_rotation(g.collimator_rotation_x,g.collimator_rotation_y)
        h=g.collimator_thickness/2
        polygons=[]
        for blade in g.collimator_blades:
            w=blade.projected_width_x/2
            points=np.array([[blade.entrance_center_x-w,0,-h],[blade.entrance_center_x+w,0,-h],
                             [blade.exit_center_x+w,0,h],[blade.exit_center_x-w,0,h]])
            polygons.append((points@rotation.T + [g.lateral_shift_mm,0,g.collimator_center_z]).tolist())
        return polygons


    @staticmethod
    def draw_entrance(g, ax):
        from matplotlib.patches import Rectangle
        w,h=g.collimator_width,g.collimator_height
        aw,ah=g.collimator_air_width,g.collimator_air_height
        ax.add_patch(Rectangle((-h/2,-w/2),h,w,fc='#bba25f',ec='#806b38'))
        ax.add_patch(Rectangle((-ah/2,-aw/2),ah,aw,fc='white',ec='#806b38'))
        for blade in g.collimator_blades:
            ax.add_patch(Rectangle((-ah/2,blade.entrance_center_x-blade.projected_width_x/2),
                ah,blade.projected_width_x,fc='#bba25f',ec='#806b38',lw=.4))
        ax.set_xlim(-h/2-10,h/2+10);ax.set_ylim(-w/2-10,w/2+10)
        ax.set_aspect('equal');ax.set_title('Slit entrance in aperture coordinates',fontsize=11)
        ax.set_xlabel('Y along slits (mm)');ax.set_ylabel('X across slits (mm)')

    @staticmethod
    def summary(g):
        return [f'Rigid shift along beam X: {g.lateral_shift_mm:g} mm; nominal CTC: {g.nominal_entrance_ctc_mm:g} mm',
            f'Aperture: {g.material}; {g.slit_count} slits / {len(g.collimator_blades)} blades',
            f'Outer X/Y/Z: {g.collimator_width:g}/{g.collimator_height:g}/{g.collimator_thickness:g} mm',
            f'Air X/Y: {g.collimator_air_width:g}/{g.collimator_air_height:g} mm',
            f'Rotation X/Y: {g.collimator_rotation_x:g}/{g.collimator_rotation_y:g} deg',
            f'Entrance slit width: {min(g.entrance_slit_widths):.3f} mm',
            f'Exit slit widths: {min(g.exit_slit_widths):.3f} to {max(g.exit_slit_widths):.3f} mm',
            f'Entrance CTC: {min(g.entrance_slit_ctcs):.3f} to {max(g.entrance_slit_ctcs):.3f} mm' if g.entrance_slit_ctcs else 'Single slit: no CTC',
            f'Edge air gaps entrance/exit: {g.entrance_edge_gap:.3f}/{g.exit_edge_gap:.3f} mm',
            'Entrance diagram is before rigid assembly rotation.']
