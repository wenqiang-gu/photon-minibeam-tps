"""Adapted geometry mathematics from vroc-photon-minibeam/TOPAS/generate.py.

Only device shapes are retained; source, phantom and dose workflow stay native.
"""
from __future__ import annotations
from dataclasses import dataclass
from types import SimpleNamespace as Geometry
import math

@dataclass(frozen=True)
class JawStage:
    axis: str
    source_to_center: float
    center_z: float
    thickness: float
    entrance_z: float
    exit_z: float
    entrance_opening: float
    exit_opening: float

@dataclass(frozen=True)
class RoundedMLCTip:
    radius: float
    circle_center_x: float
    circle_center_z: float
    center_z_offset: float
    tangent_z: float
    tangent_x: float

def _render_jaw_solid(
    g: Geometry,
    stage: JawStage,
    positive: bool,
    component_base: str,
    description: str,
) -> str:
    axis = stage.axis
    sign_name = "Positive" if positive else "Negative"
    component = f"{component_base}{sign_name}"
    opening_half_entrance = stage.entrance_opening / 2.0
    opening_half_exit = stage.exit_opening / 2.0
    if axis == "X":
        outer_half = g.field_aperture_outer_width / 2.0
        entrance_half_length = (outer_half - opening_half_entrance) / 2.0
        exit_half_length = (outer_half - opening_half_exit) / 2.0
        entrance_center = (outer_half + opening_half_entrance) / 2.0
        exit_center = (outer_half + opening_half_exit) / 2.0
        trans_x = (entrance_center + exit_center) / 2.0
        trans_y = 0.0
        hly1 = g.field_aperture_outer_height / 2.0
        hly2 = hly1
        hlx1 = entrance_half_length
        hlx2 = entrance_half_length
        hlx3 = exit_half_length
        hlx4 = exit_half_length
        phi = 0.0 if positive else 180.0
    else:
        outer_half = g.field_aperture_outer_height / 2.0
        entrance_half_length = (outer_half - opening_half_entrance) / 2.0
        exit_half_length = (outer_half - opening_half_exit) / 2.0
        entrance_center = (outer_half + opening_half_entrance) / 2.0
        exit_center = (outer_half + opening_half_exit) / 2.0
        trans_x = 0.0
        trans_y = (entrance_center + exit_center) / 2.0
        hly1 = entrance_half_length
        hly2 = exit_half_length
        hlx1 = g.field_aperture_outer_width / 2.0
        hlx2 = hlx1
        hlx3 = hlx1
        hlx4 = hlx1
        phi = 90.0 if positive else 270.0
    if not positive:
        trans_x = -trans_x
        trans_y = -trans_y
    if math.isclose(trans_x, 0.0, abs_tol=1e-12):
        trans_x = 0.0
    if math.isclose(trans_y, 0.0, abs_tol=1e-12):
        trans_y = 0.0
    theta = math.degrees(
        math.atan((exit_center - entrance_center) / stage.thickness)
    )
    return f'''# {description} {sign_name.lower()} bank
s:Ge/{component}/Type = "G4GTrap"
s:Ge/{component}/Parent = "World"
s:Ge/{component}/Material = "{g.field_aperture_material}"
d:Ge/{component}/HLZ = {stage.thickness / 2.0:.6f} mm
d:Ge/{component}/HLY1 = {hly1:.6f} mm
d:Ge/{component}/HLX1 = {hlx1:.6f} mm
d:Ge/{component}/HLX2 = {hlx2:.6f} mm
d:Ge/{component}/HLY2 = {hly2:.6f} mm
d:Ge/{component}/HLX3 = {hlx3:.6f} mm
d:Ge/{component}/HLX4 = {hlx4:.6f} mm
d:Ge/{component}/Theta = {theta:.9f} deg
d:Ge/{component}/Phi = {phi:.6f} deg
d:Ge/{component}/Alp1 = 0 deg
d:Ge/{component}/Alp2 = 0 deg
d:Ge/{component}/TransX = {trans_x:.6f} mm
d:Ge/{component}/TransY = {trans_y:.6f} mm
d:Ge/{component}/TransZ = {stage.center_z:.6f} mm
'''

def _rounded_mlc_inner_x(g: Geometry, z: float) -> float:
    if g.mlc is None or g.mlc_round_tip is None:
        raise ValueError("Rounded MLC geometry is incomplete")
    axial_offset = z - g.mlc_round_tip.circle_center_z
    radicand = (
        g.mlc_round_tip.radius ** 2 - axial_offset ** 2
    )
    if radicand < -1e-9:
        raise ValueError("MLC rounded tip does not span the requested z position")
    return (
        g.mlc_round_tip.circle_center_x
        - math.sqrt(max(0.0, radicand))
    )

def _rounded_mlc_polygon(
    g: Geometry,
    positive: bool,
    arc_segments: int = 64,
) -> tuple[tuple[float, float], ...]:
    if g.mlc is None or g.mlc_round_tip is None:
        raise ValueError("Rounded MLC geometry is incomplete")
    outer_x = g.field_aperture_outer_width / 2.0
    half_thickness = g.mlc.thickness / 2.0
    arc = []
    for index in range(arc_segments + 1):
        fraction = index / arc_segments
        local_z = -half_thickness + fraction * g.mlc.thickness
        global_z = g.mlc.center_z + local_z
        arc.append((_rounded_mlc_inner_x(g, global_z), local_z))
    polygon = [
        arc[0],
        (outer_x, -half_thickness),
        (outer_x, half_thickness),
        arc[-1],
        *reversed(arc[1:-1]),
    ]
    if positive:
        return tuple(polygon)
    return tuple(reversed([(-x, z) for x, z in polygon]))

def _render_rounded_mlc_solid(
    g: Geometry,
    positive: bool,
) -> str:
    if g.mlc is None or g.mlc_round_tip is None:
        raise ValueError("Rounded MLC geometry is incomplete")
    sign_name = "Positive" if positive else "Negative"
    component = f"MLC{sign_name}"
    polygon = _rounded_mlc_polygon(g, positive)
    coordinates = " ".join(
        f"{x:.6f} {z:.6f}" for x, z in polygon
    )
    return f'''# Rounded MLC {sign_name.lower()} bank
s:Ge/{component}/Type = "G4ExtrudedSolid"
s:Ge/{component}/Parent = "World"
s:Ge/{component}/Material = "{g.field_aperture_material}"
dv:Ge/{component}/Polygons = {2 * len(polygon)} {coordinates} mm
d:Ge/{component}/HLZ = {g.field_aperture_outer_height / 2.0:.6f} mm
dv:Ge/{component}/Off1 = 2 0 0 mm
u:Ge/{component}/Scale1 = 1
dv:Ge/{component}/Off2 = 2 0 0 mm
u:Ge/{component}/Scale2 = 1
# G4PVPlacement interprets the component rotation as a coordinate-frame
# transform.  RotX = -90 deg maps the polygon's second coordinate to world Z
# without swapping the upstream and downstream rounded-tip profiles.
d:Ge/{component}/RotX = -90 deg
d:Ge/{component}/RotY = 0 deg
d:Ge/{component}/RotZ = 0 deg
d:Ge/{component}/TransX = 0 mm
d:Ge/{component}/TransY = 0 mm
d:Ge/{component}/TransZ = {g.mlc.center_z:.6f} mm
'''

