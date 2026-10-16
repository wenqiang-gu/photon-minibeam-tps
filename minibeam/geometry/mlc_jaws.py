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

@dataclass(frozen=True)
class RoundedJawTip:
    radius: float
    circle_center_y: float
    circle_center_z: float
    center_z_offset: float
    tangent_z: float
    tangent_y: float

    # Shared transverse-bank mathematics; serialized records retain Y labels.
    @property
    def circle_center_x(self):
        return self.circle_center_y

    @property
    def tangent_x(self):
        return self.tangent_y

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
        outer_half = g.mlc_outer_width / 2.0
        entrance_half_length = (outer_half - opening_half_entrance) / 2.0
        exit_half_length = (outer_half - opening_half_exit) / 2.0
        entrance_center = (outer_half + opening_half_entrance) / 2.0
        exit_center = (outer_half + opening_half_exit) / 2.0
        trans_x = (entrance_center + exit_center) / 2.0
        trans_y = 0.0
        hly1 = g.mlc_outer_height / 2.0
        hly2 = hly1
        hlx1 = entrance_half_length
        hlx2 = entrance_half_length
        hlx3 = exit_half_length
        hlx4 = exit_half_length
        phi = 0.0 if positive else 180.0
    else:
        outer_half = g.jaw_outer_height / 2.0
        entrance_half_length = (outer_half - opening_half_entrance) / 2.0
        exit_half_length = (outer_half - opening_half_exit) / 2.0
        entrance_center = (outer_half + opening_half_entrance) / 2.0
        exit_center = (outer_half + opening_half_exit) / 2.0
        trans_x = 0.0
        trans_y = (entrance_center + exit_center) / 2.0
        hly1 = entrance_half_length
        hly2 = exit_half_length
        hlx1 = g.jaw_outer_width / 2.0
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

def _bank_geometry(g, axis):
    if axis == 'X':
        return g.mlc, g.mlc_round_tip, g.mlc_outer_width, g.mlc_outer_height
    if axis == 'Y':
        return g.lower_jaw, g.jaw_round_tip, g.jaw_outer_height, g.jaw_outer_width
    raise ValueError('Rounded bank axis must be X or Y')


def _rounded_bank_inner(g: Geometry, z: float, axis: str) -> float:
    stage, tip, outer, _ = _bank_geometry(g, axis)
    if stage is None or tip is None:
        raise ValueError("Rounded bank geometry is incomplete")
    axial_offset = z - tip.circle_center_z
    radicand = (
        tip.radius ** 2 - axial_offset ** 2
    )
    if radicand < -1e-9:
        raise ValueError("Rounded bank tip does not span the requested z position")
    return (
        tip.circle_center_x
        - math.sqrt(max(0.0, radicand))
    )

def _rounded_bank_polygon(
    g: Geometry,
    positive: bool,
    arc_segments: int = 64,
    axis: str = "X",
) -> tuple[tuple[float, float], ...]:
    stage, tip, outer, _ = _bank_geometry(g, axis)
    if stage is None or tip is None:
        raise ValueError("Rounded bank geometry is incomplete")
    outer_x = outer / 2.0
    half_thickness = stage.thickness / 2.0
    arc = []
    for index in range(arc_segments + 1):
        fraction = index / arc_segments
        local_z = -half_thickness + fraction * stage.thickness
        global_z = stage.center_z + local_z
        arc.append((_rounded_bank_inner(g, global_z, axis), local_z))
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

def _render_rounded_bank_solid(
    g: Geometry,
    positive: bool,
    axis: str = "X",
) -> str:
    stage, tip, _, extrusion_width = _bank_geometry(g, axis)
    if stage is None or tip is None:
        raise ValueError("Rounded bank geometry is incomplete")
    sign_name = "Positive" if positive else "Negative"
    label = 'MLC' if axis == 'X' else 'LowerJaw'
    component = f"{label}{sign_name}"
    polygon = _rounded_bank_polygon(g, positive, axis=axis)
    # Local polygon (transverse, axial) maps to (X,Z) or (Y,Z).
    # The latter cyclic mapping sends extrusion to beam X, without axial reversal.
    rx, ry, rz = (-90, 0, 0) if axis == 'X' else (-90, -90, 0)
    coordinates = " ".join(
        f"{x:.6f} {z:.6f}" for x, z in polygon
    )
    return f'''# Rounded {label} {sign_name.lower()} bank
s:Ge/{component}/Type = "G4ExtrudedSolid"
s:Ge/{component}/Parent = "World"
s:Ge/{component}/Material = "{g.field_aperture_material}"
dv:Ge/{component}/Polygons = {2 * len(polygon)} {coordinates} mm
d:Ge/{component}/HLZ = {extrusion_width / 2.0:.6f} mm
dv:Ge/{component}/Off1 = 2 0 0 mm
u:Ge/{component}/Scale1 = 1
dv:Ge/{component}/Off2 = 2 0 0 mm
u:Ge/{component}/Scale2 = 1
# G4PVPlacement interprets the component rotation as a coordinate-frame
# transform. These angles preserve upstream/downstream ordering while mapping
# the polygon to the bank transverse axis and beam Z.
d:Ge/{component}/RotX = {rx} deg
d:Ge/{component}/RotY = {ry} deg
d:Ge/{component}/RotZ = {rz} deg
d:Ge/{component}/TransX = 0 mm
d:Ge/{component}/TransY = 0 mm
d:Ge/{component}/TransZ = {stage.center_z:.6f} mm
'''


# Retain the established MLC helpers for callers and regression fixtures.
def _rounded_mlc_inner_x(g, z):
    return _rounded_bank_inner(g, z, 'X')


def _rounded_mlc_polygon(g, positive, arc_segments=64):
    return _rounded_bank_polygon(g, positive, arc_segments, 'X')


def _render_rounded_mlc_solid(g, positive):
    return _render_rounded_bank_solid(g, positive, 'X')
