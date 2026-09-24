"""Device composition contract and fixed central beam frame."""
from dataclasses import dataclass
import re
import numpy as np
from pyRadPlan.geometry import lps
from .coordinates import source_frame, topas_angles

@dataclass(frozen=True)
class GeometryInclude:
    text: str
    assets: tuple = ()

def beam_frame_parameters(context):
    beam = context.beam
    frame = source_frame(-np.asarray(beam.source_point),
        lps.get_beam_rotation_matrix(beam.gantry_angle, beam.couch_angle)[:, 0])
    position = np.asarray(beam.iso_center) - context.patient_center
    lines = ['s:Ge/BeamFrame/Type = "Group"', 's:Ge/BeamFrame/Parent = "World"']
    for axis, pos, rot in zip("XYZ", position, topas_angles(frame)):
        lines += [f'd:Ge/BeamFrame/Trans{axis} = {pos:.12g} mm',
                  f'd:Ge/BeamFrame/Rot{axis} = {rot:.12g} deg']
    return "\n".join(lines)


def validate_device_fragment(name, fragment):
    """Require a named component with an explicit placement in the beam frame."""
    prefix = r"Ge/" + re.escape(name)
    if not re.search(r's:' + prefix + r'/Type\s*=\s*"[^"\n]+"', fragment.text):
        raise ValueError("Device must define its named component Type")
    if not re.search(r's:' + prefix + r'/Parent\s*=\s*"BeamFrame"', fragment.text):
        raise ValueError("Device parent must be BeamFrame")
    for axis in "XYZ":
        match = re.search(r'd:' + prefix + '/Trans' + axis + r'\s*=\s*([^\s]+)\s+mm\b', fragment.text)
        try:
            valid = match is not None and np.isfinite(float(match[1]))
        except ValueError:
            valid = False
        if not valid:
            raise ValueError("Device components require explicit finite XYZ positions in mm")
