"""Optional beam-perpendicular phase-space surfaces, independent of dose scoring."""
from copy import deepcopy
import math
from numbers import Real
import re

import numpy as np

from ..geometry.assembly import TreatmentHead
from ..geometry.coordinates import topas_angles
from ..geometry.spatial import beam_basis, corners


FILM_THICKNESS_MM = 0.002
LOCATIONS = {'mlc_exit': 'mlc', 'jaw_exit': 'jaws', 'collimator_exit': 'SlitCollimator'}


def _number(value, label, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f'{label} must be a finite number')
    if value < 0 or (positive and value == 0):
        raise ValueError(f'{label} must be {"positive" if positive else "nonnegative"}')
    return float(value)


def validate_recorders(config):
    if not isinstance(config, (list, tuple)):
        raise ValueError('PHASE_SPACE_RECORDERS must be a list of recorder dictionaries')
    result, names = [], set()
    for entry in config:
        if not isinstance(entry, dict) or set(entry) - {'name', 'location', 'offset_mm', 'source_distance_mm', 'size_mm'}:
            raise ValueError('Invalid phase-space recorder fields')
        name, location = entry.get('name'), entry.get('location')
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', name) or name in names:
            raise ValueError('Phase-space recorder names must be unique identifiers')
        names.add(name)
        if not isinstance(location, str) or location not in {*LOCATIONS, 'source_distance'}:
            raise ValueError(f'Unknown phase-space recorder location: {location}')
        item = dict(name=name, location=location)
        if location == 'source_distance':
            if 'offset_mm' in entry or 'source_distance_mm' not in entry:
                raise ValueError('source_distance requires source_distance_mm and forbids offset_mm')
            item['source_distance_mm'] = _number(entry['source_distance_mm'], 'source_distance_mm', positive=True)
        else:
            if 'source_distance_mm' in entry:
                raise ValueError('Device exits forbid source_distance_mm')
            item['offset_mm'] = _number(entry.get('offset_mm', .01), 'offset_mm')
        size = entry.get('size_mm', [300., 300.])
        if not isinstance(size, (list, tuple)) or len(size) != 2:
            raise ValueError('Recorder size_mm must contain two positive dimensions')
        item['size_mm'] = [_number(v, 'size_mm', positive=True) for v in size]
        result.append(item)
    return result


def resolve_recorders(config, beam, devices, patient_center):
    """Resolve the upstream scoring face; tilted devices use their maximum beam Z."""
    config = validate_recorders(config)
    if not config:
        return []
    head = next((d for d in devices if isinstance(d, TreatmentHead)), None)
    envelopes = head.resolved(beam)[2] if head and any(c['location'] in LOCATIONS for c in config) else []
    basis = beam_basis(beam)
    iso = np.asarray(beam.iso_center, dtype=float)
    records = []
    for item in config:
        if item['location'] in LOCATIONS:
            envelope = next((e for e in envelopes if e['name'] == LOCATIONS[item['location']]), None)
            if envelope is None:
                raise ValueError(f"Recorder {item['name']} requires an enabled {item['location']} device; use source_distance for an absent-device comparison")
            z = float(corners(envelope)[:, 2].max()) + item['offset_mm']
            distance = z + float(beam.sad)
        else:
            distance = item['source_distance_mm']
            z = distance - float(beam.sad)
        if distance <= 0 or not math.isfinite(distance):
            raise ValueError('Recorder surface must be downstream of the source')
        if any(abs(z - r['surface_position_beam_mm'][2]) <= FILM_THICKNESS_MM + 1e-9 for r in records):
            raise ValueError('Coincident or overlapping phase-space recording surfaces')
        surface = np.array([0., 0., z])
        center = surface + [0., 0., FILM_THICKNESS_MM / 2]
        center_world = iso + basis @ center - patient_center
        half = [item['size_mm'][0] / 2, item['size_mm'][1] / 2, FILM_THICKNESS_MM / 2]
        bounds = corners(dict(center=center_world, rotation=basis, half_size=half))
        record = dict(deepcopy(item), resolved_source_distance_mm=distance,
                      component='PhspFilm_' + item['name'],
                      surface_position_beam_mm=surface.tolist(),
                      surface_center_lps_mm=(iso + basis @ surface).tolist(),
                      surface_center_topas_mm=(iso + basis @ surface - patient_center).tolist(),
                      center_topas_mm=center_world.tolist(), rotation_deg=topas_angles(basis).tolist(),
                      normal_lps=basis[:, 2].tolist(), film_thickness_mm=FILM_THICKNESS_MM,
                      bounds_topas_mm=[bounds.min(axis=0).tolist(), bounds.max(axis=0).tolist()],
                      output_type='Binary', include_empty_histories='InSequence',
                      coordinates='Particle positions/directions use TOPAS World coordinates; surface is beam-perpendicular',
                      crossing='Downstream crossings only; no particle-species filter; tracks continue')
        records.append(record)
    return records


def film_parameters(records):
    lines = []
    for r in records:
        component = r['component']
        lines += [f's:Ge/{component}/Type = "TsBox"', f's:Ge/{component}/Parent = "World"',
                  f'b:Ge/{component}/IsParallel = "True"',
                  f's:Ge/{component}/ParallelWorldName = "{component}World"']
        for axis, half, pos, rot in zip('XYZ', [r['size_mm'][0]/2, r['size_mm'][1]/2, FILM_THICKNESS_MM/2],
                                         r['center_topas_mm'], r['rotation_deg']):
            lines += [f'd:Ge/{component}/HL{axis} = {half:.12g} mm',
                      f'd:Ge/{component}/Trans{axis} = {pos:.12g} mm',
                      f'd:Ge/{component}/Rot{axis} = {rot:.12g} deg']
    return '\n'.join(lines)


def scorer_parameters(records):
    lines = []
    for r in records:
        prefix = 's:Sc/' + r['scorer']
        for key, value in [('Quantity', 'PhaseSpace'), ('Surface', r['component'] + '/ZMinusSurface'),
                           ('OnlyIncludeParticlesGoing', 'In'), ('OutputType', 'Binary'),
                           ('OutputFile', r['output_base']), ('IfOutputFileAlreadyExists', 'Exit'),
                           ('IncludeEmptyHistories', 'InSequence')]:
            lines.append(f'{prefix}/{key} = "{value}"')
        lines.append(f'b:Sc/{r["scorer"]}/OutputToConsole = "False"')
    return '\n'.join(lines)
