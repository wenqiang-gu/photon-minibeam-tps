"""Native steering enriched with independent per-beam hardware snapshots.

Use validate_minibeam_stf for extension dictionaries. Native pyRadPlan's dict
validator ignores extension fields; live subclasses pass through native APIs.
"""
from copy import deepcopy
import numpy as np
from pyRadPlan.stf import Beam, SteeringInformation, validate_stf
from .geometry.models import BeamGeometry


class MinibeamBeam(Beam):
    geometry: BeamGeometry


class MinibeamSteeringInformation(SteeringInformation):
    beams: list[MinibeamBeam]


def has_geometry(stf):
    beams = stf.get('beams', []) if isinstance(stf, dict) else stf.beams
    return any(('geometry' in b if isinstance(b, dict) else hasattr(b, 'geometry')) for b in beams)


def validate_minibeam_stf(stf):
    """Reconstruct, rather than trust model_copy(update=...) bypasses."""
    if not isinstance(stf, dict):
        stf = stf.model_dump(exclude_computed_fields=True)
    result = MinibeamSteeringInformation.model_validate(deepcopy(stf))
    from .geometry.assembly import resolve
    for beam in result.beams:
        resolve(beam.geometry.as_config(), float(beam.sad))
    return result


def enrich_stf(stf, geometry=None, *, overrides=None):
    """Copy native steering; overrides is one section-update dictionary per beam.

    Existing attached snapshots are retained when geometry is omitted. Beam
    associations follow list order, including after reordering or deep copying.
    """
    attached = has_geometry(stf)
    if attached and geometry is not None:
        raise ValueError('Attached geometry is authoritative; use per-beam overrides to edit it')
    native = validate_minibeam_stf(stf) if attached else validate_stf(stf)
    base = geometry if isinstance(geometry, BeamGeometry) else BeamGeometry.from_config(geometry) if not attached else None
    if overrides is not None and len(overrides) != len(native.beams):
        raise ValueError('Provide exactly one geometry override dictionary per beam')
    beams = []
    for index, beam in enumerate(native.beams):
        snapshot = beam.geometry if attached else base
        snapshot = snapshot.replace(**(overrides[index] if overrides is not None else {}))
        beams.append(MinibeamBeam(**deepcopy(beam.model_dump(
            exclude={'geometry'}, exclude_computed_fields=True)), geometry=snapshot))
    return validate_minibeam_stf(MinibeamSteeringInformation(beams=beams))


def beam_geometry_records(stf):
    """Resolved hardware and current 1-based native beam associations."""
    from .geometry.assembly import TreatmentHead, serializable
    from .geometry.spatial import beam_basis
    records = []
    head = TreatmentHead(None)
    for index, beam in enumerate(stf.beams, 1):
        if not hasattr(beam, 'geometry'):
            continue
        g, aperture, envelopes, _ = head.resolved(beam)
        basis = beam_basis(beam)
        shift = beam.geometry.aperture.lateral_shift_mm
        records.append(dict(beam_index=index, gantry_angle_deg=float(beam.gantry_angle),
            couch_angle_deg=float(beam.couch_angle), geometry=beam.geometry.model_dump(),
            configuration_sha256=beam.geometry.as_config().sha256,
            translation_lps_mm=(basis @ np.array([shift, 0., 0.])).tolist(),
            resolved=dict(mlc_jaws=serializable(g), aperture=serializable(aperture), envelopes=envelopes),
            component_centers_lps_mm={e['name']: (basis @ e['center'] + beam.iso_center).tolist() for e in envelopes}))
    return records


def matrad_steering(stf):
    """Standard native stf plus an explicit companion; never mutate native data."""
    native = SteeringInformation(beams=[Beam.model_validate(beam.model_dump(
        exclude={'geometry'}, exclude_computed_fields=True)) for beam in stf.beams])
    records = beam_geometry_records(stf)
    # scipy MATLAB structs cannot contain None (disabled hardware).
    def matlab(value):
        if value is None: return np.empty(0)
        if isinstance(value, dict): return {k: matlab(v) for k,v in value.items()}
        if isinstance(value, list): return [matlab(v) for v in value]
        return value
    result = {'stf': native.to_matrad()}
    if records: result['beam_geometry'] = matlab(records)
    return result
