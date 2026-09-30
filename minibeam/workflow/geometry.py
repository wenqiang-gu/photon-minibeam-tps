"""Resolve explicit patient-workflow switches into immutable beam geometry."""
import math
from numbers import Real
from minibeam.geometry.configuration import GeometryConfig
from minibeam.geometry.models import BeamGeometry


def validate_collimator_switch(enable_collimator):
    if type(enable_collimator) is not bool:
        raise ValueError('ENABLE_COLLIMATOR must be True or False')


_UNSET = object()

def resolve_geometry(*, config_path, enable_collimator, collimator_rotation_deg=_UNSET):
    """Load once; the returned snapshot, not the TOML, controls preparation."""
    validate_collimator_switch(enable_collimator)
    geometry = BeamGeometry.from_config(GeometryConfig.load(config_path))
    if enable_collimator and not geometry.assembly.enabled:
        raise ValueError('ENABLE_COLLIMATOR=True conflicts with assembly.enabled=false in the geometry configuration')
    overrides = {'enabled': enable_collimator}
    if enable_collimator and collimator_rotation_deg is not _UNSET:
        if isinstance(collimator_rotation_deg, bool) or not isinstance(collimator_rotation_deg, Real) or not math.isfinite(collimator_rotation_deg):
            raise ValueError('COLLIMATOR_ROTATION_DEG must be finite numeric, not Boolean')
        overrides['rotation_z_deg'] = geometry.aperture.rotation_z_deg + float(collimator_rotation_deg)
    return geometry.replace(aperture=overrides)
