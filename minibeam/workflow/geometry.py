"""Resolve explicit patient-workflow switches into immutable beam geometry."""
from minibeam.geometry.configuration import GeometryConfig
from minibeam.geometry.models import BeamGeometry


def validate_collimator_switch(enable_collimator):
    if type(enable_collimator) is not bool:
        raise ValueError('ENABLE_COLLIMATOR must be True or False')


def resolve_geometry(*, config_path, enable_collimator):
    """Load once; the returned snapshot, not the TOML, controls preparation."""
    validate_collimator_switch(enable_collimator)
    geometry = BeamGeometry.from_config(GeometryConfig.load(config_path))
    if enable_collimator and not geometry.assembly.enabled:
        raise ValueError('ENABLE_COLLIMATOR=True conflicts with assembly.enabled=false in the geometry configuration')
    return geometry.replace(aperture={'enabled': enable_collimator})
