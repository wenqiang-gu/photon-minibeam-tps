"""Aperture implementations supply resolved geometry, bounds and drawing data."""
from .slits import SlitAperture

APERTURE_TYPES = {"slits": SlitAperture}
