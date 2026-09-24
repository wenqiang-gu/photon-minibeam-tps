"""Packaged reference Schneider materials and provenance."""
from pathlib import Path
from importlib import resources

def material_bytes(override=None, water=False):
    if water:
        return b'# Synthetic homogeneous water phantom; no HU conversion.\n'
    return (Path(override).read_bytes() if override else
            resources.files(__package__).joinpath("data/HUtoMaterialSchneider.txt").read_bytes())
