"""Register the photon extension without replacing native TOPAS."""
from pyRadPlan.dose.engines import register_engine, get_available_engines
from .engines.topas import TOPASPhotonEngine, ResultsPendingError

if "TOPASPhoton" not in get_available_engines("photons"):
    register_engine(TOPASPhotonEngine)

from .steering import MinibeamBeam, MinibeamSteeringInformation, enrich_stf
from .square_field import StfGeneratorPhotonSquareField

__all__ = ["TOPASPhotonEngine", "ResultsPendingError", "MinibeamBeam",
           "MinibeamSteeringInformation", "enrich_stf", "StfGeneratorPhotonSquareField"]
