"""Small internal contracts; native pyRadPlan objects remain authoritative."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol
from pyRadPlan.stf import Beam, Ray, Beamlet

class ResultsPendingError(RuntimeError):
    """Required externally calculated dose is unavailable."""

@dataclass(frozen=True)
class RunSettings:
    bundle_dir: str = "runs/patient"
    beamlet_execution: str = "separate"
    histories: int = 10000
    seed: int = 12345
    num_threads: int = 1
    enable_opengl: bool = False
    dose_spacing_mm: tuple | None = None
    material_file: str | None = None
    dicom_dir: str | None = None
    water: bool = False
    max_matrix_bytes: int = 4 * 1024**3

@dataclass(frozen=True)
class JobContext:
    beam: Beam
    ray: Ray
    beamlet: Beamlet
    patient_center: Any
    bixel_index: int
    beam_index: int
    ray_index: int
    beamlet_index: int

@dataclass(frozen=True)
class ParameterFragment:
    text: str
    assets: tuple[tuple[str, Path], ...] = ()
    record: dict = field(default_factory=dict)
    bounds_topas_mm: Any = None

@dataclass(frozen=True)
class SourceDescription:
    # Bounds always use patient-centered TOPAS world millimeters.
    coordinate_frame: str
    bounds_topas_mm: Any
    source_name: str = "Beam"
    job_modes: tuple[str, ...] = ("beamlet",)
    normalization: str = "independent_primary_photon"
    # Independence of original histories within a column, not between columns.
    independent_histories: bool = True


class SourceModel(Protocol):
    def describe(self, context: JobContext) -> SourceDescription: ...
    def render(self, context: JobContext) -> ParameterFragment: ...


class EnergyModel(Protocol):
    def render(self, context: JobContext) -> ParameterFragment: ...


class Device(Protocol):
    name: str
    def render(self, context: JobContext) -> ParameterFragment: ...
