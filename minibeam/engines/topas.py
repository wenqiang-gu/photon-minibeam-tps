"""Thin pyRadPlan TOPASPhoton adapter; execution remains external."""
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from pyRadPlan.dose.engines import DoseEngineBase
from ..topas.manifest import json_data
from ..topas.contracts import RunSettings, ResultsPendingError
from ..topas.writer import JobWriter
from ..topas import results
from ..sources.beamlet import PointBeamletSource
from ..geometry.devices import GeometryInclude

class TOPASPhotonEngine(DoseEngineBase):
    short_name = "TOPASPhoton"
    name = "Portable OpenTOPAS photon engine"
    possible_radiation_modes = ["photons"]

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
    geometry_config: str | bool | None = None
    source_config: dict | None = None
    max_matrix_bytes: int = 4 * 1024**3

    def __init__(self, pln=None, *, source_model=None, geometry=None, devices=None, **options):
        # Upstream's assign_properties_from_pln pops "engine" from the caller's
        # plan. Keep caller-owned plans intact, including repeated calculations.
        super().__init__()
        self._plan_info = {}
        if pln is not None:
            if pln.radiation_mode != "photons" or pln.mult_scen.tot_num_scen != 1:
                raise ValueError("Photon TOPAS supports photons and one nominal scenario")
            self._plan_info = json_data({
                "radiation_mode": pln.radiation_mode,
                "num_of_fractions": pln.num_of_fractions,
                "prescribed_dose": pln.prescribed_dose,
                "machine": pln.machine,
                "steering_settings": pln.prop_stf,
            })
            config = dict(pln.prop_dose_calc)
            if config.pop("engine", "TOPASPhoton") != "TOPASPhoton":
                raise ValueError("Plan selects a different dose engine")
        else:
            config = {}
        config.update(options)
        allowed = set(TOPASPhotonEngine.__annotations__)
        for key, value in config.items():
            if key not in allowed:
                raise ValueError(f"Unknown photon TOPAS setting: {key}")
            setattr(self, key, value)
        for name in ["histories", "seed", "num_threads", "max_matrix_bytes"]:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.histories < 2 or self.histories > 2**31 - 1 or self.seed > 2**31 - 1:
            raise ValueError("Histories must be 2..2^31-1 and seed within 1..2^31-1")
        if type(self.enable_opengl) is not bool:
            raise ValueError("enable_opengl must be a boolean")
        if source_model is not None and self.source_config is not None:
            raise ValueError('Use injected source_model or source_config, not both')
        if self.source_config is not None:
            if not isinstance(self.source_config,dict):
                raise ValueError('source_config must be a dictionary')
            kind=self.source_config.get('type')
            if kind=='point' and set(self.source_config)=={'type'}:
                source_model=PointBeamletSource()
            elif kind=='phase_space' and set(self.source_config)=={'type','file_base'}:
                from ..sources.phase_space import PhaseSpaceBeamletSource
                source_model=PhaseSpaceBeamletSource(self.source_config['file_base'])
            else:
                raise ValueError('source_config requires type=point, or type=phase_space and file_base')
            self._plan_info['source_configuration']=json_data(self.source_config)
        if self.beamlet_execution not in {"separate", "combined"}:
            raise ValueError("beamlet_execution must be separate or combined")
        self.source_model = source_model or PointBeamletSource()
        if geometry is not None and devices is not None:
            raise ValueError("Use geometry or devices, not both")
        self.geometry = geometry or GeometryInclude("")
        self._injected_geometry = geometry is not None or devices is not None
        self.geometry_snapshot = None
        if self.geometry_config is True:
            raise ValueError("geometry_config must be a path, None, or False")
        self.devices = tuple(devices or ())
        # Explicit legacy TOML paths retain construction-time snapshot semantics.
        if isinstance(self.geometry_config, str) and not self._injected_geometry:
            from ..geometry.configuration import GeometryConfig
            self.geometry_snapshot = GeometryConfig.load(self.geometry_config)
        self._using_attached_geometry = False

    def _writer(self, stf=None):
        settings = RunSettings(**{name: getattr(self, name) for name in RunSettings.__dataclass_fields__})
        devices = self.devices
        snapshot = self.geometry_snapshot
        if stf is not None and any(hasattr(b, 'geometry') for b in stf.beams):
            from ..geometry.assembly import TreatmentHead
            devices = (TreatmentHead(None),)
            if self._using_attached_geometry: snapshot = None
        if devices and (self.geometry.text or self.geometry.assets):
            raise ValueError("Use geometry or devices, not both")
        return JobWriter(settings, self.source_model, self.geometry, devices, self._plan_info, snapshot)

    def _steering(self, stf):
        from ..steering import has_geometry, validate_minibeam_stf, enrich_stf
        self._using_attached_geometry = has_geometry(stf)
        if self._using_attached_geometry:
            if self._injected_geometry:
                raise ValueError('Attached geometry conflicts with injected geometry/devices')
            return validate_minibeam_stf(stf)
        if not self._injected_geometry and self.geometry_config is not False:
            from ..geometry.configuration import GeometryConfig
            if self.geometry_snapshot is None:
                self.geometry_snapshot = GeometryConfig.load(self.geometry_config)
            from ..geometry.assembly import TreatmentHead
            self.devices = (TreatmentHead(self.geometry_snapshot),) if self.geometry_snapshot.data['assembly']['enabled'] else ()
            return enrich_stf(stf, self.geometry_snapshot)
        from pyRadPlan.stf import validate_stf
        return validate_stf(stf)

    def _definition(self, ct, stf):
        stf = self._steering(stf)
        return self._writer(stf)._definition(ct, stf)

    def prepare_jobs(self, ct, cst, stf, *, bundle_dir=None, provenance=None):
        stf = self._steering(stf)
        return self._writer(stf).prepare_jobs(ct, cst, stf, bundle_dir=bundle_dir, provenance=provenance)

    def iter_scores(self, bundle_dir=None):
        return results.iter_scores(bundle_dir or self.bundle_dir)

    def collect_results(self, bundle_dir=None):
        return results.collect_results(bundle_dir or self.bundle_dir, self.max_matrix_bytes)

    def collect_forward(self, weights, bundle_dir=None):
        return results.collect_forward(weights, bundle_dir or self.bundle_dir)

    def _ready(self, ct, cst, stf):
        root = Path(self.bundle_dir).resolve()
        if not (root / "manifest.json").is_file():
            raise ResultsPendingError("Prepare a bundle and execute its TOPAS jobs before calculating dose")
        from ..topas.manifest import load_manifest
        from ..topas.validation import validate_request
        manifest = load_manifest(root)
        if manifest.get('beamlet_execution','separate') != self.beamlet_execution:
            raise ValueError('Native dose execution mode differs from saved bundle')
        validate_request(manifest, ct, stf, self.dose_spacing_mm)
        return root

    def _calc_dose(self, ct, cst, stf):
        return self.collect_results(self._ready(ct, cst, stf))

    def calc_dose_forward(self, ct, cst, stf, w=None):
        root = self._ready(ct,cst,stf)
        from ..topas.manifest import load_manifest
        manifest = load_manifest(root)
        if w is None:
            w = np.array([bl.weight for beam in stf.beams for ray in beam.rays for bl in ray.beamlets])
        w = np.asarray(w, dtype=float)
        if manifest.get('beamlet_execution') == 'combined' and w.shape == (stf.total_number_of_bixels,):
            weights = []
            for job in manifest['jobs']:
                values = w[[m['bixel_index']-1 for m in job['members']]]
                if not np.all(values == values[0]):
                    raise ValueError('Aggregate columns require equal exposure for every member bixel; use separate jobs for independent weights')
                weights.append(values[0])
            w = np.asarray(weights)
        result = results.collect_forward(w, root, manifest=manifest)
        dose = result["physical_dose"]
        if dose.GetSize() != ct.cube_hu.GetSize() or dose.GetSpacing() != ct.cube_hu.GetSpacing():
            # CT-grid dose is for viewing. Statistical errors remain on the
            # scoring grid because spatial covariance is not stored.
            result["physical_dose_dose_grid"] = dose
            if 'physical_dose_std_error' in result:
                result["physical_dose_std_error_dose_grid"] = result.pop("physical_dose_std_error")
            result["physical_dose"] = sitk.Resample(dose, ct.cube_hu, sitk.Transform(), sitk.sitkLinear, 0., sitk.sitkFloat64)
        return result
