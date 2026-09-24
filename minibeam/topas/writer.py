"""Compose TOPAS jobs; sources and devices supply parameter fragments."""
from importlib import metadata, resources
from pathlib import Path
import hashlib
import json
import shutil
import re
from dataclasses import asdict
import numpy as np
import SimpleITK as sitk
from . import manifest as bundle_io
from .manifest import planning_snapshot, implementation_hashes, json_data
from ..geometry.patient import dicom_inputs, patient_parameters, axial_slice_for_isocenter
from ..geometry.coordinates import grid_dict, patient_center, scoring_grid
from ..materials import material_bytes
from .scoring import scorer_parameters
from .contracts import JobContext
from ..geometry.devices import beam_frame_parameters, validate_device_fragment

class JobWriter:
    def __init__(self, settings, source_model, geometry, devices, plan_info, geometry_snapshot=None):
        self.__dict__.update(vars(settings))
        self.source_model = source_model
        self.geometry = geometry
        self.devices = tuple(devices)
        self._plan_info = plan_info
        self.geometry_snapshot = geometry_snapshot

    def _definition(self, ct, stf):
        center = patient_center(ct)
        dose_grid = scoring_grid(ct, self.dose_spacing_mm)
        if any(b.radiation_mode != "photons" for b in stf.beams):
            raise ValueError("Only photon steering is supported")
        if not stf.total_number_of_bixels:
            raise ValueError("Steering contains no beamlets")
        raw = sitk.GetArrayFromImage(ct.cube_hu)
        if not np.isfinite(raw).all():
            raise ValueError("CT contains nonfinite HU")
        if raw.min() < -32768 or raw.max() > 32767:
            raise ValueError("HU values exceed signed 16-bit ImageCube range")
        dicom_files, dicom_info = ({}, {}) if self.water else dicom_inputs(self.dicom_dir, ct)
        material = material_bytes(self.material_file, self.water)
        assets, owners = {}, {}
        reserved = {"geometry-config.toml", "beam_geometry.json", "dicom", "materials.txt", "common.txt", "devices.txt",
                    "OpenTOPAS-LICENSE.txt", "PROVENANCE.md"}
        def add_assets(items, owner):
            for name, path in items:
                parts = Path(name).parts
                if len(parts) != 2 or parts[0] != "inputs" or parts[1] in reserved or ".." in parts:
                    raise ValueError("Assets require unique flat paths below inputs/")
                path = Path(path).resolve(strict=True)
                if name in assets and (owners[name] != owner or assets[name] != path):
                    raise ValueError(f"Conflicting asset destination: {name}")
                assets[name], owners[name] = path, owner
        add_assets(self.geometry.assets, "geometry")
        names = [d.name for d in self.devices]
        if len(set(names)) != len(names) or any(not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", n) for n in names):
            raise ValueError("Devices require unique identifier names")
        world_half = float(np.max(np.asarray(ct.size) * ct.grid.resolution_vector) / 2)
        def extent(bounds):
            bounds = np.asarray(bounds, dtype=float)
            if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or np.any(bounds[0] > bounds[1]):
                raise ValueError("Source/device bounds must be finite ordered TOPAS world bounds")
            return float(np.abs(bounds).max())
        jobs = []
        sources = []
        from ..sources.phase_space import PhaseSpaceBeamletSource
        phase_space = isinstance(self.source_model, PhaseSpaceBeamletSource)
        from ..sources.beamlet import PointBeamletSource
        from ..sources.bixels import groups
        builtin = isinstance(self.source_model, (PhaseSpaceBeamletSource, PointBeamletSource))
        grouped = groups(stf, self.beamlet_execution) if builtin else []
        group_map = {members[0]['bixel_index']: members for members in grouped}
        if self.beamlet_execution == 'combined' and not builtin:
            raise ValueError('Combined mode requires a square point or phase-space source')
        job_count = len(grouped) if builtin else stf.total_number_of_bixels
        nvox = int(np.prod(dose_grid.dimensions))
        print(f"TOPAS: {job_count} {self.beamlet_execution} jobs, {nvox:,} dose voxels; "
              f"dense matrix {nvox*job_count*8/1024**3:.3f} GiB, "
              f"CSV estimate {nvox*job_count*100/1024**3:.3f} GiB.")
        if self.seed + job_count - 1 > 2**31-1:
            raise ValueError('Per-job random seed overflows TOPAS integer range')
        if builtin:
            # Reject physical placement errors before scanning a potentially large file.
            for bi, beam in enumerate(stf.beams,1):
                for device in self.devices:
                    if hasattr(device,'validate_patient'): device.validate_patient(ct,beam,bi)
            self.source_model.prepare(stf, self.histories, execution=self.beamlet_execution, seed=self.seed, patient_center=center)
        native_index = 0
        for bi, beam in enumerate(stf.beams, 1):
            for device in self.devices:
                if hasattr(device, "validate_patient"):
                    device.validate_patient(ct, beam, bi)
            for ri, ray in enumerate(beam.rays, 1):
                for li, bixel in enumerate(ray.beamlets, 1):
                    if getattr(bixel, "is_field_based", False):
                        raise ValueError("A field_shape needs a physical geometry/source implementation")
                    native_index += 1
                    if self.beamlet_execution == 'combined' and native_index not in group_map:
                        continue
                    members = group_map.get(native_index, [dict(bixel_index=native_index,beam_index=bi,ray_index=ri,beamlet_index=li)])
                    index = len(jobs) + 1
                    context = JobContext(beam, ray, bixel, center, native_index, bi, ri, li)
                    description = self.source_model.describe(context)
                    if self.beamlet_execution == "combined" and "combined" not in description.job_modes:
                        raise ValueError("Source does not support combined jobs")
                    if "beamlet" not in description.job_modes:
                        raise ValueError("Source does not support per-beamlet jobs/particle selection")
                    expected_normalization = 'original_accelerator_history' if phase_space else 'independent_primary_photon'
                    if description.normalization != expected_normalization or not description.independent_histories:
                        raise ValueError("Unsupported source normalization or dependent histories")
                    if not description.coordinate_frame or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", description.source_name):
                        raise ValueError("Source must declare a coordinate frame and valid source name")
                    world_half = max(world_half, extent(description.bounds_topas_mm))
                    fragment = self.source_model.render(context)
                    add_assets(fragment.assets, "source")
                    fragments = [fragment.text]
                    device_records = []
                    if self.devices:
                        fragments = [beam_frame_parameters(context)]
                        for device in self.devices:
                            part = device.render(context)
                            validate_device_fragment(device.name, part)
                            add_assets(part.assets, "device:" + device.name)
                            if part.bounds_topas_mm is not None:
                                world_half = max(world_half, extent(part.bounds_topas_mm))
                            fragments.append(part.text)
                            device_records.append({"name": device.name, "parameters": part.record})
                        fragments.append(fragment.text)
                    text = "\n".join(fragments)
                    components = re.findall(r's:Ge/([^/]+)/Type\s*=', text + "\n" + self.geometry.text)
                    if len(components) != len(set(components)) or set(components) & {"Patient", "World", "CTOutline"}:
                        raise ValueError("Duplicate or reserved geometry component name")
                    source = fragment.record
                    if self.seed + index - 1 > 2**31 - 1:
                        raise ValueError("Per-job random seed overflows TOPAS integer range")
                    name = f"beam_{bi:06d}" if self.beamlet_execution == 'combined' else f"bixel_{index:06d}"
                    transport_histories = self.histories
                    job = {"job_id": name, "bixel_index": index, "beam_index": bi,
                           "ray_index": 0 if self.beamlet_execution == 'combined' else ri,
                           "beamlet_index": 0 if self.beamlet_execution == 'combined' else li,
                           "column_index": index, "members": members,
                           "normalization_histories": self.histories,
                           "gantry_angle_deg": float(beam.gantry_angle),
                           "couch_angle_deg": float(beam.couch_angle),
                           "iso_center_lps_mm": np.asarray(beam.iso_center).tolist(),
                           "sad_mm": float(beam.sad), "bixel_width_mm": float(beam.bixel_width),
                           "histories": transport_histories, "seed": self.seed + index - 1,
                           "parameter_file": f"jobs/{name}.txt", "scorer": f"Dose_{name}",
                           "output": f"results/{name}.csv",
                           "source": source, "source_name": description.source_name,
                           "source_description": json_data(asdict(description)), "devices": device_records}
                    if self.enable_opengl and not self.water:
                        job["visualization_slice"] = axial_slice_for_isocenter(ct, beam.iso_center)
                    jobs.append(job)
                    sources.append(text)
        from ..steering import beam_geometry_records
        signature = {
            "beamlet_execution": self.beamlet_execution,
            "beam_geometry": beam_geometry_records(stf),
            "normalization": 'original_accelerator_history' if phase_space else 'independent_primary_photon',
            "implementation_sha256": implementation_hashes(),
            "ct_grid": grid_dict(ct.grid), "ct_hu_sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
            "dicom": dicom_info, "hu_range": [float(raw.min()), float(raw.max())],
            "dose_grid": grid_dict(dose_grid), "jobs": jobs, "sources": sources,
            "material_sha256": hashlib.sha256(material).hexdigest(), "water": self.water,
            "geometry_text": self.geometry.text,
            "geometry_configuration": self.geometry_snapshot.text if self.geometry_snapshot else None,
            "assets": {k: bundle_io.sha256(v) for k, v in assets.items()},
            "num_threads": self.num_threads, "enable_opengl": self.enable_opengl, "world_half_mm": world_half + 500.,
        }
        return signature, dicom_files, material, assets, center

    def prepare_jobs(self, ct, cst, stf, *, bundle_dir=None, provenance=None):
        """Write a portable bundle, or validate/reuse an identical existing one."""
        if cst.ct_image.grid != ct.grid:
            raise ValueError("Structure set and CT grids differ")
        from ..geometry.assembly import GeometryCollisionError, TreatmentHead
        try:
            definition, dicom_files, material, assets, center = self._definition(ct, stf)
        except GeometryCollisionError as error:
            from ..geometry.diagnostics import write_collision_diagnostics
            head = next(d for d in self.devices if isinstance(d, TreatmentHead))
            output = write_collision_diagnostics(ct, cst, stf, head, bundle_dir or self.bundle_dir,
                                                 patient_report=not self.water)
            conflict = GeometryCollisionError(
                f'{error}\nNo simulation bundle was written. Patient diagnostics:\n'
                f'  Report: {output / "geometry.pdf"}\n'
                f'  Summary: {output / "collisions.txt"}\n'
                f'  Details: {output / "collisions.json"}')
            conflict.diagnostics_dir = output
            raise conflict from None
        definition["planning"] = planning_snapshot(
            self._plan_info, stf, cst)
        for record in definition['planning']['beams']:
            record['job_ids'] = [j['job_id'] for j in definition['jobs'] if j['beam_index']==record['beam_index']]
        request_id = bundle_io.digest_json(definition)
        # Hash the configuration before adding its fingerprint to scorer names.
        for job in definition["jobs"]:
            job["scorer"] += "_" + request_id
        root = Path(bundle_dir or self.bundle_dir).resolve()
        jobs = definition["jobs"]
        if root.exists() and any(root.iterdir()):
            old = bundle_io.load_manifest(root)
            if old["request_id"] != request_id:
                raise ValueError("Existing bundle describes different inputs/settings; choose a new directory")
            self._report(ct, cst, stf, root)
            return root
        root.mkdir(parents=True, exist_ok=True)
        for name in ("inputs", "jobs", "results", "derived"):
            (root / name).mkdir(exist_ok=True)
        for name, source in dicom_files.items():
            target = bundle_io.bundle_path(root, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            if bundle_io.sha256(target) != definition["dicom"]["files"][name]:
                raise ValueError("Source DICOM changed during preparation; prepare a new bundle")
        if self.geometry_snapshot:
            (root / "inputs/geometry-config.toml").write_text(self.geometry_snapshot.text)
        (root / "inputs/beam_geometry.json").write_text(json.dumps(definition["beam_geometry"], indent=2, allow_nan=False))
        (root / "inputs/materials.txt").write_bytes(material)
        for name in ("OpenTOPAS-LICENSE.txt", "PROVENANCE.md"):
            (root / "inputs" / name).write_bytes(resources.files("minibeam.materials").joinpath("data", name).read_bytes())
        for name, content in assets.items():
            target = bundle_io.bundle_path(root, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(content, target)
            if bundle_io.sha256(target) != definition["assets"][name]:
                raise ValueError("Asset changed during preparation; prepare a new bundle")
        (root / "inputs/devices.txt").write_text("includeFile = inputs/common.txt\n" + self.geometry.text + "\n")
        common = patient_parameters(ct, water=self.water, num_threads=self.num_threads,
                                    world_half=definition["world_half_mm"], enable_opengl=self.enable_opengl)
        (root / "inputs/common.txt").write_text(common)
        for job, source in zip(jobs, definition["sources"]):
            lines = ['includeFile = inputs/devices.txt', source,
                     f'i:Ts/Seed = {job["seed"]}',
                     f'i:So/{job["source_name"]}/NumberOfHistoriesInRun = {job["histories"]}',
                     scorer_parameters(job, definition["dose_grid"])]
            if "visualization_slice" in job:
                # Graphics visibility only: never restrict transport voxels.
                index = job["visualization_slice"]["slice_index_1based"]
                lines += [f'iv:Ge/Patient/ShowSpecificSlicesZ = 1 {index}',
                          'i:Gr/ShowOnlyOutlineIfVoxelCountExceeds = 2147483647']
            (root / job["parameter_file"]).write_text("\n".join(lines) + "\n")
        # Input-only inventory. The manifest signs this inventory and all job metadata.
        files = {p.relative_to(root).as_posix(): bundle_io.sha256(p)
                 for p in sorted(root.rglob("*")) if p.is_file()}
        versions = {name: metadata.version(name) for name in
                    ("pyRadPlan", "numpy", "scipy", "SimpleITK", "pydicom")}
        manifest = {"schema_version": 3, "history_budget": "per_job", "beamlet_execution": self.beamlet_execution, "request_id": request_id, "files": files,
                    "implementation_sha256": definition["implementation_sha256"],
                    "enable_opengl": self.enable_opengl,
                    "versions": versions, "topas_target": "4.2.p3", "jobs": jobs,
                    "planning": definition["planning"],
                    "beam_geometry": definition["beam_geometry"],
                    "geometry_configuration": {"input": "inputs/geometry-config.toml",
                        "sha256": self.geometry_snapshot.sha256} if self.geometry_snapshot else None,
                    "ct_grid": definition["ct_grid"], "dose_grid": definition["dose_grid"],
                    "normalization": definition['normalization'],
                    "units": "Gy/original accelerator history" if definition['normalization']=='original_accelerator_history' else "Gy/primary photon",
                    "weight_units": "original accelerator histories" if definition['normalization']=='original_accelerator_history' else "primary photons",
                    "forward_uncertainty": "unavailable: shared phase-space histories" if definition['normalization']=='original_accelerator_history' else "independent columns",
                    "native_voxel_order": "C on [z,y,x]", "exchange_index_base": 1,
                    "lps_to_topas_translation_mm": (-center).tolist(),
                    "patient_component": "TsBox" if self.water else "TsDicomPatient",
                    "dicom": definition["dicom"],
                    "hu_conversion": "TOPAS native DICOM rescaling; Schneider HU-to-material mapping",
                    "material_mode": "water" if self.water else "Schneider",
                    "hu_range": definition["hu_range"],
                    "provenance": provenance or {}}
        manifest['column_mapping'] = [dict(column_index=j['column_index'], beam_index=j['beam_index'],
            job_id=j['job_id'], members=j['members'], kind='beam_aggregate' if self.beamlet_execution=='combined' else 'beamlet') for j in jobs]
        manifest["bundle_id"] = bundle_io.digest_json(manifest)
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))
        self._report(ct, cst, stf, root)
        return root

    def _report(self, ct, cst, stf, root):
        from ..geometry.assembly import TreatmentHead
        heads = [d for d in self.devices if isinstance(d, TreatmentHead)]
        if not self.water or self.geometry_snapshot or heads:
            path = root / "derived/geometry.pdf"
            if not path.exists():
                head = heads[0] if heads else None
                if self.water:
                    from ..geometry.report import write_geometry_report
                    write_geometry_report(ct, stf, head, path)
                else:
                    from ..geometry.patient_report import write_patient_geometry_report
                    write_patient_geometry_report(ct, cst, stf, head, path,
                        target_names=tuple(v.name for v in cst.vois if v.voi_type == 'TARGET'),
                        source_records=[dict(j['source'],beam_index=j['beam_index']) for j in
                            bundle_io.load_manifest(root)['jobs']] if (root/'manifest.json').exists() else None)
