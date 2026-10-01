"""Patient import, native planning, crop previews and geometry setup stages."""
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pydicom
import SimpleITK as sitk
from pyRadPlan.cst import create_voi
from pyRadPlan import load_patient, PhotonPlan, generate_stf

from minibeam.geometry.assembly import GeometryCollisionError
from minibeam.workflow.artifacts import prepare_project
from minibeam.workflow.study import run_study, effective_shift_fractions, effective_rotations
from minibeam.workflow.geometry import resolve_geometry
from minibeam.workflow.planning import (photon_machine, topas_settings, positive_number,
    boolean, check_matrad_grids, workload_estimate)
from minibeam.steering import enrich_stf




def read_roi_metadata(directory, cst):
    """Read original ROI identifiers without importing or rasterizing contours."""
    rois = []
    for path in sorted(Path(directory).rglob("*")):
        if not path.is_file():
            continue
        try:
            dataset = pydicom.dcmread(path, stop_before_pixels=True)
        except pydicom.errors.InvalidDicomError:
            continue
        if getattr(dataset, "Modality", None) == "RTSTRUCT":
            rois.extend({"name": str(roi.ROIName), "number": int(roi.ROINumber)}
                        for roi in dataset.StructureSetROISequence)
    imported_names = {voi.name for voi in cst.vois}
    return {
        "original_dicom_rois": rois,
        "omitted_rois": [roi for roi in rois if roi["name"] not in imported_names],
    }


def select_target(cst, target_name):
    """Make one nonempty ROI the sole TARGET in the native StructureSet, in place.

    Other imported TARGET structures become OARs so they cannot enlarge the
    steering field or shift the default isocenter. Masks and names are retained.
    """
    matches = [voi for voi in cst.vois if voi.name == target_name]
    if len(matches) != 1 or not np.any(sitk.GetArrayViewFromImage(matches[0].mask)):
        raise ValueError("Selected target must exist uniquely and be nonempty")
    for index, voi in enumerate(cst.vois):
        if voi.name == target_name:
            kind = "TARGET"
        elif voi.voi_type == "TARGET":
            kind = "OAR"
        else:
            continue
        if kind != voi.voi_type:
            data = voi.model_dump(exclude_computed_fields=True)
            data.update(voi_type=kind, overlap_priority=0 if kind == "TARGET" else 5)
            cst.vois[index] = create_voi(data)


@dataclass
class PatientSettings:
    """Explicit planning inputs; validation remains in the planning functions."""
    dicom_dir: str
    target: str | None
    gantry_angles: list[float]
    couch_angles: list[float] | None
    sad_mm: float
    bixel_width_mm: float
    only_central_beamlet: bool
    beamlet_execution: str
    iso_center_lps_mm: tuple[float, float, float] | None
    source_type: str
    phase_space_file_bases: list[str]
    histories_per_job: int
    topas_threads_per_job: int
    enable_collimator: bool
    geometry_config: str | None
    collimator_rotation_deg: list[float]
    slit_entrance_width_mm: float
    nominal_entrance_ctc_mm: float
    collimator_shift_fractions: list[float]
    ct_crop_voxels: dict | None
    enforce_ct_crop_protection: bool
    dose_spacing_mm: tuple[float, float, float] | None
    enable_opengl: bool
    phase_space_selection: str = "field"


# SHARED PLANNING HELPERS

def configure_plan(settings, *, project_dir):
    """Configure a native PhotonPlan from explicit patient settings.

    This is an idealized geometry-only machine: 6.0 is a nominal steering label.
    TOPAS uses the selected source independently of that label. No clinical machine kernels or MU calibration are involved.
    """
    if settings.target is None or settings.gantry_angles is None:
        raise SystemExit("Set TARGET and GANTRY_ANGLES in patient_workflow.py first")
    boolean('ONLY_CENTRAL_BEAMLET', settings.only_central_beamlet)
    gantry = np.asarray(settings.gantry_angles, dtype=float)
    couch = np.zeros_like(gantry) if settings.couch_angles is None else np.asarray(settings.couch_angles, dtype=float)
    if gantry.ndim != 1 or not gantry.size or couch.shape != gantry.shape or not np.isfinite([gantry, couch]).all():
        raise ValueError("Provide matching, nonempty finite gantry/couch angle lists")
    positive_number('BIXEL_WIDTH_MM', settings.bixel_width_mm)
    plan = PhotonPlan(machine=photon_machine(settings.sad_mm), num_of_fractions=1)
    plan.prop_stf = {"generator": "photonSingleBixel" if settings.only_central_beamlet else "photonIMRT",
                     "gantry_angles": gantry.tolist(),
                     "couch_angles": couch.tolist(), "bixel_width": settings.bixel_width_mm, "add_margin": False}
    if settings.iso_center_lps_mm is not None:
        iso = np.asarray(settings.iso_center_lps_mm, dtype=float)
        if iso.shape != (3,) or not np.isfinite(iso).all():
            raise ValueError("Isocenter must be three finite LPS coordinates in mm")
        plan.prop_stf["iso_center"] = iso.reshape(1, 3)
    # Without an override generate_stf uses native cst.target_center_of_mass().
    plan.prop_dose_calc = topas_settings(
        project_dir=project_dir, source_type=settings.source_type, phase_space_file_bases=settings.phase_space_file_bases,
        phase_space_selection=settings.phase_space_selection,
        histories=settings.histories_per_job, num_threads=settings.topas_threads_per_job, execution=settings.beamlet_execution,
        water=False, enable_opengl=settings.enable_opengl, dose_spacing_mm=settings.dose_spacing_mm)
    plan.prop_dose_calc["dicom_dir"] = str(settings.dicom_dir)
    boolean('ENFORCE_CT_CROP_PROTECTION', settings.enforce_ct_crop_protection)
    plan.prop_dose_calc["enforce_ct_crop_protection"] = settings.enforce_ct_crop_protection
    if settings.ct_crop_voxels is not None:
        plan.prop_dose_calc["ct_crop_voxels"] = settings.ct_crop_voxels
    return plan


def planning_directory(settings, project_dir):
    """Check the requested single-project/study layout without writing anything."""
    root = Path(project_dir).resolve()
    fractions = effective_shift_fractions(
        enable_collimator=settings.enable_collimator, values=settings.collimator_shift_fractions)
    rotations = effective_rotations(enable_collimator=settings.enable_collimator, values=settings.collimator_rotation_deg)
    if not (fractions.size or rotations.size) and (root / "study.json").exists():
        raise SystemExit("This is a study directory; restore its enabled collimator and rotation/shift settings or choose a new project directory")
    if (fractions.size or rotations.size) and (root / "manifest.json").exists():
        raise SystemExit("This is a single-project directory; use its study parent or a new study root")
    print("Collimator: enabled" if settings.enable_collimator else
          "Collimator: disabled; collimator rotations, shifts and slit overrides are inactive")
    print(f"TOPAS threads per job: {settings.topas_threads_per_job}")
    return root, fractions, rotations


def import_patient(settings):
    """Load native anatomy and retain original DICOM ROI identifiers."""
    ct, cst = load_patient(str(settings.dicom_dir))
    metadata = read_roi_metadata(settings.dicom_dir, cst)
    print(f"Native import: CT {ct.size}; {len(cst.vois)} structures. Omitted: {metadata['omitted_rois']}")
    return ct, cst, metadata


def build_steering(settings, ct, cst, project_dir):
    """Select the target and generate native steering once for all setups."""
    plan = configure_plan(settings, project_dir=project_dir)
    select_target(cst, settings.target)
    from minibeam.workflow.crop_preview import preview_crop
    try:
        _, grid, crop = preview_crop(ct, cst, project_dir, settings.ct_crop_voxels, settings.dose_spacing_mm,
                                     (settings.target,), settings.enforce_ct_crop_protection)
    except ValueError as error:
        raise SystemExit(f"CT crop / dose-grid preparation stopped: {error}") from None
    print(f"Transport crop: {crop['retained_ranges']}; dose spacing: {crop['actual_dose_spacing_mm']} mm")
    check_matrad_grids(ct.grid, grid)
    stf = generate_stf(ct, cst, plan)
    if settings.iso_center_lps_mm is None:
        x, y, z = stf.beams[0].iso_center
        print(f"Isocenter from target '{settings.target}' centroid (LPS, mm): "
              f"x={x:.3f}, y={y:.3f}, z={z:.3f}")
    history_unit = "original accelerator histories" if settings.source_type == "phase_space" else "primary photons"
    print(f"Source: {settings.source_type}; {settings.histories_per_job:,} {history_unit} per job")
    return plan, stf, grid


def process_setups(settings, stage, root, fractions, rotations, ct, cst, metadata):
    """Inspect or prepare resolved hardware setups with shared native steering."""
    geometry = resolve_geometry(config_path=settings.geometry_config, enable_collimator=settings.enable_collimator)
    plan, stf, grid = build_steering(settings, ct, cst, root)
    try:
        if fractions.size or rotations.size:
            run_study(stage, root, ct, cst, plan, stf, metadata, target=settings.target,
                      slit_width_mm=settings.slit_entrance_width_mm, nominal_ctc_mm=settings.nominal_entrance_ctc_mm,
                      collimator_shift_fractions=fractions.tolist(), geometry=geometry,
                      collimator_rotations=rotations.tolist())
        else:
            if settings.enable_collimator:
                print(f"Collimator rotation: baseline Z {geometry.aperture.rotation_z_deg:g} deg + additional 0 deg = resolved Z {geometry.aperture.rotation_z_deg:g} deg")
            stf = enrich_stf(stf, geometry)
            if stage == "inspect":
                count = len(stf.beams) if settings.beamlet_execution == "combined" else stf.total_number_of_bixels
                print(f"Single configured run: {workload_estimate(grid, count)}")
            else:
                prepare_project(root, ct, cst, plan, stf, metadata)
    except GeometryCollisionError as error:
        raise SystemExit(f"Geometry preparation stopped: {error}") from None


# INSPECT: list structures and optionally estimate the configured simulation.
def inspect(settings, project_dir):
    """List anatomy, preview a configured crop, and estimate without simulation inputs."""
    root, fractions, rotations = planning_directory(settings, project_dir)
    ct, cst, metadata = import_patient(settings)
    for roi in metadata["original_dicom_rois"]:
        print(f"{roi['number']:3d}  {roi['name']}")
    if settings.target is not None:
        process_setups(settings, "inspect", root, fractions, rotations, ct, cst, metadata)
    elif settings.ct_crop_voxels is not None:
        from minibeam.workflow.crop_preview import preview_crop
        try:
            preview_crop(ct, cst, root, settings.ct_crop_voxels, settings.dose_spacing_mm, enforce_protection=settings.enforce_ct_crop_protection)
        except ValueError as error:
            raise SystemExit(f"CT crop preparation stopped: {error}") from None


# PREPARE: create portable TOPAS inputs; execution happens on the cluster.
def prepare(settings, project_dir):
    """Import, plan, and write one run or the configured collimator setups."""
    root, fractions, rotations = planning_directory(settings, project_dir)
    ct, cst, metadata = import_patient(settings)
    process_setups(settings, "prepare", root, fractions, rotations, ct, cst, metadata)
