"""Patient planning: edit the settings below, then choose a workflow stage.

    python patient_workflow.py inspect
    python patient_workflow.py prepare --project projects/patient-example
    # Execute the generated TOPAS jobs on the cluster and return their CSV files.
    python patient_workflow.py collect --project projects/patient-example
    python patient_workflow.py forward --project projects/patient-example

ENABLE_COLLIMATOR=False produces one run without the slit collimator.
When enabled, COLLIMATOR_SHIFT_FRACTIONS selects the layout: [] writes one configured run;
nonempty arrays create one shift folder per fraction.

Collect and forward read the saved run, independently of current planning settings.
"""
import argparse
from pathlib import Path

import numpy as np
from pyRadPlan import load_patient, PhotonPlan, generate_stf
from pyRadPlan.machines import PhotonLINAC

from minibeam import TOPASPhotonEngine
from minibeam.geometry.assembly import GeometryCollisionError
from minibeam.geometry.coordinates import scoring_grid
from minibeam.workflow.patient import read_roi_metadata, select_target
from minibeam.workflow.execution import run_stage
from minibeam.workflow.study import run_study, effective_shift_fractions
from minibeam.workflow.geometry import resolve_geometry
from minibeam.steering import enrich_stf


# EDITABLE SETTINGS
# Paths are relative to the directory from which you run this script.

# Patient and target
DICOM_DIR = "dicom_9306087_fine"
TARGET = "PTV2017fw"
WATER = False

# Native steering and job grouping
GANTRY_ANGLES = [45.0, 135.0, 225.0, 315.0]
COUCH_ANGLES = None  # zero for each gantry angle
SAD_MM = 1000.0
BIXEL_WIDTH_MM = 5.0
ONLY_CENTRAL_BEAMLET = False  # False: all target-selected beamlets
BEAMLET_EXECUTION = "combined"  # "separate" or "combined": one aggregate job per beam
# Physical coordinates in mm: +x Left, +y Posterior, +z Superior.
# None uses the selected target's centroid, not the center of the CT volume.
ISO_CENTER_LPS_MM = None

# Source and simulation statistics
SOURCE_TYPE = "phase_space"  # "point" or "phase_space"
PHASE_SPACE_FILE_BASE = "~/Local/MCGPU/TOPAS/Elekta_Precise_6MV/ELEKTA_PRECISE_6mv_part1"
# Photons for point; original accelerator histories for phase_space.
# Try 1_000_000 for an initial phase-space validation run.
HISTORIES_PER_JOB = 10_000_000

# Hardware: disabling the collimator ignores all shift/slit overrides.
ENABLE_COLLIMATOR = False  # overrides TOML aperture.enabled; MLC and jaws unchanged
GEOMETRY_CONFIG = None  # TOML path for every setup; None uses geometry/config.toml
# When enabled, [] uses TOML aperture dimensions and placement.
# Each shift contains all GANTRY_ANGLES / COUCH_ANGLES configured above.
SLIT_ENTRANCE_WIDTH_MM = 2.0
NOMINAL_ENTRANCE_CTC_MM = 6.0  # slit width + configured physical blade thickness
# Slit collimator only; MLC and jaws remain fixed.
# Fractions of NOMINAL_ENTRANCE_CTC_MM, translated along beam-frame X.
COLLIMATOR_SHIFT_FRACTIONS = [0.0, 0.25, 0.50, 0.75]

# Scoring and visualization
DOSE_SPACING_MM = None  # preserve the native CT scoring grid
ENABLE_OPENGL = False  # interactive TOPAS viewer; keep False for cluster batch jobs

# Forward dose: scalar or vector in saved job order, using saved exposure units.
FORWARD_WEIGHT_PER_BIXEL = 1.0


# SHARED PLANNING HELPERS

def configure_plan(*, project_dir):
    """Configure a native PhotonPlan from the editable settings above.

    This is an idealized geometry-only machine: 6.0 is a nominal steering label.
    TOPAS uses the selected source independently of that label. No clinical machine kernels or MU calibration are involved.
    """
    if TARGET is None or GANTRY_ANGLES is None:
        raise SystemExit("Set TARGET and GANTRY_ANGLES in patient_workflow.py first")
    if BEAMLET_EXECUTION not in {"separate", "combined"}:
        raise ValueError("BEAMLET_EXECUTION must be separate or combined")
    if type(ONLY_CENTRAL_BEAMLET) is not bool:
        raise ValueError("ONLY_CENTRAL_BEAMLET must be a Boolean: True (central beamlet) or False (all target-selected beamlets)")
    gantry = np.asarray(GANTRY_ANGLES, dtype=float)
    couch = np.zeros_like(gantry) if COUCH_ANGLES is None else np.asarray(COUCH_ANGLES, dtype=float)
    if gantry.ndim != 1 or not gantry.size or couch.shape != gantry.shape or not np.isfinite([gantry, couch]).all():
        raise ValueError("Provide matching, nonempty finite gantry/couch angle lists")
    if not all(np.isfinite(x) and x > 0 for x in (SAD_MM, BIXEL_WIDTH_MM)):
        raise ValueError("SAD and bixel width must be positive and finite")
    # Native steering defaults to 6.0; this label does not set TOPAS photon energies.
    machine = PhotonLINAC(version=2, name="IdealSpectrumPhoton", energies=np.array([6.0]),
                          sad=SAD_MM, scd=SAD_MM / 2)
    machine_data = machine.model_dump(exclude_none=True, exclude_computed_fields=True)
    machine_data["meta"] = {"radiation_mode": "photons"}
    plan = PhotonPlan(machine=machine_data, num_of_fractions=1)
    plan.prop_stf = {"generator": "photonSingleBixel" if ONLY_CENTRAL_BEAMLET else "photonIMRT",
                     "gantry_angles": gantry.tolist(),
                     "couch_angles": couch.tolist(), "bixel_width": BIXEL_WIDTH_MM, "add_margin": False}
    if ISO_CENTER_LPS_MM is not None:
        iso = np.asarray(ISO_CENTER_LPS_MM, dtype=float)
        if iso.shape != (3,) or not np.isfinite(iso).all():
            raise ValueError("Isocenter must be three finite LPS coordinates in mm")
        plan.prop_stf["iso_center"] = iso.reshape(1, 3)
    # Without an override generate_stf uses native cst.target_center_of_mass().
    plan.prop_dose_calc = {"engine": "TOPASPhoton", "bundle_dir": str(project_dir),
                          "histories": HISTORIES_PER_JOB, "beamlet_execution": BEAMLET_EXECUTION, "water": WATER, "enable_opengl": ENABLE_OPENGL}
    if SOURCE_TYPE not in {'point','phase_space'}:
        raise ValueError('SOURCE_TYPE must be point or phase_space')
    plan.prop_dose_calc['source_config'] = {'type': SOURCE_TYPE}
    if SOURCE_TYPE == 'phase_space':
        plan.prop_dose_calc['source_config']['file_base'] = str(Path(PHASE_SPACE_FILE_BASE).expanduser())
    if not WATER:
        plan.prop_dose_calc["dicom_dir"] = str(DICOM_DIR)
    if DOSE_SPACING_MM is not None:
        plan.prop_dose_calc["dose_spacing_mm"] = DOSE_SPACING_MM
    return plan



def planning_directory(project_dir):
    """Check the requested single-project/study layout without writing anything."""
    root = Path(project_dir).resolve()
    fractions = effective_shift_fractions(
        enable_collimator=ENABLE_COLLIMATOR, values=COLLIMATOR_SHIFT_FRACTIONS)
    if not fractions.size and (root / "study.json").exists():
        raise SystemExit("This is a study directory; restore its enabled collimator and shift settings or choose a new project directory")
    if fractions.size and (root / "manifest.json").exists():
        raise SystemExit("This is a single-project directory; use its study parent or a new study root")
    print("Collimator: enabled" if ENABLE_COLLIMATOR else
          "Collimator: disabled; collimator shifts and slit overrides are inactive")
    return root, fractions


def import_patient():
    """Load native anatomy and retain original DICOM ROI identifiers."""
    ct, cst = load_patient(str(DICOM_DIR))
    metadata = read_roi_metadata(DICOM_DIR, cst)
    print(f"Native import: CT {ct.size}; {len(cst.vois)} structures. Omitted: {metadata['omitted_rois']}")
    return ct, cst, metadata


def build_steering(ct, cst, project_dir):
    """Select the target and generate native steering once for all setups."""
    plan = configure_plan(project_dir=project_dir)
    select_target(cst, TARGET)
    grid = scoring_grid(ct, DOSE_SPACING_MM)
    if ct.size[0] != ct.size[1] or grid.dimensions[0] != grid.dimensions[1]:
        raise ValueError("matRad export requires square transverse CT and dose grids with pyRadPlan 0.5.0")
    stf = generate_stf(ct, cst, plan)
    if ISO_CENTER_LPS_MM is None:
        x, y, z = stf.beams[0].iso_center
        print(f"Isocenter from target '{TARGET}' centroid (LPS, mm): "
              f"x={x:.3f}, y={y:.3f}, z={z:.3f}")
    history_unit = "original accelerator histories" if SOURCE_TYPE == "phase_space" else "primary photons"
    print(f"Source: {SOURCE_TYPE}; {HISTORIES_PER_JOB:,} {history_unit} per job")
    return plan, stf, grid


def process_setups(stage, root, fractions, ct, cst, metadata):
    """Inspect or prepare resolved hardware setups with shared native steering."""
    geometry = resolve_geometry(config_path=GEOMETRY_CONFIG, enable_collimator=ENABLE_COLLIMATOR)
    plan, stf, grid = build_steering(ct, cst, root)
    try:
        if fractions.size:
            run_study(stage, root, ct, cst, plan, stf, metadata, target=TARGET,
                      slit_width_mm=SLIT_ENTRANCE_WIDTH_MM, nominal_ctc_mm=NOMINAL_ENTRANCE_CTC_MM,
                      collimator_shift_fractions=fractions.tolist(), geometry=geometry,
                      weight_per_bixel=FORWARD_WEIGHT_PER_BIXEL)
        else:
            stf = enrich_stf(stf, geometry)
            if stage == "inspect":
                count = len(stf.beams) if BEAMLET_EXECUTION == "combined" else stf.total_number_of_bixels
                voxels = int(np.prod(grid.dimensions))
                print(f"Single configured run: {count} jobs; dense {voxels*count*8/1024**3:.3f} GiB; "
                      f"CSV estimate {voxels*count*100/1024**3:.3f} GiB")
            else:
                run_stage("prepare", root, ct, cst, plan, stf, metadata,
                          weight_per_bixel=FORWARD_WEIGHT_PER_BIXEL)
    except GeometryCollisionError as error:
        raise SystemExit(f"Geometry preparation stopped: {error}") from None


# INSPECT: list structures and optionally estimate the configured simulation.
def inspect(project_dir):
    """Read patient data and display estimates without creating run files."""
    root, fractions = planning_directory(project_dir)
    ct, cst, metadata = import_patient()
    for roi in metadata["original_dicom_rois"]:
        print(f"{roi['number']:3d}  {roi['name']}")
    if TARGET is not None:
        process_setups("inspect", root, fractions, ct, cst, metadata)


# PREPARE: create portable TOPAS inputs; execution happens on the cluster.
def prepare(project_dir):
    """Import, plan, and write one run or the configured collimator setups."""
    root, fractions = planning_directory(project_dir)
    ct, cst, metadata = import_patient()
    process_setups("prepare", root, fractions, ct, cst, metadata)


# COLLECT: assemble dose matrices from saved bundles and completed CSV outputs.
def collect(project_dir):
    """Use saved planning snapshots, independent of current planning settings."""
    from minibeam.workflow.collection import collect_saved_run
    try:
        collect_saved_run(project_dir, "collect", weight_per_bixel=FORWARD_WEIGHT_PER_BIXEL)
    except (ValueError, OSError, RuntimeError) as error:
        raise SystemExit(str(error)) from None


# FORWARD: apply exposure weights to saved dose columns and sum the dose.
def forward(project_dir):
    """Reconstruct dose using saved source units and the editable weights."""
    from minibeam.workflow.collection import collect_saved_run
    try:
        collect_saved_run(project_dir, "forward", weight_per_bixel=FORWARD_WEIGHT_PER_BIXEL)
    except (ValueError, OSError, RuntimeError) as error:
        raise SystemExit(str(error)) from None


# COMMAND-LINE ENTRY POINT
def parse_arguments(argv=None):
    """Choose the stage and output folder; patient settings live above."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["inspect", "prepare", "collect", "forward"])
    parser.add_argument("--project", dest="project_dir", help="project directory path: shift subfolders only when the collimator is enabled with nonempty shift fractions")
    args = parser.parse_args(argv)
    if args.stage in {"collect", "forward"} and args.project_dir is None:
        parser.error("--project is required for collect and forward")
    if args.project_dir is None:
        args.project_dir = "projects/patient-slit-study" if effective_shift_fractions(enable_collimator=ENABLE_COLLIMATOR, values=COLLIMATOR_SHIFT_FRACTIONS).size else "projects/patient"
    return args


def main(argv=None):
    args = parse_arguments(argv)
    stages = {"inspect": inspect, "prepare": prepare, "collect": collect, "forward": forward}
    stages[args.stage](args.project_dir)


if __name__ == "__main__":
    main()
