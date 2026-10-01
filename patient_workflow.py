"""Patient planning: edit the settings below, then choose a workflow stage.

    python patient_workflow.py inspect
    python patient_workflow.py prepare --project projects/patient-example
    # Execute the generated TOPAS jobs on the cluster and return their CSV files.
    python patient_workflow.py collect --project projects/patient-example
    python patient_workflow.py forward --project projects/patient-example

ENABLE_COLLIMATOR=False produces one run without the slit collimator.
When enabled, rotation and shift arrays select the setup combinations.
Both arrays empty write one configured run; otherwise each combination has its own folder.

Collect and forward read the saved run, independently of current planning settings."""
from copy import deepcopy
from minibeam.workflow import patient
from minibeam.workflow.collection import collect_saved_run
from minibeam.workflow.cli import parse_stage_arguments, dispatch


# EDITABLE SETTINGS
# Paths are relative to the directory from which you run this script.

# Patient and target
DICOM_DIR = "dicom_9306087_fine"
TARGET = "PTV2017fw"

# Native steering and job grouping
GANTRY_ANGLES = [0.0] # [45.0, 135.0, 225.0, 315.0]
COUCH_ANGLES = None  # zero for each gantry angle
SAD_MM = 1000.0
BIXEL_WIDTH_MM = 5.0
ONLY_CENTRAL_BEAMLET = False  # False: all target-selected beamlets
BEAMLET_EXECUTION = "separate"  # "separate" or "combined": one aggregate job per beam
# Physical coordinates in mm: +x Left, +y Posterior, +z Superior.
# None uses the selected target's centroid, not the center of the CT volume.
ISO_CENTER_LPS_MM = None

# Source and simulation statistics
SOURCE_TYPE = "phase_space"  # "point" or "phase_space"
# Ordered, distinct original-history batches; each basename has .header and .phsp.
# Shared-history particle splits are unsupported. Validation cannot prove independence.
# Applies only to SOURCE_TYPE = "phase_space".
# "field": replay particles projected into the selected bixels.
# "all_forward": replay every downstream particle in the original-history prefix.
# Requires combined execution; physical MLC/jaws/collimator define irradiation.
# Transport scatter is included in BOTH modes. Empty histories still count in
# HISTORIES_PER_JOB and dose normalization; this does not select N particles.
PHASE_SPACE_SELECTION = "field"
PHASE_SPACE_FILE_BASES = [
    "~/Local/MCGPU/TOPAS/Elekta_Precise_6MV/ELEKTA_PRECISE_6mv_part1",
    "~/Local/MCGPU/TOPAS/Elekta_Precise_6MV/ELEKTA_PRECISE_6mv_part2",
    "~/Local/MCGPU/TOPAS/Elekta_Precise_6MV/ELEKTA_PRECISE_6mv_part3",
    "~/Local/MCGPU/TOPAS/Elekta_Precise_6MV/ELEKTA_PRECISE_6mv_part4",
]
# Photons for point; total original histories across the ordered files for phase_space.
# Consume part1 first, then later parts as needed; no recycling.
# Try 1_000_000 for an initial phase-space validation run.
HISTORIES_PER_JOB = 680_000_000
# CPU threads per TOPAS job, not the number of simultaneous jobs.
TOPAS_THREADS_PER_JOB = 4

# Hardware: disabling the collimator ignores all shift/slit overrides.
ENABLE_COLLIMATOR = True  # overrides TOML aperture.enabled; MLC and jaws unchanged
GEOMETRY_CONFIG = None  # TOML path for every setup; None uses geometry/config.toml
# Rotate only the slit collimator (frame and blades); MLC/jaws stay fixed.
# Entries are additional Z rotations relative to the TOML baseline.
# With packaged baseline -90° and zero aperture X/Y tilt:
#   0°: slit length along beam X, between the MLC banks.
#  90°: original orientation, with slit length along beam Y.
#
# Rotate about the collimator center FIRST, then shift across the rotated slits.
# Positive shift: beam +Y at 0°, beam +X at 90°.
# Each rotation is combined with every shift; e.g. [0.0, 45.0, 90.0].
# [] and [0.0] have identical geometry; [] adds no rotation folder label.
COLLIMATOR_ROTATION_DEG = [0.0]
# Empty shifts keep TOML slit dimensions/translation; rotation entries still apply.
# Each rotation/shift setup contains all GANTRY_ANGLES / COUCH_ANGLES above.
SLIT_ENTRANCE_WIDTH_MM = 2.0
NOMINAL_ENTRANCE_CTC_MM = 6.0  # slit width + configured physical blade thickness
# Slit collimator only; MLC and jaws remain fixed.
# Shift distance = fraction × NOMINAL_ENTRANCE_CTC_MM.
# Translation follows the rotated slit-spacing axis, across the slits.
COLLIMATOR_SHIFT_FRACTIONS = [0.0] # [0.0, 0.25, 0.50, 0.75]

# Scoring and visualization
# Optional CT crop in original voxel indices, ordered X/Y/Z.
# Uses Python indexing: ZERO-BASED, start INCLUDED, stop EXCLUDED.
# Example: x=(40, 300) retains indices 40..299 (260 voxels).
# Equivalent MATLAB indices and inclusive TOPAS limits: 41..300.
# These indices are NOT DICOM InstanceNumber values.
# CT_CROP_VOXELS = None retains the complete CT. Original DICOM files remain unchanged.
CT_CROP_VOXELS = {"x": (60, 280), "y": (60, 280), "z": (0, 160)}
# Syntax illustration only, NOT safe bounds for this patient: it cuts CouchSurface.
# {"x": (40, 300), "y": (30, 290), "z": (0, 160)}
# True: reject clipping any ROI or material flagged by the air checks.
# False: allow non-target ROI/material clipping; still report all findings.
# Selected targets remain protected in both modes.
ENFORCE_CT_CROP_PROTECTION = False
DOSE_SPACING_MM = None  # None -> preserve the native CT scoring grid; (1.0, 1.0, 0.5) -> request finer spacing
ENABLE_OPENGL = False  # interactive TOPAS viewer; keep False for cluster batch jobs

# Forward dose: scalar or vector in saved job order, using saved exposure units.
FORWARD_WEIGHT_PER_BIXEL = 1.0


def planning_settings():
    """Capture editable planning inputs; saved-result stages never call this."""
    return deepcopy(patient.PatientSettings(
        dicom_dir=DICOM_DIR,
        target=TARGET,
        gantry_angles=GANTRY_ANGLES,
        couch_angles=COUCH_ANGLES,
        sad_mm=SAD_MM,
        bixel_width_mm=BIXEL_WIDTH_MM,
        only_central_beamlet=ONLY_CENTRAL_BEAMLET,
        beamlet_execution=BEAMLET_EXECUTION,
        iso_center_lps_mm=ISO_CENTER_LPS_MM,
        source_type=SOURCE_TYPE,
        phase_space_file_bases=PHASE_SPACE_FILE_BASES,
        phase_space_selection=PHASE_SPACE_SELECTION,
        histories_per_job=HISTORIES_PER_JOB,
        topas_threads_per_job=TOPAS_THREADS_PER_JOB,
        enable_collimator=ENABLE_COLLIMATOR,
        geometry_config=GEOMETRY_CONFIG,
        collimator_rotation_deg=COLLIMATOR_ROTATION_DEG,
        slit_entrance_width_mm=SLIT_ENTRANCE_WIDTH_MM,
        nominal_entrance_ctc_mm=NOMINAL_ENTRANCE_CTC_MM,
        collimator_shift_fractions=COLLIMATOR_SHIFT_FRACTIONS,
        ct_crop_voxels=CT_CROP_VOXELS,
        enforce_ct_crop_protection=ENFORCE_CT_CROP_PROTECTION,
        dose_spacing_mm=DOSE_SPACING_MM,
        enable_opengl=ENABLE_OPENGL,
    ))


def inspect(project_dir):
    """Inspect the configured setup."""
    return patient.inspect(planning_settings(), project_dir)


def prepare(project_dir):
    """Prepare portable TOPAS inputs; execution remains external."""
    return patient.prepare(planning_settings(), project_dir)


def collect(project_dir):
    """Collect using saved planning and completed CSVs only."""
    return collect_saved_run(project_dir, "collect")


def forward(project_dir):
    """Reconstruct saved dose with the explicit exposure weights."""
    return collect_saved_run(project_dir, "forward", weight_per_bixel=FORWARD_WEIGHT_PER_BIXEL)


def parse_arguments(argv=None):
    return parse_stage_arguments(argv, description=__doc__, inspect_directory="projects/patient")


def main(argv=None):
    dispatch(parse_arguments(argv), {"inspect": inspect, "prepare": prepare,
                                     "collect": collect, "forward": forward})


if __name__ == "__main__":
    main()
