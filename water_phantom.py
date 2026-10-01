"""Square-field water measurement: inspect, prepare, collect, or forward.

Edit the settings below. Prepare with --project projects/water-phantom, execute
its TOPAS job externally, then collect or forward from that saved project."""
from copy import deepcopy
from minibeam.workflow import water
from minibeam.workflow.collection import collect_saved_run
from minibeam.workflow.cli import parse_stage_arguments, dispatch


# EDITABLE SETTINGS
# Water boundaries: X/Z centered on zero; entrance Y=0, downstream face Y=size.
PHANTOM_SIZE_MM = 200.0
PHANTOM_SPACING_MM = 2.0
SSD_MM = 1000.0
BIXEL_WIDTH_MM = 5.0
# One perpendicular beam, with planning isocenter at the entrance center.
# Combined mode simulates the entire tiled field in ONE job.

SOURCE_TYPE = 'phase_space'  # 'point' uses the existing empirical spectrum
# Ordered distinct history batches, not particle splits sharing original histories.
# Applies only to SOURCE_TYPE = "phase_space".
# "field": replay particles projected into the selected bixels.
# "all_forward": replay every downstream particle in the original-history prefix.
# Requires combined execution; physical MLC/jaws/collimator define irradiation.
# Transport scatter is included in BOTH modes. Empty histories still count in
# HISTORIES_PER_JOB and dose normalization; this does not select N particles.
PHASE_SPACE_SELECTION = "all_forward"
PHASE_SPACE_FILE_BASES = [
    '~/Local/MCGPU/TOPAS/Elekta_Precise_6MV/ELEKTA_PRECISE_6mv_part1',
    '~/Local/MCGPU/TOPAS/Elekta_Precise_6MV/ELEKTA_PRECISE_6mv_part2',
    '~/Local/MCGPU/TOPAS/Elekta_Precise_6MV/ELEKTA_PRECISE_6mv_part3',
    '~/Local/MCGPU/TOPAS/Elekta_Precise_6MV/ELEKTA_PRECISE_6mv_part4',
]
# Total original histories (including empty histories) for the combined job;
# primary photons for point mode. NOT multiplied by the number of bixels.
HISTORIES_PER_JOB = 100_000
TOPAS_THREADS_PER_JOB = 4

# None selects the independent packaged water_config.toml.
# Its field.width_mm/height_mm define the square field AT THE WATER SURFACE.
GEOMETRY_CONFIG = None
ENABLE_COLLIMATOR = False  # MLC/jaws retain their TOML settings
DOSE_SPACING_MM = None  # native spacing; optionally (dx, dy, dz) in mm
ENABLE_OPENGL = False
# Exposure of the combined field, in saved source units (not MU).
FORWARD_WEIGHT = 1.0


def planning_settings():
    """Capture editable planning inputs; saved-result stages never call this."""
    return deepcopy(water.WaterSettings(
        phantom_size_mm=PHANTOM_SIZE_MM,
        phantom_spacing_mm=PHANTOM_SPACING_MM,
        ssd_mm=SSD_MM,
        bixel_width_mm=BIXEL_WIDTH_MM,
        source_type=SOURCE_TYPE,
        phase_space_file_bases=PHASE_SPACE_FILE_BASES,
        phase_space_selection=PHASE_SPACE_SELECTION,
        histories_per_job=HISTORIES_PER_JOB,
        topas_threads_per_job=TOPAS_THREADS_PER_JOB,
        geometry_config=GEOMETRY_CONFIG,
        enable_collimator=ENABLE_COLLIMATOR,
        dose_spacing_mm=DOSE_SPACING_MM,
        enable_opengl=ENABLE_OPENGL,
    ))


def inspect(project_dir):
    """Inspect the configured setup."""
    return water.inspect(planning_settings(), project_dir)


def prepare(project_dir):
    """Prepare portable TOPAS inputs; execution remains external."""
    return water.prepare(planning_settings(), project_dir)


def collect(project_dir):
    """Collect using saved planning and completed CSVs only."""
    return collect_saved_run(project_dir, "collect")


def forward(project_dir):
    """Reconstruct saved dose with the explicit exposure weights."""
    return collect_saved_run(project_dir, "forward", weight_per_bixel=FORWARD_WEIGHT)


def parse_arguments(argv=None):
    return parse_stage_arguments(argv, description=__doc__, inspect_directory="projects/water-phantom")


def main(argv=None):
    dispatch(parse_arguments(argv), {"inspect": inspect, "prepare": prepare,
                                     "collect": collect, "forward": forward})


if __name__ == "__main__":
    main()
