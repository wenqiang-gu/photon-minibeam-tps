"""Water phantom construction and native square-field planning."""

from importlib.resources import files
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import SimpleITK as sitk
from pyRadPlan import PhotonPlan, generate_stf
from pyRadPlan.ct import CT
from pyRadPlan.cst import StructureSet, create_voi

from minibeam.steering import enrich_stf
from minibeam.geometry.coordinates import scoring_grid
from minibeam.workflow.geometry import resolve_geometry
from minibeam.workflow.artifacts import prepare_project
from minibeam.workflow.planning import (photon_machine, topas_settings, positive_number,
    check_matrad_grids, workload_estimate)


@dataclass
class WaterSettings:
    """Explicit planning inputs; validation remains in the planning functions."""
    phantom_size_mm: float
    phantom_spacing_mm: float
    ssd_mm: float
    bixel_width_mm: float
    source_type: str
    phase_space_file_bases: list[str]
    histories_per_job: int
    topas_threads_per_job: int
    geometry_config: str | None
    enable_collimator: bool
    dose_spacing_mm: tuple[float, float, float] | None
    enable_opengl: bool
    phase_space_selection: str = "field"


# SHARED PLANNING HELPERS

def create_water(settings):
    """Native CT/CST with exact physical boundaries and voxel-center origin."""
    size = positive_number('PHANTOM_SIZE_MM', settings.phantom_size_mm)
    spacing = positive_number('PHANTOM_SPACING_MM', settings.phantom_spacing_mm)
    count = int(round(size / spacing))
    if count < 1 or not np.isclose(count * spacing, size, rtol=1e-10, atol=1e-9):
        raise ValueError('Phantom size must be an integer multiple of native spacing')
    shape = (count,) * 3
    image = sitk.GetImageFromArray(np.zeros(shape, dtype=np.float32))
    image.SetSpacing((spacing,) * 3)
    image.SetOrigin((-size / 2 + spacing / 2, spacing / 2, -size / 2 + spacing / 2))
    ct = CT(cube_hu=image)
    mask = np.ones(shape, dtype=np.uint8)
    cst = StructureSet(ct_image=ct, vois=[
        create_voi(name='BODY', grid=ct.grid, mask=mask, voi_type='EXTERNAL'),
        # Interface classification only: ray positions do not depend on this ROI.
        create_voi(name='WATER', grid=ct.grid, mask=mask, voi_type='TARGET'),
    ])
    return ct, cst


def build_planning(settings, project_dir):
    """Load hardware once, generate native steering once, and attach its snapshot."""
    root = Path(project_dir).expanduser().resolve()
    if (root / 'study.json').exists():
        raise ValueError('Water preparation requires a single project, not an existing study directory')
    config = settings.geometry_config or files('minibeam.geometry').joinpath('water_config.toml')
    geometry = resolve_geometry(config_path=config, enable_collimator=settings.enable_collimator)
    width = geometry.field.width_mm
    if not np.isclose(width, geometry.field.height_mm, rtol=0, atol=1e-9):
        raise ValueError('Water workflow requires equal field width and height')
    sad = positive_number('SSD_MM', settings.ssd_mm)
    positive_number('BIXEL_WIDTH_MM', settings.bixel_width_mm)
    ct, cst = create_water(settings)
    plan = PhotonPlan(machine=photon_machine(sad), num_of_fractions=1)
    plan.prop_stf = dict(generator='photonSquareField', gantry_angles=[0.], couch_angles=[0.],
                         iso_center=np.zeros((1, 3)), bixel_width=settings.bixel_width_mm,
                         field_width_mm=width, add_margin=False)
    plan.prop_dose_calc = topas_settings(
        project_dir=root, source_type=settings.source_type, phase_space_file_bases=settings.phase_space_file_bases,
        phase_space_selection=settings.phase_space_selection,
        histories=settings.histories_per_job, num_threads=settings.topas_threads_per_job, execution='combined',
        water=True, enable_opengl=settings.enable_opengl, dose_spacing_mm=settings.dose_spacing_mm)
    grid = scoring_grid(ct, settings.dose_spacing_mm)
    check_matrad_grids(ct.grid, grid)
    stf = enrich_stf(generate_stf(ct, cst, plan), geometry)
    metadata = dict(synthetic='water', selected_targets=['WATER'], water_measurement=dict(
        ssd_mm=sad, field_reference='water entrance surface / native isocenter plane',
        surface_center_lps_mm=[0., 0., 0.], phantom_size_mm=settings.phantom_size_mm,
        phantom_spacing_mm=settings.phantom_spacing_mm, field_width_mm=width,
        bixel_width_mm=settings.bixel_width_mm, number_of_bixels=stf.total_number_of_bixels,
        water_roi_role='Whole-water calculation region; does not select beamlets'))
    print(f'Water: {settings.phantom_size_mm:g} mm cube; entrance Y=0, downstream Y={settings.phantom_size_mm:g} mm')
    print(f'SSD/native SAD: {sad:g} mm; source LPS {stf.beams[0].source_point.tolist()}; isocenter [0, 0, 0] mm')
    print(f'Entrance field: {width:g} x {width:g} mm; {stf.total_number_of_bixels} bixels; one combined TOPAS job')
    print(f'Collimator: {"enabled" if settings.enable_collimator else "disabled"}; geometry: {config}')
    unit = 'original accelerator histories' if settings.source_type == 'phase_space' else 'primary photons'
    print(f'Source: {settings.source_type}; {settings.histories_per_job:,} {unit} TOTAL per job; {settings.topas_threads_per_job} threads')
    print(f'Dose grid: {grid.dimensions}; {workload_estimate(grid, 1)}')
    return root, ct, cst, plan, stf, metadata


# INSPECT: no simulation inputs or original phase-space scanning.
def inspect(settings, project_dir):
    build_planning(settings, project_dir)


# PREPARE: portable inputs; TOPAS execution remains external.
def prepare(settings, project_dir):
    root, ct, cst, plan, stf, metadata = build_planning(settings, project_dir)
    prepare_project(root, ct, cst, plan, stf, metadata)
