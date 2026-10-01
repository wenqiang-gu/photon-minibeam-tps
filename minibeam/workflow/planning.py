"""Small planning utilities; callers own native plans, steering and anatomy."""
from copy import deepcopy
from numbers import Real
import numpy as np
from pyRadPlan.machines import PhotonLINAC


def positive_number(name, value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be a positive finite number')
    return float(value)


def positive_integer(name, value):
    if type(value) is not int or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


def boolean(name, value):
    if type(value) is not bool:
        raise ValueError(f'{name} must be a Boolean')
    return value


def photon_machine(sad_mm):
    """Native machine dictionary; 6.0 labels steering, not source energy.

    SCD retains the existing SAD/2 convention. Physical devices use their own
    resolved geometry, rather than this nominal machine parameter.
    """
    sad = positive_number('SAD/SSD', sad_mm)
    machine = PhotonLINAC(version=2, name='IdealSpectrumPhoton', energies=np.array([6.0]),
                          sad=sad, scd=sad / 2)
    data = machine.model_dump(exclude_none=True, exclude_computed_fields=True)
    data['meta'] = {'radiation_mode': 'photons'}
    return data


def topas_settings(*, project_dir, source_type, phase_space_file_bases, histories,
                   num_threads, execution, water, enable_opengl, dose_spacing_mm, phase_space_selection="field"):
    """Build independent engine settings without opening source files."""
    positive_integer('HISTORIES_PER_JOB', histories)
    positive_integer('TOPAS_THREADS_PER_JOB', num_threads)
    boolean('water', water)
    boolean('ENABLE_OPENGL', enable_opengl)
    if execution not in {'separate', 'combined'}:
        raise ValueError('BEAMLET_EXECUTION must be separate or combined')
    if source_type not in {'point', 'phase_space'}:
        raise ValueError('SOURCE_TYPE must be point or phase_space')
    source = {'type': source_type}
    if source_type == 'phase_space':
        from ..sources.phase_space import validate_selection
        validate_selection(phase_space_selection, execution)
        source['file_bases'] = deepcopy(phase_space_file_bases)
        source['selection'] = phase_space_selection
        print(f'Phase-space selection: {phase_space_selection}')
        if phase_space_selection == 'all_forward':
            print('All downstream source particles: physical hardware defines irradiation; planning bixels do not restrict fluence.')
    settings = dict(engine='TOPASPhoton', bundle_dir=str(project_dir), histories=histories,
                    num_threads=num_threads, beamlet_execution=execution, water=water,
                    enable_opengl=enable_opengl, source_config=source)
    if dose_spacing_mm is not None:
        settings['dose_spacing_mm'] = deepcopy(dose_spacing_mm)
    return settings


def check_matrad_grids(ct_grid, dose_grid):
    if any(grid.dimensions[0] != grid.dimensions[1] for grid in (ct_grid, dose_grid)):
        raise ValueError('matRad export requires square transverse CT and dose grids with pyRadPlan 0.5.0')


def workload_estimate(grid, jobs):
    """Same dense float64 / approximate CSV accounting used by both workflows."""
    voxels = int(np.prod(grid.dimensions))
    return (f'{jobs} jobs; dense {voxels*jobs*8/1024**3:.3f} GiB; '
            f'CSV estimate {voxels*jobs*100/1024**3:.3f} GiB')
