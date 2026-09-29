"""Rotation/shift setup preparation and bookkeeping for the unified patient workflow."""
import json
import numpy as np
from minibeam.steering import enrich_stf, beam_geometry_records
from minibeam.geometry.assembly import GeometryCollisionError
from minibeam.geometry.coordinates import scoring_grid
from minibeam.workflow.execution import run_stage


def validate_shift_fractions(values):
    """An empty one-dimensional array selects an ordinary configured run."""
    fractions = np.asarray(values, dtype=float)
    if fractions.ndim != 1 or not np.isfinite(fractions).all() or np.any((fractions < 0) | (fractions >= 1)):
        raise ValueError('Collimator shift fractions must be a one-dimensional array in [0,1)')
    names = [f'shift_{round(f*100):03d}' for f in fractions]
    if len(set(names)) != len(names):
        raise ValueError('Shift fractions must produce unique percentage directory names')
    return fractions


def effective_shift_fractions(*, enable_collimator, values):
    """Inactive overrides are neither inspected nor validated."""
    from .geometry import validate_collimator_switch
    validate_collimator_switch(enable_collimator)
    return validate_shift_fractions(values) if enable_collimator else np.empty(0)


def rotation_name(angle):
    from decimal import Decimal
    text = format(Decimal(str(abs(float(angle)))), 'f')
    integer, _, fraction = text.partition('.')
    fraction = fraction.rstrip('0')
    return 'rotation_' + ('m' if angle < 0 else '') + integer.zfill(3) + ('p'+fraction if fraction else '')


def effective_rotations(*, enable_collimator, values):
    from .geometry import validate_collimator_switch
    from numbers import Real
    validate_collimator_switch(enable_collimator)
    if not enable_collimator:
        return np.empty(0)
    raw = np.asarray(values, dtype=object)
    if raw.ndim != 1 or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Real) for v in raw):
        raise ValueError('Collimator rotations must be a one-dimensional finite numeric array, without Booleans')
    rotations = raw.astype(float)
    if not np.isfinite(rotations).all():
        raise ValueError('Collimator rotations must be finite')
    names = [rotation_name(a) for a in rotations]
    if len(set(names)) != len(names):
        raise ValueError('Collimator rotations must have unique values and directory names')
    return rotations


def setup_settings(*, slit_width_mm, nominal_ctc_mm, collimator_shift_fractions, geometry,
                   collimator_rotations=()):
    """Resolve rotation-major Cartesian products; None denotes no override."""
    fractions = validate_shift_fractions(collimator_shift_fractions)
    rotations = effective_rotations(enable_collimator=True, values=collimator_rotations)
    if not fractions.size and not rotations.size:
        return []
    if not geometry.assembly.enabled or not geometry.aperture.enabled:
        raise ValueError('The collimator study requires enabled assembly and aperture')
    setups = []
    for angle in rotations.tolist() if rotations.size else [None]:
        for fraction in fractions.tolist() if fractions.size else [None]:
            parts, overrides = [], {}
            if angle is not None:
                parts.append(rotation_name(angle))
                overrides['rotation_z_deg'] = angle - 90.
            if fraction is not None:
                parts.append(f'shift_{round(fraction*100):03d}')
                overrides.update(slit_entrance_width_mm=slit_width_mm,
                    slit_entrance_ctc_mm=nominal_ctc_mm, lateral_shift_mm=fraction*nominal_ctc_mm)
            setups.append(('_'.join(parts), fraction, angle, geometry.replace(aperture=overrides)))
    if len({s[0] for s in setups}) != len(setups):
        raise ValueError('Duplicate collimator setup directories')
    return setups


def run_study(stage, root, ct, cst, plan, native, metadata, *, target,
              slit_width_mm, nominal_ctc_mm, collimator_shift_fractions, geometry,
              weight_per_bixel=1.0, collimator_rotations=()):
    """Run the selected stage for every shift; each setup remains independent."""
    setups = setup_settings(slit_width_mm=slit_width_mm, nominal_ctc_mm=nominal_ctc_mm,
                            collimator_shift_fractions=collimator_shift_fractions, geometry=geometry,
                            collimator_rotations=collimator_rotations)
    from ..geometry.cropping import resolve_grids
    _, grid, _ = resolve_grids(ct, plan.prop_dose_calc.get('ct_crop_voxels'), plan.prop_dose_calc.get('dose_spacing_mm'), cst, plan.prop_dose_calc.get('enforce_ct_crop_protection', True))
    count = len(native.beams) if plan.prop_dose_calc.get('beamlet_execution')=='combined' else native.total_number_of_bixels
    voxels = int(np.prod(grid.dimensions))
    total = len(setups)*count
    for name, fraction, angle, config in setups:
        print(f'{name}: rotation {angle if angle is not None else "TOML"}; shift {config.aperture.lateral_shift_mm:g} mm; '
              f'{count} jobs; dense {voxels*count*8/1024**3:.3f} GiB; CSV estimate {voxels*count*100/1024**3:.3f} GiB')
    print(f'Aggregate: {total} jobs; dense {voxels*total*8/1024**3:.3f} GiB; CSV estimate {voxels*total*100/1024**3:.3f} GiB')
    steering = [(name, fraction, angle, enrich_stf(native, config)) for name,fraction,angle,config in setups]
    for name, _, _, stf in steering:
        aperture = beam_geometry_records(stf)[0]['resolved']['aperture']
        pitches = aperture['entrance_slit_ctcs']
        print(f'{name}: actual entrance pitch {min(pitches):.6f}..{max(pitches):.6f} mm; physical blades {stf.beams[0].geometry.aperture.blade_thickness_mm:g} mm')
    if stage == 'inspect':
        return  # read-only steering/settings inspection, no replay files
    if stage in {'collect','forward'}:
        for name,_,_,_ in steering:
            if not (root/name/'manifest.json').is_file():
                raise SystemExit(f'{name}: prepare successfully and return TOPAS results before {stage}')
    root.mkdir(parents=True, exist_ok=True)
    index_path = root/'study.json'
    settings = dict(target=target, gantry_angles=plan.prop_stf['gantry_angles'], couch_angles=plan.prop_stf['couch_angles'],
        slit_entrance_width_mm=slit_width_mm if len(collimator_shift_fractions) else None,
        nominal_entrance_ctc_mm=nominal_ctc_mm if len(collimator_shift_fractions) else None,
        rotations=list(collimator_rotations),
        shifts=collimator_shift_fractions, target_covering_bixels=plan.prop_stf['generator'] == 'photonIMRT', plan=plan.prop_dose_calc,
        sad_mm=float(native.beams[0].sad), bixel_width_mm=float(native.beams[0].bixel_width))
    # Portable settings must not include the absolute study directory.
    settings['plan'] = {k:v for k,v in settings['plan'].items() if k != 'bundle_dir'}
    from minibeam.topas.manifest import json_data
    settings = json_data(settings)
    if index_path.exists():
        index = json.loads(index_path.read_text())
        if index['settings'] != settings:
            raise SystemExit('Study settings changed; use a fresh --project')
    else:
        index = dict(settings=settings, setups=[dict(directory=name, shift_fraction=fraction,
            rotation_deg=angle, rotation_override_applied=angle is not None, shift_override_applied=fraction is not None,
            lateral_shift_mm=stf.beams[0].geometry.aperture.lateral_shift_mm,
            preparation_status='pending') for name,fraction,angle,stf in steering])
    def save_index():
        temporary = index_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(index,indent=2,allow_nan=False)+'\n')
        temporary.replace(index_path)
    save_index()
    failures = []
    for entry, (name, fraction, angle, stf) in zip(index['setups'],steering):
        run = root/name
        current_plan = plan.model_copy(deep=True)
        current_plan.prop_dose_calc['bundle_dir'] = str(run)
        entry['beam_geometry'] = beam_geometry_records(stf)
        try:
            run_stage(stage, run, ct, cst, current_plan, stf, metadata,
                      weight_per_bixel=weight_per_bixel)
            if stage == 'prepare': entry['preparation_status'] = 'prepared'
            entry['last_stage'] = stage
            entry['last_stage_status'] = 'complete'
            entry.pop('error',None)
        except (ValueError, OSError, RuntimeError) as error:
            entry['last_stage_status'] = 'failed'
            entry['error'] = str(error)
            if stage == 'prepare': entry['preparation_status'] = 'blocked' if isinstance(error,GeometryCollisionError) else 'failed'
            if isinstance(error,GeometryCollisionError):
                entry['diagnostics_directory'] = str(error.diagnostics_dir.relative_to(root))
            failures.append(name)
            print(f'{name}: {error}')
        save_index()
    print(f'Study index: {index_path}')
    if failures:
        raise SystemExit('Incomplete setups: '+', '.join(failures))

