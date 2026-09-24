"""Shift setup preparation and bookkeeping for the unified patient workflow."""
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


def setup_settings(*, slit_width_mm, nominal_ctc_mm, collimator_shift_fractions, geometry):
    """Validate the study settings and snapshot the hardware once."""
    fractions = validate_shift_fractions(collimator_shift_fractions)
    if not fractions.size:
        return []
    names = [f'shift_{round(f*100):03d}' for f in fractions]
    config = geometry
    if not config.assembly.enabled or not config.aperture.enabled:
        raise ValueError('The slit study requires enabled assembly and aperture')
    return [(name, float(f), config.replace(aperture={
        'slit_entrance_width_mm': slit_width_mm,
        'slit_entrance_ctc_mm': nominal_ctc_mm,
        'lateral_shift_mm': float(f*nominal_ctc_mm)})) for name,f in zip(names,fractions)]


def run_study(stage, root, ct, cst, plan, native, metadata, *, target,
              slit_width_mm, nominal_ctc_mm, collimator_shift_fractions, geometry,
              weight_per_bixel=1.0):
    """Run the selected stage for every shift; each setup remains independent."""
    setups = setup_settings(slit_width_mm=slit_width_mm, nominal_ctc_mm=nominal_ctc_mm,
                            collimator_shift_fractions=collimator_shift_fractions, geometry=geometry)
    grid = scoring_grid(ct, plan.prop_dose_calc.get('dose_spacing_mm'))
    count = len(native.beams) if plan.prop_dose_calc.get('beamlet_execution')=='combined' else native.total_number_of_bixels
    voxels = int(np.prod(grid.dimensions))
    total = len(setups)*count
    for name, fraction, _ in setups:
        print(f'{name}: {fraction*100:g}% nominal CTC; shift {fraction*nominal_ctc_mm:g} mm; '
              f'{count} jobs; dense {voxels*count*8/1024**3:.3f} GiB; CSV estimate {voxels*count*100/1024**3:.3f} GiB')
    print(f'Aggregate: {total} jobs; dense {voxels*total*8/1024**3:.3f} GiB; CSV estimate {voxels*total*100/1024**3:.3f} GiB')
    steering = [(name, fraction, enrich_stf(native, config)) for name,fraction,config in setups]
    for name, _, stf in steering:
        aperture = beam_geometry_records(stf)[0]['resolved']['aperture']
        pitches = aperture['entrance_slit_ctcs']
        print(f'{name}: actual entrance pitch {min(pitches):.6f}..{max(pitches):.6f} mm; physical blades {stf.beams[0].geometry.aperture.blade_thickness_mm:g} mm')
    if stage == 'inspect':
        return  # read-only steering/settings inspection, no replay files
    if stage in {'collect','forward'}:
        for name,_,_ in steering:
            if not (root/name/'manifest.json').is_file():
                raise SystemExit(f'{name}: prepare successfully and return TOPAS results before {stage}')
    root.mkdir(parents=True, exist_ok=True)
    index_path = root/'study.json'
    settings = dict(target=target, gantry_angles=plan.prop_stf['gantry_angles'], couch_angles=plan.prop_stf['couch_angles'],
        slit_entrance_width_mm=slit_width_mm, nominal_entrance_ctc_mm=nominal_ctc_mm,
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
            lateral_shift_mm=fraction*nominal_ctc_mm, preparation_status='pending') for name,fraction,_ in steering])
    def save_index():
        temporary = index_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(index,indent=2,allow_nan=False)+'\n')
        temporary.replace(index_path)
    save_index()
    failures = []
    for entry, (name, fraction, stf) in zip(index['setups'],steering):
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

