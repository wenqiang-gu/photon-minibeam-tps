"""History-preserving square beamlet selection from a fixed IAEA source plane."""
from pathlib import Path
from numbers import Real
import tempfile
import numpy as np
from .bixels import contains, groups, rectangle_bounds
from .iaea import read_header, validated_parts, TOPAS_DTYPE, finish_guarded_replay
from ..geometry.spatial import beam_basis
from ..geometry.coordinates import topas_angles
from ..topas.contracts import ParameterFragment, SourceDescription
from ..topas.manifest import digest_json


# Domain separation from TOPAS transport. Keep this seed namespace stable.
ROULETTE_SEED_DOMAIN = 0x52524C54


def validate_selection(selection, execution=None):
    """Reject incompatible replay before opening potentially large source files."""
    if selection not in ('field', 'outside_field', 'all_forward'):
        raise ValueError('Phase-space selection must be field, outside_field or all_forward')
    if selection in ('all_forward', 'outside_field') and execution not in (None, 'combined'):
        raise ValueError(f'{selection} phase-space selection requires combined execution')


def resolve_outside_field_survival_probability(selection, value):
    if selection == 'field':
        return 1.0
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not 0 < value <= 1:
        raise ValueError('PHASE_SPACE_OUTSIDE_FIELD_SURVIVAL_PROBABILITY must be a finite number in (0, 1], excluding Booleans')
    probability = float(value)
    if probability == 0:
        raise ValueError('PHASE_SPACE_OUTSIDE_FIELD_SURVIVAL_PROBABILITY is too small to represent')
    return probability


def _roulette_threshold(probability):
    # The denominator of a float's exact rational value is a power of two.
    # Use enough 64-bit words to represent the probability without rounding it.
    numerator, denominator = probability.as_integer_ratio()
    bits = denominator.bit_length() - 1
    word_count = (bits + 63) // 64
    threshold = numerator << (64*word_count - bits)
    return tuple(np.uint64((threshold >> shift) & (2**64 - 1))
                 for shift in range(64*(word_count - 1), -1, -64))


def _roulette_survivors(bit_generator, count, threshold):
    if len(threshold) == 1:
        return bit_generator.random_raw(count) < threshold[0]
    words = bit_generator.random_raw((count, len(threshold))).T
    survives = words[-1] < threshold[-1]
    # Higher words override the comparison of lower words unless equal.
    for draw, limit in reversed(list(zip(words[:-1], threshold[:-1]))):
        survives = (draw < limit) | ((draw == limit) & survives)
    return survives


class PhaseSpaceBeamletSource:
    def __init__(self, file_base=None, *, file_bases=None, selection="field",
                 outside_field_survival_probability=1.0):
        validate_selection(selection)
        self.selection_mode = selection
        self.outside_field_survival_probability = resolve_outside_field_survival_probability(
            selection, outside_field_survival_probability)
        if file_base is not None and file_bases is not None:
            raise ValueError('Specify file_base or file_bases, not both')
        values = [file_base] if file_bases is None else file_bases
        if not isinstance(values, (list, tuple)) or not values or any(
                not isinstance(v, (str, Path)) or not str(v).strip() for v in values):
            raise ValueError('file_bases must be a nonempty ordered list of basenames')
        self.file_bases = [str(Path(v).expanduser().resolve()) for v in values]
        if len(set(self.file_bases)) != len(self.file_bases):
            raise ValueError('Duplicate phase-space file paths')
        self.file_base = self.file_bases[0]  # Legacy single-file attribute.
        self._temporary=None
        self.selections={}
        self.job_keys={}
        self.audit={}

    from .bixels import square as selection
    selection = staticmethod(selection)

    def prepare(self, stf, histories, *, execution='separate', seed=12345, patient_center=None):
        validate_selection(self.selection_mode, execution)
        probability = self.outside_field_survival_probability
        roulette = probability < 1.0
        multiplier = 1.0/probability
        if roulette:
            threshold = _roulette_threshold(probability)
            bit_generator = np.random.PCG64(np.random.SeedSequence([seed, ROULETTE_SEED_DOMAIN]))
        if not isinstance(histories,int) or isinstance(histories,bool) or not 2<=histories<=1_000_000_000:
            raise ValueError('Phase-space histories must be 2..1e9 original histories')
        infos = [read_header(base) for base in self.file_bases]
        info = infos[0]
        for part in infos[1:]:
            if (part['plane_distance_mm'] != info['plane_distance_mm'] or
                    part['source_description'] != info['source_description']):
                raise ValueError('Incompatible phase-space source plane or source/transport descriptions')
        available = sum(p['counts']['ORIG_HISTORIES'] for p in infos)
        if histories > available:
            raise ValueError('Requested histories exceed the phase-space files; particle recycling is disabled')
        allocation = []
        remaining = histories
        for part in infos:
            used = min(remaining, part['counts']['ORIG_HISTORIES'])
            allocation.append(used)
            remaining -= used
        print(f'Phase space: {available:,} available original histories across {len(infos)} distinct batches; {histories:,} requested per job.')
        for part, used in zip(infos, allocation):
            print(f"  {part['base'].name}: {used:,} / {part['counts']['ORIG_HISTORIES']:,} histories")
        selections={};job_keys={};job=0
        for members in groups(stf, execution):
            squares = [m['selection'] for m in members]
            if squares[0]['sad_mm'] <= info['plane_distance_mm']:
                raise ValueError('Phase-space plane must be upstream of isocenter')
            item = squares[0] if execution == 'separate' else {'squares': squares}
            if roulette:
                # Beam orientation and member order do not change the beam-local union.
                squares.sort(key=lambda square: square['center_xy_mm'])
                item = dict(squares=squares, selection_mode=self.selection_mode,
                            outside_field_roulette=dict(survival_probability=probability, seed=int(seed),
                                                        sampler='pcg64_forward_record_order_v1'))
            elif self.selection_mode == 'all_forward':
                item = {'selection_mode': 'all_forward'}  # Shared local replay for every beam.
            elif self.selection_mode == 'outside_field':
                item = dict(item, selection_mode='outside_field')
            key=digest_json(dict(selection=item,replay_policy='nonempty_guard_v1')); selections[key]=item
            job += 1; job_keys[members[0]['bixel_index']]=key
        if not job:raise ValueError('No phase-space bixels to prepare')
        if self._temporary:self._temporary.cleanup()
        self._temporary=tempfile.TemporaryDirectory(prefix='minibeam-phase-space-')
        directory=Path(self._temporary.name)
        states={key:{'selection':item,'last_history':0,'particles':0,'reached':0,
                     'selected_per_file':np.zeros(len(infos),dtype=np.int64),
                     'member_counts':np.zeros(0 if self.selection_mode != 'field' else len(item.get('squares',[item])),dtype=np.int64),
                     'species':np.zeros(3,dtype=np.int64),'bounds':np.array([[np.inf,np.inf],[-np.inf,-np.inf]])}
                for key,item in selections.items()}
        if roulette:
            for state in states.values():
                state['rectangle'] = rectangle_bounds(state['selection']['squares'])
                state['roulette'] = dict(eligible_particles=0, retained_particles=0, in_field_particles=0,
                                         eligible_weight_sum=0.0, retained_original_weight_sum=0.0,
                                         retained_reweighted_weight_sum=0.0)
        audits=[];prefix_particles=backward=0
        prefix_per_file=np.zeros(len(infos),dtype=np.int64)
        backward_per_file=np.zeros(len(infos),dtype=np.int64)
        print(f'Phase space: validating original files and selecting {len(states)} unique replay selections for {job} jobs; '
              f'{histories:,} original histories per job, no recycling.')
        for file_index,a,ids in validated_parts(infos,audits):
            prefix=ids<=histories
            prefix_particles+=int(prefix.sum())
            prefix_per_file[file_index]+=int(prefix.sum())
            forward=(a['code']>0)&(a['u'].astype(float)**2+a['v'].astype(float)**2<1)
            backward+=int((prefix & ~forward).sum())
            backward_per_file[file_index]+=int((prefix & ~forward).sum())
            use=prefix & forward
            if not use.any():continue
            a=a[use];ids=ids[use]
            u=a['u'].astype(float);v=a['v'].astype(float);w=np.sqrt(1-u*u-v*v)
            # Draw once per forward record, including in-field records. This keeps
            # the stream aligned across chunks, beams and replay modes.
            survives = _roulette_survivors(bit_generator, len(a), threshold) if roulette else None
            for key,state in states.items():
                item=state['selection']
                # Filtering is applied only at the source plane. Transport scatter
                # remains enabled in either mode. Do not duplicate outside-field
                # particles into individual bixel jobs: they belong to one beam dose.
                if self.selection_mode == 'all_forward' and not roulette:
                    selected = np.ones(len(a), dtype=bool)
                else:
                    squares = item.get('squares', [item])
                    distance=squares[0]['sad_mm']-info['plane_distance_mm']
                    px=a['x'].astype(float)*10+distance*u/w
                    py=a['y'].astype(float)*10+distance*v/w
                    if roulette and state['rectangle'] is not None:
                        xmin, xmax, ymin, ymax = state['rectangle']
                        in_field = (px >= xmin) & (px < xmax) & (py >= ymin) & (py < ymax)
                    else:
                        in_field = np.zeros(len(a), dtype=bool)
                        for i, selection in enumerate(squares):
                            mask = contains(selection, px, py)
                            if self.selection_mode == 'field':
                                state['member_counts'][i] += int(mask.sum())
                            in_field |= mask
                    if roulette:
                        retained = ~in_field & survives
                        selected = retained | in_field if self.selection_mode == 'all_forward' else retained
                        stats = state['roulette']
                        in_field_count = int(in_field.sum())
                        stats['in_field_particles'] += in_field_count
                        stats['eligible_particles'] += len(a) - in_field_count
                        stats['retained_particles'] += int(retained.sum())
                        stats['eligible_weight_sum'] += float(a['weight'][~in_field].sum(dtype=np.float64))
                    else:
                        selected = ~in_field if self.selection_mode == 'outside_field' else in_field
                rows=a[selected];history_ids=ids[selected]
                state['selected_per_file'][file_index]+=len(rows)
                if not len(rows):continue
                new=np.r_[history_ids[0]!=state['last_history'],history_ids[1:]!=history_ids[:-1]]
                out=np.zeros(len(rows),dtype=TOPAS_DTYPE)
                for field in ('x','y','u','v','weight'):out[field]=rows[field]
                if roulette:
                    outside_selected = retained[selected]
                    weights = rows['weight'][outside_selected].astype(np.float64)
                    with np.errstate(over='ignore'):
                        scaled = weights/probability
                    if np.any(~np.isfinite(scaled) | (scaled > np.finfo(np.float32).max)):
                        raise ValueError('Outside-field roulette survivor weight overflows the float32 TOPAS record')
                    out['weight'][outside_selected] = scaled
                    stats['retained_original_weight_sum'] += float(weights.sum())
                    stats['retained_reweighted_weight_sum'] += float(out['weight'][outside_selected].sum(dtype=np.float64))
                out['energy']=np.abs(rows['energy'])
                out['pdg']=np.array([0,22,11,-11])[rows['code']]
                out['new_history']=new
                with (directory/f'{key}.phsp').open('ab') as stream:out.tofile(stream)
                state['last_history']=int(history_ids[-1]);state['reached']+=int(new.sum());state['particles']+=len(rows)
                for code in (1,2,3):state['species'][code-1]+=int((rows['code']==code).sum())
                state['bounds'][0]=np.minimum(state['bounds'][0],[rows['x'].min()*10,rows['y'].min()*10])
                state['bounds'][1]=np.maximum(state['bounds'][1],[rows['x'].max()*10,rows['y'].max()*10])
        for i,audit in enumerate(audits):
            audit.update(consumed_original_histories=allocation[i], prefix_particles=int(prefix_per_file[i]),
                         excluded_backward_or_tangent_particles=int(backward_per_file[i]))
        bounds=np.asarray([a['xy_bounds_mm'] for a in audits])
        audit=dict(xy_bounds_mm=[bounds[:,0].min(axis=0).tolist(),bounds[:,1].max(axis=0).tolist()])
        for key,state in states.items():
            if state['particles']==0 or np.any(state['member_counts']==0):
                if roulette:
                    raise ValueError('Outside-field roulette selection has no survivors to replay; '
                                     'increase the history budget or survival probability, or check field coverage')
                raise ValueError(f"Phase-space bixel at {state['selection']} selects no particles; "
                                 'increase HISTORIES_PER_JOB or check selection coverage. No dose was fabricated.')
            replay = finish_guarded_replay(directory/key,histories,state['reached'],state['particles'])
            print(f"Replay: {histories:,} original histories; {state['reached']:,} transported nonempty histories")
            state['record']={'replay':replay, 'model':{'all_forward':'phase_space_all_forward', 'outside_field':'phase_space_outside_field', 'field':'phase_space_square_beamlet'}[self.selection_mode],
                'selection_mode':self.selection_mode,**({'selections':state['selection']['squares']} if 'squares' in state['selection'] else {'selection':state['selection']}),
                'boundary_rule':('lower-inclusive, upper-exclusive square union for roulette' if roulette and self.selection_mode == 'all_forward' else
                                 'none; all downstream particles' if self.selection_mode == 'all_forward' else
                                 'complement of lower-inclusive, upper-exclusive square union' if self.selection_mode == 'outside_field' else
                                 'lower inclusive, upper exclusive'),'represented_original_histories':histories,
                'nonempty_selected_histories':state['reached'],'empty_histories':histories-state['reached'],
                'particles_per_member_bixel':state['member_counts'].tolist(),
                'selected_particles':state['particles'],'prefix_particles':prefix_particles,
                'excluded_particles':prefix_particles-state['particles'],
                **({'retained_outside_field_particles':state['particles'],
                    'rejected_inside_field_particles':(state['roulette']['in_field_particles'] if roulette else prefix_particles-backward-state['particles'])}
                   if self.selection_mode == 'outside_field' else {}),
                'excluded_backward_or_tangent_particles':backward,
                'selected_species_counts':dict(zip(('photons','electrons','positrons'),map(int,state['species']))),
                'selected_xy_bounds_mm':state['bounds'].tolist(),
                **({'original_file':audits[0]} if len(audits)==1 else {}),
                'original_files':[dict(a, selected_particles=int(state['selected_per_file'][i]),
                    excluded_particles=int(prefix_per_file[i]-state['selected_per_file'][i])) for i,a in enumerate(audits)],
                'available_original_histories':available,
                'history_allocation':'sequential prefix of distinct original-history batches',
                'recycling':False,'cross_column_covariance':'not estimated; shared original histories'}
            if roulette:
                stats = state['roulette']
                stats['discarded_particles'] = stats['eligible_particles'] - stats['retained_particles']
                state['record']['outside_field_roulette'] = dict(
                    **state['selection']['outside_field_roulette'], **stats,
                    survivor_weight_multiplier=multiplier if np.isfinite(multiplier) else f'1/{probability!r}')
        if roulette:
            print(f'Outside-field roulette: p={probability:g}; surviving weights ×{multiplier:g}; seed {seed}.')
            for beam_index, key in enumerate(job_keys.values(), 1):
                stats = states[key]['roulette']
                print(f"Beam {beam_index}: {stats['eligible_particles']:,} eligible, "
                      f"{stats['retained_particles']:,} retained, {stats['discarded_particles']:,} discarded.")
        self.selections=states;self.job_keys=job_keys;self.audit=audit;self.info=info

    def _plane(self, context):
        basis=beam_basis(context.beam)
        focal=np.asarray(context.beam.iso_center)+context.beam.source_point
        origin=focal+basis[:,2]*self.info['plane_distance_mm']
        lower,upper=self.audit['xy_bounds_mm']
        corners=np.array([[x,y,0] for x,y in [(lower[0],lower[1]),(upper[0],lower[1]),
                                            (upper[0],upper[1]),(lower[0],upper[1])]])
        return origin,basis,corners@basis.T+origin

    def describe(self, context):
        origin,basis,points=self._plane(context)
        return SourceDescription('fixed_beam_phase_space_plane',
            (points.min(axis=0)-context.patient_center,points.max(axis=0)-context.patient_center),
            normalization='original_accelerator_history', job_modes=('combined',) if self.selection_mode != 'field' else ('beamlet','combined'))

    def render(self, context):
        key=self.job_keys[context.bixel_index];state=self.selections[key]
        origin,basis,points=self._plane(context);angles=topas_angles(basis)
        stem=f'inputs/phase_{key}'
        lines=['s:Ge/Source/Type = "Group"','s:Ge/Source/Parent = "World"']
        for axis,pos,rot in zip('XYZ',origin-context.patient_center,angles):
            lines += [f'd:Ge/Source/Trans{axis} = {pos:.12g} mm',f'd:Ge/Source/Rot{axis} = {rot:.12g} deg']
        lines += ['s:So/Beam/Type = "PhaseSpace"','s:So/Beam/Component = "Source"',
                  f's:So/Beam/PhaseSpaceFileName = "{stem}"',
                  f'b:So/Beam/PhaseSpacePreCheck = "{context.phase_space_precheck}"',
                  'b:So/Beam/PhaseSpaceIncludeEmptyHistories = "False"',
                  'i:So/Beam/PhaseSpaceMultipleUse = 1']
        assets=tuple((stem+suffix,Path(self._temporary.name)/(key+suffix)) for suffix in ('.header','.phsp'))
        return ParameterFragment('\n'.join(lines)+'\n',assets=assets,record={**state['record'],
            'phase_space_plane_lps_mm':origin.tolist(),'phase_space_plane_corners_lps_mm':points.tolist(),
            'nominal_focal_point_lps_mm':(np.asarray(context.beam.iso_center)+context.beam.source_point).tolist(),
            'local_to_world':basis.tolist(),'topas_euler_deg':angles.tolist()})
