"""History-preserving square beamlet selection from a fixed IAEA source plane."""
from pathlib import Path
import tempfile
import numpy as np
from .iaea import read_header, validated_parts, TOPAS_DTYPE, write_topas_header
from ..geometry.spatial import beam_basis
from ..geometry.coordinates import topas_angles
from ..topas.contracts import ParameterFragment, SourceDescription
from ..topas.manifest import digest_json


class PhaseSpaceBeamletSource:
    def __init__(self, file_base=None, *, file_bases=None):
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
        from .bixels import groups
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
            key=digest_json(item); selections[key]=item
            job += 1; job_keys[members[0]['bixel_index']]=key
        if not job:raise ValueError('No phase-space bixels to prepare')
        if self._temporary:self._temporary.cleanup()
        self._temporary=tempfile.TemporaryDirectory(prefix='minibeam-phase-space-')
        directory=Path(self._temporary.name)
        states={key:{'selection':item,'last_history':0,'particles':0,'reached':0,
                     'selected_per_file':np.zeros(len(infos),dtype=np.int64),
                     'member_counts':np.zeros(len(item.get('squares',[item])),dtype=np.int64),
                     'species':np.zeros(3,dtype=np.int64),'bounds':np.array([[np.inf,np.inf],[-np.inf,-np.inf]])}
                for key,item in selections.items()}
        audits=[];prefix_particles=backward=0
        prefix_per_file=np.zeros(len(infos),dtype=np.int64)
        backward_per_file=np.zeros(len(infos),dtype=np.int64)
        print(f'Phase space: validating original files and selecting {len(states)} unique square selections/unions for {job} jobs; '
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
            for key,state in states.items():
                from .bixels import contains
                item=state['selection']
                squares = item.get('squares', [item])
                distance=squares[0]['sad_mm']-info['plane_distance_mm']
                px=a['x'].astype(float)*10+distance*u/w
                py=a['y'].astype(float)*10+distance*v/w
                selected=np.zeros(len(a),dtype=bool)
                for i,selection in enumerate(squares):
                    mask = contains(selection, px, py)
                    state['member_counts'][i] += int(mask.sum())
                    selected |= mask
                rows=a[selected];history_ids=ids[selected]
                state['selected_per_file'][file_index]+=len(rows)
                if not len(rows):continue
                new=np.r_[history_ids[0]!=state['last_history'],history_ids[1:]!=history_ids[:-1]]
                out=np.zeros(len(rows),dtype=TOPAS_DTYPE)
                for field in ('x','y','u','v','weight'):out[field]=rows[field]
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
                raise ValueError(f"Phase-space bixel at {state['selection']} selects no particles; "
                                 'increase HISTORIES_PER_JOB or check coverage. No dose was fabricated.')
            write_topas_header(directory/f'{key}.header',histories,state['reached'],state['particles'])
            state['record']={'model':'phase_space_square_beamlet',**({'selections':state['selection']['squares']} if 'squares' in state['selection'] else {'selection':state['selection']}),
                'boundary_rule':'lower inclusive, upper exclusive','represented_original_histories':histories,
                'nonempty_selected_histories':state['reached'],'empty_histories':histories-state['reached'],
                'particles_per_member_bixel':state['member_counts'].tolist(),
                'selected_particles':state['particles'],'prefix_particles':prefix_particles,
                'excluded_particles':prefix_particles-state['particles'],
                'excluded_backward_or_tangent_particles':backward,
                'selected_species_counts':dict(zip(('photons','electrons','positrons'),map(int,state['species']))),
                'selected_xy_bounds_mm':state['bounds'].tolist(),
                **({'original_file':audits[0]} if len(audits)==1 else {}),
                'original_files':[dict(a, selected_particles=int(state['selected_per_file'][i]),
                    excluded_particles=int(prefix_per_file[i]-state['selected_per_file'][i])) for i,a in enumerate(audits)],
                'available_original_histories':available,
                'history_allocation':'sequential prefix of distinct original-history batches',
                'recycling':False,'cross_column_covariance':'not estimated; shared original histories'}
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
            normalization='original_accelerator_history', job_modes=('beamlet','combined'))

    def render(self, context):
        key=self.job_keys[context.bixel_index];state=self.selections[key]
        origin,basis,points=self._plane(context);angles=topas_angles(basis)
        stem=f'inputs/phase_{key}'
        lines=['s:Ge/Source/Type = "Group"','s:Ge/Source/Parent = "World"']
        for axis,pos,rot in zip('XYZ',origin-context.patient_center,angles):
            lines += [f'd:Ge/Source/Trans{axis} = {pos:.12g} mm',f'd:Ge/Source/Rot{axis} = {rot:.12g} deg']
        lines += ['s:So/Beam/Type = "PhaseSpace"','s:So/Beam/Component = "Source"',
                  f's:So/Beam/PhaseSpaceFileName = "{stem}"',
                  'b:So/Beam/PhaseSpacePreCheck = "True"',
                  'b:So/Beam/PhaseSpaceIncludeEmptyHistories = "True"',
                  'i:So/Beam/PhaseSpaceMultipleUse = 1']
        assets=tuple((stem+suffix,Path(self._temporary.name)/(key+suffix)) for suffix in ('.header','.phsp'))
        return ParameterFragment('\n'.join(lines)+'\n',assets=assets,record={**state['record'],
            'phase_space_plane_lps_mm':origin.tolist(),'phase_space_plane_corners_lps_mm':points.tolist(),
            'nominal_focal_point_lps_mm':(np.asarray(context.beam.iso_center)+context.beam.source_point).tolist(),
            'local_to_world':basis.tolist(),'topas_euler_deg':angles.tolist()})
