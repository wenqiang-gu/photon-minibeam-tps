"""Point photons sampled uniformly over disjoint native squares, replayed by TOPAS."""
from pathlib import Path
import tempfile
import numpy as np
from .energy import EmpiricalSpectrum
from .bixels import groups, contains
from .iaea import TOPAS_DTYPE, write_topas_header
from ..geometry.spatial import beam_basis
from ..geometry.coordinates import topas_angles
from ..topas.contracts import JobContext, ParameterFragment, SourceDescription
from ..topas.manifest import digest_json


class PointBeamletSource:
    def __init__(self, energy_model=None):
        self.energy_model = energy_model or EmpiricalSpectrum()
        self._temporary = None
        self.states = {}

    def prepare(self, stf, histories, *, execution='separate', seed=12345, patient_center=None):
        patient_center = np.zeros(3) if patient_center is None else np.asarray(patient_center)
        selections = groups(stf, execution)
        if type(histories) is not int or histories < 2:
            raise ValueError('Point histories must be an integer >= 2')
        if histories > 2**31-1:
            raise ValueError('Combined point histories exceed TOPAS integer limit; reduce histories or use separate jobs')
        if not callable(getattr(self.energy_model, 'sample', None)):
            raise ValueError('Point replay energy models must implement sample(context, rng, count)')
        total = len(selections)*histories
        print(f'Point source: {total:,} photons; binary replay estimate {total*TOPAS_DTYPE.itemsize/1024**3:.3f} GiB; uniform square fluence, no recycling.')
        if self._temporary:
            self._temporary.cleanup()
        self._temporary = tempfile.TemporaryDirectory(prefix='minibeam-point-')
        self.states = {}
        directory = Path(self._temporary.name)
        for members in selections:
            # Draw membership counts from a uniform-area mixture; total is exactly
            # the job budget, including when it is smaller than the bixel count.
            areas = np.array([m['selection']['width_mm']**2 for m in members])
            probabilities = areas / areas.sum()
            allocation_rng = np.random.default_rng(np.random.SeedSequence(
                [seed, members[0]['bixel_index'], 193]))
            counts = allocation_rng.multinomial(histories, probabilities)
            energy_records = []
            contexts = []
            energy_assets = []
            for member in members:
                beam = stf.beams[member['beam_index']-1]
                ray = beam.rays[member['ray_index']-1]
                context = JobContext(beam,ray,ray.beamlets[0],patient_center,member['bixel_index'],member['beam_index'],member['ray_index'],1)
                contexts.append(context)
                fragment = self.energy_model.render(context)
                energy_records.append(fragment.record)
                energy_assets.extend(fragment.assets)
            record = dict(model='point_square_beamlet', selections=[m['selection'] for m in members],
                          sampling='uniform area on common isocenter plane', boundary_rule='lower inclusive, upper exclusive',
                          histories_per_job=histories, selected_particles=histories,
                          photons_per_member=counts.tolist(), member_probabilities=probabilities.tolist(),
                          source_seed=seed, energy_models=energy_records, recycling=False)
            key = digest_json(dict(record, members=members))
            path = directory/(key+'.phsp')
            with path.open('wb') as stream:
                for member, context, member_count in zip(members,contexts,counts):
                    rng = np.random.default_rng(np.random.SeedSequence([seed, member['bixel_index']]))
                    selection = member['selection']
                    basis = beam_basis(context.beam)
                    focal = context.beam.iso_center + context.beam.source_point - patient_center
                    world_origin = focal.astype(np.float32).astype(float)
                    iso = context.beam.iso_center - patient_center
                    remaining = int(member_count)
                    while remaining:
                        count = min(65536, remaining)
                        out = np.zeros(count, dtype=TOPAS_DTYPE)
                        pending = np.arange(count)
                        for attempt in range(100):
                            xy = (rng.random((len(pending),2))-.5)*selection['width_mm'] + selection['center_xy_mm']
                            direction = np.column_stack((xy, np.full(len(xy),selection['sad_mm'])))
                            direction /= np.linalg.norm(direction,axis=1)[:,None]
                            uv = direction[:,:2].astype(np.float32)
                            w = np.sqrt(np.maximum(0.,1.-np.sum(uv.astype(float)**2,axis=1)))
                            with np.errstate(divide='ignore', invalid='ignore'):
                                projected = uv.astype(float)/w[:,None]*selection['sad_mm']
                            # OpenTOPAS 4.2.p3 reconstructs Z then stores both the
                            # rotated direction and translated source position as floats.
                            w_topas = np.sqrt(np.maximum(0., 1.-(uv[:,0]*uv[:,0]).astype(float)
                                                       -(uv[:,1]*uv[:,1]).astype(float))).astype(np.float32)
                            world_direction = (np.column_stack((uv,w_topas)).astype(float) @ basis.T).astype(np.float32).astype(float)
                            with np.errstate(divide='ignore', invalid='ignore'):
                                travel = np.dot(iso-world_origin,basis[:,2]) / (world_direction @ basis[:,2])
                                actual = (world_origin + travel[:,None]*world_direction - iso) @ basis
                            valid = contains(selection, projected[:,0], projected[:,1]) & (w > 0)
                            valid &= contains(selection, actual[:,0], actual[:,1]) & (travel > 0)
                            out['u'][pending[valid]], out['v'][pending[valid]] = uv[valid,0], uv[valid,1]
                            pending = pending[~valid]
                            if not len(pending): break
                        else:
                            raise ValueError('Point bixel cannot be represented safely in TOPAS float directions')
                        energies = np.asarray(self.energy_model.sample(context,rng,count),dtype=float)
                        if energies.shape != (count,) or not np.isfinite(energies).all() or np.any(energies<=0):
                            raise ValueError('Energy sampler must return finite positive photon energies')
                        out['energy']=energies;out['weight']=1.;out['pdg']=22;out['new_history']=1
                        if not np.isfinite(out['energy']).all() or np.any(out['energy']<=0):
                            raise ValueError('Photon energy cannot be represented in TOPAS binary format')
                        out.tofile(stream);remaining-=count
            write_topas_header(directory/(key+'.header'),histories,histories,histories)
            self.states[members[0]['bixel_index']] = (key,record,tuple(energy_assets))

    def describe(self, context):
        position = np.asarray(context.beam.iso_center)+context.beam.source_point-context.patient_center
        return SourceDescription('fixed_beam_focal_point', (position,position), job_modes=('beamlet','combined'))

    def render(self, context):
        if context.bixel_index not in self.states:
            raise ValueError('Prepare square point-source replay before rendering')
        key, record, energy_assets = self.states[context.bixel_index]
        focal = np.asarray(context.beam.iso_center)+context.beam.source_point
        basis = beam_basis(context.beam);angles=topas_angles(basis)
        stem='inputs/point_'+key
        lines=['s:Ge/Source/Type = "Group"','s:Ge/Source/Parent = "World"']
        for axis,pos,rot in zip('XYZ',focal-context.patient_center,angles):
            lines += [f'd:Ge/Source/Trans{axis} = {pos:.12g} mm', f'd:Ge/Source/Rot{axis} = {rot:.12g} deg']
        lines += ['s:So/Beam/Type = "PhaseSpace"','s:So/Beam/Component = "Source"',
                  f's:So/Beam/PhaseSpaceFileName = "{stem}"','b:So/Beam/PhaseSpacePreCheck = "True"',
                  'b:So/Beam/PhaseSpaceIncludeEmptyHistories = "True"','i:So/Beam/PhaseSpaceMultipleUse = 1']
        assets=tuple((stem+suffix,Path(self._temporary.name)/(key+suffix)) for suffix in ('.header','.phsp'))
        return ParameterFragment('\n'.join(lines)+'\n',assets=assets+energy_assets,record=dict(record, **record['energy_models'][0],
            source_lps_mm=focal.tolist(),aim_lps_mm=(context.beam.iso_center+context.ray.ray_pos).tolist(),
            local_to_world=basis.tolist(),topas_euler_deg=angles.tolist()))
