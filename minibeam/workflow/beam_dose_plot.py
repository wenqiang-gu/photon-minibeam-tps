"""Beam-specific forward dose displays; retains only 2D data between beams."""
from itertools import product
import numpy as np
import SimpleITK as sitk
from pyRadPlan.geometry import lps
from .dose_plot import snapshot_images
from ..topas.manifest import member_jobs


def beam_display_grid(ct, parameters):
    """Orthogonal U,V,depth grid; U=X, V=Z, depth=Y at zero angles."""
    rotation = lps.get_beam_rotation_matrix(parameters['gantry_angle'], parameters['couch_angle'])
    depth = -np.asarray(parameters['source_point'], dtype=float)
    depth /= np.linalg.norm(depth)
    u = rotation[:, 0]
    if not np.isfinite(depth).all() or not np.allclose(np.dot(u, depth), 0., atol=1e-6):
        raise ValueError('Saved beam source and transverse axes are inconsistent')
    v = np.cross(u, depth)
    v /= np.linalg.norm(v)
    # Display axes deliberately retain +Superior vertically at gantry/couch zero.
    # U,V,depth is a reflection of the right-handed transport frame.
    basis = np.column_stack((u, v, depth))
    iso = np.asarray(parameters['iso_center'], dtype=float)
    corners = np.array([ct.TransformContinuousIndexToPhysicalPoint(tuple(c))
                        for c in product(*[(-.5, n-.5) for n in ct.GetSize()])])
    local = (corners-iso) @ basis
    low, high = local.min(axis=0), local.max(axis=0)
    spacing = float(min(ct.GetSpacing()))
    dimensions = np.ceil((high-low)/spacing - 1e-10).astype(int)
    center = low + spacing/2
    ref = sitk.Image(dimensions.tolist(), sitk.sitkFloat32)
    ref.SetSpacing((spacing,)*3)
    ref.SetDirection(tuple(basis.ravel()))
    ref.SetOrigin(tuple(iso+basis @ center))
    return ref, center, basis


def bixel_centers(snapshot, manifest, beam_index, basis):
    """Project saved ray offsets; keep one dot per ray and all bixel associations."""
    def items(value):
        return [value] if isinstance(value, dict) else np.asarray(value).ravel()

    beam = items(snapshot['stf'])[beam_index-1]
    rays = items(beam['ray'])
    iso = np.asarray(beam['isoCenter'], dtype=float).reshape(3)
    centers = {}
    for member in member_jobs(manifest):
        if member['beam_index'] != beam_index:
            continue
        ray_index = member['ray_index']
        if ray_index not in centers:
            offset = np.asarray(rays[ray_index-1]['rayPos'], dtype=float).reshape(3)
            if not np.isfinite(offset).all():
                raise ValueError('Saved bixel center is not finite')
            centers[ray_index] = dict(beam_index=beam_index, ray_index=ray_index,
                uv_mm=(offset @ basis[:, :2]).tolist(),
                center_lps_mm=(iso+offset).tolist(), bixels=[])
        association = dict(bixel_index=member['bixel_index'],
                           beamlet_index=member['beamlet_index'], job_id=member['job_id'])
        if association not in centers[ray_index]['bixels']:
            centers[ray_index]['bixels'].append(association)
    return list(centers.values())


def project_beam(ct, masks, dose, parameters):
    ref, center, basis = beam_display_grid(ct, parameters)
    # Match the existing forward CT-grid resampling convention for coarse scores.
    if (dose.GetSize(), dose.GetOrigin(), dose.GetSpacing(), dose.GetDirection()) != (
            ct.GetSize(), ct.GetOrigin(), ct.GetSpacing(), ct.GetDirection()):
        dose = sitk.Resample(dose, ct, sitk.Transform(), sitk.sitkLinear, 0., sitk.sitkFloat64)
    rotated = sitk.Resample(dose, ref, sitk.Transform(), sitk.sitkLinear, 0., sitk.sitkFloat64)
    values = sitk.GetArrayViewFromImage(rotated)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('Beam plot requires finite nonnegative dose')
    occupied = np.zeros(ref.GetSize()[2], dtype=bool)
    outlines = {}
    for name, mask in masks.items():
        transformed = sitk.Resample(mask, ref, sitk.Transform(), sitk.sitkNearestNeighbor, 0., sitk.sitkUInt8)
        array = sitk.GetArrayViewFromImage(transformed)
        rows = array.any(axis=(1, 2))
        if rows.any():
            occupied |= rows
            outlines[name] = array.max(axis=0)
    rows = np.flatnonzero(occupied)
    first, last = (int(rows[0]), int(rows[-1])) if rows.size else (0, ref.GetSize()[2]-1)
    projected = values[first:last+1].max(axis=0)
    spacing = ref.GetSpacing()[0]
    mid = int(np.clip(round(-center[2]/spacing), 0, ref.GetSize()[2]-1))
    # Resample only the reference CT cross-section, not an extra 3D CT cube.
    section = sitk.Image([ref.GetSize()[0], ref.GetSize()[1], 1], sitk.sitkFloat32)
    section.SetSpacing(ref.GetSpacing())
    section.SetDirection(ref.GetDirection())
    section.SetOrigin(ref.TransformIndexToPhysicalPoint((0, 0, mid)))
    anatomy = sitk.Resample(ct, section, sitk.Transform(), sitk.sitkLinear, -1000., sitk.sitkFloat32)
    slab = dict(slice_indices_zero_based_inclusive=[first, last],
                depth_voxel_centers_mm=(center[2]+np.array([first,last])*spacing).tolist(),
                depth_voxel_boundaries_mm=(center[2]+np.array([first-.5,last+.5])*spacing).tolist(),
                target_names=list(outlines),
                fallback=None if rows.size else 'No nonempty target in beam frame; using full depth')
    return dict(dose=projected, ct=sitk.GetArrayFromImage(anatomy)[0], outlines=outlines,
                center=center, spacing=spacing,
                record=dict(depth_slab=slab, reference_slice_depth_mm=float(center[2]+mid*spacing),
                    display_grid=dict(dimensions=list(ref.GetSize()), spacing_mm=list(ref.GetSpacing()),
                                      origin_lps_mm=list(ref.GetOrigin()), axes_lps=basis.tolist()),
                    reference_isocenter_lps_mm=parameters['iso_center']))


class BeamDosePlots:
    """Catch display failures per beam; dose reconstruction continues."""
    def __init__(self, snapshot, manifest, weights, diagnostics):
        self.ct, self.masks, self.missing = snapshot_images(snapshot, manifest)
        self.manifest, self.weights, self.diagnostics = manifest, weights, diagnostics
        self.views, self.records = {}, []
        self.snapshot = snapshot

    def consume(self, beam_index, dose):
        jobs = [(i,j) for i,j in enumerate(self.manifest['jobs']) if j['beam_index']==beam_index]
        parameters = self.manifest['planning']['beams'][beam_index-1]['parameters']
        record = dict(beam_index=beam_index, gantry_angle_deg=parameters['gantry_angle'],
                      couch_angle_deg=parameters['couch_angle'], job_ids=[j['job_id'] for _,j in jobs],
                      weights=[float(self.weights[i]) for i,_ in jobs],
                      weight_units=self.manifest['weight_units'], missing_targets=self.missing)
        self.records.append(record)
        try:
            view = project_beam(self.ct, self.masks, dose, parameters)
            record.update(view.pop('record'))
            record['bixel_centers'] = bixel_centers(
                self.snapshot, self.manifest, beam_index, np.asarray(record['display_grid']['axes_lps']))
            self.views[beam_index] = view
        except Exception as error:
            record.update(status='failed', error=str(error))
            print(f'Beam {beam_index}: dose plot failed: {error}')

    def save(self, directory):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.lines import Line2D
        from .collection import atomic_path
        peak = max((float(v['dose'].max()) for v in self.views.values()), default=0.)
        for record in self.records:
            if record.get('status')=='failed':
                continue
            fig = None
            try:
                index = record['beam_index']
                view = self.views[index]
                delta = view['spacing']
                x = view['center'][0]+np.arange(view['dose'].shape[1])*delta
                y = view['center'][1]+np.arange(view['dose'].shape[0])*delta
                extent = [x[0]-delta/2,x[-1]+delta/2,y[0]-delta/2,y[-1]+delta/2]
                fig = Figure(figsize=(11,7),layout='constrained')
                FigureCanvasAgg(fig)
                ax = fig.subplots()
                ax.imshow(view['ct'],origin='lower',extent=extent,cmap='gray',vmin=-1000,vmax=800,interpolation='nearest')
                artist = ax.imshow(np.ma.masked_equal(view['dose'],0),origin='lower',extent=extent,
                                   cmap='inferno',vmin=0,vmax=peak or 1.,alpha=.9,interpolation='nearest')
                for name, outline in view['outlines'].items():
                    ax.contour(np.r_[x[0]-delta,x,x[-1]+delta],np.r_[y[0]-delta,y,y[-1]+delta],
                               np.pad(outline,1),levels=[.5],colors=['lime'],linewidths=1.3)
                centers = np.asarray([c['uv_mm'] for c in record['bixel_centers']]).reshape(-1, 2)
                ax.scatter(centers[:,0], centers[:,1], s=8, color='cyan', edgecolors='none', zorder=4)
                ax.plot(0,0,'+',color='cyan',ms=10,zorder=5)
                slab = record['depth_slab']
                low,high = slab['depth_voxel_centers_mm']
                ax.set(title=f"Beam {index}: gantry {record['gantry_angle_deg']:g}°, couch {record['couch_angle_deg']:g}°\n"
                             f"Maximum dose over target depth {low:.1f} to {high:.1f} mm",
                       xlabel='Beam transverse U (mm from isocenter)',ylabel='Beam transverse V (mm from isocenter)',
                       xlim=extent[:2],ylim=extent[2:])
                ax.set_aspect('equal')
                handles = [Line2D([],[],color='lime',label=name+' (projected)') for name in view['outlines']]
                handles.append(Line2D([],[],color='cyan',marker='.',linestyle='None',
                                      label='Bixel centers at isocenter plane'))
                handles.append(Line2D([],[],color='cyan',marker='+',linestyle='None',label='Isocenter'))
                ax.legend(handles=handles,loc='upper right',fontsize=8)
                fig.colorbar(artist,ax=ax,label='Maximum dose (Gy)',shrink=.75)
                notices = [f"This beam's weighted dose only. CT: perpendicular reference slice at depth {record['reference_slice_depth_mm']:.1f} mm.",
                           f"Exposure weights ({record['weight_units']}): {np.array2string(np.asarray(record['weights']),threshold=6)}",
                           'Inferno, linear shared scale, no positive-dose cutoff. Depth is positive downstream.']
                if slab['fallback']: notices.append(slab['fallback'])
                if self.missing: notices.append('Target unavailable: '+', '.join(self.missing))
                if not view['dose'].any(): notices.append('This beam has zero dose.')
                warnings = [d for d in self.diagnostics if d['job_id'] in record['job_ids']]
                record['scorer_warnings'] = warnings
                if warnings: notices.append(f'SCORER WARNINGS ({len(warnings)} jobs): validity requires user review; see indexing.md.')
                fig.supxlabel('\n'.join(notices),fontsize=9)
                path = directory/f'dose_beam_{index:03d}.png'
                with atomic_path(path) as temporary: fig.savefig(temporary,dpi=180,format='png')
                record.update(status='complete',path='derived/'+path.name,dose_color_limits_gy=[0.,peak or 1.])
                print(f'Beam dose plot: {path}')
            except Exception as error:
                record.update(status='failed',error=str(error))
                print(f"Beam {record['beam_index']}: dose plot failed: {error}")
            finally:
                if fig is not None: fig.clear()
        return dict(status='complete' if all(r.get('status')=='complete' for r in self.records) else 'partial',
                    method='per-beam weighted maximum over selected-target depth slab',beams=self.records)


def beam_plot_report(plot):
    text = ('\n## Beam dose plots\n\nEach figure uses only its beam’s weighted dose, maximized over the target depth slab. '
            'CT is the perpendicular cross-section nearest that beam’s isocenter. Depth is positive downstream; '
            'U/V are beam transverse axes (U=X, V=Z at zero gantry/couch). '
            'Small cyan dots mark selected bixel centers at the isocenter plane, including zero-exposure bixels; '
            'the larger cyan + marks isocenter. Repeated ray associations share one dot. '
            'All beam figures share a linear dose scale. Old dose_projections.png files are not refreshed.\n\n')
    if 'error' in plot:
        return text+'Plotting failed: '+plot['error']+'. Saved dose remains available.\n'
    for record in plot['beams']:
        if record['status']=='complete':
            name=record['path'].split('/')[-1]
            low,high=record['depth_slab']['depth_voxel_centers_mm']
            text+=f"- [Beam {record['beam_index']}]({name}): target depth {low:.3f} to {high:.3f} mm.\n"
            if record['depth_slab']['fallback']:text+='  '+record['depth_slab']['fallback']+'\n'
        else:text+=f"- Beam {record['beam_index']}: plotting failed: {record['error']}. Saved dose remains available.\n"
    return text
