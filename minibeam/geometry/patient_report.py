"""Patient geometry in fixed DICOM LPS coordinates; display only."""
import os
from pathlib import Path
import tempfile
import numpy as np
import SimpleITK as sitk
from scipy.spatial import ConvexHull
from .coordinates import patient_center
from .spatial import beam_basis, corners
from .patient import axial_slice_for_isocenter
from .diagnostics import material_solids


def patient_scene(ct, stf, head, source_records=None):
    """Physical vertices shared by 2D/3D views (never bounding rectangles)."""
    center = patient_center(ct)
    box = corners({'center': center, 'half_size': np.asarray(ct.size)*ct.grid.resolution_vector/2,
                   'rotation': np.eye(3)})
    beams = []
    for index, beam in enumerate(stf.beams, 1):
        iso = np.asarray(beam.iso_center)
        basis = beam_basis(beam)
        solids = [] if head is None else [(name, points@basis.T+iso)
                                         for name, points in material_solids(head, beam)]
        record=next((r for r in (source_records or []) if r.get('beam_index')==index
                     and r.get('model')=='phase_space_square_beamlet'),None)
        beams.append({'index': index, 'beam': beam, 'source': iso+beam.source_point,
                      'iso': iso, 'solids': solids,
                      'phase_plane': np.asarray(record['phase_space_plane_corners_lps_mm']) if record else None,
                      'phase_origin': np.asarray(record['phase_space_plane_lps_mm']) if record else None,
                      'slice': axial_slice_for_isocenter(ct, iso)})
    return {'ct_box': box, 'ct_center': center, 'beams': beams}


def slice_groups(scene):
    groups = {}
    for item in scene['beams']:
        groups.setdefault(item['slice']['slice_index_1based'], []).append(item)
    return groups


def projected_polygon(points, axes):
    projection = np.unique(np.asarray(points)[:,axes], axis=0)
    if len(projection)<3 or np.linalg.matrix_rank(projection-projection[0],tol=1e-8)<2:
        return projection
    return projection[ConvexHull(projection).vertices]


def clipped_segment(start, end, lower, upper):
    """Clip a line to a physical box for the 3D patient close-up."""
    start, end = np.asarray(start), np.asarray(end)
    direction = end-start
    t0, t1 = 0., 1.
    for axis in range(3):
        if abs(direction[axis]) < 1e-12:
            if not lower[axis] <= start[axis] <= upper[axis]:
                return None
        else:
            a,b = sorted(((lower[axis]-start[axis])/direction[axis],
                          (upper[axis]-start[axis])/direction[axis]))
            t0,t1 = max(t0,a),min(t1,b)
            if t0 > t1:
                return None
    return np.vstack([start+t0*direction,start+t1*direction])


def axial_data(ct, cst, slice_index, target_names):
    """Native slice and contour masks, with outer voxel edges in LPS mm."""
    z = slice_index-1
    spacing = ct.grid.resolution_vector
    lower = np.asarray(ct.origin)-spacing/2
    upper = lower+np.asarray(ct.size)*spacing
    overlays = []
    if cst is not None:
        for voi in cst.vois:
            target = voi.name in target_names
            if target or voi.voi_type == 'EXTERNAL' or voi.name.upper() == 'BODY':
                overlays.append((voi.name, target, sitk.GetArrayViewFromImage(voi.mask)[z].copy()))
    return (sitk.GetArrayViewFromImage(ct.cube_hu)[z].copy(),
            [lower[0],upper[0],lower[1],upper[1]], overlays)


def write_patient_geometry_report(ct, cst, stf, head, path, *, target_names=(), diagnostics=None, source_records=None):
    os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'minibeam-matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from matplotlib.lines import Line2D
    from matplotlib.backends.backend_pdf import PdfPages
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    scene = patient_scene(ct, stf, head, source_records)
    if not scene['beams']:
        raise ValueError('Patient report requires at least one beam')
    colors = plt.get_cmap('tab10')
    box, center = scene['ct_box'], scene['ct_center']
    labels = ['X - Left (mm)', 'Y - Posterior (mm)', 'Z - Superior (mm)']
    all_points = np.vstack([box]+[np.vstack([b['source'],b['iso']]+[v for _,v in b['solids']]
                                 + ([b['phase_plane']] if b['phase_plane'] is not None else []))
                                 for b in scene['beams']])
    lo, hi = all_points.min(axis=0), all_points.max(axis=0)
    padding = max(np.ptp(all_points,axis=0).max()*.05,20.)

    def axial(ax, index):
        pixels, extent, overlays = axial_data(ct,cst,index,target_names)
        ax.imshow(pixels,origin='lower',extent=extent,cmap='gray',vmin=-1000,vmax=1000,
                  interpolation='nearest',zorder=0)
        xs = ct.origin[0]+np.arange(ct.size[0])*ct.grid.resolution_vector[0]
        ys = ct.origin[1]+np.arange(ct.size[1])*ct.grid.resolution_vector[1]
        for name, target, mask in overlays:
            if np.any(mask):
                # Pad to close contours that touch the image boundary.
                xp=np.r_[xs[0]-ct.grid.resolution_vector[0],xs,xs[-1]+ct.grid.resolution_vector[0]]
                yp=np.r_[ys[0]-ct.grid.resolution_vector[1],ys,ys[-1]+ct.grid.resolution_vector[1]]
                ax.contour(xp,yp,np.pad(mask,1),levels=[.5],colors=['#ef3395' if target else '#45d8ad'],linewidths=1)

    def view2d(ax, beams, axes=(0,1), zoom=False, slice_index=None):
        if slice_index is not None:
            axial(ax,slice_index)
        ax.add_patch(Polygon(projected_polygon(box,axes),fill=False,ec='#555555',ls='--',lw=1.2,zorder=2))
        ax.plot(center[axes[0]],center[axes[1]],'+',color='#101010',ms=10,mew=2,zorder=5)
        for item in beams:
            color=colors((item['index']-1)%10)
            source, iso=item['source'],item['iso']
            for _,points in item['solids']:
                ax.add_patch(Polygon(projected_polygon(points,axes),fc=color,ec=color,alpha=.22,lw=.6))
            if item['phase_plane'] is not None:
                ax.add_patch(Polygon(projected_polygon(item['phase_plane'],axes),fill=False,ec='#962cb1',lw=1.5,ls=':'))
            ax.plot([source[axes[0]],iso[axes[0]]],[source[axes[1]],iso[axes[1]]],color=color,lw=1)
            delta=iso-source
            arrow=ax.annotate('',xy=(source[axes[0]]+.65*delta[axes[0]],source[axes[1]]+.65*delta[axes[1]]),
                        xytext=(source[axes[0]]+.53*delta[axes[0]],source[axes[1]]+.53*delta[axes[1]]),
                        arrowprops={'arrowstyle':'->','color':color,'lw':1.4})
            arrow.arrow_patch.set_clip_path(ax.patch)
            arrow.arrow_patch.set_clip_on(True)
            ax.plot(source[axes[0]],source[axes[1]],'*',color=color,ms=11)
            ax.plot(iso[axes[0]],iso[axes[1]],'o',mfc='none',mec=color,ms=8,mew=1.5,zorder=6)
            if not zoom:
                ax.annotate(f"B{item['index']}",(source[axes[0]],source[axes[1]]),xytext=(6,6),textcoords='offset points',fontsize=9)
        limits = (box.min(axis=0)-25,box.max(axis=0)+25) if zoom else (lo-padding,hi+padding)
        ax.set_xlim(limits[0][axes[0]],limits[1][axes[0]])
        ax.set_ylim(limits[0][axes[1]],limits[1][axes[1]])
        ax.set_xlabel(labels[axes[0]]);ax.set_ylabel(labels[axes[1]])
        ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.15)

    def view3d(ax, zoom):
        lower,upper=(box.min(axis=0)-30,box.max(axis=0)+30) if zoom else (lo-padding,hi+padding)
        # Twelve real box edges, not an axis-aligned box around rotated hardware.
        for i,a in enumerate(box):
            for b in box[i+1:]:
                if np.count_nonzero(np.abs(a-b)>1e-8)==1:
                    ax.plot(*np.vstack([a,b]).T,color='#555555',ls='--',lw=1)
        ax.scatter(*center,color='black',marker='+',s=65)
        for item in scene['beams']:
            color=colors((item['index']-1)%10)
            segment=clipped_segment(item['source'],item['iso'],lower,upper)
            if segment is not None: ax.plot(*segment.T,color=color,lw=1)
            if np.all(item['source']>=lower) and np.all(item['source']<=upper):
                ax.scatter(*item['source'],color=color,marker='*',s=60)
            ax.scatter(*item['iso'],edgecolors=[color],facecolors='none',marker='o',s=35)
            if not zoom:ax.text(*item['source'],f" B{item['index']}",color=color)
            if not zoom and item['phase_plane'] is not None:
                ax.add_collection3d(Poly3DCollection([item['phase_plane']],facecolors='#962cb1',edgecolors='#962cb1',alpha=.2))
            for _,points in ([] if zoom else item['solids']):
                hull=ConvexHull(points)
                ax.add_collection3d(Poly3DCollection(points[hull.simplices],facecolors=color,edgecolors='none',alpha=.18))
        ax.set_xlim(lower[0],upper[0]);ax.set_ylim(lower[1],upper[1]);ax.set_zlim(lower[2],upper[2])
        ax.set_box_aspect(upper-lower)
        ax.set_xlabel(labels[0]);ax.set_ylabel(labels[1]);ax.set_zlabel(labels[2])
        ax.view_init(elev=24,azim=-55)

    def footer(fig):
        fig.text(.5,.015,'Fixed DICOM LPS coordinates. Geometry projections are not slice intersections. '
                 'CT box is the transport boundary; it is not the patient surface.',ha='center',fontsize=9)

    legend=[Line2D([],[],color='#555555',ls='--',label='CT transport boundary'),
            Line2D([],[],color='black',marker='+',ls='',label='CT center'),
            Line2D([],[],color='black',marker='o',mfc='none',ls='',label='Isocenter'),
            Line2D([],[],color='#45d8ad',label='BODY / EXTERNAL'),
            Line2D([],[],color='#ef3395',label='Selected target')]
    if any(item['phase_plane'] is not None for item in scene['beams']):
        legend.append(Line2D([],[],color='#962cb1',ls=':',label='Phase-space plane'))
        legend.append(Line2D([],[],color='black',marker='*',ls='',label='Nominal focal point'))
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent,suffix='.pdf',delete=False) as stream:
        temporary=Path(stream.name)
    try:
        with PdfPages(temporary,metadata={'Title':'Patient-centered photon geometry','CreationDate':None}) as pdf:
            for index, group in slice_groups(scene).items():
                fig,axes=plt.subplots(1,2,figsize=(14,9))
                fig.subplots_adjust(left=.07,right=.96,bottom=.16,top=.86,wspace=.25)
                z=group[0]['slice']['slice_center_lps_mm']
                fig.suptitle(f'Patient axial overview | CT slice {index}, LPS Z = {z:.3f} mm',fontsize=16)
                view2d(axes[0],scene['beams'],slice_index=index);axes[0].set_title('All beam paths projected onto X-Y')
                view2d(axes[1],scene['beams'],zoom=True,slice_index=index);axes[1].set_title('Patient close-up (hardware may lie outside view)')
                names=', '.join(target_names) or 'not supplied'
                fig.text(.5,.105,f'Slice selected for beams {", ".join(str(b["index"]) for b in group)}. Target: {names}. CT display: -1000 to 1000 HU.',ha='center',fontsize=10)
                fig.legend(handles=legend,loc='lower center',bbox_to_anchor=(.5,.045),ncol=4 if len(legend)>5 else 5,fontsize=9)
                footer(fig);pdf.savefig(fig);plt.close(fig)
            fig=plt.figure(figsize=(14,9))
            fig.suptitle('Patient 3D overview | fixed LPS coordinates',fontsize=16)
            fig.subplots_adjust(left=.03,right=.93,bottom=.10,top=.9,wspace=.3)
            for k,zoom in enumerate([False,True],1):
                ax=fig.add_subplot(1,2,k,projection='3d');view3d(ax,zoom)
                ax.set_title('Sources and enabled hardware' if not zoom else 'CT boundary and isocenters (no anatomy surface)')
            fig.legend(handles=legend[:3]+legend[5:],loc='lower center',bbox_to_anchor=(.5,.045),ncol=3 if len(legend)>5 else 4,fontsize=9)
            footer(fig);pdf.savefig(fig);plt.close(fig)
            for item in scene['beams']:
                beam=item['beam']
                fig=plt.figure(figsize=(14,10))
                fig.suptitle(f"Beam {item['index']} | Gantry {beam.gantry_angle:g} deg, couch {beam.couch_angle:g} deg",fontsize=16)
                gs=fig.add_gridspec(2,3,height_ratios=[3,1],left=.065,right=.97,bottom=.08,top=.88,wspace=.3,hspace=.2)
                for k,(axes,title) in enumerate([((0,1),'Axial X-Y'),((0,2),'Coronal X-Z'),((1,2),'Sagittal Y-Z')]):
                    ax=fig.add_subplot(gs[0,k]);view2d(ax,[item],axes,slice_index=item['slice']['slice_index_1based'] if k==0 else None)
                    ax.set_title(title+' projection')
                ax=fig.add_subplot(gs[1,:]);ax.axis('off')
                lines=[f"Source LPS (mm): {np.round(item['source'],3).tolist()}",
                       f"Isocenter LPS (mm): {np.round(item['iso'],3).tolist()} | CT center: {np.round(center,3).tolist()}",
                       f"SAD: {beam.sad:g} mm | Bixel width: {beam.bixel_width:g} mm | Axial slice: {item['slice']['slice_index_1based']}"]
                if item['phase_plane'] is not None:
                    lines[0]=lines[0].replace('Source LPS','Nominal focal point LPS')
                    lines.append(f"Phase-space frame origin LPS (mm): {np.round(item['phase_origin'],3).tolist()}")
                    lines.append('Stars mark nominal focal points; particles start on the purple plane, not at the stars.')
                if head:
                    for key in ['mlc','jaws','aperture']:
                        cfg=head.config_for(beam).data[key]
                        lines.append(f"{key.upper()}: source-center {cfg['source_to_center_mm']:g} mm; thickness {cfg['thickness_mm']:g} mm" if cfg['enabled'] and head.config_for(beam).data['assembly']['enabled'] else f'{key.upper()}: disabled')
                    _, aperture, _, _ = head.resolved(beam)
                    if aperture:
                        pitches = aperture.entrance_slit_ctcs
                        lines.append(f"Slits: width {min(aperture.entrance_slit_widths):g} mm; nominal CTC {aperture.nominal_entrance_ctc_mm:g} mm; shift +X {aperture.lateral_shift_mm:g} mm")
                        if pitches: lines.append(f"Actual entrance pitch: {min(pitches):.6f} to {max(pitches):.6f} mm (rigid translation, no refocusing)")
                else:lines.append('No configured treatment head shown; custom geometry is not rendered.')
                ax.text(0,1,'\n'.join(lines),va='top',fontsize=9,linespacing=1.25)
                footer(fig);pdf.savefig(fig);plt.close(fig)
                if diagnostics is not None:
                    record=diagnostics[item['index']-1]
                    lines=['Intersections are with the CT transport box, not necessarily tissue.',
                           'Witness coordinates are DICOM LPS millimeters. Inradius is not penetration depth.','']
                    for key in ['envelopes','material_solids']:
                        lines.append(key.replace('_',' ').upper())
                        hits=[hit for hit in record[key] if hit['intersects']]
                        for hit in hits:
                            lines.append(f"{hit['component']}: {np.round(hit['witness_lps_mm'],3).tolist()}; inradius {hit['intersection_inradius_mm']:.3f} mm")
                        if not hits:lines.append('No intersections found.')
                        lines.append('')
                    for start in range(0,len(lines),32):
                        fig,ax=plt.subplots(figsize=(14,10));ax.axis('off')
                        fig.suptitle(f"Beam {item['index']} | collision diagnostics - preparation blocked",fontsize=16)
                        ax.text(0,1,'\n'.join(lines[start:start+32]),va='top',fontsize=11,linespacing=1.5)
                        pdf.savefig(fig);plt.close(fig)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
