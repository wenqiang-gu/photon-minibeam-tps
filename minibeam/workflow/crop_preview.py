"""Diagnostic slices for explicit crop requests, including rejected crops."""
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from ..geometry.cropping import resolve_grids
from .collection import atomic_path, write_json


def preview_crop(ct, cst, project_dir, crop, spacing=None, target_names=(), enforce_protection=True):
    """Write diagnostics outside the bundle, then propagate validation failures."""
    error = None
    diagnostic_masks = {}
    try:
        resolved = resolve_grids(ct, crop, spacing, cst, enforce_protection, diagnostic_masks=diagnostic_masks)
        audit = resolved[2]
    except ValueError as exc:
        error = exc
        audit = getattr(exc, 'audit', dict(roi_clipping=[], conflicts=[str(exc)], enforce_protection=enforce_protection))
    if crop is None:
        if error is not None:
            raise error
        return resolved
    root = Path(project_dir).resolve()
    output = root.with_name(root.name + '-crop-preview')
    output.mkdir(parents=True, exist_ok=True)
    summary = dict(audit, validation='rejected' if error else 'accepted', requested_ranges=crop,
                   error=str(error) if error else '', target_names=list(target_names))
    for row in audit.get('roi_clipping', []):
        if row['removed_voxels']:
            label = 'BLOCKING target conflict' if row['blocking'] and row['is_target'] else ('BLOCKING ROI conflict' if row['blocking'] else 'BYPASSED ROI clipping')
            print(f"{row['name']}: {row['removed_voxels']:,}/{row['total_voxels']:,} voxels "
                  f"({row['removed_fraction']:.2%}); {row['clipping_extent']}; {label}")
    if audit.get('material_flagged_voxels'):
        state = 'BYPASSED material findings' if not audit.get('enforce_protection', True) else 'BLOCKING material findings'
        print(f"Excluded region: {audit['material_flagged_voxels']:,} flagged voxels; {state}")
    if 'protection_counts' in audit:
        print(f"Protection categories: {audit['protection_counts']} (ROI and material categories may overlap)")
    try:
        summary['plot'] = draw_preview(ct, cst, crop, audit, target_names, output/'crop_preview.png', error, diagnostic_masks)
        print(f"Crop preview: {output / 'crop_preview.png'}")
    except Exception as plot_error:
        summary['plot'] = dict(status='failed', error=str(plot_error))
        print(f'Crop preview could not be plotted: {plot_error}')
    write_json(output/'clipping_summary.json', summary)
    print(f"Clipping summary: {output / 'clipping_summary.json'}")
    if error is not None:
        raise error
    return resolved


def draw_preview(ct, cst, crop, audit, target_names, path, error, diagnostic_masks=None):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.patches import Rectangle
    from matplotlib.lines import Line2D
    image = sitk.GetArrayFromImage(ct.cube_hu)
    spacing = ct.grid.resolution_vector
    origin = np.asarray(ct.origin)
    size = np.asarray(ct.size)
    targets = np.zeros(image.shape, dtype=bool)
    clipped = np.zeros_like(targets)
    clipped_names = {r['name'] for r in audit.get('roi_clipping',[]) if r['removed_voxels']}
    for voi in cst.vois:
        if voi.name in target_names: targets |= sitk.GetArrayViewFromImage(voi.mask)>0
        if voi.name in clipped_names: clipped |= sitk.GetArrayViewFromImage(voi.mask)>0
    center = np.argwhere(targets).mean(axis=0)[::-1] if targets.any() else (size-1)/2
    indices = np.clip(np.floor(center+.5).astype(int),0,size-1)
    lo = np.array([crop[a][0] for a in 'xyz'],dtype=int)
    hi = np.array([crop[a][1] for a in 'xyz'],dtype=int)
    if np.any(lo<0) or np.any(hi>size) or np.any(hi<=lo):
        raise ValueError('Invalid crop ranges; see clipping_summary.json')
    lower = origin-spacing/2; upper = lower+size*spacing
    crop_lower = lower+lo*spacing; crop_upper=lower+hi*spacing
    excluded = np.ones(image.shape,dtype=bool)
    excluded[lo[2]:hi[2],lo[1]:hi[1],lo[0]:hi[0]]=False
    diagnostic_masks = diagnostic_masks or {}
    categories = [('hu_above_threshold', 'gold', 'HU > -950 in excluded region'),
                  ('enclosed_low_hu', 'magenta', 'Low-HU enclosure/connectivity flags'),
                  ('clipped_roi', 'red', 'Clipped ROIs (target clipping always blocks)')]
    # Later layers have display priority; categories can overlap in space/depth.
    def overlay(ax, arrays, extent, alpha):
        from matplotlib.colors import to_rgb
        for key, color, _ in categories:
            values = arrays[key]
            rgba = np.zeros((*values.shape,4))
            rgba[:,:,:3] = to_rgb(color); rgba[:,:,3] = values*alpha
            ax.imshow(rgba,origin='lower',extent=extent,interpolation='nearest')
    masks = {key: diagnostic_masks.get(key, np.zeros_like(excluded)) for key,_,_ in categories}
    fig=Figure(figsize=(16,11));FigureCanvasAgg(fig)
    axes=fig.subplots(2,3)
    records=[]
    for col,(ax,(name,h,v,fixed)) in enumerate(zip(axes[0],[('XY',0,1,2),('XZ',0,2,1),('YZ',1,2,0)])):
        def plane(a):return np.take(a,indices[fixed],axis=2-fixed)
        extent=[lower[h],upper[h],lower[v],upper[v]]
        ax.imshow(plane(image),origin='lower',extent=extent,cmap='gray',vmin=-1000,vmax=1000)
        shade=np.zeros((*plane(image).shape,4));shade[:,:,:3]=[.6,.6,.6];shade[:,:,3]=plane(excluded)*.18
        ax.imshow(shade,origin='lower',extent=extent)
        overlay(ax, {key: plane(mask) for key,mask in masks.items()}, extent, .6)
        intersects=bool(lo[fixed]<=indices[fixed]<hi[fixed])
        ax.add_patch(Rectangle((crop_lower[h],crop_lower[v]),crop_upper[h]-crop_lower[h],crop_upper[v]-crop_lower[v],
                              fill=False,ec='lime',lw=1.6,ls='-' if intersects else '--'))
        xs=origin[h]+np.arange(size[h])*spacing[h];ys=origin[v]+np.arange(size[v])*spacing[v]
        for mask,color in [(targets,'cyan'),(clipped,'red')]:
            section=plane(mask)
            if section.any():
                xp=np.r_[xs[0]-spacing[h],xs,xs[-1]+spacing[h]]
                yp=np.r_[ys[0]-spacing[v],ys,ys[-1]+spacing[v]]
                ax.contour(xp,yp,np.pad(section.astype(float),1),levels=[.5],colors=[color],linewidths=1.)
        axis='XYZ'[fixed];position=float(origin[fixed]+indices[fixed]*spacing[fixed])
        ax.set_title(f'{name} slice | {axis}={position:.2f} mm'+('' if intersects else '\nSlice outside retained depth'),fontsize=10)
        ax.set_xlabel(['X — Left (mm)','Y — Posterior (mm)','Z — Superior (mm)'][h])
        ax.set_ylabel(['X — Left (mm)','Y — Posterior (mm)','Z — Superior (mm)'][v])
        ax.set_xlim(extent[:2]);ax.set_ylim(extent[2:]);ax.set_aspect('equal')
        projected = {key: mask.any(axis=2-fixed) for key,mask in masks.items()}
        projection = axes[1,col]
        projection.set_facecolor('#202020')
        overlay(projection, projected, extent, .9)
        projection.add_patch(Rectangle((crop_lower[h],crop_lower[v]),crop_upper[h]-crop_lower[h],crop_upper[v]-crop_lower[v],fill=False,ec='lime',lw=1.3))
        projection.set_xlim(extent[:2]);projection.set_ylim(extent[2:]);projection.set_aspect('equal')
        projection.set_xlabel(ax.get_xlabel());projection.set_ylabel(ax.get_ylabel())
        projection.set_title(f'{name}: protection masks projected through ALL depths\nNo CT background; red > magenta > yellow display priority',fontsize=9)
        records.append(dict(slice_category_counts={key:int(plane(mask).sum()) for key,mask in masks.items()},
                            projection_pixel_counts={key:int(mask.sum()) for key,mask in projected.items()},plane=name,slice_index_zero_based=int(indices[fixed]),slice_position_lps_mm=position,extent_mm=extent,intersects_crop=intersects))
    status='REJECTED' if error else 'ACCEPTED by crop checks'
    enforcement = 'ENFORCED' if audit.get('enforce_protection', True) else 'BYPASSED — selected targets remain protected'
    fig.suptitle(f'CT crop preview — {status} | ROI/material protection: {enforcement}\nRetained XYZ ranges: {crop} (zero-based, stop-exclusive)',fontsize=12)
    handles = [Line2D([],[],color='lime',label='Retained crop (dashed when outside reference slice)'),
        Line2D([],[],color='gray',lw=7,alpha=.3,label='Excluded region (reference slices)'),
        Line2D([],[],color='cyan',label='Selected target'),Line2D([],[],color='red',label='Clipped ROI contours')]
    handles += [Line2D([],[],color=color,lw=5,label=label) for _,color,label in categories]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.032),ncol=3,fontsize=9)
    fig.text(.5,.012,'Top: reference CT slices. Bottom: whole-volume mask projections. Flags remain visible when bypassed. Counts may overlap; see clipping_summary.json.',ha='center',fontsize=9)
    fig.subplots_adjust(top=.88,bottom=.16,wspace=.32,hspace=.35)
    with atomic_path(path) as temp:fig.savefig(temp,format='png',dpi=150)
    return dict(status='complete',planes=records, projection_method='any flagged voxel along omitted axis; no CT background',
                category_colors={key:color for key,color,_ in categories}, display_priority='clipped_roi > enclosed_low_hu > hu_above_threshold', reference='selected target centroid' if targets.any() else 'CT center')
