"""Saved forward-dose projections in fixed DICOM LPS coordinates."""
from itertools import product
import numpy as np
import SimpleITK as sitk


def _uncell(value):
    # plotting accepts both squeezed loadmat data and the preserved export cells.
    while isinstance(value, np.ndarray) and value.size == 1 and value.dtype == object:
        value = value.flat[0]
    return value


def snapshot_images(snapshot, manifest):
    """Reconstruct CT and selected masks from MATLAB [Y,X,Z] snapshots."""
    grid = manifest['ct_grid']
    nx, ny, nz = grid['dimensions']
    shape = (ny, nx, nz)

    def image(cube):
        result = sitk.GetImageFromArray(cube.transpose(2, 0, 1))
        result.SetOrigin(tuple(grid['origin']))
        result.SetSpacing(tuple(grid['resolution'][a] for a in 'xyz'))
        result.SetDirection(tuple(np.asarray(grid['direction']).ravel()))
        return result

    ct = image(np.asarray(_uncell(snapshot['ct']['cubeHU'])).reshape(shape))
    rows = np.asarray(snapshot['cst'], dtype=object).reshape(-1, 6)
    targets = manifest['planning']['plan'].get('target_names', [])
    if isinstance(targets, str):
        targets = [targets]
    masks, missing = {}, []
    for name in targets:
        matches = [r for r in rows if r[1] == name]
        if len(matches) != 1:
            missing.append(name)
            continue
        indices = np.asarray(_uncell(matches[0][3]), dtype=float).ravel()
        if not indices.size:
            missing.append(name)
            continue
        if not np.isfinite(indices).all() or np.any(indices != np.floor(indices)) or np.any((indices < 1) | (indices > np.prod(shape))):
            raise ValueError(f'Invalid saved ROI indices: {name}')
        flat = np.zeros(np.prod(shape), dtype=np.uint8)
        flat[indices.astype(np.int64)-1] = 1
        masks[name] = image(flat.reshape(shape, order='F'))
    if not targets:
        missing.append('(no saved selected target)')
    return ct, masks, missing


def display_images(ct, dose, masks):
    """Resample oblique data into an axis-aligned LPS grid, only for display."""
    if np.allclose(ct.GetDirection(), np.eye(3).ravel(), atol=1e-10):
        reference = ct
    else:
        corners = np.array([ct.TransformContinuousIndexToPhysicalPoint(tuple(v))
                            for v in product(*[(-.5, n-.5) for n in ct.GetSize()])])
        spacing = np.full(3, min(ct.GetSpacing()))
        lower, upper = corners.min(axis=0), corners.max(axis=0)
        dimensions = np.ceil((upper-lower)/spacing).astype(int)
        reference = sitk.Image(dimensions.tolist(), sitk.sitkFloat32)
        reference.SetSpacing(tuple(spacing))
        reference.SetOrigin(tuple(lower+spacing/2))

    def resample(image, interpolation, background, pixel):
        return sitk.Resample(image, reference, sitk.Transform(), interpolation, background, pixel)

    return (resample(ct, sitk.sitkLinear, -1000., sitk.sitkFloat32),
            resample(dose, sitk.sitkLinear, 0., sitk.sitkFloat64),
            {name: resample(mask, sitk.sitkNearestNeighbor, 0., sitk.sitkUInt8)
             for name, mask in masks.items()})


def plane_views(array, indices):
    """XY, XZ, YZ maximum projections and reference slices; input is [Z,Y,X]."""
    x, y, z = indices
    return [(array.max(axis=0), array[z, :, :]),
            (array.max(axis=1), array[:, y, :]),
            (array.max(axis=2), array[:, :, x])]



def target_y_slab(masks, reference):
    """Inclusive target Y range on the axis-aligned display grid."""
    occupied = np.zeros(reference.GetSize()[1], dtype=bool)
    names = []
    for name, mask in masks.items():
        present = np.any(sitk.GetArrayViewFromImage(mask), axis=(0, 2))
        if present.any():
            occupied |= present
            names.append(name)
    indices = np.flatnonzero(occupied)
    first, last = (int(indices[0]), int(indices[-1])) if indices.size else (0, len(occupied)-1)
    origin, spacing = reference.GetOrigin()[1], reference.GetSpacing()[1]
    return dict(slice_indices_zero_based_inclusive=[first, last],
                y_voxel_centers_lps_mm=[origin+first*spacing, origin+last*spacing],
                y_voxel_boundaries_lps_mm=[origin+(first-.5)*spacing, origin+(last+.5)*spacing],
                target_names=names,
                fallback=None if names else 'No nonempty target on display grid; using full Y range')


def dose_views(values, index, slab):
    views = plane_views(values, index)
    first, last = slab['slice_indices_zero_based_inclusive']
    views[1] = (values[:, first:last+1, :].max(axis=1), views[1][1])
    return views


def projection_description(plot):
    """Shared caption for the saved indexing guide."""
    slab = plot['xz_y_slab']
    low, high = slab['y_voxel_centers_lps_mm']
    text = (f'XZ dose uses maximum over Y voxel centers {low:.3f} to {high:.3f} mm, '
            'inclusive, spanning the selected targets. Dose outside the ROI shape within that slab is included. '
            'XY and YZ dose projections and all target outlines use full depth. '
            'CT backgrounds and display extents remain unchanged; reference slices use the first saved isocenter.')
    if slab['fallback']:
        text += ' Fallback: '+slab['fallback']+'.'
    return text


def save_dose_projections(path, snapshot, manifest, dose, weights, diagnostics):
    """Write a three-panel PNG without GUI state or modifying saved images."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.lines import Line2D
    from .collection import atomic_path

    ct, masks, missing = snapshot_images(snapshot, manifest)
    ct, dose, masks = display_images(ct, dose, masks)
    anatomy = sitk.GetArrayViewFromImage(ct)
    values = sitk.GetArrayViewFromImage(dose)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('Dose plot requires finite nonnegative dose')
    isos = np.asarray([j['iso_center_lps_mm'] for j in manifest['jobs']])
    iso = isos[0]
    index = np.clip(ct.TransformPhysicalPointToIndex(tuple(iso)), 0, np.asarray(ct.GetSize())-1)
    slab = target_y_slab(masks, ct)
    projections = dose_views(values, index, slab)
    reference_slices = plane_views(anatomy, index)
    outlines = {name: plane_views(sitk.GetArrayViewFromImage(mask), index) for name, mask in masks.items()}
    coordinates = [ct.GetOrigin()[i]+np.arange(ct.GetSize()[i])*ct.GetSpacing()[i] for i in range(3)]
    peak = max(float(view[0].max()) for view in projections)
    fig = Figure(figsize=(17, 7), layout='constrained')
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, 3)
    labels = ['X (mm) → Left', 'Y (mm) → Posterior', 'Z (mm) → Superior']
    for k, (ax, (horizontal, vertical, depth), title) in enumerate(zip(
            axes, [(0, 1, 2), (0, 2, 1), (1, 2, 0)], ['XY', 'XZ', 'YZ'])):
        xx, yy = coordinates[horizontal], coordinates[vertical]
        dx, dy = ct.GetSpacing()[horizontal], ct.GetSpacing()[vertical]
        extent = [xx[0]-dx/2, xx[-1]+dx/2, yy[0]-dy/2, yy[-1]+dy/2]
        ax.imshow(reference_slices[k][1], origin='lower', extent=extent,
                  cmap='gray', vmin=-1000, vmax=800, interpolation='nearest')
        projected = projections[k][0]
        artist = ax.imshow(np.ma.masked_less_equal(projected, 0), origin='lower',
                           extent=extent, cmap='inferno', vmin=0, vmax=peak or 1.,
                           alpha=.9, interpolation='nearest')
        for name, views in outlines.items():
            # Pad masks so targets touching display boundaries still have closed outlines.
            padded = np.pad(views[k][0], 1)
            ax.contour(np.r_[xx[0]-dx, xx, xx[-1]+dx],
                       np.r_[yy[0]-dy, yy, yy[-1]+dy], padded,
                       levels=[.5], colors=['lime'], linewidths=1.2)
        ax.plot(iso[horizontal], iso[vertical], '+', color='cyan', ms=9)
        location = coordinates[depth][index[depth]]
        if k == 1:
            low, high = slab['y_voxel_centers_lps_mm']
            title += f' (Y = {low:.1f} to {high:.1f} mm)'
        ax.set(title=f'{title}: maximum along {"XYZ"[depth]}\nCT reference {"XYZ"[depth]} = {location:.1f} mm',
               xlabel=labels[horizontal], ylabel=labels[vertical],
               xlim=extent[:2], ylim=extent[2:])
        ax.set_aspect('equal')
    fig.colorbar(artist, ax=axes, label='Maximum dose (Gy)', shrink=.65)
    handles = [Line2D([], [], color='lime', label=f'{name} (projected)') for name in masks]
    handles.append(Line2D([], [], color='cyan', marker='+', linestyle='None', label='First isocenter projection'))
    fig.legend(handles=handles, loc='outside upper center', ncols=min(3, len(handles)), fontsize=9)
    notes = ['Inferno • linear scale • no dose cutoff (exact zero is transparent).',
             'XZ dose: target Y slab only. XY/YZ dose and ROI outlines: full depth. CT backgrounds unchanged.',
             f'Exposure weights ({manifest["weight_units"]}): {np.array2string(np.asarray(weights), threshold=8)}']
    if slab['fallback']:
        notes.append(slab['fallback'])
    if missing:
        notes.append('Target unavailable: '+', '.join(missing))
    if not np.allclose(isos, iso):
        notes.append('Beams have different isocenters; CT references use the first saved beam.')
    if peak == 0:
        notes.append('All dose values are zero.')
    if diagnostics:
        notes.append(f'SCORER WARNINGS ({len(diagnostics)} jobs): validity requires user review; see indexing.md.')
    fig.supxlabel('\n'.join(notes), fontsize=9)
    try:
        with atomic_path(path) as temporary:
            fig.savefig(temporary, dpi=180, format='png')
    finally:
        fig.clear()
    return dict(status='complete', path='derived/'+path.name, colormap='inferno', gamma=1.,
                method='XZ maximum over target Y slab; XY/YZ full-depth maximum; CT reference slices at first saved isocenter',
                xz_y_slab=slab,
                reference_isocenter_lps_mm=iso.tolist(), missing_targets=missing,
                dose_color_limits_gy=[0., peak or 1.])
