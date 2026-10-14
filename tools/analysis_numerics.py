"""Coordinate-aware profile extraction, normalization, ratios, and sampled PVDR."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks

from tools.analysis_io import Grid, Plane


@dataclass
class Curve:
    x: np.ndarray
    value: np.ndarray
    spread: np.ndarray | None = None
    counts: np.ndarray | None = None

    def scaled(self, scale: float) -> Curve:
        return Curve(self.x, self.value * scale,
                     None if self.spread is None else self.spread * abs(scale), self.counts)


def roi_mask(coordinates: np.ndarray, half_width: float) -> np.ndarray:
    mask = np.abs(coordinates) <= half_width + 1e-9
    if not mask.any():
        raise ValueError(f"central ROI ±{half_width:g} mm contains no bin centers")
    return mask


def average(values: np.ndarray, axes, sd: np.ndarray | None = None,
            spatial: bool = False) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    finite = np.isfinite(values)
    count = finite.sum(axis=axes)
    total = np.where(finite, values, 0).sum(axis=axes)
    mean = np.divide(total, count, out=np.full_like(total, np.nan, dtype=float), where=count > 0)
    spread = None
    if sd is not None:
        valid_sd = np.isfinite(sd) & (sd >= 0)
        variance = np.where(finite & valid_sd, sd ** 2, 0).sum(axis=axes)
        known = ((~finite) | valid_sd).all(axis=axes) & (count > 0)
        spread = np.divide(np.sqrt(variance), count, out=np.full_like(mean, np.nan), where=known)
    elif spatial:
        expanded = mean
        for axis in sorted((axes,) if isinstance(axes, int) else axes):
            expanded = np.expand_dims(expanded, axis)
        variance = np.where(finite, (values - expanded) ** 2, 0).sum(axis=axes)
        spread = np.sqrt(np.divide(variance, count - 1, out=np.full_like(mean, np.nan), where=count > 1))
    return mean, spread, count


def mc_pdd(grid: Grid, hx: float, hy: float) -> Curve:
    ix, iy = np.flatnonzero(roi_mask(grid.x, hx)), np.flatnonzero(roi_mask(grid.y, hy))
    selection = np.ix_(ix, iy, np.arange(len(grid.z)))
    v, s, n = average(grid.dose[selection], (0, 1), None if grid.sd is None else grid.sd[selection])
    return Curve(grid.z, v, s, n)


def plane_mean(plane: Plane, hx: float, hy: float) -> tuple[float, int]:
    region = plane.dose[np.ix_(roi_mask(plane.y, hy), roi_mask(plane.x, hx))]
    finite = region[np.isfinite(region)]
    if not finite.size:
        raise ValueError(f"no finite doses in reference ROI: {plane.path}")
    return float(finite.mean()), int(finite.size)


def reference_value(x: np.ndarray, values: np.ndarray, depth: float) -> float:
    if depth < x[0] or depth > x[-1]:
        raise ValueError(f"reference depth {depth:g} mm outside available range {x[0]:g}–{x[-1]:g} mm")
    exact = np.flatnonzero(np.isclose(x, depth, atol=1e-9, rtol=0))
    if exact.size:
        result = float(values[exact[0]])
    else:
        right = int(np.searchsorted(x, depth))
        if not np.isfinite(values[right-1:right+1]).all():
            raise ValueError("reference depth brackets contain invalid dose values")
        result = float(np.interp(depth, x, values))
    if not np.isfinite(result) or result <= 0:
        raise ValueError("reference ROI mean must be finite and greater than zero")
    return result


def peak_normalize(curve: Curve) -> tuple[Curve, dict]:
    eligible = np.flatnonzero((curve.x >= 0) & np.isfinite(curve.x) & np.isfinite(curve.value))
    if not eligible.size:
        raise ValueError("peak normalization requires finite samples at nonnegative depth")
    index = eligible[np.argmax(curve.value[eligible])]
    peak = float(curve.value[index])
    if peak <= 0:
        raise ValueError("peak normalization requires a positive peak")
    scale = 100.0 / peak
    return curve.scaled(scale), dict(peak_value=peak, peak_depth_mm=float(curve.x[index]),
                                   normalization_factor=scale)


def ssd_shape_factor(depths, source_ssd: float, target_ssd: float, reference_depth: float):
    """Inverse-square shape correction anchored at reference_depth, not dmax."""
    depths = np.asarray(depths, dtype=float)
    if not all(np.isfinite(v) and v > 0 for v in (source_ssd, target_ssd)):
        raise ValueError("SSDs must be finite and positive")
    if not np.isfinite(reference_depth) or reference_depth < 0:
        raise ValueError("SSD correction reference depth must be finite and nonnegative")
    if not np.isfinite(depths).all() or np.any(source_ssd + depths <= 0) or np.any(target_ssd + depths <= 0):
        raise ValueError("SSD correction requires finite depths and positive source-to-point distances")
    return ((target_ssd + reference_depth) / (source_ssd + reference_depth))**2 * (
        (source_ssd + depths) / (target_ssd + depths))**2


def normalization(grid: Grid, planes: list[Plane], hx: float, hy: float,
                  fraction: float, reference_depth: float) -> tuple[float, dict]:
    nx, ny = hx * fraction, hy * fraction
    mc = mc_pdd(grid, nx, ny)
    mc_ref = reference_value(mc.x, mc.value, reference_depth)
    ordered = sorted(planes, key=lambda p: p.depth)
    means_counts = [plane_mean(p, nx, ny) for p in ordered]
    measured = reference_value(np.array([p.depth for p in ordered]),
                               np.array([v for v, _ in means_counts]), reference_depth)
    scale = measured / mc_ref
    return scale, dict(reference_depth_mm=reference_depth, roi_fraction=fraction,
                       roi_x_half_width_mm=nx, roi_y_half_width_mm=ny,
                       mc_reference_mean=mc_ref, measurement_reference_mean=measured, mc_scale=scale,
                       mc_roi_bin_counts=np.unique(mc.counts).tolist(),
                       measurement_roi=[dict(depth_mm=p.depth, mean=v, finite_bins=n)
                                        for p, (v, n) in zip(ordered, means_counts)])


def profile_normalization(grid: Grid, planes: list[Plane], hx: float, hy: float,
                          axis: str, half_width: float, reference_depth: float) -> tuple[float, dict]:
    """Average 1D profile samples equally, then interpolate their means in depth."""
    if axis not in ("x", "y") or not np.isfinite(half_width) or half_width <= 0:
        raise ValueError("profile normalization requires axis x/y and a finite positive half-width")

    def window(coords):
        if coords[0] > -half_width + 1e-9 or coords[-1] < half_width - 1e-9:
            raise ValueError(f"normalization window ±{half_width:g} mm outside {axis.upper()} coordinate coverage")
        return roi_mask(coords, half_width)

    ix = window(grid.x) if axis == "x" else roi_mask(grid.x, hx)
    iy = window(grid.y) if axis == "y" else roi_mask(grid.y, hy)
    selected = grid.dose[np.ix_(ix, iy, np.arange(len(grid.z)))]
    profiles, _, _ = average(selected, 1 if axis == "x" else 0)
    mc_means, _, mc_counts = average(profiles, 0)
    mc_ref = reference_value(grid.z, mc_means, reference_depth)
    ordered = sorted(planes, key=lambda p: p.depth)
    measured_rows = []
    for plane in ordered:
        curve = data_profile(plane, axis, hx, hy)
        values = curve.value[window(curve.x)]
        finite = values[np.isfinite(values)]
        measured_rows.append(dict(depth_mm=plane.depth,
                                  mean=float(finite.mean()) if finite.size else np.nan,
                                  finite_profile_bins=int(finite.size)))
    measured = reference_value(np.array([p.depth for p in ordered]),
                               np.array([r["mean"] for r in measured_rows]), reference_depth)
    scale = measured / mc_ref
    return scale, dict(mode=axis, reference_depth_mm=reference_depth,
                       normalization_half_width_mm=half_width,
                       perpendicular_roi_half_width_mm=hy if axis == "x" else hx,
                       roi_x_half_width_mm=half_width if axis == "x" else hx,
                       roi_y_half_width_mm=half_width if axis == "y" else hy,
                       mc_reference_mean=mc_ref, measurement_reference_mean=measured, mc_scale=scale,
                       mc_profile_means=[dict(depth_mm=float(z), mean=float(v), finite_profile_bins=int(n))
                                         for z, v, n in zip(grid.z, mc_means, mc_counts)],
                       measurement_profile_means=measured_rows)


def mc_profile(grid: Grid, axis: str, depth: float, hx: float, hy: float) -> tuple[Curve, float]:
    # Boundary requests (e.g. the water surface at 0 mm) use the nearest voxel center.
    dz = float(np.median(np.diff(grid.z))) if len(grid.z) > 1 else 2 * float(grid.z[0])
    if depth < grid.z[0] - dz / 2 - 1e-9 or depth > grid.z[-1] + dz / 2 + 1e-9:
        raise ValueError(f"profile depth {depth:g} mm outside MC scoring volume")
    zi = int(np.argmin(np.abs(grid.z - depth)))
    if axis == "x":
        mask = roi_mask(grid.y, hy)
        data = grid.dose[:, mask, zi]
        sd = None if grid.sd is None else grid.sd[:, mask, zi]
        v, s, n = average(data, 1, sd)
        coordinates = grid.x
    else:
        mask = roi_mask(grid.x, hx)
        data = grid.dose[mask, :, zi]
        sd = None if grid.sd is None else grid.sd[mask, :, zi]
        v, s, n = average(data, 0, sd)
        coordinates = grid.y
    return Curve(coordinates, v, s, n), float(grid.z[zi])


def data_profile(plane: Plane, axis: str, hx: float, hy: float) -> Curve:
    if axis == "x":
        v, s, n = average(plane.dose[roi_mask(plane.y, hy), :], 0, spatial=True)
        coordinates = plane.x
    else:
        v, s, n = average(plane.dose[:, roi_mask(plane.x, hx)], 1, spatial=True)
        coordinates = plane.y
    return Curve(coordinates, v, s, n)


def ratio(numerator: Curve, denominator: Curve, threshold_percent: float = 1) -> Curve:
    # Only interpolate within contiguous finite segments; never bridge missing samples.
    nv = np.full_like(denominator.x, np.nan)
    ns = np.full_like(nv, np.nan) if numerator.spread is not None else None
    indices = np.flatnonzero(np.isfinite(numerator.value))
    segments = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
    for segment in segments:
        if len(segment) < 2:
            continue
        inside = (denominator.x >= numerator.x[segment[0]]) & (denominator.x <= numerator.x[segment[-1]])
        nv[inside] = np.interp(denominator.x[inside], numerator.x[segment], numerator.value[segment])
        if ns is not None:
            # Linear interpolation propagates independent endpoint variances.
            xx = denominator.x[inside]
            j = np.clip(np.searchsorted(numerator.x[segment], xx, side="right"), 1, len(segment)-1)
            left, right = segment[j-1], segment[j]
            w = (xx - numerator.x[left]) / (numerator.x[right] - numerator.x[left])
            ns[inside] = np.sqrt((1-w)**2 * numerator.spread[left]**2 + w**2 * numerator.spread[right]**2)
    finite = np.isfinite(denominator.value)
    if not finite.any():
        raise ValueError("ratio denominator has no finite samples")
    cutoff = np.max(denominator.value[finite]) * threshold_percent / 100
    valid = finite & np.isfinite(nv) & (denominator.value > max(0, cutoff))
    value = np.divide(nv, denominator.value, out=np.full_like(nv, np.nan), where=valid)
    spread = None
    if ns is not None and denominator.spread is not None:
        spread = np.full_like(value, np.nan)
        spread[valid] = np.sqrt((ns[valid] / denominator.value[valid])**2 +
                                (nv[valid] * denominator.spread[valid] / denominator.value[valid]**2)**2)
    return Curve(denominator.x, value, spread)


def pvdr(curve: Curve, *, ignore_edge_peaks: bool = True, prominence_fraction: float = 0.5,
         expected_ctc_mm: float | None = None) -> tuple[dict, list[dict]]:
    v = curve.value
    finite = np.isfinite(v)
    summary = dict(pvdr=None, mean_peak=None, mean_valley=None, retained_peaks=0,
                   retained_valleys=0, ignore_edge_peaks=ignore_edge_peaks, failure_reason=None,
                   peak_sd=None, valley_sd=None, mean_peak_se=None, mean_valley_se=None,
                   pvdr_uncertainty=None, uncertainty_failure_reason="PVDR unavailable",
                   uncertainty_method="Retained-feature sample SD (ddof=1) / sqrt(N); independent means and samples")
    if finite.sum() < 3 or np.ptp(v[finite]) <= 0:
        summary["failure_reason"] = "profile has no measurable modulation"
        return summary, []
    prominence = prominence_fraction * np.ptp(v[finite])
    indices = np.flatnonzero(finite)
    segments = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
    distance = None
    if expected_ctc_mm is not None:
        distance = max(1, int(np.ceil(0.6 * expected_ctc_mm / np.median(np.diff(curve.x)))))
    peaks = []
    for segment in segments:
        if len(segment) >= 3:
            found, _ = find_peaks(v[segment], prominence=prominence, distance=distance)
            peaks.extend(segment[found].tolist())
    retained = peaks[1:-1] if ignore_edge_peaks else peaks
    keep = set(retained)
    features = [dict(kind="peak", index=i, position_mm=float(curve.x[i]), dose=float(v[i]),
                     retained=i in keep) for i in peaks]
    valleys = []
    for left, right in zip(peaks, peaks[1:]):
        if not finite[left:right+1].all():
            continue
        i = left + 1 + int(np.argmin(v[left+1:right]))
        used = left in keep and right in keep
        features.append(dict(kind="valley", index=i, position_mm=float(curve.x[i]), dose=float(v[i]), retained=used))
        if used:
            valleys.append(i)
    summary.update(retained_peaks=len(retained), retained_valleys=len(valleys))
    for kind, indices in (("peak", retained), ("valley", valleys)):
        if len(indices) >= 2:
            sd = float(np.std(v[indices], ddof=1))
            summary[f"{kind}_sd"] = sd
            summary[f"mean_{kind}_se"] = sd / np.sqrt(len(indices))
    if len(retained) < 2 or len(valleys) != len(retained) - 1:
        summary["failure_reason"] = "need at least two retained peaks and finite intervening valleys"
    else:
        mp, mv = float(np.mean(v[retained])), float(np.mean(v[valleys]))
        summary.update(mean_peak=mp, mean_valley=mv)
        if mp <= 0 or mv <= 0:
            summary["failure_reason"] = "mean peak and valley doses must be positive"
        else:
            summary["pvdr"] = mp / mv
            if len(valleys) < 2:
                summary["uncertainty_failure_reason"] = "need at least two retained peaks and two retained valleys"
            else:
                summary["pvdr_uncertainty"] = float(summary["pvdr"] * np.hypot(
                    summary["mean_peak_se"] / mp, summary["mean_valley_se"] / mv))
                summary["uncertainty_failure_reason"] = None
    return summary, features
