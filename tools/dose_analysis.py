"""Command-line workflows and reproducible saved analysis artifacts."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import textwrap

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "minibeam-dose-analysis-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from tools.analysis_config import DEFAULT_CONFIG, field_settings, load_config, validate_settings
from tools.analysis_io import read_agl, read_grid, read_plane
from tools.analysis_report import write_report
from tools.analysis_numerics import (Curve, data_profile, mc_pdd, mc_profile, normalization,
                       plane_mean, profile_normalization, pvdr, ratio, reference_value, ssd_shape_factor, peak_normalize)


COMMANDS = {
    "pdd": "Compare depth-dose curves, or plot MC alone",
    "lateral": "Compare X/Y lateral profiles, or plot MC alone",
    "mc-fields-ratio": "Collimator/open MC dose per history and ratios",
    "data-fields-ratio": "Slit/open measured dose per MU and ratios",
    "output-factor": "Open-field output factors relative to 10 x 10 cm",
    "pvdr-depth": "Overlay MC and measured slit PVDR versus depth",
}

def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="python tools/dose_analysis.py", description="Standalone MC and measured dose analysis")
    commands = root.add_subparsers(dest="command", required=True)
    for name, description in COMMANDS.items():
        p = commands.add_parser(name, help=description, description=description)
        p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        if name == "pvdr-depth":
            selection = p.add_mutually_exclusive_group()
            selection.add_argument("--field", type=int, default=10, help="field width in cm (default: 10)")
            selection.add_argument("--all-fields", action="store_true")
            selection.add_argument("--fields", type=int, nargs="+",
                                   help="field widths in cm to calculate and include in the aggregate plot")
        elif name != "output-factor":
            p.add_argument("--field", type=int, default=10, help="field width in cm (default: 10)")
        p.add_argument("--output-dir", type=Path, help="output root; field/command/source folders are added")
        if name in {"pdd", "lateral", "output-factor"}:
            p.add_argument("--reference-depth-mm", type=float)
            p.add_argument("--normalization-roi-fraction", type=float)
        p.add_argument("--x-roi-half-width-mm", type=float)
        p.add_argument("--y-roi-half-width-mm", type=float)
        if name not in {"output-factor", "pdd"}:
            p.add_argument("--depth-mm", type=float, nargs="+", help="profile or PVDR depths (mm)")
        if name in {"pdd", "lateral"}:
            p.add_argument("--source", choices=("open", "collimator"), default="open")
            p.add_argument("--mc-only", action="store_true")
        if name == "pdd":
            p.add_argument("--agl-ssd-correction", action="store_true",
                           help="show the approximate SSD-corrected AGL curve and ratio (hidden by default)")
        if name == "lateral":
            p.add_argument("--combine-depths", action="store_true", help="overlay depths in one lateral figure")
            p.add_argument("--normalization-mode", choices=("2d", "x", "y"))
            p.add_argument("--axis", choices=("x", "y", "both"), default="both")
            p.add_argument("--mc-shift-x-mm", type=float,
                           help="MC translation in comparison X (mm; positive moves right)")
            p.add_argument("--mc-shift-y-mm", type=float,
                           help="MC translation in comparison Y (mm; positive moves right)")
        if name in {"mc-fields-ratio", "data-fields-ratio"}:
            p.add_argument("--axis", choices=("x", "y"))
            p.add_argument("--ratio-threshold-percent", type=float)
        if name == "pvdr-depth":
            p.add_argument("--mc-axis", choices=("x", "y"), help="MC slit-modulation axis")
            p.add_argument("--data-axis", choices=("x", "y"), help="measured slit-modulation axis")
            edges = p.add_mutually_exclusive_group()
            edges.add_argument("--include-edge-peaks", dest="ignore_edge_peaks", action="store_false")
            edges.add_argument("--ignore-edge-peaks", dest="ignore_edge_peaks", action="store_true")
            p.set_defaults(ignore_edge_peaks=None)
            p.add_argument("--peak-prominence-fraction", type=float)
            p.add_argument("--expected-ctc-mm", type=float)
    report = commands.add_parser("report", help="Rebuild Markdown report without recalculation")
    report.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "outputs/dose_analysis")
    return root


def settings_for(config: dict, field: int, args: argparse.Namespace) -> dict:
    s = field_settings(config, field)
    for key in ("reference_depth_mm", "x_roi_half_width_mm", "y_roi_half_width_mm",
                "normalization_roi_fraction", "ratio_threshold_percent",
                "mc_shift_x_mm", "mc_shift_y_mm",
                "ignore_edge_peaks", "peak_prominence_fraction", "expected_ctc_mm"):
        value = getattr(args, key, None)
        if value is not None:
            s[key] = value
            if key in ("mc_shift_x_mm", "mc_shift_y_mm"):
                s["mc_shift_origins"][key] = "CLI"
    for argument, key in (("depth_mm", "depths_mm"), ("mc_axis", "pvdr_mc_axis"), ("data_axis", "pvdr_data_axis"),
                          ("normalization_mode", "lateral_normalization_mode")):
        value = getattr(args, argument, None)
        if value is not None:
            s[key] = value
    if getattr(args, "axis", None) in ("x", "y") and args.command != "lateral":
        s["profile_axis"] = args.axis
    if args.output_dir:
        s["output_root"] = args.output_dir.expanduser().resolve()
    validate_settings(s)
    return s


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [plain(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(plain(rows))


class Run:
    def __init__(self, settings: dict, args: argparse.Namespace, config_path: Path, *, aggregate=False):
        self.s, self.args = settings, args
        default_source = {"output-factor": "open", "pvdr-depth": "collimator"}.get(args.command, "open-collimator")
        source = getattr(args, "source", default_source)
        if getattr(args, "mc_only", False):
            source += "-mc-only"
        self.out = settings["output_root"]
        if not aggregate:
            self.out /= f"field_{settings['field_cm']}cm"
        self.out = self.out / args.command / source
        self.meta = dict(command=args.command, field_cm=None if aggregate else settings["field_cm"],
                         generated_at=datetime.now(timezone.utc).isoformat(), config=config_path,
                         settings=settings, arguments=vars(args), inputs=[], centering=[], results={},
                         uncertainty="MHA has no MC uncertainty; ROI voxel covariance unavailable. Measurement bands describe spatial spread; PVDR estimates describe retained-feature variation, not MC or measurement uncertainty.")
        self.grids = {}
        self.planes = {}

    def input(self, path: Path):
        stat = path.stat()
        self.meta["inputs"].append(dict(path=path.resolve(), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns))

    def grid(self, source: str):
        if source not in self.grids:
            key = "mc_open" if source == "open" else "mc_collimator"
            if key not in self.s:
                raise ValueError(f"{key} is not configured; set a dose_scoring_grid.mha path in {self.meta['config']}")
            grid = read_grid(self.s[key])
            if not np.isclose(grid.provenance["field_width_mm"], self.s["field_cm"] * 10):
                raise ValueError("Saved water field size does not match configured field")
            if not np.isclose(grid.provenance["ssd_mm"], self.s["mc_ssd_mm"]):
                raise ValueError("Saved water SSD does not match configured mc_ssd_mm")
            for path in grid.provenance["input_files"]:
                self.input(Path(path))
            self.meta.setdefault("saved_dose", {})[source] = grid.provenance
            if grid.provenance["review_required"]:
                print(f"WARNING: {source} dose has scorer warnings; validity requires user review")
            self.grids[source] = grid
        return self.grids[source]

    def maps(self, source: str):
        if source not in self.planes:
            entries = self.s["open_maps" if source == "open" else "slit_maps"]
            if not entries:
                raise ValueError(f"no {source} measurement maps configured")
            planes = []
            prefix = "slit" if source == "collimator" else "open"
            for entry in entries:
                p = read_plane(entry["file"], entry["depth_mm"], slit=source == "collimator",
                               offset_x=self.s.get(f"{prefix}_map_x_offset_mm"), offset_y=self.s.get(f"{prefix}_map_y_offset_mm"))
                self.input(entry["file"])
                planes.append(p)
                if p.centering:
                    self.meta["centering"].append(dict(field_cm=self.s["field_cm"], depth_mm=p.depth,
                                                       path=p.path, source=source, **p.centering))
            self.planes[source] = planes
        return self.planes[source]

    def prepare(self):
        self.out.mkdir(parents=True, exist_ok=True)

    def save(self, fig, name: str):
        self.prepare()
        fig.savefig(self.out / f"{name}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    def finish(self):
        self.prepare()
        (self.out / "metadata.json").write_text(json.dumps(plain(self.meta), indent=2, allow_nan=False) + "\n")
        for entry in self.meta["centering"]:
            print(f"Field {entry['field_cm']} cm, depth {entry['depth_mm']:g} mm: "
                  f"map center X={entry['center_x_mm']:.6g}, Y={entry['center_y_mm']:.6g} mm; "
                  f"applied X={entry['offset_x_mm']:.6g} ({entry['x_mode']}), "
                  f"Y={entry['offset_y_mm']:.6g} ({entry['y_mode']})")
        if self.meta["centering"]:
            # Print one copy-ready block per distinct offset pair, with depth applicability.
            grouped = {}
            for e in self.meta["centering"]:
                grouped.setdefault((e["field_cm"], e["source"], e["offset_x_mm"], e["offset_y_mm"]), []).append(e["depth_mm"])
            for (field, source, ox, oy), depths in grouped.items():
                prefix = "slit" if source == "collimator" else "open"
                print(f'# Add inside [fields."{field}"] (depths {", ".join(f"{d:g}" for d in depths)} mm):\n'
                      f'{prefix}_map_x_offset_mm = {ox:.8g}\n{prefix}_map_y_offset_mm = {oy:.8g}')
        print(f"Saved: {self.out}")
        print(f"Report: {write_report(self.s['output_root'])}")


def draw(ax, curve: Curve, label: str, color=None, *, points=False, band_label=None, linestyle=None, band_alpha=.15):
    line, = ax.plot(curve.x, curve.value, "o" if points else "-", label=label, color=color, markersize=4)
    if linestyle is not None:
        line.set_linestyle(linestyle)
    if curve.spread is not None and np.isfinite(curve.spread).any():
        ax.fill_between(curve.x, curve.value-curve.spread, curve.value+curve.spread,
                        color=line.get_color(), alpha=band_alpha, label=band_label)
    ax.grid(alpha=.25)


def rows_curve(curve: Curve, **labels) -> list[dict]:
    return [dict(**labels, coordinate_mm=float(x), value=float(v),
                 spread=None if curve.spread is None else float(curve.spread[i]),
                 finite_bins=None if curve.counts is None else int(curve.counts[i]))
            for i, (x, v) in enumerate(zip(curve.x, curve.value))]


def map_at(planes, depth):
    matches = [p for p in planes if np.isclose(p.depth, depth, atol=1e-9, rtol=0)]
    if not matches:
        raise ValueError(f"no configured measurement map at {depth:g} mm")
    return matches[0]


def dimensions(s):
    return s["x_roi_half_width_mm"], s["y_roi_half_width_mm"]


def run_pdd(run: Run):
    a, s = run.args, run.s
    grid = run.grid(a.source)
    hx, hy = dimensions(s)
    curve = mc_pdd(grid, hx, hy)
    rows = []
    if a.mc_only:
        fig, ax = plt.subplots(figsize=(8, 5))
        draw(ax, curve, "MC")
        ax.set(xlabel="Depth in water (mm)", ylabel=f"Accumulated dose ({grid.unit})",
               title=f"{s['field_cm']} cm {a.source} MC depth dose")
        ax.legend()
        run.save(fig, "pdd-mc")
        rows.extend(rows_curve(curve, comparison="mc-only", source="mc", unit=grid.unit))
    else:
        if "agl" not in s:
            raise ValueError("open-field AGL PDD input is not configured")
        x, values, agl_meta = read_agl(s["agl"])
        run.input(s["agl"])
        ref = s["reference_depth_mm"]
        fraction = s["normalization_roi_fraction"]
        scaled, mc_peak = peak_normalize(curve)
        agl, agl_peak = peak_normalize(Curve(x, values))
        r = ratio(scaled, agl, 0)
        # Any constant anchor cancels in peak normalization; use the surface.
        factor = ssd_shape_factor(x, s["agl_ssd_mm"], s["mc_ssd_mm"], 0)
        approximate, approx_peak = peak_normalize(Curve(x, values * factor))
        approximate_ratio = ratio(scaled, approximate, 0)
        fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1]}, layout="constrained")
        draw(axes[0], scaled, "MC (peak normalized)")
        draw(axes[0], agl, f"AGL — SSD {s['agl_ssd_mm']:g} mm")
        show_correction = a.agl_ssd_correction
        if show_correction:
            axes[0].plot(x, approximate.value, "--", color="tab:green",
                         label=f"AGL → SSD {s['mc_ssd_mm']:g} mm (approx.)")
        axes[0].set(ylabel="PDD (% of peak)", title=f"{s['field_cm']} cm open PDD · peak normalized")
        axes[0].legend()
        draw(axes[1], r, "MC / AGL")
        if show_correction:
            axes[1].plot(approximate_ratio.x, approximate_ratio.value, "--", color="tab:green",
                         label="MC / AGL approximation")
        axes[1].legend(fontsize=8)
        if show_correction:
            fig.supxlabel("Peak-normalized inverse-square approximation; no scatter or field-size correction",
                          fontsize=8)
        axes[1].axhline(1, color="gray", linewidth=.8)
        axes[1].set(xlabel="Depth in water (mm)", ylabel="MC / AGL", xlim=(0, 150), ylim=(.95, 1.05))
        run.save(fig, "pdd-agl")
        rows += rows_curve(scaled, comparison="agl", source="mc", unit="percent")
        rows += rows_curve(agl, comparison="agl", source="measurement", unit="percent")
        rows += rows_curve(r, comparison="agl-ratio", source="mc/agl", unit="ratio")
        approximation_rows = rows_curve(approximate, comparison="agl-ssd-approximation",
                                       source="measurement-approximation", unit="percent")
        for row, correction in zip(approximation_rows, factor):
            row["ssd_correction_factor"] = float(correction)
        rows += approximation_rows
        rows += rows_curve(approximate_ratio, comparison="agl-ssd-approximation-ratio",
                           source="mc/agl-approximation", unit="ratio")
        run.meta["results"]["agl_ssd_approximation"] = dict(
            source_ssd_mm=s["agl_ssd_mm"], target_ssd_mm=s["mc_ssd_mm"],
            formula="(target_ssd/source_ssd)^2 * ((source_ssd+d)/(target_ssd+d))^2; then normalize to own peak",
            method="Peak-normalized inverse-square shape approximation",
            limitations="No scatter, field-size, or electron-contamination correction",
            source_agl_file_preserved=True, shown_in_plot=show_correction)
        run.meta["results"]["agl"] = dict(metadata=agl_meta, normalization="independent sampled peaks at nonnegative depths",
            peaks=dict(mc=mc_peak, measurement=agl_peak, approximation=approx_peak),
            mc_scale=mc_peak["normalization_factor"], roi_x_half_width_mm=hx, roi_y_half_width_mm=hy,
            display_limits=dict(depth_mm=[0,150], ratio=[.95,1.05]),
            mc_roi_bin_counts=np.unique(curve.counts).tolist())
        planes = run.maps("open")
        scale, info = normalization(grid, planes, hx, hy, fraction, ref)
        vals_counts = [plane_mean(p, hx, hy) for p in planes]
        measured = Curve(np.array([p.depth for p in planes]), np.array([v for v, _ in vals_counts]),
                         counts=np.array([n for _, n in vals_counts]))
        fig, ax = plt.subplots(figsize=(8, 5))
        draw(ax, curve.scaled(scale), "MC matched to Sphinx")
        draw(ax, measured, "Sphinx ROI mean", points=True)
        ax.set(xlabel="Depth in water (mm)", ylabel="Measured dose (cGy)",
               title=f"{s['field_cm']} cm open PDD · reference {ref:g} mm")
        ax.legend()
        run.save(fig, "pdd-sphinx")
        rows += rows_curve(curve.scaled(scale), comparison="sphinx", source="mc", unit="cGy")
        rows += rows_curve(measured, comparison="sphinx", source="measurement", unit="cGy")
        run.meta["results"]["sphinx_normalization"] = info
    run.prepare()
    write_rows(run.out / "pdd.csv", rows)


def run_lateral(run: Run):
    a, s = run.args, run.s
    unshifted_grid = run.grid(a.source)
    sx, sy = s["mc_shift_x_mm"], s["mc_shift_y_mm"]
    grid = unshifted_grid.shift_xy(sx, sy)
    origins = s["mc_shift_origins"]
    run.meta["results"]["mc_shifts"] = dict(
        x_mm=sx, y_mm=sy, origins=origins,
        coordinate_system="analysis X = MHA X; analysis Y = MHA Z",
        applied_before="profile ROI extraction and reference-ROI normalization",
    )
    print(f"MC shifts: comparison X={sx:+g} mm ({origins['mc_shift_x_mm']}), "
          f"comparison Y={sy:+g} mm ({origins['mc_shift_y_mm']})")
    hx, hy = dimensions(s)
    planes = [] if a.mc_only else run.maps(a.source)
    scale = 1.0
    norm_label = "MC only; no measurement normalization"
    if not a.mc_only:
        mode = s["lateral_normalization_mode"]
        if mode == "2d":
            scale, info = normalization(grid, planes, hx, hy, s["normalization_roi_fraction"], s["reference_depth_mm"])
            info["mode"] = "2d"
        else:
            scale, info = profile_normalization(grid, planes, hx, hy, mode,
                                                (hx if mode == "x" else hy) * s["normalization_roi_fraction"],
                                                s["reference_depth_mm"])
            info["roi_fraction"] = s["normalization_roi_fraction"]
        run.meta["results"]["normalization"] = info
        norm_label = (f"Normalization {mode.upper()} (fraction {s['normalization_roi_fraction']:g}) at {s['reference_depth_mm']:g} mm: "
                      f"X ±{info['roi_x_half_width_mm']:g}, Y ±{info['roi_y_half_width_mm']:g} mm")
        print(f"{norm_label}; MC mean={info['mc_reference_mean']:.6g} {grid.unit}; "
              f"measurement mean={info['measurement_reference_mean']:.6g} cGy; scale={scale:.6g}")
    mapping = {"x": "x", "y": "y"}
    run.meta["results"]["mc_axis_mapping"] = mapping
    print(f"Lateral axis mapping: comparison X ← MC {mapping['x'].upper()}, comparison Y ← MC {mapping['y'].upper()}")
    axes_to_plot = ["x", "y"] if a.axis == "both" else [a.axis]
    rows = []
    figures, depth_mapping = [], []
    for index, depth in enumerate(s["depths_mm"]):
        plane = None if a.mc_only else map_at(planes, depth)
        if not a.combine_depths or index == 0:
            fig, axes = plt.subplots(1 if a.mc_only else 2, len(axes_to_plot),
                                     figsize=(7*len(axes_to_plot), 5 if a.mc_only else 7),
                                     sharex="col", squeeze=False, layout="constrained",
                                     gridspec_kw=None if a.mc_only else {"height_ratios": [3, 1]})
        color = plt.get_cmap("tab10")(index % 10) if a.combine_depths else None
        suffix = f" · {depth:g} mm" if a.combine_depths else ""
        fig.suptitle(f"MC shifts in comparison coordinates: X={sx:+g} mm, Y={sy:+g} mm\n{norm_label}")
        for column, (axis, ax) in enumerate(zip(axes_to_plot, axes[0])):
            raw, actual = mc_profile(grid, axis, depth, hx, hy)
            mc = raw.scaled(scale)
            draw(ax, mc, f"MC {mapping[axis].upper()}{suffix}", color=color,
                 band_alpha=.08 if a.combine_depths else .15)
            unit = grid.unit if a.mc_only else "cGy"
            mc_rows = rows_curve(mc, depth_mm=depth, source="mc", axis=axis, original_axis=mapping[axis],
                                 actual_mc_depth_mm=actual, raw_scale=scale, unit=unit)
            original_coordinates = unshifted_grid.x if axis == "x" else unshifted_grid.y
            for row, coordinate in zip(mc_rows, original_coordinates):
                row.update(original_coordinate_mm=float(coordinate),
                           shifted_coordinate_mm=row["coordinate_mm"],
                           mc_shift_mm=sx if axis == "x" else sy)
            rows += mc_rows
            if plane is not None:
                measured = data_profile(plane, axis, hx, hy)
                draw(ax, measured, f"Measurement {axis.upper()}{suffix}", color=color,
                     linestyle="--" if a.combine_depths else None,
                     band_label="Measurement ROI spatial SD" if index == 0 or not a.combine_depths else None,
                     band_alpha=.08 if a.combine_depths else .15)
                rows += rows_curve(measured, depth_mm=depth, source="measurement", axis=axis,
                                   original_axis=axis, actual_mc_depth_mm=None, raw_scale=1.0, unit=unit)
                # Compare the already normalized MC to measurement; do not fit again.
                # Interpolate only inside MC coverage and omit nonpositive denominators.
                relative = ratio(mc, measured, 0)
                ratio_ax = axes[1, column]
                draw(ratio_ax, relative, f"MC / measurement{suffix}", color=color)
                if not a.combine_depths or index == 0:
                    ratio_ax.axhline(1, color="gray", linewidth=.8, linestyle="--")
                ratio_ax.set(xlabel=f"Comparison {axis.upper()} (mm)",
                             ylabel="MC / measurement", ylim=(.95, 1.05))
                if a.combine_depths:
                    ratio_ax.legend(fontsize=8)
                rows += rows_curve(relative, depth_mm=depth, source="mc/measurement", axis=axis,
                                   actual_mc_depth_mm=actual, unit="ratio")
            ax.set(xlabel=f"Comparison {axis.upper()} (mm)", ylabel=f"Dose ({unit})",
                   title=(f"{s['field_cm']} cm {a.source} · combined depths" if a.combine_depths else
                          f"{s['field_cm']} cm {a.source} · {depth:g} mm (MC {actual:g} mm)"))
            ax.legend(fontsize=8)
            if not a.mc_only:
                ax.set_xlabel("")
        depth_mapping.append(dict(requested_depth_mm=depth, actual_mc_depth_mm=actual))
        if not a.combine_depths:
            name = f"lateral-{a.axis}-depth-{depth:g}mm"
            run.save(fig, name)
            figures.append(f"{name}.png")
    if a.combine_depths:
        name = f"lateral-{a.axis}-combined-depths"
        run.save(fig, name)
        figures.append(f"{name}.png")
    run.meta["results"]["lateral_figures"] = figures
    run.meta["results"]["depth_mapping"] = depth_mapping
    if not a.mc_only:
        run.meta["results"]["lateral_ratio"] = dict(
            definition="normalized MC / measurement", display_limits=[.95, 1.05],
            interpolation="MC interpolated to measurement coordinates without extrapolation",
            exclusions="Nonpositive measurement values and unavailable MC samples",
            note="Ratios outside display limits remain in profiles.csv")
    run.prepare()
    write_rows(run.out / "profiles.csv", rows)


def run_fields_ratio(run: Run, *, mc: bool):
    s = run.s
    hx, hy = dimensions(s)
    axis = s["profile_axis"]
    if mc:
        open_input, slit_input = run.grid("open"), run.grid("collimator")
        if open_input.normalization != slit_input.normalization:
            raise ValueError("MC dose units differ between open and collimator inputs")
        denominators = (open_input.exposure, slit_input.exposure)
        unit = open_input.provenance["coefficient_units"]
    else:
        open_input, slit_input = run.maps("open"), run.maps("collimator")
        denominators = (s["open_monitor_units"], s["slit_monitor_units"])
        unit = "cGy/MU"
    run.meta["results"]["dose_divisors"] = dict(open=denominators[0], collimator=denominators[1])
    rows, ratios = [], []
    for depth in s["depths_mm"]:
        actual = {}
        if mc:
            oc, actual["open"] = mc_profile(open_input, axis, depth, hx, hy)
            sc, actual["collimator"] = mc_profile(slit_input, axis, depth, hx, hy)
        else:
            oc = data_profile(map_at(open_input, depth), axis, hx, hy)
            sc = data_profile(map_at(slit_input, depth), axis, hx, hy)
        oc, sc = oc.scaled(1/denominators[0]), sc.scaled(1/denominators[1])
        rc = ratio(sc, oc, s["ratio_threshold_percent"])
        fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
        band = "uncertainty unavailable" if mc else "ROI spatial SD"
        draw(axes[0], oc, "Open field", band_label=f"Open {band}")
        draw(axes[0], sc, "Collimator", band_label=f"Collimator {band}")
        axes[0].set(ylabel=unit, title=f"{s['field_cm']} cm · {axis.upper()} · depth {depth:g} mm")
        axes[0].legend(fontsize=8)
        draw(axes[1], rc, "Collimator/open", band_label=f"Propagated {band}")
        axes[1].set(xlabel=f"{axis.upper()} (mm)", ylabel="Collimator/open")
        run.save(fig, f"ratio-{axis}-depth-{depth:g}mm")
        for source, curve in (("open", oc), ("collimator", sc)):
            rows += rows_curve(curve, source=source, depth_mm=depth, axis=axis, unit=unit,
                               actual_mc_depth_mm=actual.get(source))
        ratios += rows_curve(rc, depth_mm=depth, axis=axis, unit="ratio")
    run.prepare()
    write_rows(run.out / "profiles.csv", rows)
    write_rows(run.out / "ratios.csv", ratios)


def plot_pvdr_profile(ax, curve, summary, features, *, source, axis, unit):
    """Display the exact samples and statistics returned by the PVDR calculation."""
    ax.plot(curve.x, curve.value, color="0.25", linewidth=1, label="Dose profile")
    for kind, marker, color, prefix in (("peak", "^", "tab:red", "P"),
                                         ("valley", "v", "tab:blue", "V")):
        for retained in (True, False):
            selected = [f for f in features if f["kind"] == kind and f["retained"] == retained]
            if not selected:
                continue
            ax.scatter([f["position_mm"] for f in selected], [f["dose"] for f in selected],
                       marker=marker, s=35, color=color if retained else "0.65",
                       facecolors=color if retained else "none", zorder=3,
                       label=f"{'Retained' if retained else 'Excluded'} {kind}s")
            if retained:
                for number, feature in enumerate(selected, 1):
                    ax.annotate(f"{prefix}{number}", (feature["position_mm"], feature["dose"]),
                                xytext=(0, 8 if kind == "peak" else -13),
                                textcoords="offset points", ha="center", fontsize=7, color=color)
        mean = summary[f"mean_{kind}"]
        if mean is not None:
            ax.axhline(mean, color=color, linestyle="--", linewidth=1,
                       label=f"Mean retained {kind} = {mean:.6g} {unit}")
    value = summary["pvdr"]
    status = f"PVDR = {value:.6g}" if value is not None else "PVDR unavailable"
    if summary.get("pvdr_uncertainty") is not None:
        status += f" ± {summary['pvdr_uncertainty']:.3g} (feature-variation SE)"
    ax.set(title=f"{source} ({axis.upper()}) — {status}",
           xlabel=f"{axis.upper()} position (mm)", ylabel=f"Dose ({unit})")
    if value is None:
        ax.text(.02, .98, textwrap.fill(summary["failure_reason"], 65),
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(facecolor="white", alpha=.9, edgecolor="0.8"))
    ax.margins(y=.18)
    ax.grid(alpha=.2)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.16), fontsize=8, ncol=2)


def draw_pvdr_uncertainty(ax, rows, source, color):
    x = np.array([r["depth_mm"] for r in rows])
    y = np.array([r["pvdr"] if r["pvdr"] is not None else np.nan for r in rows])
    se = np.array([r.get("pvdr_uncertainty") if r.get("pvdr_uncertainty") is not None else np.nan
                   for r in rows])
    valid = np.isfinite(y) & np.isfinite(se)
    if source == "mc":
        ax.fill_between(x, y-se, y+se, where=valid, color=color, alpha=.18)
    elif valid.any():
        ax.errorbar(x[valid], y[valid], yerr=se[valid], fmt="none", ecolor=color, capsize=3)


def run_pvdr(run: Run):
    s = run.s
    grid, planes = run.grid("collimator"), run.maps("collimator")
    hx, hy = dimensions(s)
    summaries, features, profile_rows = [], [], []
    diagnostic_figures = []
    for depth in s["depths_mm"]:
        mc, actual = mc_profile(grid, s["pvdr_mc_axis"], depth, hx, hy)
        data = data_profile(map_at(planes, depth), s["pvdr_data_axis"], hx, hy)
        profile_fig, axes = plt.subplots(1, 2, figsize=(16, 6), layout="constrained")
        profile_fig.suptitle(f"{s['field_cm']} cm slit profiles at {depth:g} mm depth "
                             f"(MC voxel depth {actual:g} mm)")
        for ax, (source, curve, axis) in zip(axes, (("mc", mc, s["pvdr_mc_axis"]),
                                                   ("measurement", data, s["pvdr_data_axis"]))):
            summary, found = pvdr(curve, ignore_edge_peaks=s["ignore_edge_peaks"],
                                  prominence_fraction=s["peak_prominence_fraction"],
                                  expected_ctc_mm=s.get("expected_ctc_mm"))
            labels = dict(field_cm=s["field_cm"], depth_mm=depth, source=source, axis=axis,
                          actual_mc_depth_mm=actual if source == "mc" else None,
                          dose_unit=grid.unit if source == "mc" else "cGy")
            summaries.append(dict(**labels, **summary))
            features += [dict(**labels, **feature) for feature in found]
            profile_rows += rows_curve(curve, **labels)
            plot_pvdr_profile(ax, curve, summary, found,
                              source="MC" if source == "mc" else "Measurement",
                              axis=axis, unit=labels["dose_unit"])
            value = summary["pvdr"]
            means = ", ".join(f"mean {kind}=" +
                              (f"{summary[f'mean_{kind}']:.6g} {labels['dose_unit']}"
                               if summary[f"mean_{kind}"] is not None else "N/A")
                              for kind in ("peak", "valley"))
            print(f"Field {s['field_cm']} cm, {depth:g} mm, {source} {axis.upper()}: " +
                  (f"PVDR={value:.6g} ({summary['retained_peaks']} peaks, {summary['retained_valleys']} valleys)"
                   if value is not None else f"PVDR unavailable: {summary['failure_reason']}") + f"; {means}; " +
                  (f"PVDR uncertainty (1 SE)={summary['pvdr_uncertainty']:.6g}"
                   if summary.get("pvdr_uncertainty") is not None else
                   f"uncertainty unavailable: {summary.get('uncertainty_failure_reason', 'not estimated')}"))
        name = f"pvdr-profile-depth-{depth:g}mm"
        run.save(profile_fig, name)
        diagnostic_figures.append(f"{name}.png")
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    for source, label in (("mc", f"MC ({s['pvdr_mc_axis'].upper()})"),
                          ("measurement", f"Measurement ({s['pvdr_data_axis'].upper()})")):
        selected = [row for row in summaries if row["source"] == source]
        line, = ax.plot([r["depth_mm"] for r in selected],
                [r["pvdr"] if r["pvdr"] is not None else np.nan for r in selected],
                marker="o", linestyle="none" if source == "measurement" else "-", label=label,
                **({"color": "black", "markerfacecolor": "none", "markeredgecolor": "black"}
                   if source == "measurement" else {}))
        draw_pvdr_uncertainty(ax, selected, source, line.get_color())
    fig.supxlabel("Band / error bars: ±1 SE from retained-feature variation", fontsize=9)
    unavailable = sum(row["pvdr"] is None for row in summaries)
    title = f"{s['field_cm']} cm slit PVDR versus depth"
    if unavailable:
        title += f"\n{unavailable} unavailable result(s); see PVDR summary"
    ax.set(xlabel="Depth in water (mm)", ylabel="PVDR", title=title)
    ax.set_xticks(s["depths_mm"])
    ax.grid(alpha=.25)
    ax.legend()
    run.save(fig, "pvdr-depth")
    write_rows(run.out / "pvdr.csv", summaries)
    # Always produce the features export, including a header if detection found nothing.
    if features:
        write_rows(run.out / "features.csv", features)
    else:
        (run.out / "features.csv").write_text("field_cm,depth_mm,source,axis,kind,index,position_mm,dose,retained\n")
    write_rows(run.out / "profiles.csv", profile_rows)
    run.meta["results"]["pvdr"] = summaries
    run.meta["results"]["pvdr_profile_figures"] = diagnostic_figures


def run_output_factor(config: dict, args: argparse.Namespace):
    reference_field = 10
    if reference_field not in config["fields"]:
        raise ValueError("output-factor requires a configured 10 cm reference field")
    run = Run(settings_for(config, 10, args), args, config["path"], aggregate=True)
    rows = []
    run.meta["results"]["field_settings"] = {}
    for size in sorted(config["fields"]):
        s = settings_for(config, size, args)
        # Output factors must use the same reference depth and physical ROI for all fields.
        for key in ("reference_depth_mm", "x_roi_half_width_mm", "y_roi_half_width_mm", "normalization_roi_fraction"):
            if s[key] != run.s[key]:
                raise ValueError(f"output-factor requires identical {key} across fields; supply a common CLI override")
        local = Run(s, args, config["path"])
        grid = local.grid("open")
        if rows and grid.normalization != run.meta["results"]["normalization"]:
            raise ValueError("Incompatible source normalization across fields")
        run.meta["results"]["normalization"] = grid.normalization
        hx, hy = dimensions(s)
        f = s["normalization_roi_fraction"]
        mc = mc_pdd(grid, hx*f, hy*f)
        mc_ref = reference_value(mc.x, mc.value, s["reference_depth_mm"])
        planes = local.maps("open")
        means = [plane_mean(p, hx*f, hy*f) for p in planes]
        data_ref = reference_value(np.array([p.depth for p in planes]), np.array([v for v, _ in means]),
                                   s["reference_depth_mm"])
        for source, value, divisor in (("mc", mc_ref, grid.exposure),
                                        ("measurement", data_ref, s["open_monitor_units"])):
            rows.append(dict(field_cm=size, source=source, reference_depth_mm=s["reference_depth_mm"],
                             raw_roi_mean=value, dose_divisor=divisor, dose_per_history_or_mu=value/divisor))
        run.meta["inputs"].extend(local.meta["inputs"])
        run.meta.setdefault("saved_dose", {})[str(size)] = local.meta["saved_dose"]["open"]
        run.meta["results"]["field_settings"][size] = dict(settings=s, mc_roi_bin_counts=np.unique(mc.counts).tolist(),
            measurement_roi=[dict(depth_mm=p.depth, mean=v, finite_bins=n) for p, (v, n) in zip(planes, means)])
        del local, grid, planes
    refs = {r["source"]: r["dose_per_history_or_mu"] for r in rows if r["field_cm"] == 10}
    fig, ax = plt.subplots(figsize=(8, 5))
    for source, label in (("mc", "Open-field MC"), ("measurement", "Open-field Sphinx")):
        selected = [r for r in rows if r["source"] == source]
        for row in selected:
            row["output_factor"] = row["dose_per_history_or_mu"] / refs[source]
        ax.plot([r["field_cm"] for r in selected], [r["output_factor"] for r in selected], "o-", label=label)
    reference_rows = [dict(field_cm=field, source="reference", reference_depth_mm=100.0,
                           output_factor=value) for field, value in run.s.get("output_factor_reference", {}).get("values", [])]
    show_reference = bool(reference_rows) and bool(np.isclose(run.s["reference_depth_mm"], 100.0, rtol=0, atol=1e-9))
    run.meta["results"]["reference"] = dict(
        attribution=run.s.get("output_factor_reference", {}).get("attribution"),
        reference_depth_mm=100.0, included=show_reference, values=reference_rows)
    ticks = set(config["fields"])
    if show_reference:
        ax.plot([r["field_cm"] for r in reference_rows], [r["output_factor"] for r in reference_rows],
                marker="o", linestyle="none", color="black", markerfacecolor="none",
                markeredgecolor="black", label="Reference")
        rows.extend(reference_rows)
        ticks.update(r["field_cm"] for r in reference_rows)
    else:
        print("Reference omitted: the supplied head scatter reference applies only at 100 mm depth.")
    ax.axhline(1, color="gray", linewidth=.8)
    ax.set(xlabel="Field width (cm)", ylabel="Output factor relative to 10 × 10 cm",
           title=f"Open-field output factors at {run.s['reference_depth_mm']:g} mm")
    ax.set_xticks(sorted(ticks))
    ax.grid(alpha=.25)
    ax.legend()
    run.save(fig, "output-factor")
    write_rows(run.out / "output-factor.csv", rows)
    run.meta["results"]["output_factors"] = rows
    run.finish()


def save_pvdr_overview(config, args, completed, failures):
    """Aggregate only results produced successfully during this invocation."""
    fields = sorted(set(args.fields)) if args.fields else sorted(config["fields"])
    run = Run(settings_for(config, fields[0], args), args, config["path"], aggregate=True)
    rows = [row for meta in completed.values() for row in meta["results"]["pvdr"]]
    fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")
    field_handles, unavailable = [], []
    colors = plt.get_cmap("tab10").colors
    for field in fields:
        if field not in completed:
            continue
        color = colors[sorted(config["fields"]).index(field) % len(colors)]
        field_rows = [r for r in rows if r["field_cm"] == field]
        valid = any(r["pvdr"] is not None for r in field_rows)
        if not valid:
            unavailable.append(field)
        else:
            field_handles.append(Line2D([], [], color=color, label=f"{field} × {field} cm"))
        for source in ("mc", "measurement"):
            selected = sorted((r for r in field_rows if r["source"] == source), key=lambda r: r["depth_mm"])
            ax.plot([r["depth_mm"] for r in selected],
                    [r["pvdr"] if r["pvdr"] is not None else np.nan for r in selected],
                    color=color, linestyle="-" if source == "mc" else "none",
                    marker=None if source == "mc" else "o", markerfacecolor="none",
                    markeredgewidth=1.5, label=f"{field} cm {source}")
            draw_pvdr_uncertainty(ax, selected, source, color)
    point_rows = []
    source_handles = [Line2D([], [], color="0.25", label="MC"),
                      Line2D([], [], color="0.25", marker="o", markerfacecolor="none",
                             linestyle="none", label="Measurement")]
    if 10 in fields and run.s.get("point_source_pvdr_reference"):
        point_rows = [dict(field_cm=10, depth_mm=depth, source="point-source-mc", pvdr=value)
                      for depth, value in run.s["point_source_pvdr_reference"]["values"]]
        color = colors[sorted(config["fields"]).index(10) % len(colors)]
        point_line, = ax.plot([r["depth_mm"] for r in point_rows], [r["pvdr"] for r in point_rows],
                             color=color, linestyle="--", marker="s", label="10×10 cm — point-source MC")
        source_handles.append(point_line)
        rows.extend(point_rows)
    if field_handles:
        legend = ax.legend(handles=field_handles, title="Field size", loc="upper right")
        ax.add_artist(legend)
    ax.legend(handles=source_handles, loc="upper center", fontsize=9)
    notes = ["Calculated results: band / error bars = ±1 SE from retained-feature variation"]
    if point_rows:
        notes.append("Point-source MC: user-provided values; uncertainty not supplied")
    if unavailable:
        notes.append("No valid PVDR with current peak selection: " + ", ".join(map(str, unavailable)) + " cm")
    if failures:
        notes.append("Failed fields: " + ", ".join(map(str, failures)) + " cm")
    if notes:
        fig.supxlabel("\n".join(notes), fontsize=9)
    scope = "selected field sizes" if args.fields else "all field sizes"
    ax.set(title=f"Slit PVDR versus depth — {scope}" + (" (partial results)" if failures else ""),
           xlabel="Depth in water (mm)", ylabel="PVDR")
    ax.set_xticks(sorted({r["depth_mm"] for r in rows}))
    ax.grid(alpha=.25)
    run.save(fig, "pvdr-all-fields")
    csv_path = run.out / "pvdr-all-fields.csv"
    if rows:
        write_rows(csv_path, rows)
    else:
        csv_path.write_text("field_cm,depth_mm,source,pvdr,failure_reason\n")
    run.meta["settings"] = {"scope": f"{scope}; see results.field_settings"}
    run.meta["inputs"] = [entry for meta in completed.values() for entry in meta["inputs"]]
    run.meta["saved_dose"] = {str(field): meta.get("saved_dose", {}) for field, meta in completed.items()}
    run.meta["results"] = dict(pvdr=rows, requested_fields=fields, failed_fields=failures,
                                unavailable_fields=unavailable, partial=bool(failures),
                                field_settings={field: meta["settings"] for field, meta in completed.items()})
    run.meta["results"]["point_source_reference"] = dict(
        attribution=run.s.get("point_source_pvdr_reference", {}).get("attribution"), included=bool(point_rows),
        values=point_rows, uncertainty="not supplied", depths="fixed; independent of requested depths")
    run.finish()


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "report":
            print(write_report(args.output_dir))
            return 0
        # Reject the unsupported comparison before loading any inputs or creating outputs.
        if args.command == "pdd" and args.source == "collimator" and not args.mc_only:
            raise ValueError("no dedicated collimator PDD measurement is configured; "
                             "use pdd --source collimator --mc-only to plot MC alone")
        config = load_config(args.config)
        if args.command == "output-factor":
            run_output_factor(config, args)
            return 0
        aggregate = bool(getattr(args, "all_fields", False) or getattr(args, "fields", None))
        fields = (sorted(set(args.fields)) if getattr(args, "fields", None) else
                  sorted(config["fields"]) if aggregate else [args.field])
        # Validate the complete selection before writing any results.
        for field in fields:
            field_settings(config, field)
        failures = []
        completed = {}
        for field in fields:
            try:
                run = Run(settings_for(config, field, args), args, config["path"])
                if args.command == "pdd":
                    run_pdd(run)
                elif args.command == "lateral":
                    run_lateral(run)
                elif args.command in {"mc-fields-ratio", "data-fields-ratio"}:
                    run_fields_ratio(run, mc=args.command == "mc-fields-ratio")
                elif args.command == "pvdr-depth":
                    run_pvdr(run)
                run.finish()
                if args.command == "pvdr-depth":
                    completed[field] = run.meta
            except (ValueError, OSError, KeyError, RuntimeError) as error:
                if not aggregate:
                    raise
                failures.append(field)
                print(f"error: field {field} cm: {error}", file=sys.stderr)
            finally:
                plt.close("all")
        if args.command == "pvdr-depth" and aggregate:
            save_pvdr_overview(config, args, completed, failures)
        return 1 if failures else 0
    except (ValueError, OSError, KeyError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
