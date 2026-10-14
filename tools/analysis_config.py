"""Explicit field presets; no runtime input-file discovery."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import math
import tomllib

DEFAULT_CONFIG = Path(__file__).with_name("dose_analysis.toml")
DEFAULTS = dict(reference_depth_mm=100.0, normalization_roi_fraction=0.5,
                agl_ssd_mm=900.0, mc_ssd_mm=1000.0,
                lateral_normalization_mode="2d",
                x_roi_half_width_mm=2.5, y_roi_half_width_mm=2.5,
                depths_mm=[0.0, 20.0, 50.0, 100.0], open_monitor_units=400.0,
                slit_monitor_units=2000.0, ratio_threshold_percent=1.0,
                profile_axis="x", pvdr_mc_axis="y", pvdr_data_axis="y",
                mc_shift_x_mm=0.0, mc_shift_y_mm=0.0,
                ignore_edge_peaks=True, peak_prominence_fraction=0.5)
PATH_KEYS = {"mc_open", "mc_collimator", "agl"}
OPTIONAL = {"slit_map_x_offset_mm", "slit_map_y_offset_mm", "expected_ctc_mm",
            "output_factor_reference", "point_source_pvdr_reference"}


def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    path = path.expanduser().resolve()
    with path.open("rb") as stream:
        doc = tomllib.load(stream)
    if doc.get("version") != 1:
        raise ValueError("configuration version must be 1")
    if set(doc) - {"version", "defaults", "fields"}:
        raise ValueError("unknown configuration top-level keys")
    defaults = doc.get("defaults", {})
    removed = "lateral_normalization_half_width_mm"
    migration = (f"{removed} was removed; use x_roi_half_width_mm / "
                 "y_roi_half_width_mm and normalization_roi_fraction")
    if removed in defaults:
        raise ValueError(migration)
    if set(defaults) - (set(DEFAULTS) | OPTIONAL | {"output_root"}):
        raise ValueError("unknown key in configuration defaults")
    fields = doc.get("fields")
    if not isinstance(fields, dict) or not fields:
        raise ValueError("configuration must define fields")
    result = dict(path=path, fields={})
    for name, values in fields.items():
        if removed in values:
            raise ValueError(migration)
        if not name.isdigit() or int(name) <= 0:
            raise ValueError(f"field key must be a positive integer size in cm: {name}")
        if set(values) - (set(DEFAULTS) | PATH_KEYS | OPTIONAL | {"open_maps", "slit_maps", "output_root"}):
            raise ValueError(f"unknown configuration key for field {name}")
        settings = deepcopy(DEFAULTS)
        settings.update(defaults)
        settings.update(values)
        settings["mc_shift_origins"] = {
            key: ("field configuration" if key in values else
                  "global defaults" if key in defaults else "built-in default")
            for key in ("mc_shift_x_mm", "mc_shift_y_mm")
        }
        settings["field_cm"] = int(name)
        for key in PATH_KEYS | {"output_root"}:
            if key not in settings:
                if key != "output_root":
                    continue
                settings[key] = "../outputs/dose_analysis"
            candidate = Path(settings[key]).expanduser()
            settings[key] = (path.parent / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
        for group in ("open_maps", "slit_maps"):
            entries = settings.get(group, [])
            if not isinstance(entries, list):
                raise ValueError(f"{name}.{group} must be a list")
            cooked = []
            for entry in entries:
                if not isinstance(entry, dict) or set(entry) != {"file", "depth_mm"}:
                    raise ValueError(f"{name}.{group} entries require file and depth_mm")
                depth = entry["depth_mm"]
                _number(depth, f"{name}.{group}.depth_mm", minimum=0)
                candidate = Path(entry["file"]).expanduser()
                cooked.append(dict(file=(path.parent / candidate).resolve(), depth_mm=float(depth)))
            depths = [e["depth_mm"] for e in cooked]
            if len(set(depths)) != len(depths):
                raise ValueError(f"duplicate depths in {name}.{group}")
            settings[group] = sorted(cooked, key=lambda e: e["depth_mm"])
        validate_settings(settings)
        result["fields"][int(name)] = settings
    return result


def _number(value, name: str, *, minimum: float | None = None, positive: bool = False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if positive and value <= 0 or minimum is not None and value < minimum:
        raise ValueError(f"invalid {name}: {value}")


def validate_settings(s: dict) -> None:
    for key in ("agl_ssd_mm", "mc_ssd_mm"):
        _number(s[key], key, positive=True)
    if s["lateral_normalization_mode"] not in ("2d", "x", "y"):
        raise ValueError("lateral_normalization_mode must be 2d, x, or y")
    for name in ("mc_shift_x_mm", "mc_shift_y_mm"):
        _number(s[name], name)
    for name in ("reference_depth_mm", "ratio_threshold_percent"):
        _number(s[name], name, minimum=0)
    for name in ("normalization_roi_fraction", "x_roi_half_width_mm", "y_roi_half_width_mm",
                 "open_monitor_units", "slit_monitor_units", "peak_prominence_fraction"):
        _number(s[name], name, positive=True)
    if s["normalization_roi_fraction"] > 1 or s["peak_prominence_fraction"] > 1:
        raise ValueError("ROI and peak prominence fractions must be in (0, 1]")
    if s["ratio_threshold_percent"] >= 100:
        raise ValueError("ratio_threshold_percent must be less than 100")
    for name in ("slit_map_x_offset_mm", "slit_map_y_offset_mm"):
        if name in s:
            _number(s[name], name)
    if "expected_ctc_mm" in s:
        _number(s["expected_ctc_mm"], "expected_ctc_mm", positive=True)
    for name in ("output_factor_reference", "point_source_pvdr_reference"):
        if name in s:
            ref = s[name]
            if not isinstance(ref, dict) or set(ref) != {"attribution", "values"} or not isinstance(ref["attribution"], str) or not ref["attribution"].strip():
                raise ValueError(f"{name} requires attribution and values")
            if not isinstance(ref["values"], list) or not ref["values"]:
                raise ValueError(f"{name} values must be nonempty pairs")
            for pair in ref["values"]:
                if not isinstance(pair, list) or len(pair) != 2:
                    raise ValueError(f"{name} requires coordinate/value pairs")
                for value in pair:
                    _number(value, name, positive=True)
    for name in ("ignore_edge_peaks",):
        if type(s[name]) is not bool:
            raise ValueError(f"{name} must be true or false")
    for name in ("profile_axis", "pvdr_mc_axis", "pvdr_data_axis"):
        if s[name] not in ("x", "y"):
            raise ValueError(f"{name} must be x or y")
    if not isinstance(s["depths_mm"], list) or not s["depths_mm"]:
        raise ValueError("depths_mm must be a nonempty list")
    for depth in s["depths_mm"]:
        _number(depth, "depths_mm", minimum=0)
    if len(set(s["depths_mm"])) != len(s["depths_mm"]):
        raise ValueError("depths_mm must not contain duplicates")
    s["depths_mm"] = sorted(s["depths_mm"])


def field_settings(config: dict, field: int) -> dict:
    if field not in config["fields"]:
        extra = " The supplied datasets do not contain a 5 cm field." if field == 5 else ""
        raise ValueError(f"field {field} cm is not configured.{extra}")
    return deepcopy(config["fields"][field])
