"""Build a portable Markdown gallery of the plots in an analysis output root."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import re
from urllib.parse import quote


def natural_key(value):
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", str(value))]


def write_report(output_root: Path) -> Path:
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    open_depths, slit_depths = {}, {}
    pdd, output_factor, merged, pvdr = [], [], [], []
    depth_cache = {}
    for plot in sorted(root.rglob("*.png"), key=lambda p: natural_key(p.relative_to(root))):
        parts = plot.relative_to(root).parts
        if len(parts) == 3:
            if parts == ("output-factor", "open", "output-factor.png"):
                output_factor.append((None, plot))
            elif parts == ("pvdr-depth", "collimator", "pvdr-all-fields.png"):
                merged.append((None, plot))
            continue
        if len(parts) != 4 or not re.fullmatch(r"field_(\d+)cm", parts[0]):
            continue
        field = int(re.fullmatch(r"field_(\d+)cm", parts[0])[1])
        command, source = parts[1:3]
        if command == "pdd" and source == "open" and plot.stem == "pdd-agl":
            pdd.append((field, plot))
            continue
        if command == "pvdr-depth" and source == "collimator" and plot.stem == "pvdr-depth":
            pvdr.append((field, plot))
            continue
        if not (command == "lateral" and source in ("open", "collimator")):
            continue
        if plot.parent not in depth_cache:
            metadata = plot.parent / "metadata.json"
            depth_cache[plot.parent] = json.loads(metadata.read_text()) if metadata.is_file() else {}
        meta = depth_cache[plot.parent]
        generated = meta.get("results", {}).get("lateral_figures")
        if generated is not None and plot.name not in generated:
            continue
        if plot.stem.endswith("-combined-depths"):
            if generated is None:
                continue
            depth = -1  # Combined figures precede individual depths when different fields use different modes.
        else:
            match = re.search(r"-depth-([\d.]+)mm$", plot.stem)
            if not match:
                continue
            depth = float(match[1])
            depths = meta.get("settings", {}).get("depths_mm")
            if depths is not None and depth not in depths:
                continue
        target = open_depths if source == "open" else slit_depths
        target.setdefault(depth, []).append((field, plot))
    count = sum(len(v) for v in open_depths.values()) + sum(len(v) for v in slit_depths.values())
    count += len(pdd) + len(output_factor) + len(merged) + len(pvdr)
    lines = ["# Dose analysis report", "",
             f"Report refreshed: {datetime.now(timezone.utc).isoformat()}", "",
             "Plots link directly to the PNG files. Each successful analysis refreshes this report.", "",
             "Open-field comparisons are grouped by field, with lateral plots before PDD and output factor last. "
             "Collimator comparisons are grouped by field, with lateral plots before PVDR versus depth. "
             "Metadata links describe the latest run in each source folder.", "",
             f"**{count} plots**", "", "## Contents", "",
             ]

    lines.extend(["## Interpretation and saved-dose diagnostics", "",
        "Cross-sectional X/Y mean physical MHA X/Z relative to the water entrance center. "
        "Depth in water is physical MHA Y relative to the entrance surface.", "",
        "MHA dose is already normalized and exposure-weighted. No second division by histories "
        "or batch count is applied. Ratio/output-factor calculations remove the saved common exposure weight.", "",
        "MHA alone supplies no Monte Carlo uncertainty, and ROI voxel covariance is unavailable. "
        "Measurement bands describe ROI spatial spread; PVDR bands/error bars describe retained-feature "
        "variation, not Monte Carlo or measurement uncertainty. Repeated transport batches reuse the "
        "source sample; their variation is conditional on that fixed replay.", ""])
    for metadata in sorted(root.rglob("metadata.json")):
        document = json.loads(metadata.read_text())
        saved = document.get("saved_dose", {})
        if not saved:
            continue
        lines.extend([f"Saved-dose provenance: [{metadata.parent.relative_to(root)}]"
                      f"({quote(metadata.relative_to(root).as_posix())})", ""])
        def diagnostics(value):
            if not isinstance(value, dict):
                return
            if "review_required" in value:
                if value["review_required"]:
                    lines.extend(["**Scorer warnings: validity requires user review.**", "",
                        "```json", json.dumps(value.get("scorer_warnings", []), indent=2), "```", ""])
                if value.get("batch_metadata"):
                    lines.extend(["Saved batch provenance (no additional batch scaling):", "",
                        "```json", json.dumps(value["batch_metadata"], indent=2), "```", ""])
            else:
                for child in value.values():
                    diagnostics(child)
        diagnostics(saved)

    def plot_anchor(plot):
        return "plot-" + re.sub(r"[^a-zA-Z0-9_-]", "-", plot.relative_to(root).with_suffix("").as_posix())

    def depth_label(depth):
        return "Combined depths" if depth == -1 else f"Depth {depth:g} mm"

    def field_contents(depth_groups, summary, source):
        fields = sorted({field for items in depth_groups.values() for field, _ in items} |
                        {field for field, _ in summary})
        for field in fields:
            lines.append(f"- **{source} {field}×{field} cm**")
            lateral_links = []
            for depth, items in sorted(depth_groups.items()):
                for size, plot in items:
                    if size == field:
                        axis = plot.stem.split("-")[1]
                        lateral_links.append(f"[{depth_label(depth)} ({axis})](#{plot_anchor(plot)})")
            if lateral_links:
                lines.append("  - Lateral: " + ", ".join(lateral_links))
            for size, plot in summary:
                if size == field:
                    title = "PDD" if source == "Open field" else "PVDR versus depth"
                    lines.append(f"  - [{title}](#{plot_anchor(plot)})")
        return fields

    open_fields = field_contents(open_depths, pdd, "Open field")
    if output_factor:
        lines.append(f"- [Open field: output factor](#{plot_anchor(output_factor[0][1])})")
    slit_fields = field_contents(slit_depths, pvdr, "Collimator")
    if merged:
        lines.append(f"- [Collimator: merged PVDR](#{plot_anchor(merged[0][1])})")
    lines.append("")

    def gallery(items, comparison, heading=None):
        for field, plot in sorted(items, key=lambda item: (item[0] or 0, natural_key(item[1].name))):
            label = f"Field {field} cm" if field is not None else "Aggregate comparison"
            lines.extend([f'<a id="{plot_anchor(plot)}"></a>', "", f"#### {heading or label}", ""])
            supporting = sorted(plot.parent.glob("*.csv"))
            metadata = plot.parent / "metadata.json"
            if metadata.is_file():
                supporting.insert(0, metadata)
            if supporting:
                lines.extend([" · ".join(f"[{p.name}]({quote(p.relative_to(root).as_posix())})"
                                         for p in supporting), ""])
            lines.extend([f"![{label}: {comparison}]({quote(plot.relative_to(root).as_posix())})", ""])

    lines.extend(['<a id="open-field"></a>', "", "## Open field", ""])
    for field in open_fields:
        lines.extend([f'<a id="open-field-{field}"></a>', "", f"### Field {field}×{field} cm", ""])
        for depth, items in sorted(open_depths.items()):
            gallery([(size, plot) for size, plot in items if size == field],
                    f"Open-field lateral — {depth_label(depth)}", heading=f"Lateral — {depth_label(depth)}")
        gallery([(size, plot) for size, plot in pdd if size == field],
                "Open-field PDD", heading="PDD — MC vs AGL")
    if output_factor:
        lines.extend(["### Output factor", ""])
        gallery(output_factor, "Output factor")
    lines.extend(['<a id="collimator"></a>', "", "## Collimator", ""])
    for field in slit_fields:
        lines.extend([f'<a id="collimator-field-{field}"></a>', "", f"### Field {field}×{field} cm", ""])
        for depth, items in sorted(slit_depths.items()):
            gallery([(size, plot) for size, plot in items if size == field],
                    f"Collimator lateral — {depth_label(depth)}", heading=f"Lateral — {depth_label(depth)}")
        gallery([(size, plot) for size, plot in pvdr if size == field],
                "PVDR versus depth", heading="PVDR versus depth")
    if merged:
        lines.extend(["### Merged PVDR versus depth", ""])
        gallery(merged, "Merged PVDR versus depth")
    report = root / "report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "outputs" / "dose_analysis")
    print(write_report(parser.parse_args().output_dir))
