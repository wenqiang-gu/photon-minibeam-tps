#!/usr/bin/env python3
"""Print a summary and full DICOM data set for an RT Plan.

Example:
    .venv/bin/python tools/print_rtplan_info.py \
        /Users/wgu/Local/MLC/RP.1.2.246.352.71.5.43216483251.92316.20260918102317.dcm
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pydicom
from pydicom.uid import RTPlanStorage


def value(dataset: Any, attribute: str, default: str = "—") -> Any:
    """Return a DICOM attribute or a readable placeholder when it is absent."""
    return getattr(dataset, attribute, default)


def print_plan_info(path: Path, *, summary_only: bool = False) -> None:
    """Read *path* and print its RT Plan metadata, sequences, and beam settings."""
    dataset = pydicom.dcmread(path, stop_before_pixels=True)
    if dataset.SOPClassUID != RTPlanStorage:
        raise ValueError(f"Expected an RT Plan DICOM file, found {dataset.SOPClassUID.name}.")

    print(f"File: {path}")
    print("\nPlan")
    for label, attribute in (
        ("Label", "RTPlanLabel"),
        ("Name", "RTPlanName"),
        ("Status", "ApprovalStatus"),
        ("Plan date/time", "RTPlanDate"),
        ("Geometry", "RTPlanGeometry"),
        ("Machine vendor", "Manufacturer"),
        ("Machine model", "ManufacturerModelName"),
    ):
        print(f"  {label}: {value(dataset, attribute)}")

    print("\nFraction groups")
    for group in value(dataset, "FractionGroupSequence", []):
        beam_metersets = [
            f"beam {reference.ReferencedBeamNumber}: {reference.BeamMeterset} MU"
            for reference in value(group, "ReferencedBeamSequence", [])
        ]
        print(
            f"  Group {value(group, 'FractionGroupNumber')}: "
            f"{value(group, 'NumberOfFractionsPlanned')} fraction(s), "
            f"{value(group, 'NumberOfBeams')} beam(s)"
        )
        print(f"    Metersets: {', '.join(beam_metersets) or '—'}")

    print("\nDose references")
    for reference in value(dataset, "DoseReferenceSequence", []):
        print(
            f"  {value(reference, 'DoseReferenceDescription')}: "
            f"{value(reference, 'TargetPrescriptionDose')} Gy "
            f"({value(reference, 'DoseReferenceType')})"
        )

    print("\nBeams")
    for beam in value(dataset, "BeamSequence", []):
        control_points = value(beam, "ControlPointSequence", [])
        first_cp = control_points[0] if control_points else None
        print(
            f"  {value(beam, 'BeamNumber')} — {value(beam, 'BeamName')}: "
            f"{value(beam, 'BeamType')}, {value(beam, 'RadiationType')}, "
            f"{value(first_cp, 'NominalBeamEnergy')} MV, "
            f"{len(control_points)} control point(s)"
        )
        print(
            f"    machine={value(beam, 'TreatmentMachineName')}; "
            f"gantry={value(first_cp, 'GantryAngle')}°; "
            f"collimator={value(first_cp, 'BeamLimitingDeviceAngle')}°; "
            f"couch={value(first_cp, 'PatientSupportAngle')}°"
        )
        print(f"    collimation (jaws + MLC): {aperture_summary(first_cp)}")

    if not summary_only:
        print("\nFull DICOM file metadata")
        print(dataset.file_meta)
        print("\nFull DICOM data set")
        print(dataset)


def aperture_summary(control_point: Any) -> str:
    """Summarize jaw positions and MLC leaf-pair openings for one control point."""
    if control_point is None:
        return "no control points"

    positions = {
        item.RTBeamLimitingDeviceType: [float(position) for position in item.LeafJawPositions]
        for item in value(control_point, "BeamLimitingDevicePositionSequence", [])
    }
    jaws = ", ".join(
        f"{device}={positions[device]} mm" for device in positions if device != "MLCX"
    )
    leaf_positions = positions.get("MLCX", [])
    leaf_pairs = len(leaf_positions) // 2
    openings = [
        index + 1
        for index, (bank_a, bank_b) in enumerate(
            zip(leaf_positions[:leaf_pairs], leaf_positions[leaf_pairs:])
        )
        if bank_b > bank_a
    ]
    return f"{jaws}; MLC open pairs: {format_pairs(openings)}"


def format_pairs(pairs: list[int]) -> str:
    """Render leaf-pair numbers compactly, e.g. ``[1-3, 7]``."""
    if not pairs:
        return "none"

    ranges: list[str] = []
    start = end = pairs[0]
    for pair in pairs[1:]:
        if pair == end + 1:
            end = pair
        else:
            ranges.append(str(start) if start == end else f"{start}-{end}")
            start = end = pair
    ranges.append(str(start) if start == end else f"{start}-{end}")
    return ", ".join(ranges)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path, help="path to an RT Plan DICOM file")
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="omit the complete DICOM metadata and data-set dump",
    )
    args = parser.parse_args()
    print_plan_info(args.plan, summary_only=args.summary_only)


if __name__ == "__main__":
    main()
