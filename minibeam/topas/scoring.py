"""TOPAS scorer configuration and strict CSV interpretation."""
import csv
import math
import re

def score_rows(path, job, grid, *, diagnostics=None, normalization=None):
    """Yield (native row, Gy/represented source history, variance of estimated mean).

    Parse every explicit bin. Zero bins must be present; truncated output is an
    error. Histories_with_Scorer_Active is the denominator, never Count_in_Bin.
    """
    nx, ny, nz = grid["dimensions"]
    size = nx * ny * nz
    seen = bytearray(size)
    headers = []
    count = 0
    with open(path, newline="") as stream:
        for line in stream:
            if line.startswith("#"):
                if count:
                    raise ValueError("Unexpected header after scorer data")
                headers.append(line.rstrip())
                continue
            if not line.strip():
                continue
            if not count:
                _validate_header(headers, job, grid)
                warning = _scorer_warning(headers)
                if warning:
                    warning.update(job_id=job.get("job_id", job["scorer"]), csv_path=str(path),
                                   histories=job["histories"], normalization=normalization)
                    print(f"Scorer warning for {warning['job_id']}: {path}\n"
                          f"  {warning['warning_text']}\n"
                          f"  Unscored steps: {warning['unscored_steps']}; "
                          f"unscored energy: {warning['unscored_energy_mev']:.9g} MeV\n"
                          f"  Histories: {job['histories']:,}; normalization: {normalization or 'see saved manifest'}\n"
                          "  Validity requires user review; statistical uncertainty does not account for missing energy.")
                    if diagnostics is not None:
                        diagnostics.append(warning)
            fields = next(csv.reader([line]))
            if size == 1 and len(fields) == 4 and not any(re.match(r'# [XYZ] in ', h) for h in headers):
                fields = ['0','0','0'] + fields
            if len(fields) != 7:
                raise ValueError("Expected X,Y,Z,Sum,Mean,Histories,Standard_Deviation columns")
            try:
                xyz = [int(s.strip()) for s in fields[:3]]
                total, mean, histories, sd = map(float, fields[3:])
            except ValueError as exc:
                raise ValueError("Malformed TOPAS CSV row") from exc
            if not all(math.isfinite(v) for v in [total, mean, histories, sd]):
                raise ValueError("Nonfinite TOPAS score")
            if any(v < 0 for v in [total, mean, sd]):
                raise ValueError("Negative dose/statistics")
            if histories != job["histories"] or histories < 2:
                raise ValueError("Actual scorer history count differs from the requested count")
            # OpenTOPAS 4.2.p3 stores some score accumulators in float precision;
            # real output differs by ~6e-8 even with 16-digit CSV formatting.
            if not math.isclose(total / histories, mean, rel_tol=2e-7, abs_tol=1e-30):
                raise ValueError("TOPAS Sum/Histories disagrees with Mean")
            x, y, z = xyz
            if not (0 <= x < nx and 0 <= y < ny and 0 <= z < nz):
                raise ValueError("Scorer bin outside expected grid")
            row = x + nx * (y + ny * z)
            if seen[row]:
                raise ValueError("Duplicate scorer bin")
            seen[row] = 1
            count += 1
            denominator = job.get('normalization_histories', histories)
            scale = histories / denominator
            yield row, total / denominator, sd * sd / histories * scale * scale
    if count != size:
        raise ValueError(f"Incomplete scorer grid: expected {size} bins, found {count}")


_UNSCORED_WARNING = "# Warning: Some steps were not scored due to touchable returning an invalid index number."
_UNSCORED_DETAILS = (
    "# See console log for messages starting with: Topas experienced a potentially serious error in scoring.",
    "# We believe this error is due to an unsolved bug in Geant4 parallel world handling.",
)


def _scorer_warning(headers):
    """Accept only the complete, recognized TOPAS invalid-index warning block."""
    warnings = [h for h in headers if "Warning:" in h]
    if any("Filtered by:" in h for h in headers):
        raise ValueError("Scorer reports unexpected filtering")
    related = [h for h in headers if h.startswith((
        "# Total number of steps not scored", "# Total amount of energy not scored",
        "# See console log", "# We believe this error"))]
    if not warnings and not related:
        return None
    if warnings != [_UNSCORED_WARNING]:
        raise ValueError("Unknown or duplicate scorer warning: " + "; ".join(warnings))
    index = headers.index(_UNSCORED_WARNING)
    block = headers[index:index+5]
    if len(block) != 5 or tuple(block[1:3]) != _UNSCORED_DETAILS or related != block[1:]:
        raise ValueError("Incomplete or conflicting unscored-step warning block")
    steps = re.fullmatch(r"# Total number of steps not scored for this reason: (\d+)", block[3])
    energy = re.fullmatch(r"# Total amount of energy not scored for this reason: ([\d.eE+\-]+) MeV", block[4])
    if not steps or not energy:
        raise ValueError("Malformed unscored-step warning statistics")
    value = float(energy[1])
    if not math.isfinite(value) or value < 0:
        raise ValueError("Invalid unscored energy")
    return dict(kind="invalid_voxel_index", warning_text=_UNSCORED_WARNING[2:],
                original_header=block, unscored_steps=int(steps[1]),
                unscored_energy_mev=value)


def _validate_header(headers, job, grid):
    header = "\n".join(headers)
    if f"# Results for scorer: {job['scorer']}" not in headers:
        raise ValueError("Scorer identity mismatch")
    _scorer_warning(headers)
    if not re.search(r"# TOPAS Version:.*4\.2(?:\.p?3|[ .-]+p3)", header):
        raise ValueError("Expected OpenTOPAS 4.2.p3 / 4.2.3 output")
    quantity = next((h for h in headers if h.startswith("# DoseToMedium")), "")
    if not re.fullmatch(r"# DoseToMedium\s*\(\s*Gy\s*\)\s*:\s*Sum\s+Mean\s+Histories_with_Scorer_Active\s+Standard_Deviation\s*", quantity):
        raise ValueError("Unexpected dose quantity, units or statistic columns")
    if tuple(grid['dimensions']) == (1,1,1) and not any(re.match(r'# [XYZ] in ', h) for h in headers):
        if '# Scored in component: Patient' not in headers:
            raise ValueError('Unsegmented scorer component mismatch')
        return  # TOPAS omits bin coordinates and axis headers for a single voxel.
    scales = {"mm": 1., "cm": 10., "m": 1000.}
    for axis, n in zip("XYZ", grid["dimensions"]):
        match = re.search(rf"# {axis} in (\d+) bins?\s+of\s+([\d.eE+\-]+)\s+(mm|cm|m)", header)
        if not match or int(match[1]) != n:
            raise ValueError(f"{axis} scoring-grid dimension mismatch")
        actual = float(match[2]) * scales[match[3]]
        if not math.isclose(actual, grid["resolution"][axis.lower()], rel_tol=1e-5, abs_tol=1e-6):
            raise ValueError(f"{axis} scoring-grid spacing mismatch")



def scorer_parameters(job, grid):
    scorer = job["scorer"]
    lines = [f's:Sc/{scorer}/Quantity = "DoseToMedium"',
             f's:Sc/{scorer}/Component = "Patient"',
             f's:Sc/{scorer}/OutputType = "csv"',
             f's:Sc/{scorer}/OutputFile = "results/{job["job_id"]}"',
             f's:Sc/{scorer}/IfOutputFileAlreadyExists = "Exit"',
             f'b:Sc/{scorer}/OutputToConsole = "False"',
             f'sv:Sc/{scorer}/Report = 4 "Sum" "Mean" "Histories" "Standard_Deviation"']
    for axis, bins in zip("XYZ", grid["dimensions"]):
        lines.append(f'i:Sc/{scorer}/{axis}Bins = {bins}')
    return "\n".join(lines)
