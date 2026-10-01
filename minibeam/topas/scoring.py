"""TOPAS scorer configuration and strict CSV interpretation."""
import csv
import numpy as np
import math
import re

# Bound parsing buffers independently of the dose-grid size. The per-grid seen
# mask remains necessary: files may reorder bins, even across chunk boundaries.
_CHUNK_ROWS = 100_000
_COORDINATES = re.compile(r'^\s*[+-]?[0-9]+\s*,\s*[+-]?[0-9]+\s*,\s*[+-]?[0-9]+\s*,')


def score_chunks(path, job, grid, *, diagnostics=None, normalization=None,
                 chunk_rows=_CHUNK_ROWS):
    """Yield arrays of native row indices, normalized dose and mean variance.

    Parse and validate zero bins too; sparsity never excuses missing records.
    Coordinate tokens must be integers, not silently truncated floating values.
    """
    if type(chunk_rows) is not int or chunk_rows < 1:
        raise ValueError('chunk_rows must be a positive integer')
    nx, ny, nz = grid['dimensions']
    size = nx * ny * nz
    seen = np.zeros(size, dtype=bool)
    headers, lines = [], []
    count, started = 0, False
    single = False

    def parse(lines):
        nonlocal count
        if not single and any(_COORDINATES.match(line) is None for line in lines):
            raise ValueError('Malformed TOPAS CSV row: bin coordinates must be integer tokens')
        try:
            values = np.loadtxt(lines, delimiter=',', comments=None, ndmin=2)
        except ValueError as exc:
            raise ValueError('Malformed TOPAS CSV row: expected coordinate/statistic columns') from exc
        expected = 4 if single else 7
        if values.shape != (len(lines), expected):
            raise ValueError('Expected X,Y,Z,Sum,Mean,Histories,Standard_Deviation columns')
        stats = values if single else values[:, 3:]
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite TOPAS score')
        total, mean, histories, sd = stats.T
        if np.any(stats[:, [0, 1, 3]] < 0):
            raise ValueError('Negative dose/statistics')
        if np.any(histories != job['histories']) or np.any(histories < 2):
            raise ValueError('Actual scorer history count differs from the requested count')
        per_history = total / histories
        # Match math.isclose's symmetric relative tolerance, not np.isclose's
        # asymmetric reference value or additive absolute tolerance.
        tolerance = np.maximum(1e-30, 2e-7 * np.maximum(np.abs(per_history), np.abs(mean)))
        if np.any(np.abs(per_history - mean) > tolerance):
            raise ValueError('TOPAS Sum/Histories disagrees with Mean')
        if single:
            rows = np.zeros(len(lines), dtype=np.int64)
        else:
            xyz = values[:, :3]
            if np.any(xyz < 0) or np.any(xyz >= [nx, ny, nz]):
                raise ValueError('Scorer bin outside expected grid')
            x, y, z = xyz.astype(np.int64).T
            rows = x + nx * (y + ny * z)
        if np.any(seen[rows]) or np.unique(rows).size != rows.size:
            raise ValueError('Duplicate scorer bin')
        seen[rows] = True
        count += len(rows)
        denominator = job.get('normalization_histories', job['histories'])
        scale = histories / denominator
        return rows, total / denominator, sd * sd / histories * scale * scale

    with open(path, newline='') as stream:
        for line in stream:
            if line.startswith('#'):
                if started:
                    raise ValueError('Unexpected header after scorer data')
                headers.append(line.rstrip())
                continue
            if not line.strip():
                continue
            if not started:
                _validate_header(headers, job, grid)
                warning = _scorer_warning(headers)
                if warning:
                    warning.update(job_id=job.get('job_id', job['scorer']), csv_path=str(path),
                                   histories=job['histories'], normalization=normalization)
                    print(f"Scorer warning for {warning['job_id']}: {path}\n"
                          f"  {warning['warning_text']}\n"
                          f"  Unscored steps: {warning['unscored_steps']}; "
                          f"unscored energy: {warning['unscored_energy_mev']:.9g} MeV\n"
                          f"  Histories: {job['histories']:,}; normalization: {normalization or 'see saved manifest'}\n"
                          '  Validity requires user review; statistical uncertainty does not account for missing energy.')
                    if diagnostics is not None:
                        diagnostics.append(warning)
                single = size == 1 and not any(re.match(r'# [XYZ] in ', h) for h in headers) and len(line.split(',')) == 4
                started = True
            # TOPAS normally writes unquoted numbers. Preserve the legacy CSV
            # reader's quoted-number support without slowing ordinary numeric rows.
            if '"' in line:
                try:
                    fields = next(csv.reader([line], strict=True))
                    if any(',' in field or '\n' in field or '\r' in field for field in fields):
                        raise ValueError('Malformed TOPAS CSV numeric field')
                    line = ','.join(fields) + '\n'
                except csv.Error as exc:
                    raise ValueError('Malformed TOPAS CSV row') from exc
            lines.append(line)
            if len(lines) == chunk_rows:
                yield parse(lines)
                lines = []
        if lines:
            yield parse(lines)
    if count != size:
        raise ValueError(f'Incomplete scorer grid: expected {size} bins, found {count}')


def score_rows(path, job, grid, *, diagnostics=None, normalization=None):
    """Compatibility row iterator; production assembly consumes score_chunks."""
    for rows, dose, variance in score_chunks(path, job, grid, diagnostics=diagnostics,
                                            normalization=normalization):
        for row, value, var in zip(rows, dose, variance):
            yield int(row), float(value), float(var)


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
