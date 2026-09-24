"""Portable bundle validation and TOPAS CSV reader (Python standard library only).

Used locally when preparing and collecting cluster jobs.
Native vector row = x + nx*(y + ny*z).
"""
import hashlib
import json
from pathlib import Path
import numpy as np


def sha256(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def bundle_path(root, relative):
    root = Path(root).resolve()
    result = (root / relative).resolve()
    if not result.is_relative_to(root) or Path(relative).is_absolute():
        raise ValueError(f"Path escapes bundle: {relative}")
    return result


def load_manifest(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema_version") not in {2, 3}:
        raise ValueError("Unsupported bundle schema; prepare a new bundle in a new directory")
    signed = {k: v for k, v in manifest.items() if k != "bundle_id"}
    if digest_json(signed) != manifest["bundle_id"]:
        raise ValueError("Manifest content hash mismatch")
    if manifest.get('normalization') not in {'independent_primary_photon', 'original_accelerator_history', 'photons_per_member_bixel'}:
        raise ValueError('Unsupported dose normalization')
    expected_units = ('Gy/original accelerator history', 'original accelerator histories') if manifest['normalization'] == 'original_accelerator_history' else ('Gy/primary photon', 'primary photons')
    if manifest['normalization'] == 'photons_per_member_bixel':
        expected_units = ('Gy/(photon per member bixel)', 'photons per member bixel')
    if (manifest.get('units'), manifest.get('weight_units')) != expected_units:
        raise ValueError('Dose units disagree with normalization')
    jobs = manifest["jobs"]
    if not jobs or [j["bixel_index"] for j in jobs] != list(range(1, len(jobs) + 1)):
        raise ValueError("Missing, duplicate or reordered beamlet identities")
    for key in ("job_id", "parameter_file", "output", "scorer"):
        if len({j[key] for j in jobs}) != len(jobs):
            raise ValueError(f"Duplicate {key}")
    combined = manifest.get('beamlet_execution', 'separate') == 'combined'
    if manifest.get('beamlet_execution', 'separate') not in {'separate','combined'}:
        raise ValueError('Unsupported beamlet execution mode')
    if combined and (manifest['schema_version'] != 3 or any(j['ray_index'] != 0 or j['beamlet_index'] != 0 for j in jobs)):
        raise ValueError('Aggregate jobs must use non-native zero ray/bixel sentinels')
    members = [m for j in jobs for m in j.get('members', [j])]
    identities = [(j['beam_index'], j['ray_index'], j['beamlet_index']) for j in members]
    if len(set(identities)) != len(members) or any(type(v) is not int or v < 1 for row in identities for v in row):
        raise ValueError('Invalid or duplicate beam/ray/bixel association')
    if identities != sorted(identities):
        raise ValueError('Reordered beam/ray/bixel associations')
    if manifest['schema_version'] == 3:
        if manifest.get('history_budget') not in {None, 'per_job'}:
            raise ValueError('Unsupported history budget')
        legacy_point_combined = combined and manifest['normalization']!='original_accelerator_history' and manifest.get('history_budget') != 'per_job'
        if (manifest['normalization']=='photons_per_member_bixel') != legacy_point_combined:
            raise ValueError('Normalization is inconsistent with execution mode')
        if [j['column_index'] for j in jobs] != list(range(1,len(jobs)+1)):
            raise ValueError('Invalid column indices')
        if [m['bixel_index'] for m in members] != list(range(1,len(members)+1)):
            raise ValueError('Missing or reordered member bixels')
        if combined and len({j['beam_index'] for j in jobs}) != len(jobs):
            raise ValueError('Combined mode requires one job per beam')
        for j in jobs:
            if not j['members'] or any(m['beam_index'] != j['beam_index'] for m in j['members']):
                raise ValueError('Job members must belong to its beam')
            denominator = j['normalization_histories']
            multiplier = len(j['members']) if manifest['normalization']=='photons_per_member_bixel' else 1
            if type(denominator) is not int or denominator < 2 or j['histories'] != denominator*multiplier:
                raise ValueError('Incorrect history normalization denominator')
            if not combined and len(j['members']) != 1:
                raise ValueError('Separate job must contain one member')
        expected_mapping = [dict(column_index=j['column_index'],beam_index=j['beam_index'],job_id=j['job_id'],members=j['members'],kind='beam_aggregate' if combined else 'beamlet') for j in jobs]
        if manifest.get('column_mapping') != expected_mapping:
            raise ValueError('Column membership mapping mismatch')
    for job in jobs:
        if job['parameter_file'] not in manifest['files']:
            raise ValueError('Unsigned job parameter file')
        bundle_path(root, job['output'])
    for path, expected in manifest["files"].items():
        if sha256(bundle_path(root, path)) != expected:
            raise ValueError(f"Bundle input was modified: {path}. Prepare a new bundle.")
    return manifest



def json_data(value):
    """Copy scientific Python values to strict JSON; reject ambiguous custom objects."""
    def convert(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(f"Unsupported planning metadata type: {type(item).__name__}")
    return json.loads(json.dumps(value, default=convert, allow_nan=False))


def planning_snapshot(plan_info, stf, cst):
    """Record plan settings and generated native beam geometry automatically."""
    beams = []
    first_job = 1
    for index, beam in enumerate(stf.beams, 1):
        count = beam.total_number_of_bixels
        beams.append({
            'beam_index': index,
            # Retain all native non-ray fields, including future native beam fields.
            'parameters': beam.model_dump(exclude={'rays'}, exclude_computed_fields=True),
            'num_rays': beam.num_of_rays,
            'num_bixels': count,
            'energies_mev': sorted({float(b.energy) for r in beam.rays for b in r.beamlets}),
            'job_ids': [f'bixel_{j:06d}' for j in range(first_job, first_job + count)],
        })
        first_job += count
    return json_data({
        'units': {'angles': 'deg', 'lengths': 'mm', 'energies': 'MeV'},
        'coordinate_system': 'DICOM LPS',
        'source_point_reference': 'Native source_point is relative to iso_center; job source_lps_mm is absolute LPS',
        'plan': dict(plan_info, target_names=[v.name for v in cst.vois if v.voi_type == 'TARGET']),
        'beams': beams,
    })


def implementation_hashes(package_root=None):
    root = Path(package_root) if package_root else Path(__file__).parents[1]
    return {p.relative_to(root).as_posix(): sha256(p) for p in sorted(root.rglob("*.py"))}


def member_jobs(manifest):
    """Expand saved job membership for native steering validation, never dose columns."""
    for job in manifest['jobs']:
        for member in job.get('members', [job]):
            item = dict(job, **{k:v for k,v in member.items() if k != 'selection'})
            if 'selection' in member:
                source = dict(job['source'], selection=member['selection'])
                if manifest.get('beamlet_execution') == 'combined':
                    source.pop('aim_lps_mm',None)
                item['source'] = source
            yield item
