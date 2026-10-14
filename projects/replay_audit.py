"""Read-only replay checks, usable on Python 3.8 without planning dependencies."""
import json
from pathlib import Path
import re
import struct

POLICY = 'nonempty_guard_v1'
RECORD = struct.Struct('<7fiBB')


def _inside(root, name):
    root = Path(root).resolve()
    path = (root/name).resolve()
    if Path(name).is_absolute() or root not in path.parents:
        raise ValueError('Replay path escapes setup: '+name)
    return path


def audit_setup(root, manifest=None):
    root = Path(root)
    manifest = manifest or json.loads((root/'manifest.json').read_text())
    rows = []
    cache = {}
    for job in manifest['jobs']:
        row = dict(project=str(root.resolve()), job_id=job['job_id'],
                   column_index=job.get('column_index',job.get('bixel_index')))
        try:
            text = _inside(root, job['parameter_file']).read_text()
            match = re.search(r's:So/[^/]+/PhaseSpaceFileName\s*=\s*"([^"]+)"',text)
            if not match:
                row.update(status='not_phase_space_replay');rows.append(row);continue
            stem = match[1]
            include = re.search(r'b:So/[^/]+/PhaseSpaceIncludeEmptyHistories\s*=\s*"(True|False)"',text)
            include = bool(include and include[1]=='True')
            if stem not in cache:
                header = _inside(root,stem+'.header').read_text()
                if not header.startswith('TOPAS Binary Phase Space'):
                    raise ValueError('Unsupported replay layout')
                counts = []
                for label in ('Number of Original Histories','Number of Original Histories that Reached Phase Space','Number of Scored Particles','Number of Bytes per Particle'):
                    value = re.search(r'^'+re.escape(label)+r':\s*(\d+)\s*$',header,re.M)
                    if not value:raise ValueError('Malformed replay header')
                    counts.append(int(value[1]))
                path = _inside(root,stem+'.phsp');size=path.stat().st_size
                if counts[-1]!=RECORD.size or size<RECORD.size or size%RECORD.size:
                    raise ValueError('Malformed replay record size')
                with path.open('rb') as stream:
                    stream.seek(-RECORD.size,2);last=RECORD.unpack(stream.read(RECORD.size))
                cache[stem] = (counts,size,last)
            (original,nonempty,particles,_),size,last = cache[stem]
            row.update(replay=stem,header_histories=original,nonempty_histories=nonempty,
                       physical_particles=particles,last_new_history=bool(last[-1]),
                       includes_empty_histories=include,checked='header, byte size, terminal record, configuration; not full replay hash')
            policy=job.get('replay_policy')
            if policy==POLICY:
                multiple = re.search(r'i:So/[^/]+/PhaseSpaceMultipleUse\s*=\s*(\d+)',text)
                requested = re.search(r'i:So/[^/]+/NumberOfHistoriesInRun\s*=\s*(\d+)',text)
                precheck = re.search(r'b:So/[^/]+/PhaseSpacePreCheck\s*=\s*"True"',text)
                valid=(not include and multiple and int(multiple[1])==1 and requested
                       and int(requested[1])==nonempty and precheck and original==nonempty+1 and nonempty==job['histories']
                       and size==(particles+1)*RECORD.size and last[6]==-1 and last[-1]==1
                       and job['normalization_histories']>=nonempty>=1)
                if not valid:raise ValueError('Invalid guarded replay contract')
                row['status']='guarded'
            elif policy is not None:
                raise ValueError('Unknown replay policy')
            elif include and original>nonempty and size==particles*RECORD.size and not last[-1]:
                row['status']='known_trigger'
                row['approximate_inflation']=original/nonempty if nonempty else None
            else:
                row['status']='legacy_unverified'
        except (OSError,ValueError,KeyError,struct.error) as error:
            row.update(status='unavailable_or_invalid',reason=str(error))
        rows.append(row)
    return rows


def enforce(root, manifest, write_report=True):
    rows=audit_setup(root,manifest)
    bad=[r for r in rows if r['status']=='known_trigger' or
         (r['status']=='unavailable_or_invalid' and next(j for j in manifest['jobs'] if j['job_id']==r['job_id']).get('replay_policy'))]
    if bad:
        # Separate sibling diagnostics never modify an immutable prepared setup.
        folder=Path(root).with_name(Path(root).name+'-replay-audit')
        report=folder/'audit.json'
        if write_report:
            folder.mkdir(exist_ok=True)
            report.write_text(json.dumps(rows,indent=2)+'\n')
        raise ValueError('Unsafe or invalid phase-space replay: '+', '.join(r['job_id'] for r in bad)+
                         '. Prepare a fresh guarded project; no dose rescaling was applied. '+('Audit: '+str(report) if write_report else 'Dry run: no audit file written.'))
    return rows
