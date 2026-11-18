#!/usr/bin/env python3
"""Standard-library submission helper. Use submit_topas.sh for editable defaults."""
import sys

if sys.version_info < (3, 8):
    sys.exit("TOPAS submission requires Python 3.8 or newer; pyRadPlan is not required.")

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import uuid


# This sibling module is standard-library only; cluster nodes need no package.
try:
    import add_batches as batches
except ImportError:
    from projects import add_batches as batches


def positive(value):
    if not re.fullmatch(r'[1-9][0-9]*', str(value)):
        raise argparse.ArgumentTypeError('must be a positive integer')
    return int(value)


def clean(value):
    value = str(value)
    if any(c in value for c in '\t\n\r\x00'):
        raise ValueError('Paths and options cannot contain tabs, newlines or NUL')
    return value


def read_json(path):
    with path.open() as stream:
        return json.load(stream)


def within(root, name):
    path = (root / clean(name)).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        raise ValueError(f'Path escapes project: {name}') from None
    return path


def thread_setting(job, root, cache=None):
    """Resolve literal threads, honoring local overrides and shared include chains."""
    cache = {} if cache is None else cache
    def visit(path, active):
        if path in active:
            raise ValueError(f'Include cycle: {path}')
        if path in cache:
            return cache[path]
        if not path.is_file():
            raise ValueError(f'Missing TOPAS input/include: {path}')
        local = None
        inherited = set()
        for raw in path.read_text().splitlines():
            tokens = shlex.split(raw, comments=True)
            line = ' '.join(tokens)
            if not line:
                continue
            if re.match(r'^includeFile\s*=', line, re.I):
                rhs = raw.split('=', 1)[1]
                includes = shlex.split(rhs, comments=True)
                if not includes:
                    raise ValueError(f'Empty includeFile in {path}')
                for name in includes:
                    setting = visit((root / name).resolve(), active | {path})
                    if setting is not None:
                        inherited.add(setting)
            elif re.search(r'\bTs/NumberOfThreads\s*=', line, re.I):
                match = re.fullmatch(r'i:Ts/NumberOfThreads\s*=\s*([1-9][0-9]*)', line, re.I)
                if not match or local is not None:
                    raise ValueError(f'Unsupported/duplicate thread definition in {path}: {raw}')
                local = (str(path), int(match[1]))
        if local is None and len(inherited) > 1:
            raise ValueError(f'Ambiguous thread definitions in include branches: {path}')
        result = local if local is not None else next(iter(inherited), None)
        cache[path] = result
        return result
    found = visit(job, set())
    return None if found is None else found[1]


def resolve_selection(root, additional_batches):
    base, selected_batch = batches.locate(root)
    selected_ids = [selected_batch] if selected_batch else []
    if additional_batches and not selected_batch:
        raise ValueError('For multiple directories, pass individual batch directories from the same setup')
    for path in additional_batches:
        other_base, batch_id = batches.locate(path)
        if other_base != base or not batch_id:
            raise ValueError('For multiple directories, pass individual batch directories from the same setup')
        if batch_id in selected_ids:
            raise ValueError(f'Duplicate batch directory: {path}')
        selected_ids.append(batch_id)
    if (base/'study.json').is_file() and (base/'manifest.json').is_file():
        raise ValueError('Ambiguous directory: both study.json and manifest.json exist')
    roots = [within(base, e['directory']) for e in read_json(base/'study.json')['setups']] if (base/'study.json').is_file() else [base]
    if not roots or len(set(roots)) != len(roots):
        raise ValueError('Study has no setups or duplicate setup directories')
    return roots, selected_ids


def print_worker_status(root, additional_batches):
    roots, selected_ids = resolve_selection(root, additional_batches)
    for bundle in roots:
        if not (bundle/'manifest.json').is_file():
            raise ValueError(f'Missing setup manifest: {bundle}')
        if len(roots) > 1:
            print(bundle)
        status_dir = bundle/'job_status'
        folders = ([status_dir / batch_id for batch_id in selected_ids] if selected_ids
                   else sorted(status_dir.glob('batch_*')))
        found = False
        for folder in folders:
            for path in sorted(folder.glob('*.status')):
                print(f'{folder.name}/{path.stem}\t{path.read_text().strip()}')
                found = True
        if not found:
            print('No recorded worker statuses.')


def discover(root, dry_run=False, additional_batches=()):
    roots, selected_ids = resolve_selection(root, additional_batches)
    tasks = []
    for bundle in roots:
        manifest = read_json(bundle/'manifest.json')
        try:
            from replay_audit import enforce
        except ImportError:
            from projects.replay_audit import enforce
        enforce(bundle, manifest, write_report=not dry_run)
        records = batches.inventory(bundle, manifest) if (bundle/'batches.json').exists() else [dict(batch_id='batch_000',jobs=manifest['jobs'])]
        if selected_ids:
            by_id = {record['batch_id']: record for record in records}
            for batch_id in selected_ids:
                if batch_id not in by_id:
                    raise ValueError(f'Batch is not indexed: {batch_id}')
            records = [by_id[batch_id] for batch_id in selected_ids]
        seen = set()
        include_cache = {}
        for record in records:
            print('Selected %s: %s (%d jobs)' % (bundle, record['batch_id'], len(record['jobs'])))
            for job in record['jobs']:
                input_file = within(bundle, job['parameter_file'])
                key = clean(job['job_id'])
                if (record['batch_id'], key) in seen or any(t['bundle']==str(bundle) and t['input']==str(input_file) for t in tasks):
                    raise ValueError('Duplicate job identity/input')
                seen.add((record['batch_id'], key))
                tasks.append(dict(bundle=clean(bundle), input=clean(input_file), job_id=key,
                    batch_id=record['batch_id'], bundle_id=manifest['bundle_id'],
                    inferred_threads=thread_setting(input_file,bundle,include_cache)))
        if not manifest['jobs']: raise ValueError('No prepared jobs: '+str(bundle))
    return tasks


def scheduler_status(record):
    job = str(record['slurm_job_id'])
    if not re.fullmatch(r'[0-9]+',job):
        return 'status unavailable (invalid saved job ID)'
    cluster = record.get('cluster')
    extra = ['--clusters',cluster] if cluster else []
    commands = [('squeue',['-h','-j',job,'-o','%i %T']),
                ('sacct',['-n','-X','-j',job,'--format=JobID,State','--parsable2'])]
    for name, arguments in commands:
        if not shutil.which(name):
            continue
        try:
            p = subprocess.run([name,*extra,*arguments],text=True,capture_output=True,timeout=5)
            if p.returncode==0 and p.stdout.strip():
                return '; '.join(p.stdout.strip().splitlines()[:8])
        except (OSError,subprocess.TimeoutExpired):
            pass
    return 'status unavailable'


def show_history(tasks):
    statuses = {}
    for bundle in dict.fromkeys(t['bundle'] for t in tasks):
        history = Path(bundle)/'slurm/submissions.jsonl'
        if not history.exists():
            continue
        for line in history.read_text().splitlines():
            record = json.loads(line)
            if record['bundle_id'] != next(t['bundle_id'] for t in tasks if t['bundle']==bundle):
                continue
            key = (record.get('cluster'),record['slurm_job_id'])
            if key not in statuses:
                statuses[key] = scheduler_status(record)
            print('  Batches in submission: '+', '.join(sorted({j.get('batch_id','batch_000') for j in record.get('jobs',[])})))
            print(f"Previously submitted {bundle}: job {key[1]} at {record['submitted_at']}; {statuses[key]}")
    if statuses:
        print('History is informational: this invocation submits all selected jobs again unless --dry-run is used.')


def main(argv=None):
    parser = argparse.ArgumentParser(description='Submit TOPAS jobs as one Slurm array.')
    parser.add_argument('directories', type=Path, nargs='+', metavar='DIR',
                        help='One setup or study, or one or more batch directories from the same setup')
    parser.add_argument('--throttle',type=positive,default=10,help='Maximum simultaneous array tasks (default: 10)')
    parser.add_argument('--time',default='48:00:00',help='Time limit per task (default: 48:00:00)')
    parser.add_argument('--mem',default='',help='Memory per task, e.g. 12G or 32000M')
    parser.add_argument('--partition',default='',help='Slurm partition')
    parser.add_argument('--exclude',default='',help='Excluded node list, e.g. rohpc[9003-9005]')
    parser.add_argument('--job-name',default='topas_general')
    parser.add_argument('--wait-for', type=positive, metavar='JOB_ID',
                        help='Wait until this Slurm job or entire array finishes, regardless of exit status')
    parser.add_argument('--topas-env',default=os.environ.get('TOPAS_ENV',''),help='Environment script; defaults to TOPAS_ENV')
    parser.add_argument('--cpus-per-task',type=positive,help='Require CPUs to match inferred TOPAS threads; fallback when absent')
    parser.add_argument('--default-cpus',type=positive,default=1,help=argparse.SUPPRESS)
    parser.add_argument('--dry-run',action='store_true',help='Validate and show submission; write nothing and do not call sbatch')
    parser.add_argument('--status', action='store_true',
                        help='Print the last recorded worker status for each job with a status file and exit')
    args = parser.parse_args(argv)
    root, *additional = [path.expanduser().resolve() for path in args.directories]
    if args.status:
        print_worker_status(root, additional)
        return
    if not args.topas_env:
        raise ValueError('Set --topas-env or TOPAS_ENV (or DEFAULT_TOPAS_ENV in the shell script)')
    env = Path(args.topas_env).expanduser().resolve()
    if not env.is_file() or not os.access(env,os.R_OK):
        raise ValueError(f'Environment script is not readable: {env}')
    clean(env)
    tasks = discover(root, dry_run=args.dry_run, additional_batches=additional)
    if additional:
        root = Path(tasks[0]['bundle'])
    for task in tasks:
        inferred = task['inferred_threads']
        if inferred is not None and args.cpus_per_task is not None and inferred != args.cpus_per_task:
            raise ValueError(f"CPU mismatch: {task['input']} specifies {inferred}, CLI requests {args.cpus_per_task}")
        task['cpus'] = inferred or args.cpus_per_task or args.default_cpus
        if inferred is None and any(c in task['input'] for c in ('"', '\\')):
            raise ValueError('Fallback wrapper cannot quote this input path')
    counts = {t['cpus'] for t in tasks}
    if len(counts)!=1:
        raise ValueError(f'Mixed thread counts {sorted(counts)}: submit matching setups separately')
    cpus = counts.pop()
    if not re.fullmatch(r'\d+(?:-\d+)?(?::\d+){0,2}',args.time):
        raise ValueError('Invalid Slurm time limit')
    if args.mem and not re.fullmatch(r'\d+(?:[KMGT]B?)?',args.mem,re.I):
        raise ValueError('Memory must be an integer with optional K/M/G/T suffix')
    show_history(tasks)
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]
    destination = root/'slurm'/stamp
    worker = Path(__file__).with_name('topas_worker.sh')
    if not worker.is_file():
        raise ValueError(f'Missing worker script: {worker}')
    command = ['sbatch','--parsable',f'--array=0-{len(tasks)-1}%{args.throttle}',
               '--ntasks=1',f'--cpus-per-task={cpus}',f'--time={args.time}',
               f'--job-name={clean(args.job_name)}',f'--output={destination}/task_%A_%a.out',
               f'--error={destination}/task_%A_%a.err']
    for name in ['mem','partition','exclude']:
        if getattr(args,name):
            command.append(f'--{name}={clean(getattr(args,name))}')
    if args.wait_for is not None:
        command.append(f'--dependency=afterany:{args.wait_for}')
    command += [str(destination/'worker.sh'),str(destination/'tasks.tsv'),str(env)]
    print(f'{len(tasks)} tasks; {cpus} CPUs/task; throttle {args.throttle}; time {args.time}')
    print(f"Thread inference: {sum(t['inferred_threads'] is not None for t in tasks)} inputs; others use explicit fallback wrappers.")
    print('Submission:',shlex.join(command))
    if args.dry_run:
        return
    if not shutil.which('sbatch'):
        raise ValueError('sbatch not found; use --dry-run off-cluster')
    destination.mkdir(parents=True)
    shutil.copy2(worker,destination/'worker.sh')
    lines = []
    for index, task in enumerate(tasks):
        input_file = Path(task['input'])
        if task['inferred_threads'] is None:
            # TOPAS include parsing supports quoted names. Refuse embedded quotes.
            if '"' in str(input_file) or '\\' in str(input_file):
                raise ValueError('Fallback wrapper cannot quote this input path')
            wrapper = destination/f'input_{index}.txt'
            wrapper.write_text(f'includeFile = "{input_file}"\ni:Ts/NumberOfThreads = {cpus}\n')
            input_file = wrapper
        lines.append('\t'.join([task['bundle'],str(input_file),task['batch_id']+'/'+task['job_id']]))
    (destination/'tasks.tsv').write_text('\n'.join(lines)+'\n')
    settings = dict(command=command,environment=str(env),cpus_per_task=cpus,throttle=args.throttle,
                    time=args.time,mem=args.mem,partition=args.partition,exclude=args.exclude,job_name=args.job_name)
    (destination/'submission.json').write_text(json.dumps(dict(settings=settings,tasks=tasks),indent=2))
    result = subprocess.run(command,text=True,capture_output=True)
    if result.returncode:
        (destination/'sbatch.stdout').write_text(result.stdout)
        (destination/'sbatch.stderr').write_text(result.stderr)
        raise ValueError(f'sbatch failed: {result.stderr.strip()}; diagnostics: {destination}')
    match = re.fullmatch(r'(\d+)(?:;([\w.-]+))?',result.stdout.strip())
    if not match:
        raise ValueError(f'sbatch returned success but an unrecognized job ID: {result.stdout!r}. Check Slurm before retrying; {destination}')
    job_id, cluster = match.groups()
    print(f'Submitted Slurm job {job_id}; records/logs: {destination}',flush=True)
    record = dict(slurm_job_id=job_id,cluster=cluster,submitted_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                  submission_directory=str(destination),settings=settings)
    try:
        (destination/'sbatch.stdout').write_text(result.stdout)
        (destination/'sbatch.stderr').write_text(result.stderr)
        (destination/'receipt.json').write_text(json.dumps(record,indent=2))
        for bundle in dict.fromkeys(t['bundle'] for t in tasks):
            selected = [(i,t) for i,t in enumerate(tasks) if t['bundle']==bundle]
            entry = dict(record,bundle_id=selected[0][1]['bundle_id'],
                         jobs=[dict(array_index=i,job_id=t['job_id'],batch_id=t['batch_id']) for i,t in selected])
            history = Path(bundle)/'slurm/submissions.jsonl'
            history.parent.mkdir(exist_ok=True)
            with history.open('a') as stream:
                stream.write(json.dumps(entry)+'\n')
    except OSError as error:
        raise ValueError(f'SLURM JOB {job_id} WAS SUBMITTED but recording failed: {error}. Do not blindly resubmit; check Slurm. Records: {destination}') from error


if __name__ == '__main__':
    try:
        main()
    except (ValueError,OSError,KeyError,TypeError) as error:
        print(f'Error: {error}',file=sys.stderr)
        sys.exit(1)
