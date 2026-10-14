"""Portable repeated-transport batch inventories. Python 3.8+, standard library only.

Canonical implementation used by the installed package and cluster scripts.
Every parameter file is executed from the original setup root, never the batch directory.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from contextlib import contextmanager


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def within(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or root not in path.parents:
        raise ValueError('Path escapes setup: ' + str(relative))
    return path


def read(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    fd, name = tempfile.mkstemp(dir=str(path.parent), prefix='.' + path.name)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def signed(value, key):
    if digest({k: v for k, v in value.items() if k != key}) != value.get(key):
        raise ValueError('Integrity mismatch: ' + key)


def base_manifest(root, write_report=True):
    manifest = read(Path(root) / 'manifest.json')
    if manifest.get('schema_version') not in (2, 3, 4):
        raise ValueError('Unsupported base manifest schema')
    signed(manifest, 'bundle_id')
    for name, expected in manifest['files'].items():
        if sha256(within(root, name)) != expected:
            raise ValueError('Modified saved input: ' + name)
    try:
        from replay_audit import enforce
    except ImportError:
        from projects.replay_audit import enforce
    enforce(root, manifest, write_report=write_report)
    return manifest


def locate(path):
    """Return setup root and optional selected batch, without trusting arbitrary parents."""
    path = Path(path).resolve()
    marker = path / 'batch.json'
    if marker.is_file():
        record = read(marker)
        root = path.parent.parent
        if path.parent.name != 'batches' or path.name != record.get('batch_id'):
            raise ValueError('Invalid batch location')
        return root, record['batch_id']
    return path, None


def inventory(root, manifest, verify=True):
    """Logical batch zero remains the immutable legacy bundle."""
    zero = dict(batch_id='batch_000', jobs=manifest['jobs'], files={}, base_bundle_id=manifest['bundle_id'])
    path = Path(root) / 'batches.json'
    if not path.exists():
        return [zero]
    index = read(path)
    signed(index, 'index_id')
    if index.get('schema_version') != 1 or index.get('base_bundle_id') != manifest['bundle_id']:
        raise ValueError('Batch index does not match base bundle')
    marker = Path(root)/'batches/batch_000/batch.json'
    if read(marker) != dict(batch_id='batch_000', base_bundle_id=manifest['bundle_id']):
        raise ValueError('Invalid original-batch reference')
    batches = [zero]
    used = {j['seed'] for j in zero['jobs']}
    for number, entry in enumerate(index['batches'], 1):
        bid = 'batch_%03d' % number
        if entry['batch_id'] != bid or entry['directory'] != 'batches/' + bid:
            raise ValueError('Missing, duplicate or reordered batch IDs')
        folder = within(root, entry['directory'])
        record = read(folder / 'batch.json')
        signed(record, 'batch_fingerprint')
        if record.get('schema_version') != 1:
            raise ValueError('Unsupported batch schema')
        if record['batch_fingerprint'] != entry['batch_fingerprint'] or record['base_bundle_id'] != manifest['bundle_id'] or record['batch_id'] != bid:
            raise ValueError('Batch inventory mismatch')
        if len(record['jobs']) != len(zero['jobs']):
            raise ValueError('Batch column count mismatch')
        expected_files = set()
        for original, job in zip(zero['jobs'], record['jobs']):
            changes = {'seed', 'scorer', 'output', 'parameter_file', 'batch_id'}
            if {k:v for k,v in job.items() if k not in changes} != {k:v for k,v in original.items() if k not in changes}:
                raise ValueError('Batch physics or column mapping differs from base')
            seed = job['seed']
            if type(seed) is not int or not 1 <= seed <= 2**31-1 or seed in used:
                raise ValueError('Invalid or repeated batch seed')
            used.add(seed)
            prefix = entry['directory']
            if job.get('batch_id') != bid or job['parameter_file'] != prefix + '/jobs/' + Path(original['parameter_file']).name or job['output'] != prefix + '/results/' + Path(original['output']).name:
                raise ValueError('Invalid batch job paths')
            if job['scorer'] != original['scorer'] + '_' + bid:
                raise ValueError('Invalid batch scorer identity')
            expected_files.add(job['parameter_file'])
            if verify:
                text = within(root, job['parameter_file']).read_text()
                if text != render(within(root, original['parameter_file']).read_text(), original, job):
                    raise ValueError('Batch parameters differ beyond seed/scorer/output')
        if set(record['files']) != expected_files:
            raise ValueError('Batch input inventory mismatch')
        if verify:
            for name, expected in record['files'].items():
                if sha256(within(root, name)) != expected:
                    raise ValueError('Modified batch input: ' + name)
        batches.append(record)
    return batches


def render(text, original, job):
    # Fail closed for external/custom templates. Change only these known parameters.
    pattern = r'(?m)^i:Ts/Seed\s*=\s*' + str(original['seed']) + r'\s*$'
    text, count = re.subn(pattern, 'i:Ts/Seed = ' + str(job['seed']), text)
    if count != 1:
        raise ValueError('Expected exactly one supported TOPAS seed definition')
    old = 's:Sc/%s/OutputFile = "%s"' % (original['scorer'], str(Path(original['output']).with_suffix('')))
    new = 's:Sc/%s/OutputFile = "%s"' % (original['scorer'], str(Path(job['output']).with_suffix('')))
    if text.count(old) != 1:
        raise ValueError('Unsupported scorer output definition')
    text = text.replace(old, new)
    if 'Sc/' + original['scorer'] + '/' not in text:
        raise ValueError('Missing scorer parameters')
    return text.replace('Sc/' + original['scorer'] + '/', 'Sc/' + job['scorer'] + '/')


@contextmanager
def locked(root):
    lock = Path(root) / '.batch-add.lock'
    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise ValueError('Batch addition already locked: %s (remove only after confirming no writer is active)' % lock)
    try:
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        yield
    finally:
        lock.unlink()


def _add(root, count, dry_run, total=None):
    manifest = base_manifest(root, write_report=not dry_run)
    batches = inventory(root, manifest)
    if total is not None:
        count = max(0, total - len(batches))
    print('%s: %d existing + %d additional = %d batches (%d jobs)' %
          (root, len(batches), count, len(batches)+count, (len(batches)+count)*len(manifest['jobs'])))
    shared_bytes = sum(within(root, name).stat().st_size for name in manifest['files'] if name.startswith('inputs/'))
    voxels = 1
    for dimension in manifest['dose_grid']['dimensions']: voxels *= dimension
    print('Shared inputs: %.3f GiB; total CSV estimate: %.3f GiB; merged columns: %d' %
          (shared_bytes/1024**3, voxels*len(manifest['jobs'])*(len(batches)+count)*100/1024**3, len(manifest['jobs'])))
    if not count:
        return
    used = {j['seed'] for b in batches for j in b['jobs']}
    if len(used) + count*len(manifest['jobs']) > 2**31-1:
        raise ValueError('Requested batches exhaust the valid TOPAS seed space')
    base_seed = manifest['jobs'][0]['seed']
    zero_directory = Path(root)/'batches/batch_000'
    if len(batches)==1 and zero_directory.exists():
        raise ValueError('Unindexed original-batch directory exists: '+str(zero_directory))
    prepared = []
    for number in range(len(batches), len(batches)+count):
        bid = 'batch_%03d' % number
        directory = 'batches/' + bid
        destination = within(root, directory)
        if destination.exists():
            raise ValueError('Unindexed batch directory exists; inspect it before retrying: ' + str(destination))
        jobs, texts = [], {}
        for original in manifest['jobs']:
            identity = [base_seed, manifest['bundle_id'], number, original['job_id']]
            seed = int(digest(identity)[:16], 16) % (2**31-1) + 1
            while seed in used:
                seed = seed % (2**31-1) + 1
            used.add(seed)
            job = dict(original, seed=seed, batch_id=bid, scorer=original['scorer']+'_'+bid,
                       parameter_file=directory+'/jobs/'+Path(original['parameter_file']).name,
                       output=directory+'/results/'+Path(original['output']).name)
            texts[job['parameter_file']] = render(within(root, original['parameter_file']).read_text(), original, job)
            jobs.append(job)
        record = dict(schema_version=1, batch_id=bid, base_bundle_id=manifest['bundle_id'], jobs=jobs,
                      generator_sha256=sha256(__file__),
                      source_reuse='fixed replay sample; transport repeats only',
                      files={name: hashlib.sha256(text.encode()).hexdigest() for name,text in texts.items()})
        record['batch_fingerprint'] = digest(record)
        prepared.append((directory, record, texts))
    if dry_run:
        return
    entries = [dict(batch_id=b['batch_id'], directory='batches/'+b['batch_id'], batch_fingerprint=b['batch_fingerprint']) for b in batches[1:]]
    published = []
    try:
        (Path(root)/'batches').mkdir(exist_ok=True)
        if not zero_directory.exists():
            zero_directory.mkdir()
            published.append(zero_directory)
            atomic_json(zero_directory/'batch.json',dict(batch_id='batch_000',base_bundle_id=manifest['bundle_id']))
        for directory, record, texts in prepared:
            destination = within(root, directory)
            staging = Path(tempfile.mkdtemp(prefix='.batch-', dir=str(Path(root)/'batches')))
            try:
                (staging/'jobs').mkdir()
                (staging/'results').mkdir()
                for name, text in texts.items():
                    (staging/'jobs'/Path(name).name).write_text(text)
                atomic_json(staging/'batch.json', record)
                os.rename(str(staging), str(destination))
                published.append(destination)
            finally:
                if staging.exists(): shutil.rmtree(staging)
            entries.append(dict(batch_id=record['batch_id'], directory=directory, batch_fingerprint=record['batch_fingerprint']))
        index = dict(schema_version=1, base_bundle_id=manifest['bundle_id'], batches=entries)
        index['index_id'] = digest(index)
        atomic_json(Path(root)/'batches.json', index)
    except BaseException:
        for folder in published: shutil.rmtree(folder)
        raise


def add(root, count=0, dry_run=False, total=None):
    root = Path(root).resolve()
    value = count if total is None else total
    if type(value) is not int or value < 1:
        raise ValueError('Batch count must be a positive integer')
    if dry_run:
        _add(root, count, True, total)
    else:
        with locked(root):
            _add(root, count, False, total)


def setups(path):
    path = Path(path).resolve()
    if (path/'study.json').exists():
        if (path/'manifest.json').exists(): raise ValueError('Ambiguous study/setup directory')
        roots = [within(path, e['directory']) for e in read(path/'study.json')['setups']]
        if not roots or len(set(roots)) != len(roots): raise ValueError('Empty or duplicate study setups')
        return roots
    if (path/'batch.json').exists(): raise ValueError('Add batches at the setup or study root')
    return [path]


def main(argv=None):
    parser = argparse.ArgumentParser(description='Add repeated TOPAS transport batches using saved inputs only.')
    parser.add_argument('project', type=Path)
    parser.add_argument('--count', type=int, default=3, help='Additional batches, not final total (default: 3)')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    try:
        roots = setups(args.project)
        # Preflight all setups before modifying any. Locks revalidate each setup.
        for root in roots: add(root, args.count, dry_run=True)
        if not args.dry_run:
            for root in roots: add(root, args.count)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, str(error)+'\n')


if __name__ == '__main__':
    main()
