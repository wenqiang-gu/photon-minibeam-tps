#!/usr/bin/env python3
"""Audit saved TOPAS replay without original source files or pyRadPlan."""
import argparse
import csv
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from projects.replay_audit import audit_setup


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('project',type=Path)
    p.add_argument('--output-dir',type=Path,default=Path('outputs/replay-audit'))
    args=p.parse_args(argv);root=args.project
    manifests=[root/'manifest.json'] if (root/'manifest.json').exists() else sorted(root.rglob('manifest.json'))
    rows=[]
    for path in manifests:
        manifest=json.loads(path.read_text());base_rows=audit_setup(path.parent,manifest)
        for row in base_rows:row['batch_id']='batch_000'
        rows.extend(base_rows)
        markers=sorted((path.parent/'batches').glob('batch_*/batch.json'))
        if markers and not (path.parent/'batches.json').exists():
            print('Missing batch inventory:',path.parent/'batches.json', '; copied batches are audited but not automatically collectable.')
        for marker in markers:
            record=json.loads(marker.read_text())
            if record.get('batch_id')=='batch_000':continue
            from projects.add_batches import signed
            try:
                signed(record,'batch_fingerprint')
                if record['base_bundle_id']!=manifest['bundle_id']:
                    raise ValueError('Copied batch belongs to another setup')
                batch_rows=audit_setup(path.parent,dict(manifest,jobs=record['jobs']))
                for row in batch_rows:
                    row['batch_id']=record['batch_id']
                    row['batch_index_present']=(path.parent/'batches.json').exists()
                rows.extend(batch_rows)
            except (KeyError,ValueError) as error:
                rows.append(dict(project=str(path.parent.resolve()),batch_id=marker.parent.name,
                                 status='unavailable_or_invalid',reason=str(error)))
    for derived in sorted(root.rglob('derived')):
        if not (derived.parent/'manifest.json').exists() and (derived/'result.mat').exists():
            rows.append(dict(project=str(derived.parent.resolve()),status='unavailable_or_invalid',reason='Derived results only; no manifest or replay inputs'))
    for row in rows:print(row['project'],row.get('job_id',''),row['status'])
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'audit.json').write_text(json.dumps(rows,indent=2)+'\n')
    keys=sorted({k for row in rows for k in row})
    with (args.output_dir/'audit.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    print('Audit:',args.output_dir.resolve())

if __name__=='__main__':main()
