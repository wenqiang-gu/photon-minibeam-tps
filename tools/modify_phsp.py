#!/usr/bin/env python3
"""Create reusable modified IAEA sources, before beamlet selection."""
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile
import tomllib

import numpy as np

# Support direct invocation from a repository checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from minibeam.sources.iaea import read_header, validated_chunks
from tools.phsp_modifications import BASE, PRESETS, VERSION, Spectrum, photon_masks, spread, transition


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def numeric(value, name):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value):
        raise ValueError(f'{name} must be finite numeric')
    return float(value)


def configuration(path):
    path=Path(path).resolve()
    with path.open('rb') as stream:cfg=tomllib.load(stream)
    unknown=set(cfg)-{'file_bases','output_dir','seed','variants'}
    if unknown:raise ValueError(f'Unknown configuration keys: {sorted(unknown)}')
    def resolve(value):
        p=Path(value).expanduser()
        return (p if p.is_absolute() else path.parent/p).resolve()
    bases=cfg.get('file_bases')
    if not isinstance(bases,list) or not bases or any(not isinstance(v,str) or not v.strip() for v in bases):
        raise ValueError('file_bases must be a nonempty ordered list')
    bases=[resolve(v) for v in bases]
    if len(set(bases))!=len(bases) or len({b.name for b in bases})!=len(bases):
        raise ValueError('Duplicate input paths or basenames')
    if not isinstance(cfg.get('output_dir'),str) or not cfg['output_dir'].strip():raise ValueError('output_dir is required')
    output=resolve(cfg['output_dir'])
    if output.exists():raise ValueError(f'Output already exists: {output}; choose a fresh directory')
    if any(output==b.parent or output in b.parents for b in bases):raise ValueError('Output overlaps input directory')
    seed=cfg.get('seed',12345)
    if isinstance(seed,bool) or not isinstance(seed,int) or not 0<=seed<2**64:raise ValueError('seed must be an unsigned 64-bit integer')
    variants=cfg.get('variants')
    if not isinstance(variants,dict) or not variants:raise ValueError('At least one named variant is required')
    resolved={}
    for name,v in variants.items():
        if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]*',name):raise ValueError(f'Unsafe variant name: {name}')
        if not isinstance(v,dict) or set(v)-{'spectrum','C1','C2','C3','source_sigma_x_mm','source_sigma_y_mm'}:raise ValueError(f'Invalid variant keys: {name}')
        choice=v.get('spectrum','baseline')
        if choice=='custom':
            p=tuple(numeric(v[k],k) for k in ('C1','C2','C3'))
            if not (0<=p[0]<=10 and 0<=p[1]<=20 and -1.999<=p[2]<=10):raise ValueError('Custom parameters outside study bounds: C1 0..10, C2 0..20, C3 -1.999..10')
        else:
            if choice not in PRESETS:raise ValueError(f'Unknown spectrum: {choice}')
            if any(k in v for k in ('C1','C2','C3')):raise ValueError('C1/C2/C3 require spectrum="custom"')
            p=PRESETS[choice]
        sx=numeric(v.get('source_sigma_x_mm',0),'source_sigma_x_mm')
        sy=numeric(v.get('source_sigma_y_mm',0),'source_sigma_y_mm')
        if min(sx,sy)<0:raise ValueError('Source sigmas must be nonnegative')
        resolved[name]=dict(spectrum=choice,parameters=p,source_sigma_x_mm=sx,source_sigma_y_mm=sy)
    if len({n.casefold() for n in resolved})!=len(resolved):raise ValueError('Case-insensitive variant name collision')
    return bases,output,seed,resolved


def stat_init():
    return [dict(count=0,weight=0.,weighted_energy=0.,energy=0.,minimum_weight=float('inf'),maximum_weight=0.,minimum_energy=float('inf'),maximum_energy=0.) for _ in range(3)]


def stats_update(stats,a):
    for code,s in enumerate(stats,1):
        mask=np.abs(a['code'].astype(int))==code
        if not mask.any():continue
        w=a['weight'][mask].astype(float);e=np.abs(a['energy'][mask].astype(float))
        s['count']+=len(w);s['weight']+=float(w.sum());s['weighted_energy']+=float(w@e);s['energy']+=float(e.sum())
        s['minimum_weight']=min(s['minimum_weight'],float(w.min()));s['maximum_weight']=max(s['maximum_weight'],float(w.max()))
        s['minimum_energy']=min(s['minimum_energy'],float(e.min()));s['maximum_energy']=max(s['maximum_energy'],float(e.max()))


def update_header(raw,stats,marker):
    # Preserve original source/transport descriptions. Replace the affected summary.
    lines=['// Weight Wmin Wmax <E> Emin Emax Particle; <E> is weight-averaged']
    for s,name in zip(stats,['PHOTONS','ELECTRONS','POSITRONS']):
        if s['count']:
            mean=s['weighted_energy']/s['weight'] if s['weight'] else s['energy']/s['count']
            lines.append(' '.join(f'{v:.12g}' for v in [s['weight'],s['minimum_weight'],s['maximum_weight'],mean,s['minimum_energy'],s['maximum_energy']])+' '+name)
    section='$STATISTICAL_INFORMATION_PARTICLES:\n'+'\n'.join(lines)+'\n\n'
    pattern=r'(?ms)^\$STATISTICAL_INFORMATION_PARTICLES:.*?(?=^\$|\Z)'
    raw=re.sub(pattern,lambda _:section,raw) if re.search(pattern,raw) else raw+'\n'+section
    return raw+'\n// Modified source; original generation descriptions above retained as provenance.\n$MINIBEAM_MODIFICATION:\n'+marker+'\n'


def modify(config, *, dry_run=False, chunk_size=100_000):
    bases,output,seed,variants=configuration(config)
    infos=[read_header(b) for b in bases]
    if any(i.get('modification') for i in infos):raise ValueError('Use original inputs; chaining already modified sources is unsupported')
    first=infos[0]
    if any(i['plane_distance_mm']!=first['plane_distance_mm'] or i['source_description']!=first['source_description'] for i in infos):raise ValueError('Incompatible source geometry or descriptions')
    size=sum(i['data'].stat().st_size for i in infos)
    print(f'{len(infos)} parts; {sum(i["counts"]["ORIG_HISTORIES"] for i in infos):,} original histories; {len(variants)} variants; about {size*len(variants)/2**30:.2f} GiB particle output')
    print(f'Output: {output}')
    if dry_run:
        print('Dry run: headers checked; particle validation and output writing not performed.')
        return output
    models={n:Spectrum(v['parameters']) for n,v in variants.items()}
    weighted={n:0. for n in variants};tapered={n:0. for n in variants};original=protected_total=0.;protected_count=0
    ranges={n:[float('inf'),0.] for n in variants}
    positive_weight_ranges={n:[float('inf'),0.] for n in variants}
    audits=[];hashes=set()
    for i,info in enumerate(infos):
        print(f'Pass 1: validating {i+1}/{len(infos)}: {info["base"].name}',flush=True)
        audit={}
        for a,ids in validated_chunks(info,audit,chunk_size):
            photon,protected,use=photon_masks(a)
            w=a['weight'][use].astype(float);e=np.abs(a['energy'][use].astype(float))
            original+=float(w.sum());protected_total+=float(a['weight'][protected].astype(float).sum());protected_count+=int(protected.sum())
            for n,model in models.items():
                # Check all photons, including the protected band, for nonbaseline models.
                if not model.identity:
                    energies=np.abs(a['energy'][photon].astype(float))
                    if np.any((energies < .01) | (energies > 5.75)):
                        raise ValueError('Spectrum modification requires photon energies in 0.01–5.75 MeV')
                r=model.ratio(e);weighted[n]+=float(w@r)
                tapered[n]+=float(w@(r*transition(e)))
                positive=(w*r)[w>0]
                if len(positive):
                    positive_weight_ranges[n][0]=min(positive_weight_ranges[n][0],float(positive.min()))
                    positive_weight_ranges[n][1]=max(positive_weight_ranges[n][1],float(positive.max()))
                if len(r):ranges[n]=[min(ranges[n][0],float(r.min())),max(ranges[n][1],float(r.max()))]
        if audit['phsp_sha256'] in hashes:raise ValueError('Duplicate particle-file content; distinct batches required')
        hashes.add(audit['phsp_sha256']);audits.append(audit)
    correction={n:(1. if models[n].identity or original==0 else 1+(original-weighted[n])/tapered[n]) for n in variants}
    if any(not np.isfinite(c) or c<=0 for c in correction.values()):raise ValueError('Invalid continuum correction')
    for n in variants:
        if not np.isfinite(ranges[n][0]):ranges[n]=[1.,1.]
        print(f'{n}: global unprotected-photon correction {correction[n]:.12g}',flush=True)
    # Bound the corrected extrema before creating any particle outputs.
    # Tiny fitted baseline densities can make a finite model ratio unusable.
    for n, (low, high) in positive_weight_ranges.items():
        if high == 0:
            continue
        corrected=np.array([low*min(1.,correction[n]),high*max(1.,correction[n])])
        with np.errstate(over='ignore',under='ignore'):
            stored=corrected.astype(np.float32)
        if np.any(~np.isfinite(stored)) or np.any(stored<=0):
            raise ValueError(
                f'{n}: spectrum reweighting cannot be represented in float32; '
                f'corrected positive-weight bounds {corrected[0]:.6g}..{corrected[1]:.6g}, '
                f'raw multipliers span {ranges[n][0]:.6g}..{ranges[n][1]:.6g}. '
                'A near-zero fitted baseline density can cause extreme ratios at low energy. '
                'No outputs were written. Do not clamp weights or bypass this check; '
                'the low-energy reweighting policy must be revised. '
                'Use spectrum="baseline" for focal-spot-only studies in the meantime.')
    output.parent.mkdir(parents=True,exist_ok=True)
    lock=output.parent/(output.name+'.lock')
    with lock.open('x') as stream:stream.write('modifier staging in progress\n')
    staging=None
    try:
        if output.exists():raise ValueError('Output appeared during preparation')
        staging=Path(tempfile.mkdtemp(prefix=output.name+'.tmp-',dir=output.parent))
        for n in variants:(staging/n).mkdir()
        offset=0;all_outputs={n:[] for n in variants};after={n:0. for n in variants};exceptions={n:0 for n in variants}
        common={n:dict(schema=1,algorithm=VERSION,seed=seed,variant=variants[n],inputs=audits,
            normalization='Preserve full ordered-input photon weight; low-energy and 511-keV protected weights unchanged; smoothly tapered global correction; original histories unchanged',
            global_continuum_correction=correction[n],original_unprotected_weight=original,
            protected_photon_weight=protected_total,protected_photon_count=protected_count,
            raw_multiplier_range=ranges[n],model_endpoint_MeV=5.75,protected_band_MeV=[.51049895,.51149895],
            low_energy_policy={'unchanged_through_MeV':.1,'full_ratio_from_MeV':.2,
                'transition':'smoothstep cubic; exp(t*log(model_ratio)) * (1+(correction-1)*t)'}) for n in variants}
        for i,info in enumerate(infos):
            print(f'Pass 2: writing variants for {i+1}/{len(infos)}: {info["base"].name}',flush=True)
            stats={n:stat_init() for n in variants};sha={n:hashlib.sha256() for n in variants};audit={}
            with ExitStack() as stack:
                streams={n:stack.enter_context((staging/n/(info['base'].name+'.phsp')).open('wb')) for n in variants}
                for a,ids in validated_chunks(info,audit,chunk_size):
                    _,_,use=photon_masks(a)
                    for n,v in variants.items():
                        b=a.copy()
                        if not models[n].identity:
                            w=a['weight'][use].astype(float)*models[n].ratio(np.abs(a['energy'][use].astype(float)),correction[n])
                            with np.errstate(over='ignore',under='ignore'):stored=w.astype(np.float32)
                            if np.any(~np.isfinite(stored)) or np.any((a['weight'][use]>0)&(stored<=0)):raise ValueError('Output weight overflow/underflow')
                            b['weight'][use]=stored
                        exceptions[n]+=spread(b,ids+offset,v['source_sigma_x_mm'],v['source_sigma_y_mm'],info['plane_distance_mm'],seed)
                        after[n]+=float(b['weight'][use].astype(float).sum())
                        stats_update(stats[n],b);raw=b.tobytes();streams[n].write(raw);sha[n].update(raw)
            if audit!=audits[i]:raise ValueError('Input changed between passes')
            for n in variants:
                part=info['base'].name
                provenance=dict(common[n],set_id=digest(common[n]),part_index=i,output_basename=part,output_phsp_sha256=sha[n].hexdigest())
                sidecar=staging/n/(part+'.modification.json')
                sidecar.write_text(json.dumps(provenance,indent=2,allow_nan=False)+'\n')
                marker=sidecar.name+' '+hashlib.sha256(sidecar.read_bytes()).hexdigest()
                (staging/n/(part+'.header')).write_text(update_header(info['header'].read_text(),stats[n],marker))
                all_outputs[n].append(dict(basename=part,phsp_sha256=sha[n].hexdigest()))
            offset+=info['counts']['ORIG_HISTORIES']
        for n in variants:
            if not np.isclose(after[n],original,rtol=1e-7,atol=1e-12):raise ValueError('Float32 output failed photon-fluence conservation')
            summary=dict(common[n],outputs=all_outputs[n],output_unprotected_weight=after[n],
                photon_weight_rounding_residual=after[n]-original,direction_exceptions=exceptions[n])
            (staging/n/'modification.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
        if output.exists():raise ValueError('Output already exists')
        staging.rename(output);staging=None
    finally:
        if staging is not None:shutil.rmtree(staging)
        lock.unlink()
    for n in variants:
        print(f'\n{n}: PHASE_SPACE_FILE_BASES = '+repr([str(output/n/b.name) for b in bases]))
    return output


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path(__file__).with_suffix('.toml'))
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args(argv)
    try:modify(args.config,dry_run=args.dry_run)
    except (ValueError,OSError,KeyError,TypeError) as error:parser.exit(1,f'Error: {error}\n')


if __name__=='__main__':main()
