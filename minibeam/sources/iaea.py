"""Streaming validation of the explicitly supported Elekta IAEA record layout."""
from pathlib import Path
import hashlib
import numpy as np

IAEA_DTYPE = np.dtype([('code','i1'),('energy','<f4'),('x','<f4'),('y','<f4'),
    ('u','<f4'),('v','<f4'),('weight','<f4'),('increment','<i4'),('latch','<i4')])
TOPAS_DTYPE = np.dtype([(name,'<f4') for name in ('x','y','z','u','v','energy','weight')]
                       + [('pdg','<i4'),('negative_z','u1'),('new_history','u1')])


def read_header(base):
    base=Path(base).expanduser().resolve()
    header=Path(str(base)+'.header');data=Path(str(base)+'.phsp')
    raw=header.read_bytes();sections={};key=None
    for line in raw.decode('utf-8').splitlines():
        line=line.split('//',1)[0].strip()
        if line.startswith('$') and line.endswith(':'):
            key=line[1:-1];sections[key]=[]
        elif line and key is not None:sections[key].append(line)
    def numbers(name):
        try:return [float(v) for line in sections[name] for v in line.split()]
        except (KeyError,ValueError) as error:raise ValueError(f'Invalid IAEA header section {name}') from error
    def integer(name):
        values=numbers(name)
        if len(values)!=1 or not np.isfinite(values[0]) or values[0]!=int(values[0]):
            raise ValueError(f'Invalid IAEA integer {name}')
        return int(values[0])
    if (integer('FILE_TYPE')!=0 or integer('RECORD_LENGTH')!=33 or integer('BYTE_ORDER')!=1234
        or numbers('RECORD_CONTENTS') != [1,1,0,1,1,1,1,0,2,1,2]):
        raise ValueError('Unsupported IAEA layout: require little-endian 33-byte constant-Z Elekta records with history increments and LATCH')
    constants=numbers('RECORD_CONSTANT')
    if len(constants)!=1 or not np.isfinite(constants[0]) or constants[0]<=0:
        raise ValueError('Invalid constant phase-space Z')
    counts={name:integer(name) for name in ('ORIG_HISTORIES','PARTICLES','PHOTONS','ELECTRONS','POSITRONS')}
    if min(counts.values())<0 or counts['ORIG_HISTORIES']<2 or counts['PARTICLES']<1:
        raise ValueError('Invalid IAEA counts')
    size=data.stat().st_size
    if size!=counts['PARTICLES']*33 or size!=integer('CHECKSUM'):
        raise ValueError('IAEA file size, record count or byte-count CHECKSUM mismatch')
    if sum(counts[k] for k in ('PHOTONS','ELECTRONS','POSITRONS'))!=counts['PARTICLES']:
        raise ValueError('IAEA species counts do not sum to particle count')
    return {'base':base,'header':header,'data':data,'plane_distance_mm':constants[0]*10,
            'counts':counts,'header_sha256':hashlib.sha256(raw).hexdigest()}


def validated_chunks(info, audit, chunk_size=1_000_000):
    """Yield records and original history IDs, validating the entire file.

    Consumers must exhaust this generator, even after their selected prefix.
    """
    counts=info['counts'];total=history=nonempty=0;species=np.zeros(3,dtype=np.int64)
    bounds=np.array([[np.inf,np.inf],[-np.inf,-np.inf]])
    weight_min=np.inf;weight_max=-np.inf;digest=hashlib.sha256()
    before=info['data'].stat()
    with info['data'].open('rb') as stream:
        while True:
            raw=stream.read(chunk_size*33)
            if not raw:break
            if len(raw)%33:raise ValueError('Truncated IAEA record')
            digest.update(raw);a=np.frombuffer(raw,dtype=IAEA_DTYPE)
            finite=np.ones(len(a),dtype=bool)
            for field in ('energy','x','y','u','v','weight'):finite &= np.isfinite(a[field])
            code=np.abs(a['code'].astype(np.int16));new=a['energy']<0
            if (not finite.all() or np.any(a['weight']<0) or np.any(a['energy']==0)
                or np.any(~np.isin(code,[1,2,3])) or np.any(a['increment']<0)
                or np.any(new != (a['increment']>0))
                or np.any(a['u'].astype(float)**2+a['v'].astype(float)**2>1+1e-6)):
                raise ValueError('Malformed IAEA particle, direction, weight or history flag/increment')
            if total==0 and not new[0]:raise ValueError('First IAEA particle must start a history')
            ids=history+np.cumsum(a['increment'],dtype=np.int64)
            history=int(ids[-1]);total+=len(a);nonempty+=int(new.sum())
            if history>counts['ORIG_HISTORIES']:raise ValueError('History increments exceed original histories')
            for i in range(3):species[i]+=np.count_nonzero(code==i+1)
            bounds[0]=np.minimum(bounds[0],[a['x'].min(),a['y'].min()])
            bounds[1]=np.maximum(bounds[1],[a['x'].max(),a['y'].max()])
            weight_min=min(weight_min,float(a['weight'].min()));weight_max=max(weight_max,float(a['weight'].max()))
            yield a,ids
    after=info['data'].stat()
    if (before.st_size,before.st_mtime_ns,before.st_ino)!=(after.st_size,after.st_mtime_ns,after.st_ino):
        raise ValueError('IAEA file changed while reading')
    if total!=counts['PARTICLES'] or species.tolist()!=[counts[k] for k in ('PHOTONS','ELECTRONS','POSITRONS')]:
        raise ValueError('IAEA particle/species counts disagree with header')
    if hashlib.sha256(info['header'].read_bytes()).hexdigest()!=info['header_sha256']:
        raise ValueError('IAEA header changed while reading')
    audit.update(file_basename=info['base'].name,record_layout='IAEA little-endian 33-byte constant-Z with history increments and LATCH',
        header_sha256=info['header_sha256'],phsp_sha256=digest.hexdigest(),
        records=total,original_histories=counts['ORIG_HISTORIES'],nonempty_histories=nonempty,
        last_recorded_history=history,trailing_empty_histories=counts['ORIG_HISTORIES']-history,
        species_counts=dict(zip(('photons','electrons','positrons'),map(int,species))),
        xy_bounds_mm=(bounds*10).tolist(),weight_range=[weight_min,weight_max],
        phase_space_plane_distance_mm=info['plane_distance_mm'])


def write_topas_header(path, histories, reached, particles):
    # Empty histories are encoded in the two history totals. Native TOPAS
    # precheck appends them on replay with IncludeEmptyHistories=True.
    Path(path).write_text('TOPAS Binary Phase Space\n\n'
        f'Number of Original Histories: {histories}\n'
        f'Number of Original Histories that Reached Phase Space: {reached}\n'
        f'Number of Scored Particles: {particles}\n'
        f'Number of Bytes per Particle: {TOPAS_DTYPE.itemsize}\n')
