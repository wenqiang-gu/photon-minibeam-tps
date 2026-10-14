import json
from pathlib import Path
import numpy as np
import pytest
from tools.modify_phsp import modify
from tools.phsp_modifications import Spectrum,BASE,PRESETS,photon_masks,spread,gaussian_pairs
from minibeam.sources.iaea import IAEA_DTYPE,read_header,validated_chunks,validated_parts
from test_phase_space import make_iaea


def config(tmp_path,bases,variants='[variants.base]\nspectrum="baseline"',name='out'):
    p=tmp_path/(name+'.toml')
    p.write_text('file_bases='+json.dumps([str(b) for b in bases])+'\noutput_dir='+json.dumps(name)+'\nseed=12345\n'+variants+'\n')
    return p


def read(base):
    audit={};rows=[]
    for a,ids in validated_chunks(read_header(base),audit,2):rows.append(a)
    return np.concatenate(rows),audit


def sample(tmp_path,name='part',energy=1.):
    return make_iaea(tmp_path/name,[(1,-energy,.5,.1,0.,0.,.05,2,7),
        (1,.51099895,0,0,0,0,.05,0,8),(-1,-2.,0,0,.1,0,.05,3,9),
        (2,.4,0,0,0,0,.05,0,10),(3,-1.,0,0,0,0,.05,2,11)])


def test_identity_and_provenance(tmp_path):
    b=sample(tmp_path);out=modify(config(tmp_path,[b]),chunk_size=1)
    a,audit=read(out/'base'/'part')
    assert a.tobytes()==Path(str(b)+'.phsp').read_bytes()
    assert audit['trailing_empty_histories']==5
    assert audit['modification']['variant']['spectrum']=='baseline'
    summary=json.loads((out/'base'/'modification.json').read_text())
    assert summary['photon_weight_rounding_residual']==0
    with pytest.raises(ValueError,match='already exists'):modify(config(tmp_path,[b]))


def test_multi_variants_global_correction_and_joint(tmp_path):
    bases=[sample(tmp_path),sample(tmp_path,'second',.2)]
    variants='''[variants.energy]
spectrum="peak_plus_0p1"
[variants.width]
source_sigma_x_mm=0.5
source_sigma_y_mm=0.2
[variants.joint]
spectrum="peak_plus_0p1"
source_sigma_x_mm=0.5
source_sigma_y_mm=0.2
'''
    out=modify(config(tmp_path,bases,variants),chunk_size=2)
    before=after=0
    for b in bases:
        original,_=read(b);e,_=read(out/'energy'/b.name);w,_=read(out/'width'/b.name);j,_=read(out/'joint'/b.name)
        photon,protected,use=photon_masks(original)
        assert np.array_equal(e['weight'][protected],original['weight'][protected])
        assert np.array_equal(e['weight'][~photon],original['weight'][~photon])
        assert np.array_equal(j['weight'],e['weight'])
        assert np.array_equal(j[['u','v']],w[['u','v']])
        for key in ['code','energy','x','y','increment','latch']:assert np.array_equal(j[key],original[key])
        before+=original['weight'][photon].astype(float).sum();after+=e['weight'][photon].astype(float).sum()
    assert after==pytest.approx(before,rel=1e-7)
    audits=[];list(validated_parts([read_header(out/'joint'/b.name) for b in bases],audits))
    assert len(audits)==2
    with pytest.raises(ValueError,match='complete ordered'):
        list(validated_parts([read_header(out/'joint'/bases[1].name),read_header(out/'joint'/bases[0].name)],[]))
    with pytest.raises(ValueError,match='complete ordered'):
        list(validated_parts([read_header(out/'joint'/bases[0].name),read_header(out/'energy'/bases[1].name)],[]))
    # Chunking changes neither history draws nor float32 output for this fixture.
    out2=modify(config(tmp_path,bases,variants,name='out2'),chunk_size=1)
    for b in bases:assert Path(str(out/'joint'/b.name)+'.phsp').read_bytes()==Path(str(out2/'joint'/b.name)+'.phsp').read_bytes()


def test_rotation_residual_and_repeat_history():
    a=np.zeros(3,dtype=IAEA_DTYPE);a['code']=1;a['energy']=1;a['weight']=.05
    a['x']=[0,0,1];ids=np.array([2,2,5]);gx,gy=gaussian_pairs(ids,12345)
    a['u'][2]=.02 # Off-axis direction has residual relative to source-to-record ray.
    original=a.copy();spread(a,ids,.5,.2,272.1,12345)
    assert a['u'][0]==a['u'][1]
    for i in range(3):
        nominal=np.array([original['x'][i]*10,0,272.1]);new=nominal-[gx[i]*.5,gy[i]*.2,0]
        nominal/=np.linalg.norm(nominal);new/=np.linalg.norm(new)
        d=np.array([original['u'][i],0,np.sqrt(1-original['u'][i]**2)])
        moved=np.array([a['u'][i],a['v'][i],np.sqrt(1-float(a['u'][i])**2-float(a['v'][i])**2)])
        assert d@nominal==pytest.approx(moved@new,abs=1e-7)


@pytest.mark.parametrize('variant',['[variants.x]\nspectrum="bad"','[variants.x]\nsource_sigma_x_mm=-1', '[variants.x]\nsource_sigma_x_mm=true','[variants.x]\nspectrum="custom"\nC1=1\nC2=2\nC3=-2'])
def test_invalid(tmp_path,variant):
    with pytest.raises((ValueError,KeyError)):modify(config(tmp_path,[sample(tmp_path)],variant))
    assert not (tmp_path/'out').exists()


def test_hash_tamper_and_chain(tmp_path):
    b=sample(tmp_path);out=modify(config(tmp_path,[b]))
    modified=out/'base'/'part'
    with pytest.raises(ValueError,match='chaining'):modify(config(tmp_path,[modified],name='again'))
    p=Path(str(modified)+'.phsp');a=np.fromfile(p,dtype=IAEA_DTYPE);a['weight'][0]=.1;a.tofile(p)
    with pytest.raises(ValueError,match='hash mismatch'):read(modified)


def test_domain_dryrun_and_duplicate_content(tmp_path):
    b=sample(tmp_path,energy=6)
    cfg=config(tmp_path,[b],'[variants.x]\nspectrum="peak_plus_0p1"')
    modify(cfg,dry_run=True);assert not (tmp_path/'out').exists()
    with pytest.raises(ValueError,match='energies'):modify(cfg)
    other=sample(tmp_path,'other',6)
    with pytest.raises(ValueError,match='Duplicate particle'):modify(config(tmp_path,[b,other]))


def test_identity_ratio_and_protection():
    assert np.array_equal(Spectrum(BASE).ratio(np.array([1.,2.])),[1,1])
    for p in PRESETS.values():assert np.isfinite(Spectrum(p).ratio(np.array([.01,.3,5.75]))).all()
    a=np.zeros(5,dtype=IAEA_DTYPE);a['code']=1;a['energy']=[.5104,.5106,.51099895,.5114,.5116]
    assert photon_masks(a)[1].tolist()==[False,True,True,True,False]


def test_modified_selection_partition(case,tmp_path):
    from minibeam.sources.phase_space import PhaseSpaceBeamletSource
    from minibeam.sources.iaea import TOPAS_DTYPE
    b=make_iaea(tmp_path/'partition',[(1,-1,0,0,0,0,.05,2,1),
        (1,2,2,0,0,0,.05,0,2), (2,-.3,3,0,0,0,.05,2,3)])
    out=modify(config(tmp_path,[b],'[variants.x]\nspectrum="peak_minus_0p1"\nsource_sigma_x_mm=0.01'))
    counts={}
    sources=[]
    for selection in ['field','outside_field','all_forward']:
        src=PhaseSpaceBeamletSource(file_base=str(out/'x'/b.name),selection=selection)
        src.prepare(case[3],10,execution='combined');sources.append(src)
        state=next(iter(src.selections.values()))
        counts[selection]=state['record']['selected_particles']
        assert state['record']['original_file']['modification']['variant']['spectrum']=='peak_minus_0p1'
    assert counts['field']+counts['outside_field']==counts['all_forward']


def test_cleanup_on_write_failure(tmp_path,monkeypatch):
    import tools.modify_phsp as module
    b=sample(tmp_path)
    def fail(*args):raise ValueError('synthetic write failure')
    monkeypatch.setattr(module,'spread',fail)
    with pytest.raises(ValueError,match='synthetic'):modify(config(tmp_path,[b]))
    assert not (tmp_path/'out').exists()
    assert not list(tmp_path.glob('out.tmp-*'))
    assert not (tmp_path/'out.lock').exists()


@pytest.mark.topas
def test_modified_topas_replay(case,tmp_path):
    import os,subprocess
    from minibeam import TOPASPhotonEngine
    from minibeam.topas.manifest import load_manifest
    exe=os.environ.get('PHOTON_TPS_TOPAS')
    if not exe:pytest.skip('Set PHOTON_TPS_TOPAS for modified phase-space transport')
    b=make_iaea(tmp_path/'transport')
    out=modify(config(tmp_path,[b],'[variants.x]\nspectrum="peak_plus_0p1"\nsource_sigma_x_mm=0.1\nsource_sigma_y_mm=0.1'))
    ct,plan,cst,stf,_=case
    engine=TOPASPhotonEngine(plan,water=True,source_config={'type':'phase_space','file_base':str(out/'x'/b.name)})
    root=engine.prepare_jobs(ct,cst,stf)
    for job in load_manifest(root)['jobs']:
        run=subprocess.run([exe,job['parameter_file']],cwd=root,capture_output=True,text=True,timeout=120)
        assert run.returncode==0,run.stdout+run.stderr
        assert 'PreCheck showed no problems' in run.stdout
    assert engine.collect_results(root).physical_dose.flat[0].nnz>0


@pytest.mark.parametrize('spectrum',['peak_minus_0p1','peak_plus_0p1'])
def test_low_energy_protection_and_conservation(tmp_path,spectrum):
    energies=[.010001,.02,.05,.099,.1,.1001,.15,.1999,.2,.51099895,1.,5.7]
    b=make_iaea(tmp_path/'low_energy',[(1,-e,0,0,0,0,.05,1,0) for e in energies],original=20)
    out=modify(config(tmp_path,[b],f'[variants.energy]\nspectrum="{spectrum}"'))
    original,_=read(b);a,_=read(out/'energy'/'low_energy')
    protected=photon_masks(original)[1]
    assert np.array_equal(a['weight'][protected],original['weight'][protected])
    assert a['weight'].astype(float).sum()==pytest.approx(original['weight'].astype(float).sum(),rel=1e-7)
    assert np.all(a['weight']>0)
    model=Spectrum(PRESETS[spectrum])
    for boundary in [.1,.2]:
        r=model.ratio(np.array([boundary-1e-8,boundary,boundary+1e-8]),.95)
        assert np.max(np.abs(np.diff(r)))<1e-5
    assert model.ratio(np.array([.01,.05,.1]),.95).tolist()==[1,1,1]
