"""Chunk boundaries must not weaken scorer integrity or alter native row order."""
import numpy as np
import pytest
from minibeam.topas.scoring import score_chunks, score_rows


def fixture(tmp_path, single=False):
    n=1 if single else 12
    grid=dict(dimensions=[1,1,1] if single else [2,3,2],resolution=dict(x=2.,y=2.,z=2.))
    job=dict(job_id='test',scorer='dose_hash',histories=10,normalization_histories=5)
    header=['# TOPAS Version: 4.2.p3','# Results for scorer: dose_hash','# Scored in component: Patient']
    if not single:
        header += [f'# {a} in {v} bins of 2 mm' for a,v in zip('XYZ',grid['dimensions'])]
    header += ['# DoseToMedium ( Gy ) : Sum Mean Histories_with_Scorer_Active Standard_Deviation']
    rows=[]
    for i in range(n):
        prefix='' if single else f'{i%2}, {(i//2)%3}, {i//6}, '
        rows.append(prefix+f'{10*i}, {i}, 10, {2*i}')
    path=tmp_path/'dose.csv'
    def write(data):path.write_text('\n'.join(header+data)+'\n')
    write(rows)
    return path,job,grid,rows,write


@pytest.mark.parametrize('chunk_size',[1,2,5,12,100_000])
@pytest.mark.parametrize('reverse',[False,True])
def test_chunk_values(tmp_path,chunk_size,reverse):
    p,j,g,rows,write=fixture(tmp_path)
    if reverse:write(rows[::-1])
    d=np.empty(12);v=np.empty(12)
    for indices,dose,var in score_chunks(p,j,g,chunk_rows=chunk_size):
        assert len(indices)<=chunk_size
        d[indices]=dose;v[indices]=var
    np.testing.assert_array_equal(d,2*np.arange(12))
    np.testing.assert_array_equal(v,(2*np.arange(12))**2/10*4)
    assert len(list(score_rows(p,j,g)))==12


@pytest.mark.parametrize('mutation,match',[
 ('duplicate_within','Duplicate'),('duplicate_across','Duplicate'),('missing','Incomplete'),
 ('decimal_index','integer'),('exponent_index','integer'),('negative_index','outside'),
 ('outside','outside'),('extra','columns'),('short','columns'),('junk','Malformed'),
 ('comment','header'),('nan','Nonfinite'),('inf','Nonfinite'),('negative','Negative'),
 ('history','history'),('mean','Mean'),('empty','Incomplete')])
def test_reject(tmp_path,mutation,match):
    p,j,g,rows,write=fixture(tmp_path)
    if mutation=='duplicate_within':rows[1]=rows[0]
    elif mutation=='duplicate_across':rows[-1]=rows[0]
    elif mutation=='missing':rows.pop()
    elif mutation=='empty':rows=[]
    elif mutation=='comment':rows.insert(3,'# unexpected')
    else:
        f=rows[-1].split(',')
        if mutation=='decimal_index':f[0]='1.0'
        if mutation=='exponent_index':f[0]='1e0'
        if mutation=='negative_index':f[0]='-1'
        if mutation=='outside':f[0]='2'
        if mutation=='extra':f.append('0')
        if mutation=='short':f.pop()
        if mutation=='junk':f[3]='110 garbage'
        if mutation in ('nan','inf'):f[3]=mutation
        if mutation=='negative':f[6]='-1'
        if mutation=='history':f[5]='9'
        if mutation=='mean':f[4]='12'
        rows[-1]=','.join(f)
    write(rows)
    with pytest.raises(ValueError,match=match):list(score_chunks(p,j,g,chunk_rows=5))


def test_single_and_quoted(tmp_path):
    p,j,g,rows,write=fixture(tmp_path,single=True)
    write(['"0","0","10","0"'])
    assert list(score_rows(p,j,g))[0]==(0,0.,0.)
    write(['0,0,0,0,0,10,0'])  # Legacy explicit XYZ with no axis headings.
    assert list(score_rows(p,j,g))[0]==(0,0.,0.)
    write(['"0,0",0,0,10,0'])
    with pytest.raises(ValueError,match='Malformed'):list(score_chunks(p,j,g))
    write(rows*2)
    with pytest.raises(ValueError,match='Duplicate'):list(score_chunks(p,j,g,chunk_rows=1))


def test_progress_and_zero_weight_validation(case,monkeypatch,capsys):
    from test_results import fake_results
    from minibeam.topas import results
    ct,_,cst,stf,engine=case
    root=engine.prepare_jobs(ct,cst,stf);m=fake_results(root)
    original=results.score_chunks
    def small(*args,**kwargs):return original(*args,chunk_rows=500,**kwargs)
    monkeypatch.setattr(results,'score_chunks',small)
    # Each chunk advances time, exercising throttled progress without sleeping.
    import itertools
    ticks=itertools.count(0,6)
    monkeypatch.setattr(results.time,'monotonic',lambda:next(ticks))
    engine.collect_forward([0.,0.])
    out=capsys.readouterr().out
    assert 'Reading 1/2:' in out and 'Validated 2/2:' in out and '500/4,800 bins' in out
    p=root/m['jobs'][0]['output'];p.write_text(p.read_text()+'0,0,0,0,0,10,0\n')
    with pytest.raises(ValueError,match='Duplicate'):engine.collect_forward([0.,0.])
