import json
import numpy as np
import pytest
from tools.display_phsp import inspect_sources,values
from minibeam.sources.iaea import IAEA_DTYPE
from test_phase_space import make_iaea


def test_weighted_histograms_and_prefix(tmp_path):
    b=make_iaea(tmp_path/'input',[(1,-1,1,2,0,0,.1,2,0),
        (1,2,-1,0,.6,0,.2,0,0),(-1,-3,0,0,0,0,.3,3,0),
        (2,-.2,0,0,0,0,1.,2,0)])
    out=inspect_sources([b],tmp_path/'view',histories=5,bins=20)
    m=json.loads((out/'summary.json').read_text());a=np.load(out/'histograms.npz')
    assert m['selected_records']==3
    assert m['selected_nonempty_histories']==2
    assert m['histories_without_selected_species']==3
    assert m['backward_or_tangent_records']==1
    assert m['weighted_mean_energy_MeV']==pytest.approx(2.333333333)
    assert a['energy_counts'].sum()==pytest.approx(.6)
    assert a['position_weighted_counts'].sum()==pytest.approx(.6)
    assert a['projection_weighted_counts'].sum()==pytest.approx(.1)
    assert m['projection_window']['outside_records']==1
    assert m['projection_window']['outside_weight']==pytest.approx(.2)
    assert a['projected_x_edges'][[0,-1]].tolist()==[-100,100]
    assert a['projected_y_edges'][[0,-1]].tolist()==[-100,100]
    assert a['weight_counts'].sum()==3
    assert (out/'distributions.png').stat().st_size>1000


def test_coordinates():
    a=np.zeros(1,dtype=IAEA_DTYPE);a['code']=1;a['energy']=1;a['x']=1;a['y']=2;a['u']=.6
    d=values(a,272.1,1000)
    assert d['x'][0]==10 and d['y'][0]==20
    assert d['projected_x'][0]==pytest.approx(10+727.9*.75)
    assert d['angle_x'][0]==pytest.approx(36.8699,abs=.0001)


def test_modified_source_and_errors(tmp_path):
    from tools.modify_phsp import modify
    from test_phsp_modifier import config
    b=make_iaea(tmp_path/'input')
    source=modify(config(tmp_path,[b]))/'base'/'input'
    out=inspect_sources([source],tmp_path/'view',species='electron',bins=10)
    assert json.loads((out/'summary.json').read_text())['inputs'][0]['modification']
    with pytest.raises(ValueError,match='exists'):inspect_sources([b],out)
    with pytest.raises(ValueError,match='prefix'):inspect_sources([b],tmp_path/'bad',histories=100)
    with pytest.raises(ValueError,match='distinct'):inspect_sources([b,b],tmp_path/'bad')


def test_photon_divergence(tmp_path):
    a=np.zeros(3,dtype=IAEA_DTYPE);a['code']=[1,-1,2];a['energy']=1
    a['x']=3;a['y']=4;a['u']=.6
    d=values(a,272.1,1000)
    assert d['photon_forward'].tolist()==[True,False,False]
    assert d['radial'].tolist()==[50.]
    assert d['divergence'][0]==pytest.approx(.75)
    b=make_iaea(tmp_path/'input',[(1,-1,3,4,.6,0,.1,1,0),(-1,-1,0,0,0,0,.2,1,0)])
    out=inspect_sources([b],tmp_path/'view',bins=10)
    h=np.load(out/'histograms.npz')
    assert h['radial_divergence_weighted_counts'].sum()==0
    assert h['radial_edges'][[0,-1]].tolist()==[0,80]
    assert h['divergence_edges'][[0,-1]].tolist()==[0,.5]
    m=json.loads((out/'summary.json').read_text())['photon_divergence']
    assert m['outside_records']==1 and m['outside_weight']==pytest.approx(.1)
    assert m['downstream_records']==1 and m['excluded_backward_tangent_records']==1
    assert (out/'photon_divergence.png').stat().st_size>1000
