import numpy as np
import pytest
from tools.analysis_numerics import Curve, gamma_profile


def test_gamma_identity_and_constant_difference():
    x=np.array([-5.,0.,5.]);r=Curve(x,np.full(3,100.))
    g,m=gamma_profile(r,r)
    np.testing.assert_allclose(g.value,0);assert m['pass_rate_percent']==100
    g,m=gamma_profile(Curve(x,np.full(3,102.)),r)
    np.testing.assert_allclose(g.value,1);assert m['passed_points']==3
    g,_=gamma_profile(Curve(x,np.full(3,104.)),r)
    np.testing.assert_allclose(g.value,2)


def test_gamma_exact_segment_minimum():
    # At reference (0, 100), MC is D=100+2*(x-2).
    # Normalized distance gives gamma=min sqrt(x²/4+(x-2)²)=sqrt(.8).
    e=Curve(np.array([-5.,5.]),np.array([86.,106.]))
    r=Curve(np.array([0.,1.]),np.array([100.,100.]))
    g,_=gamma_profile(e,r)
    assert g.value[0]==pytest.approx(np.sqrt(.8))


def test_gamma_cutoff_and_support():
    e=Curve(np.array([0.,2.]),np.array([100.,100.]))
    r=Curve(np.array([-1.,0.,1.,2.]),np.array([100.,5.,100.,10.]))
    g,m=gamma_profile(e,r)
    assert np.isnan(g.value[:2]).all();assert m['evaluated_points']==2
    assert gamma_profile(e,r,0)[1]['evaluated_points']==3
    with pytest.raises(ValueError):gamma_profile(e,r,100)
    with pytest.raises(ValueError):gamma_profile(e,Curve(r.x,np.zeros(4)))
