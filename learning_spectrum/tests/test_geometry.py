import importlib.util,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from geometry import *

def test_preupdate_baseline_and_signed():
    g=Geometry();a=bb.bc.R_TABLE[32];g.byte(32,0)
    assert np.array_equal(g.elig,np.zeros((88,88)))
    assert np.array_equal(g.mu_a,a/16)
    r=a*np.exp(-DT/10);b=bb.bc.R_TABLE[48]
    expected=np.outer(r,b-a/16)
    g.byte(48,DT);np.testing.assert_allclose(g.elig,expected,rtol=1e-14,atol=1e-14)

def test_prefix_causality():
    a=Geometry();b=Geometry()
    for k,x in enumerate(b'   0+'):
        a.byte(x,k*DT);b.byte(x,k*DT)
    np.testing.assert_array_equal(a.feature(5*DT),b.feature(5*DT))
    saved=a.clone();b.byte(49,6*DT)
    np.testing.assert_array_equal(a.mu_a,saved.mu_a)

def test_probe_does_not_change_continuing_history():
    g=Geometry();g.feed_cue(b'        0+1=',0);a=g.mu_a.copy();r=g.mu_r.copy()
    g.panel([b'        1+2=',b'        3+0='],165)
    np.testing.assert_array_equal(g.mu_a,a);np.testing.assert_array_equal(g.mu_r,r)

def test_original_source_equivalence_without_birth():
    p=ROOT.parent.parent/'minifly-response/response_mechanisms/src/mechanism_model.py'
    spec=importlib.util.spec_from_file_location('original_mechanism',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    s=m.System.__new__(m.System);s.p=m.DEFAULT.copy();s.arm='J';s.stores=[]
    s.recent=np.zeros(88);s.elig=np.zeros((88,88));s.sens=bb.bc.FE0();s.cue_last=None
    g=Geometry(False)
    for k,b in enumerate(b'        0+1='):
        s.cue_byte(b,k*DT);g.byte(b,k*DT)
    np.testing.assert_array_equal(g.feature(12*DT),s.features(12*DT))

def test_interaction_basis():
    q=int_basis();np.testing.assert_allclose(q.T@q,np.eye(9),atol=1e-14)
    p=np.eye(4)-np.ones((4,4))/4
    np.testing.assert_allclose(q@q.T,np.kron(p,p),atol=1e-14)

def test_source_integrity():assert len(source_verify())>10
