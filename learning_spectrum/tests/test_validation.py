import hashlib,inspect,json,sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from geometry import Geometry,ROOT
from calibration import Observation,predict_baseline
from finite_horizon import kernel_metrics
from launch_gate import verify_gate,REQUIRED

def observations():return [Observation(b'        0+1=',float(i),tuple(np.arange(4)+i)) for i in range(25)]

def test_calibration_prefix_before_sixteen():
    a=observations();b=a[:5]+[Observation(o.cue,o.at,(1e8,1e8,0,0)) for o in a[5:]]
    np.testing.assert_array_equal(predict_baseline(a,b'        1+2=',5.),predict_baseline(b,b'        1+2=',5.))

def test_calibration_empty_past():
    np.testing.assert_array_equal(predict_baseline(observations(),b'        0+1=',-1.),np.zeros(4))

def test_teacherless_interfaces():
    for f in (Geometry.byte,Geometry.feed_cue,predict_baseline):
        assert not any(x in inspect.signature(f).parameters for x in ('answer','label','task','world','item','position'))
    assert list(Observation.__annotations__)==['cue','at','raw']

def test_identity_kernel_rejected():
    native=np.random.default_rng(42).normal(size=(30,20))
    assert kernel_metrics(native,np.eye(30))['pass_gate'] is False

def test_degenerate_kernel_rejected():
    with pytest.raises(ValueError):kernel_metrics(np.ones((30,5)),np.eye(30))

def test_gate_failure_and_tamper(tmp_path):
    p=tmp_path/'gate.json';p.write_text(json.dumps({'gates':{k:True for k in REQUIRED},'total_margin_prediction':'NO QUANTITATIVE THEORY PREDICTION'}));sha=hashlib.sha256(p.read_bytes()).hexdigest()
    with pytest.raises(ValueError):verify_gate(p,sha)
    with pytest.raises(ValueError):verify_gate(p,'0'*64)
    g=json.loads(p.read_text());g['gates']['static_spectrum']=False;p.write_text(json.dumps(g));sha=hashlib.sha256(p.read_bytes()).hexdigest()
    with pytest.raises(ValueError):verify_gate(p,sha)

def test_exact_ordered_replay_and_failed_gains():
    x=json.loads((ROOT/'results/CYCLE2_RESULT.json').read_text())
    for arms in x['results'].values():
        r=arms['candidate_A'];assert not r['formation_gate'] and not r['final_gain_gate']
        for c in r['dynamics'].values():
            assert c['direct_replay_max_error']<1e-10 and c['direct_clipping_coordinates']==0
            assert np.isfinite(np.array(c['operator'])).all()
    assert x['identity_negative_control']['pass_gate'] is False
