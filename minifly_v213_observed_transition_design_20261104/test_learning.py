import json
from pathlib import Path
import numpy as np
from learner import ARMS,Learner,snapshot,increments
from world import generate,seed
from train import fit

rng=np.random.default_rng(21213)
theta=rng.normal(0,.2,(2,2))
logits=rng.normal(0,.2,(2,4,4))
rows=(1,3)
def predict():
    return snapshot(theta,.3,rows,1,logits)
def loss():
    return np.log(predict().prob[2])
e,g=increments(predict(),2)
maxerr=0.
for arr,ix,want in [(theta,(1,i),g[i]) for i in range(2)]+[
    (logits,(a,rows[a],j),e[a,j]) for a in range(2) for j in range(4)]:
    old=arr[ix]
    arr[ix]=old+1e-5
    pos=loss()
    arr[ix]=old-1e-5
    neg=loss()
    arr[ix]=old
    maxerr=max(maxerr,abs(want-(pos-neg)/2e-5))
assert maxerr<1e-8
w=generate(seed("V213|ENGINEERING|UPDATE"))
models={}
for arm in ARMS:
    m=Learner(w.positions,w.opcodes,w.key,arm)
    birth=m.weights.copy()
    fit(m,w,8)
    assert np.isfinite(m.weights).all() and np.isfinite(m.theta).all()
    assert (m.weights==birth).all()==(arm=="NO-PRIOR-WRITE")
    assert m.emission_updates==(0 if arm=="NO-PRIOR-WRITE" else 16)
    assert m.gate_updates==([8,8] if arm in ARMS[:2] else [0,0])
    models[arm]=m.gate_updates
Path("CYCLE2_TEST.json").write_text(json.dumps(dict(status="PASS",derivative_max_error=maxerr,
    prediction_outcome_applications=64,models=models))+"\n")
