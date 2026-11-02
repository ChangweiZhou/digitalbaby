"""Predeclared inference lesions and gradient/step comparisons, no learner fitting."""
import json,time,hashlib
import numpy as np
import core as C
from local_rule import gradients
def cosine(a,b):
 den=np.linalg.norm(a)*np.linalg.norm(b);return float(a.ravel()@b.ravel()/den) if den else 0.
def delta(p,g,m,v,t,targets,lr):
 d=g/targets;d/=np.maximum(1,np.linalg.norm(d,axis=1,keepdims=True));mm=.9*m+.1*d;vv=.999*v+.001*d*d;step=(mm/(1-.9**(t+1)))/(np.sqrt(vv/(1-.999**(t+1)))+1e-8);return p*(1-lr*1e-4)-lr*step-p
results=[];start=time.process_time()
for w in (202611027711,202611027712):
 path=C.H/f'local/BP_{w}.npz';before=hashlib.sha256(path.read_bytes()).hexdigest();a=np.load(path);p,o,e,b=[a[k] for k in ('p','o','e','b')];hx,hy,alpha=C.data(w,'held',64);x,y,_=C.data(w,'train',64)
 les={name:C.evaluate(p,o,e,hx,hy,True,rec,cross) for name,rec,cross in [('cross_edges_off',True,False),('recurrence_off',False,False)]}
 ref=np.load(C.H/f'local/diagnostic_BP_{w}.npz');rr=[]
 for mode,name in [(0,'BS'),(1,'BR')]:
  gp,go,_=gradients(p,o,e,b,C.SRC,x[:8],y[:8],mode);lr=json.loads((C.H/'PILOT_LOCK.json').read_text())['lr']
  rr.append(dict(arm=name,hidden_gradient_cosine=cosine(gp,ref['gp']),output_gradient_cosine=cosine(go,ref['go']),hidden_applied_step_cosine=cosine(delta(p,gp,a['m0'],a['v0'],int(a['steps']),776,lr),delta(p,ref['gp'],a['m0'],a['v0'],int(a['steps']),776,lr)),output_applied_step_cosine=cosine(delta(o,go,a['m1'],a['v1'],int(a['steps']),776,lr),delta(o,ref['go'],a['m1'],a['v1'],int(a['steps']),776,lr))))
 assert before==hashlib.sha256(path.read_bytes()).hexdigest()
 results.append(dict(world=w,lesions=les,gradient_diagnostics=rr,parameter_norms=dict(hidden=float(np.linalg.norm(p)),output=float(np.linalg.norm(o)),feedback=float(np.linalg.norm(b))),distinct_body_bytes=len(alpha),checkpoint_unchanged=True))
C.save('DIAGNOSTICS.json',dict(results=results,CPU=time.process_time()-start,extra_gradient_record_exposures=48,extra_gradient_target_evaluations=4656,extra_inference_record_evaluations=256,extra_autonomous_rollouts=2816,applied_step_computed_without_parameter_mutation=True,no_candidate_training=True))
