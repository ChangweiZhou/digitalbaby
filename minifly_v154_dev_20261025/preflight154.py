"""Cycle 1 tests: numerical/causal interfaces, no addition training."""
import math,pickle,copy
import numpy as np
from scipy.optimize import linprog
import support154 as s
import native154 as n

def run():
 checks={};C=n.C;A=(C-C.mean(1,keepdims=True))/4;witness=[];port_ok=True
 for y in range(257):
  v=A[y];z=A@v
  if all(z[y]>z[j]+1e-10 for j in range(257) if j!=y):witness.append(dict(y=y,method='code_row'))
  else:
   other=np.delete(np.arange(257),y);lp=linprog(np.zeros(16),A_ub=A[other]-A[y],b_ub=-np.ones(256),bounds=[(None,None)]*16,method='highs',options={'threads':1})
   witness.append(dict(y=y,method='LP',success=bool(lp.success)))
   if not lp.success:
    port_ok=False
 checks['all_257_unique_argmax_classes_feasible_q0']=port_ok
 # Native FE endpoints and control timing, with no observation labels.
 trace_errors=[]
 for variant,old in [('I_FAST',n.fastmem.BC.FE0()),('P_SLOW',n.fastmem.FEPerm())]:
  f=n.Trace(variant)
  for i,b in enumerate(list(range(256))+[0,255,10,0]):
   t=(i+1)*n.DELTA;old.feed(b,t);f.feed(b,t);old.advance(t+.1);f.advance(t+.1)
   trace_errors.append(float(np.max(abs(old.p-f.p))));assert np.array_equal(old.read(),f.read())
 assert max(trace_errors)==0;checks['FE0_and_FEPERM_endpoints_bit_identical']=True
 # Native parity, nonplastic activity parity and safe paired-prefix optimization.
 p=b'\x00ab\xff';a=b'\x10';errors=[];query_errors=[];write_gains=[]
 for front in n.FRONTS:
  model=n.Native('N_2B',front);passive=model.clone();s.record(p,a,'technical',True);s.record(p,a,'technical',False)
  model.train(p,a,True);passive.train(p,a,False)
  for mm in [model,passive]:mm._prompt(p)
  x=n.taps(model);xp=n.taps(passive);assert all(np.array_equal(x[k],xp[k]) for k in x)
  checks['native_written_and_inhibited_activity_parity']=True
  before=s.digest(model);r=n.pair(model,p,a,'technical');assert s.digest(model)==before
  s.query('technical');old=n.parent.generate(model,p);tf=n.parent.teacher_forced(model,p,a)
  assert old['output_hex']==r['output_hex'] and old['EOS']==r['EOS'] and old['choices']==r['choices']
  query_errors.append(max(abs(u['lp']-v['lp']) for u,v in zip(tf,r['teacher_forced'])));assert query_errors[-1]==0
  # No extra teaching: log expression movement caused by the already-paid write.
  write_gains.append(dict(front=front,active_KCs=int(x['X'].sum()),observed_norm=float(np.linalg.norm(x['OBS'])),expression_displacement=float(np.linalg.norm(model.values()-passive.values()))))
 checks['query_prefix_sharing_preserves_exact_outputs']=True;checks['query_read_only']=True
 model=n.Native('N_A','P_SLOW');old=n.parent.Native('N_A');model.n=old.n=1023
 for mm in [model,old]:s.record(p,a,'technical',True);mm.train(p,a)
 err=max(float(np.max(abs(getattr(x.fly.m,k)-getattr(y.fly.m,k)))) for x,y in zip(model.bank,old.bank) for k in ['fast','slow','adapt']);assert err==0
 checks['operative_parent_A_state_timing_parity']=True
 # Serialization compares numerical semantics, not pickle memoization byte identity.
 cp=pickle.loads(pickle.dumps(model));assert cp.n==model.n and cp.time==model.time
 for x,y in zip(model.bank,cp.bank):
  for k in ['fast','slow','adapt']:assert np.array_equal(getattr(x.fly.m,k),getattr(y.fly.m,k))
 checks['semantic_serialization_roundtrip']=True
 # Local full and constrained heads use the exact gradient of their one-position loss.
 rg=np.random.default_rng(154);phi=rg.normal(size=7);phi/=np.linalg.norm(phi);coords=[]
 for constrained in [False,True]:
  B=A if constrained else np.eye(257);W=rg.normal(0,.01,(B.shape[1],7));y=31;scale=.37
  def f(w):
   z=B@w@phi;z-=z.max();lp=z-np.log(np.exp(z).sum());return -scale*lp[y]+np.sum(w*w)/(2*3072)
  z=B@W@phi;z-=z.max();prob=np.exp(z)/np.exp(z).sum();prob[y]-=1;grad=scale*np.outer(B.T@prob,phi)+W/3072
  for ix in [(0,1),(W.shape[0]-1,6)]:
   t=W[ix];ep=1e-5;W[ix]=t+ep;plus=f(W);W[ix]=t-ep;minus=f(W);W[ix]=t;coords.append(abs((plus-minus)/(2*ep)-grad[ix]))
 assert max(coords)<2e-9;checks['head_gradient_and_regularizer_parity']=True
 # Scorer-only witness and metadata roundtrip use saved synthetic bytes, no model query.
 expected=bytes([0,10,255]);synthetic=dict(expected_hex=expected.hex(),output_hex=expected.hex(),EOS=True,choices=list(expected)+[256],correct=True)
 assert json_roundtrip(synthetic)==synthetic;checks['receipt_serialization_and_payload_EOS_witness']=True
 s.save('cycles/PORT_CHECK.json',dict(status='PASS' if port_ok else 'PORT_NOT_QUALIFIED',classes=witness))
 s.save('cycles/CYCLE1_TESTS.json',dict(status='PASS' if port_ok else 'PASS_WITH_C16_PORT_FAILURE',checks=checks,trace_max_error=max(trace_errors),parent_max_error=err,paired_query_loss_error=max(query_errors),gradient_errors=coords,native_write_gain_diagnostics=write_gains,technical_counts={k:v for k,v in s.COUNTS.items() if 'technical' in k}))
 return port_ok

def json_roundtrip(x):
 import json
 return json.loads(json.dumps(x))
