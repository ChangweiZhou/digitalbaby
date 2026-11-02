"""Corrected evaluation contract, on technical records only during this exercise."""
import util as U
import learner as M
import sys,numpy as np
T=M.torch

def score(net,fixture,old_y=None,same_code=None):
 before=M.state_hash(net,rng=T.get_rng_state());net.eval()
 with T.no_grad():lp=T.log_softmax(net(fixture['idx']).double(),1).numpy()/np.log(2)
 y=fixture['y'];p=np.exp2(lp);bits=-lp[np.arange(len(y)),y];eos=y==256;bins=[]
 for i in range(10):
  sel=(p[:,256]>=i/10)&(p[:,256]<(i+1)/10 if i<9 else p[:,256]<=1);bins.append(dict(n=int(sel.sum()),predicted=float(p[sel,256].mean()) if sel.any() else None,observed=float(eos[sel].mean()) if sel.any() else None))
 out=dict(bits=float(bits.mean()),targets=len(y),payload_bits=float(bits[~eos].mean()),EOS_NLL=float(bits[eos].mean()),EOS_Brier=float(np.mean((p[:,256]-eos)**2)),false_EOS_rate=float(np.mean(p[~eos].argmax(1)==256)),entropy=float(-(p*lp).sum(1).mean()),EOS_calibration=bins,normalization_error=float(abs(p.sum(1)-1).max()),accuracy=float((p.argmax(1)==y).mean()))
 if old_y is not None:
  changed=y!=old_y;same=(~changed)&same_code;altered=(~changed)&~same_code
  out['subsets']={k:dict(n=int(v.sum()),bits=float(bits[v].mean()) if v.any() else None) for k,v in [('SAME_KC',same),('ALTERED_KC',altered),('CHANGED',changed)]};out['new_old_margin']=float((lp[np.arange(len(y)),y]-lp[np.arange(len(y)),old_y])[changed].mean()) if changed.any() else None
 assert M.state_hash(net,rng=T.get_rng_state())==before
 return out

def free_run(net,records):
 sys.path.append('/Users/pencilbard/Downloads/second_try_copy/V88/.venv-rce-v87-followup/lib/python3.11/site-packages');sys.path.insert(0,str(U.ROOT/'minifly_v176_feature_deployment_dev_20261101'));import native_stage as N
 m=N.Model();before=M.state_hash(net,rng=T.get_rng_state());net.eval();lengths=[];reference=[];truncated=emitted=0
 for r in records:
  prefix=bytes(r[:3]);generated=[];stop=False;f=m._new_front();t=m.time+2*N.F.DELTA;f.advance(t)
  for b in prefix:t+=N.F.DELTA;f.feed(b,t)
  for _ in range(128):
   view=f.clone();view.advance(t+N.F.DELTA);x=np.flatnonzero(N.F.FastMemory.encode(m,view.read())[0]);idx=np.full((1,185),5178,np.int64);idx[0,:len(x) if len(x) else 1]=x if len(x) else 5177
   with T.no_grad():y=int(net(T.from_numpy(idx)).argmax(1))
   if y==256:stop=True;emitted+=1;break
   generated.append(y);t+=N.F.DELTA;f.feed(y,t)
  truncated+=not stop;lengths.append(len(prefix)+len(generated));reference.append(len(r))
 assert M.state_hash(net,rng=T.get_rng_state())==before
 return dict(records=len(records),own_EOS=emitted,truncated=truncated,max_model_outputs=128,predicted_payload_lengths=lengths,reference_payload_lengths=reference,length_histogram={str(v):lengths.count(v) for v in sorted(set(lengths))},exact_prose_not_a_gate=True,read_only=True)
