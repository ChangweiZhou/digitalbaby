"""Cycle 2: charged numerical/optimizer/trace/restore qualification, no main fit."""
import os,sys,pickle,time,subprocess,traceback
import sharedlib as s
import numpy as np
from collections import deque
from controller171 import PARAMS,CAPS,Controller,warm
from numeric171 import forward_influence,forward_gradient,parameter_indices
def run():
 assert not (s.H/'QUALIFICATION_STARTED.json').exists(),'No automatic replay'
 s.TIER='technical';s.save('QUALIFICATION_STARTED.json',dict(CPU=s.cpu()));source=s.lock();s.save('QUALIFICATION_SOURCE_LOCK.json',source)
 checks={};gradchecks={}
 # Independent augmented-state forward sensitivity and detached-prefix finite differences.
 xs=[0,255,17,13];vs=np.random.default_rng(1721901).normal(0,2,(4,16));synthetic_losses=0
 for arm,keep in [('ADAM1',1),('ADAM3',3)]:
  c=s.Shared(202610269790,arm,np.ones(16));terms=[];prefix=None
  for i,(b,v) in enumerate(zip(xs,vs)):
   if i==len(xs)-keep:prefix=(c.h.copy(),c.r.copy())
   terms=forward_influence(c,b,v,terms,keep)
  forward=forward_gradient(c,sum(terms),183,2048);c.exact_credit(183);reverse=c.gradvector();rel=float(np.linalg.norm(forward-reverse)/max(1e-14,np.linalg.norm(reverse)));ids,_=parameter_indices(c)
  selected=[int(ix.ravel()[np.argmax(abs(reverse[ix.ravel()]))]) for ix in ids.values()];vec=c.vector();fd=[]
  for j in selected:
   losses=[]
   for sign in (1,-1):
    q=c.clone();v=vec.copy();v[j]+=sign*1e-5;q.setvector(v);q.reset();q.h=prefix[0].copy();q.r=prefix[1].copy()
    for x,value in zip(xs[-keep:],vs[-keep:]):q.step(x,value)
    losses.append(-q.read()[1][183]);synthetic_losses+=1
   fd.append((losses[0]-losses[1])/2e-5)
  fdrel=float(np.linalg.norm(np.array(fd)-reverse[selected])/max(1e-14,np.linalg.norm(reverse[selected])))
  gradchecks[arm]=dict(forward_reverse_relative=rel,finite_difference_relative=fdrel,parameters=len(selected));assert rel<1e-6 and fdrel<1e-6
 checks['numerical']=gradchecks;s.COUNTS['synthetic_loss_evaluations']+=synthetic_losses
 # Literal SGD implementation equality and explicitly post-preconditioner Adam ramp.
 g=s.Shared(202610269790,'SGD3',np.ones(16));g.step(17,vs[0]);g.exact_credit(183);legacy=g.clone();Controller.apply(legacy);g.update([warm(0)]);assert np.array_equal(g.vector(),legacy.vector());checks['literal_SGD_step_identity']=True
 a=s.Shared(202610269790,'ADAM3',np.ones(16));a.grads['c'].fill(2);old=a.c.copy();a.update([1/16]);expected=2**-10/16*2/(2+1e-8);assert np.allclose(old-a.c,expected,rtol=0,atol=1e-15);checks['Adam_bias_correction_and_actual_step_warmup']=True
 r=s.Shared(202610269790,'ADAM3-R0',np.ones(16));r.grads['R'].fill(100);r.update([1]);assert not r.R.any() and not r.mom1['R'].any() and not r.mom2['R'].any();checks['R0_cannot_regrow']=True
 a=s.Shared(202610269790,'ADAM3',np.ones(16));a.U.fill(1);a.grads['U'].fill(-2);a.update([1]);assert np.max(a.U)==1;checks['parameter_projection']=True
 assert len(set(tuple(x) for x in s.N.C))==257 and np.all(s.N.C.sum(1)==0)
 worst=min(float(np.min((s.N.C[y]-s.N.C[np.arange(257)!=y])@(1+.25*s.N.C[y])/4)) for y in range(257));assert worst>0;checks['constant_carrier_all_classes_box_witness_min_margin']=worst
 checkpoint=s.ROOT/'minifly_v171_dev_20261026/local/cal_202610269711_final.pkl';assert s.sha(checkpoint)==s.SPEC['inputs_sha256'][str(checkpoint.relative_to(s.ROOT))]
 with checkpoint.open('rb') as f:model,_=pickle.load(f)
 parent_before=s.sha(checkpoint);prompt=bytes([0,255,17]);answer=bytes([73]);t0=s.cpu();cache=s.collect(model,prompt,answer,'technical:opaque0');cache_CPU=s.cpu()-t0
 s.checkpoint(cache,'technical/CACHE.pkl');actors={arm:s.Shared(202610269790,arm,np.ones(16)) for arm in s.ARMS};births={k:v.state_hash() for k,v in actors.items() if k!='ADAM3-R0'};assert len(set(births.values()))==1
 before={k:v.vector().copy() for k,v in actors.items()};start=s.cpu();rows={}
 for arm,c in actors.items():
  out,sites=s.fit_record(c,cache);rows[arm]=dict(out=out,sites=[x[0] for x in sites]);assert c.n_train==2 and not c.h.any() and not c.frames
  if arm=='ADAM3-CONST':assert all(np.allclose(x[1]['v'],1) for x in sites)
 fit_CPU_per_five=s.cpu()-start;s.checkpoint((model,actors,rows),'technical/POST_RECORD.pkl')
 checks['birth_equality_excluding_R0']=True;checks['constant_at_all_targets']=True;checks['record_end_reset_and_target_clock']=True
 q=actors['ADAM3'];unchanged=actors['SGD3'].vector().copy();q.U[0,0]+=.001;assert np.array_equal(actors['SGD3'].vector(),unchanged);q.U[0,0]-=.001;checks['mutable_independence']=True
 t0=s.cpu();first=s.N.pair(model,actors['ADAM3'],prompt,answer);pair_CPU=s.cpu()-t0
 receipt=s.checkpoint(actors,'technical/ROUNDTRIP.pkl')
 with (s.H/receipt['path']).open('rb') as f:restored=pickle.load(f)
 second=s.N.pair(model,restored['ADAM3'],prompt,answer);assert first==second;assert restored['ADAM3'].opt_t==actors['ADAM3'].opt_t;checks['whole_actor_and_optimizer_roundtrip']=True
 checks['parent_checkpoint_file_unchanged']=s.sha(checkpoint)==parent_before;assert checks['parent_checkpoint_file_unchanged']
 s.save('local/technical/RECORD_RESULTS.json',rows)
 assert all(s.sha(s.ROOT/p)==v for p,v in source.items());assert s.COUNTS['technical_fit_records']==5 and s.COUNTS['technical_fit_targets']==10
 # Explicit upper forecast; measured reference work and a multiplicative allowance, not a fitted outcome.
 forecast=4*(160*cache_CPU+12800*fit_CPU_per_five/5+5120*pair_CPU)+180
 admission=forecast+s.cpu()<3600;assert admission,('RESOURCE_ADMISSION',forecast,s.cpu())
 s.save('QUALIFICATION.json',dict(status='PASS',checks=checks,synthetic_loss_evaluations=synthetic_losses,technical_fit_records=5,technical_fit_targets=10,technical_cache_episodes=1,technical_own_TF_pairs=2,benchmark=dict(cache_CPU=cache_CPU,five_arms_one_record_CPU=fit_CPU_per_five,pair_CPU=pair_CPU),conservative_forecast_CPU=forecast,main_admitted=True,source_lock_entries=len(source),counts=dict(s.COUNTS),CPU=s.cpu(),wall=s.wall(),peak_RSS_bytes=s.rss()))
 s.save('cycles/CYCLE2_REVIEW.md', '# Cycle 2 — qualified\n\nIndependent exact-one/three forward and finite-difference checks, original SGD parity, actual Adam warmup, R0 structural lesion, constant carrier, opaque trace, complete optimizer roundtrip and charged admission all passed. Five opaque fitted records/ten targets and two own/TF pairs are technical observations, not addition evidence. Fixed main dose/arms/caps unchanged.\n')
 s.finish('CYCLE2_QUALIFIED_MAIN_ADMITTED');print(s.plain(dict(status='PASS',forecast_CPU=forecast,CPU=s.cpu(),checks=checks)))
if __name__=='__main__':
 awake=subprocess.Popen(['caffeinate','-i','-w',str(os.getpid())],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
 try:run()
 except Exception as e:
  s.save('QUALIFICATION_FAILURE.json',dict(error=str(e),traceback=traceback.format_exc().replace(str(s.ROOT),'$REPO').replace(str(s.HOME) if hasattr(s,'HOME') else '/Users/pencilbard','$HOME'),counts=dict(s.COUNTS),CPU=s.cpu()));s.finish('QUALIFICATION_FAILED_NO_AUTOMATIC_RETRY');raise
 finally:awake.terminate();awake.wait(timeout=5)
