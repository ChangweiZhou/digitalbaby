"""Frozen-checkpoint finite-difference probe. Zero optimizer updates."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,math,hashlib,time
from pathlib import Path
V=Path(__file__).resolve().parents[2]/'minifly_v182_shallow_local_dev_20261102';sys.path.insert(0,str(V))
import torch as T,numpy as np
import parent_learner as P,evaluator as E
from sensor import Sensor
from data import C
T.set_num_threads(1);s=Sensor();out={};start=time.process_time()
for w,fn in [('M1','austen-persuasion.txt'),('M2','shakespeare-hamlet.txt')]:
 records=C.records((V/'assets/corpora'/fn).read_bytes())[:160];rows=[];ys=[]
 for r in records:
  for row,y in zip(s.codes(r),[*r,256]):
   rows.append(row);ys.append(y)
   if len(ys)==1024:break
  if len(ys)==1024:break
 x=T.from_numpy(np.array(rows));ys=T.tensor(ys);net=P.Learner(w);net.restore(T.load(V/'results/checkpoints'/f'{w}_NORMAL_A2.pt',weights_only=True));before=E.state_hash(net);out[w]=[]
 for b in range(4):
  xb=x[b*256:(b+1)*256];y=ys[b*256:(b+1)*256];rng=net.rng.get_state();_,(xx,ah,h,at,mask,z)=net.forward(xb,True);net.rng.set_state(rng)
  ah=ah.double().detach().requires_grad_();wt=net.p['T'].double();bt=net.p['Tb'].double();wo=net.p['O'].double();bo=net.p['Ob'].double();mask=mask.double();xx=xx.double()
  def losses(a):return T.nn.functional.cross_entropy((((a.clamp_min(0)@wt.T+bt).clamp_min(0)*mask)@wo.T+bo),y,reduction='none')
  base=losses(ah);true_delta=T.autograd.grad(base.mean(),ah)[0];true=T.cat([T.einsum('nj,njr->jr',true_delta,xx).flatten(),true_delta.sum(0)])
  def cos(a):return float(T.dot(a,true)/(a.norm()*true.norm()+1e-30))
  logits=(h.double()@wt.T+bt).clamp_min(0)*mask;logits=logits@wo.T+bo;e=logits.softmax(1);e[T.arange(256),y]-=1;e/=256;local=(e@net.fh.double())*(ah>0);local=T.cat([T.einsum('nj,njr->jr',local,xx).flatten(),local.sum(0)])
  sigma=.01*float(ah.detach().square().mean().sqrt());g=[]
  gen=T.Generator().manual_seed(183000+b)
  for seed in range(8):
   noise=T.randn(ah.shape,dtype=T.float64,generator=gen)
   dp=((losses(ah.detach()+sigma*noise)-losses(ah.detach()-sigma*noise))/(2*sigma))[:,None]*noise/256
   g.append(T.cat([T.einsum('nj,njr->jr',dp,xx).flatten(),dp.sum(0)]))
  out[w].append({'source_H_cosine':cos(local),'one_pair_cosine':cos(g[0]),'eight_pair_average_cosine':cos(T.stack(g).mean(0)),'sigma':sigma,'all_one_pair_cosines':[cos(v) for v in g]})
 assert E.state_hash(net)==before
print(json.dumps({'optimizer_updates':0,'checkpoint_states_unchanged':True,'CPU_seconds':time.process_time()-start,'evidence_scope':'four fixed 256-site A-train batches at A2 in two opened worlds; directional feasibility only','results':out},indent=2))
