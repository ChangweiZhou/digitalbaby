"""Read-only checkpoint analysis. No step(), fitting, or state persistence."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,math,copy,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];V=ROOT/'minifly_v182_shallow_local_dev_20261102';sys.path.insert(0,str(V))
import numpy as np,torch as T
import parent_learner as P,evaluator as E
from sensor import Sensor
from data import prepare
T.set_num_threads(1);start=time.process_time();s=Sensor();data,_=prepare(s);out={}
def batches(records,codes):
 rows=[];ys=[];ids=[]
 for ri,(r,cs) in enumerate(zip(records,codes)):
  for y,row in zip([*r,256],cs):
   rows.append(row);ys.append(y);ids.append(str(ri))
   if len(ys)==256:
    yield make(rows),T.tensor(ys),ids;rows=[];ys=[];ids=[]
 if ys:yield make(rows),T.tensor(ys),ids
def make(rows):
 x=T.zeros((len(rows),5177))
 for i,r in enumerate(rows):x[i,T.tensor(r.astype(np.int64))]=1
 return x
for w,sp in data.items():
 codes={name:E.pack(s,sp[name]) for name in ('A_train','A_held','B_train')};out[w]={}
 for stage in ('A2','B3'):
  net=P.Learner(w);net.restore(T.load(V/'results/checkpoints'/f'{w}_NORMAL_{stage}.pt',weights_only=True));before=E.state_hash(net)
  # A-train mean forecast gives a context-independent comparator without new labels.
  sump=T.zeros(257,dtype=T.float64);cnt=0
  for x,y,ids in batches(sp['A_train'],codes['A_train']):
   lp,_=net.forward(x);sump+=lp.double().softmax(1).sum(0);cnt+=len(y)
  meanp=sump/cnt
  trainy=np.array([b for r in sp['A_train'] for b in [*r,256]]);freq=T.from_numpy((np.bincount(trainy,minlength=257)+.5)/(len(trainy)+.5*257))
  own=const=cal=absent=0.;n=0;absent_mask=T.from_numpy(np.bincount(trainy,minlength=257)==0)
  for x,y,ids in batches(sp['A_held'],codes['A_held']):
   logits,_=net.forward(x);lp=logits.double().log_softmax(1);own+=float((-lp[T.arange(len(y)),y]/math.log(2)).sum());const+=float((-meanp[y].log2()).sum());q=(lp+freq.log()-meanp.log()).log_softmax(1);cal+=float((-q[T.arange(len(y)),y]/math.log(2)).sum());absent+=float(lp.exp()[:,absent_mask].sum());n+=len(y)
  marginal={'own_A_held_bits':own/n,'constant_mean_forecast_bits':const/n,'conditional_gain_over_own_marginal_bits':(const-own)/n,'training_frequency_recentered_diagnostic_bits':cal/n,'mean_mass_on_A_unseen_classes':absent/n}
  # Four earliest real batches, same dropout mask for local and exact reference.
  role='A_train' if stage=='A2' else 'B_train';geo=[]
  for bi,(x,y,ids) in enumerate(batches(sp[role],codes[role])):
   if bi==4:break
   st=copy.deepcopy(net.state());gl,_,_=net.gradients(x,y,ids);net.restore(st)
   logits,(_,ah,h,at,mask,z)=net.forward(x,True);pp={k:v.clone().requires_grad_() for k,v in net.p.items()}
   hh=((x[:,net.ix]*pp['H']).sum(2)+pp['Hb']).clamp_min(0);tt=(hh@pp['T'].T+pp['Tb']).clamp_min(0)*mask
   loss=T.nn.functional.cross_entropy(tt@pp['O'].T+pp['Ob'],y);gb=dict(zip(pp,T.autograd.grad(loss,tuple(pp.values()))));assert T.allclose(gl['O'],gb['O'],atol=1e-7,rtol=1e-5)
   net.restore(st);d={}
   for k,p in net.p.items():
    m=.9*net.m[k]+.1*gl[k];v=.999*net.v[k]+.001*gl[k]*gl[k];c=net.clock+1;d[k]=-.001*m/(1-.9**c)/((v/(1-.999**c)).sqrt()+1e-8)-.0000001*p
   obs={}
   for layer in ('H','T','O'):
    keys=[layer,layer+'b'];g=T.cat([gl[k].ravel() for k in keys]);true=T.cat([gb[k].ravel() for k in keys]);delta=T.cat([d[k].ravel() for k in keys]);obs[layer]={'local_true_gradient_cosine':float(T.dot(g,true)/(g.norm()*true.norm()+1e-30)),'predicted_CE_change_bits':float(T.dot(true,delta)/math.log(2))}
   geo.append(obs)
  assert E.state_hash(net)==before
  out[w][stage]={'marginal':marginal,'gradient_geometry':geo}
print(json.dumps({'optimizer_updates':0,'checkpoint_states_unchanged':True,'CPU_seconds':time.process_time()-start,'results':out},indent=2))
