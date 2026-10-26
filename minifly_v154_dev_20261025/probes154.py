"""Evaluator-only fixed-feature readout instruments. Never native candidate answers."""
import math,time
import numpy as np
from scipy.optimize import minimize
import support154 as s
import native154 as n
from features154 import stream_indices
A=(n.C-n.C.mean(1,keepdims=True))/4
LAMBDA=1/3072
class SolverCap(RuntimeError):pass

def softmax(z):
 z=z-z.max(1,keepdims=True);lp=z-np.log(np.exp(z).sum(1,keepdims=True));return np.exp(lp),lp

def score(W,X,y,ids,data,constrained,parts,tag):
 groups=[]
 for part in parts:
  allowed={r['id'] for r in data[part]};ix=np.array([i for i,r in enumerate(ids) if r in allowed]);z=np.asarray(X[ix]@W)
  if constrained:z=z@A.T
  prob,lp=softmax(z);pred=prob.argmax(1);by=[]
  for rec in data[part]:
   rr=[j for j,i in enumerate(ix) if ids[i]==rec['id']];s.COUNTS['probe_TF_records']+=1;by.append(dict(id=rec['id'],targets=[dict(y=int(y[ix[j]]),argmax=int(pred[j]),lp=float(lp[j,y[ix[j]]])) for j in rr]))
  s.COUNTS['probe_TF_distributions']+=len(ix);groups.append(dict(partition=part,tag=tag,records=by,BPS=float(-lp[np.arange(len(ix)),y[ix]].mean()/math.log(2)),argmax_accuracy=float(np.mean(pred==y[ix]))))
 return groups

def fit(X,y,ids,data,kind,key):
 constrained=kind.endswith('C16');offline=kind.startswith('OFF');K=16 if constrained else 257;d=X.shape[1];order,scales,phases=stream_indices(data,ids);agg=np.bincount(order,weights=scales,minlength=len(y));active=np.flatnonzero(agg);trainX=X[active];trainY=y[active];weights=agg[active]/len(order);shape=(d,K);score_rows=[];start=s.cpu();fitid=f'{key}_{kind}';W=np.zeros(shape);evaluations=0;best=None;converged=False;reason=''
 if offline:
  def objective(flat):
   nonlocal evaluations,best
   if evaluations>=120:raise SolverCap('MAX_OBJECTIVE_EVALUATIONS')
   s.budget();evaluations+=1;s.COUNTS['batch_evaluations']+=1;s.COUNTS['batch_virtual_target_reuses']+=len(order);s.COUNTS['batch_physical_target_rows']+=len(active)
   w=flat.reshape(shape);z=np.asarray(trainX@w)
   if constrained:z=z@A.T
   p,lp=softmax(z);loss=float(-weights@lp[np.arange(len(trainY)),trainY]+.5*LAMBDA*np.sum(w*w));p[np.arange(len(trainY)),trainY]-=1;p*=weights[:,None]
   if constrained:p=p@A
   grad=np.asarray(trainX.T@p)+LAMBDA*w
   if best is None or loss<best[0]:best=(loss,flat.copy(),float(np.max(abs(grad))))
   return loss,grad.ravel()
  try:
   result=minimize(objective,W.ravel(),jac=True,method='L-BFGS-B',options=dict(maxiter=60,maxfun=120,gtol=1e-6,ftol=1e-12,maxcor=5));reason=str(result.message);converged=bool(result.success and best[2]<=1e-6)
  except SolverCap as e:reason=str(e)
  assert best is not None;W=best[1].reshape(shape);gradient_inf=best[2];s.COUNTS['batch_fits']+=1
 else:
  eta=1/(1+float(np.linalg.norm(A,2))**2) if constrained else .5;scale=1.;offset=0
  for phase in ['OLD','NEW']:
   for p in range(8):
    for recix in data['orders'][phase][p]:
     rec=data[phase][recix];prompt,answer=s.unpack(rec);s.record(prompt,answer,'online_probe',True)
     for i in [j for j,r in enumerate(ids) if r==rec['id']]:
      row=X.getrow(i);idx=row.indices;val=row.data;z=scale*(val@W[idx])
      if constrained:z=A@z
      z-=z.max();pr=np.exp(z)/np.exp(z).sum();pr[y[i]]-=1;g=A.T@pr if constrained else pr
      scale*=1-eta*LAMBDA;W[idx]-=(eta*n.parent.warmup(offset)/scale)*np.outer(val,g);offset+=1;s.COUNTS['online_probe_target_updates']+=1
    s.budget()
   if phase=='OLD':score_rows+=score(scale*W,X,y,ids,data,constrained,['OLD'],'OLD_END')
  W*=scale;gradient_inf=None;converged=True;reason='FIXED_STREAM_COMPLETE';s.COUNTS['online_fits']+=1
 score_rows+=score(W,X,y,ids,data,constrained,['OLD','NEW','validation'],'FINAL');meta=dict(id=fitid,kind=kind,parameters=d*K,feature_dimension=d,output_classes=257,explicit_bias=False,CPU=s.cpu()-start,objective_evaluations=evaluations,virtual_target_uses=evaluations*len(order),numerical_gradient_inf=gradient_inf,status='QUALIFIED' if converged else 'OPTIMIZER_UNRESOLVED',stop_reason=reason,external_diagnostic_only=True,scoring=score_rows)
 s.save(f'local/PROBE_{fitid}.json',meta);s.save(f'PROBE_{fitid}.json',{k:v for k,v in meta.items() if k!='scoring'}|{'metrics':[dict(partition=x['partition'],tag=x['tag'],BPS=x['BPS'],argmax_accuracy=x['argmax_accuracy']) for x in score_rows]});s.log('PROBE_COMMIT',id=fitid,status=meta['status'],CPU=meta['CPU']);return meta
