"""Read-only same-coordinate gradient attribution on saved R_LE states."""
import pickle,time
import numpy as np
import support152 as s
import models152 as m
from harness152 import evaluate

def stats(a,b):
 x=a.ravel();y=b.ravel();nx=float(np.linalg.norm(x));ny=float(np.linalg.norm(y))
 return dict(first_norm=nx,second_norm=ny,cosine=float(x@y/(nx*ny)) if nx and ny else None,norm_ratio=ny/nx if nx else None)
def run(world,eta):
 name=f'R_LE_{world}_{eta:g}';data=s.dataset(world);results=[]
 for phase,p in [('OLD',1),('OLD',8),('NEW',1),('NEW',8)]:
  with (s.H/'local_states'/f'{name}_{phase}{p}.pkl').open('rb') as f:base=pickle.load(f)
  rec=data[phase][data['orders'][phase][0][0]];prompt,answer=s.read_record(rec);gr={}
  for method in ['R_BP','R_LE','R_LE0']:
   s.target_record(answer,'credit',prompt=prompt);q=base.clone();gr[method],_=q.gradients(prompt,answer,method)
  blocks={method:{k:stats(gr['R_BP'][k],gr[method][k]) for k in base.PARAMS} for method in ['R_LE','R_LE0']};applied={method:{k:np.clip(v,-1,1) for k,v in gg.items()} for method,gg in gr.items()}
  post=[]
  for control in ['ZERO','LE_CORE_ONLY','LE_HEAD_ONLY','LE_JOINT','BP_JOINT_BLOCK_NORM_MATCHED']:
   s.target_record(answer,'credit',prompt=prompt);q=base.clone();gg={k:np.zeros_like(getattr(q,k)) for k in q.PARAMS}
   for k in gg:
    if control=='LE_CORE_ONLY' and k not in ['R','q'] or control=='LE_HEAD_ONLY' and k in ['R','q'] or control=='LE_JOINT':gg[k]=applied['R_LE'][k].copy()
    elif control=='BP_JOINT_BLOCK_NORM_MATCHED':
     nb=np.linalg.norm(applied['R_BP'][k]);nl=np.linalg.norm(applied['R_LE'][k]);gg[k]=applied['R_BP'][k]*(nl/nb) if nb else np.zeros_like(gg[k])
   # Matched diagnostic increments are already applied-gradient magnitudes; do not reclip them.
   for k in q.PARAMS:setattr(q,k,getattr(q,k)-q.eta*gg[k])
   q.theta=np.clip(q.theta,-3,3)
   norms={k:dict(requested=float(q.eta*np.linalg.norm(gg[k])),actual=float(np.linalg.norm(getattr(q,k)-getattr(base,k)))) for k in q.PARAMS}
   probes=evaluate(q,[rec,data['OLD'][data['orders']['OLD'][0][0]]],control,'credit')
   post.append(dict(control=control,applied_norms=norms,probe=probes))
  results.append(dict(world=world,snapshot=f'{phase}{p}',gradient_blocks=blocks,applied_gradient_blocks={method:{k:stats(applied['R_BP'][k],applied[method][k]) for k in base.PARAMS} for method in ['R_LE','R_LE0']},counterfactuals=post))
 s.save(f'CREDIT_{world}.json',results);return results
