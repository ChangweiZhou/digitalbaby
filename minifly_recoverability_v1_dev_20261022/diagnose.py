"""Posthoc artifact-only payload/EOS and counterfactual diagnosis, no fitting."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import numpy as np,json
P=Path(__file__).resolve().parent
rows=[];bindings=[]
for seed in range(202610229301,202610229305):
 for std in (0,1):
  metas=[];decs=[]
  for life in (0,1):
   tag=f'{seed}_{life}_{std}_W';meta=json.loads((P/'traces'/f'{tag}.json').read_text());dec=json.loads((P/'traces'/f'{tag}_decoded.json').read_text());metas.append(meta);decs.append(dec)
   with np.load(P/'traces'/f'{tag}.npz') as raw,np.load(P/'traces'/f'{tag}_decoded.npz') as zp:
    for r in dec['results']:
     if r['lesion']!='NONE' or r['pool'] not in ('OLD_end','OLD_final','NEW_final','OLD_S_NEW'):continue
     vals=meta['labels'][meta['pools'].index(r['pool'])];z=raw[f"D{meta['pools'].index(r['pool'])}"].mean(-1) if r['readout']=='DIRECT' else zp[r['array']]
     payload=[];rank=[];validT=[]
     for a,v in zip(z,vals):
      L=len(v)
      if not L:continue
      got=[int(sum(1<<b for b in range(8) if a[j,b]>0)) for j in range(L)]
      payload.append(got==v);A=float(a[:L,8].max());B=float(a[L,8]);rank.append(A<B);validT.append(A<=0<B)
     rows.append(dict(pair=seed,life=life,STD=std,view=r['readout'],head_seed=r.get('head_seed'),pool=r['pool'],payload_exact_nonempty=float(np.mean(payload)),EOS_rank_feasible_nonempty=float(np.mean(rank)),EOS_zero_threshold_valid_nonempty=float(np.mean(validT)),bit_accuracy=r['metrics']['bits']))
  # Pair binding must be evaluated on same key, not just equal average accuracy.
  for cohort in ('OLD_final','NEW_final'):
   v0=metas[0]['labels'][metas[0]['pools'].index(cohort)];v1=metas[1]['labels'][metas[1]['pools'].index(cohort)]
   for view in ('MEAN','SEP'):
    for hs in range(202610229401,202610229404):
     rr=[next(r for r in decs[l]['results'] if r['pool']==cohort and r['readout']==view and r['head_seed']==hs and r['lesion']=='NONE') for l in (0,1)]
     o0,o1=[r['metrics']['outputs'] for r in rr];ids=[i for i in range(18) if v0[i] and v1[i] and v0[i]!=v1[i]]
     eq=lambda o,v:o['EOS'] and o['bytes']==v
     bindings.append(dict(pair=seed,STD=std,view=view,head_seed=hs,cohort=cohort,changed_nonempty_keys=len(ids),both_lives_correct=sum(eq(o0[i],v0[i]) and eq(o1[i],v1[i]) for i in ids),current_target_matches=sum(eq(o0[i],v0[i])+eq(o1[i],v1[i]) for i in ids),cross_life_target_matches=sum(eq(o0[i],v1[i])+eq(o1[i],v0[i]) for i in ids)))
out=dict(status='POSTHOC_ARTIFACT_ONLY_NO_FITTING',rows=rows,counterfactual_binding=bindings)
(P/'PAYLOAD_EOS_DIAGNOSIS.json').write_text(json.dumps(out,indent=2)+'\n')
for std in (0,1):
 for view in ('DIRECT','MEAN','SEP'):
  for pool in ('OLD_end','OLD_final','NEW_final','OLD_S_NEW'):
   z=[r for r in rows if r['STD']==std and r['view']==view and r['pool']==pool]
   print(std,view,pool,'payload',round(np.mean([r['payload_exact_nonempty'] for r in z])*100,2),'bits',round(np.mean([r['bit_accuracy'] for r in z])*100,2),'EOSrank',round(np.mean([r['EOS_rank_feasible_nonempty'] for r in z])*100,2))
