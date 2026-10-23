"""Posthoc diagnostic: unchanged native payload + frozen head EOS; no fitting."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,numpy as np
from pathlib import Path
P=Path(__file__).resolve().parent;rows=[]
for seed in range(202610229301,202610229305):
 for life in (0,1):
  for std in (0,1):
   tag=f'{seed}_{life}_{std}_W';meta=json.loads((P/'traces'/f'{tag}.json').read_text());dec=json.loads((P/'traces'/f'{tag}_decoded.json').read_text())
   with np.load(P/'traces'/f'{tag}.npz') as raw,np.load(P/'traces'/f'{tag}_decoded.npz') as z:
    for r in dec['results']:
     if r['lesion']!='NONE' or r['readout']=='DIRECT' or r['pool'] not in ('OLD_final','NEW_final','OLD_S_NEW'):continue
     i=meta['pools'].index(r['pool']);vals=meta['labels'][i];m=raw[f'D{i}'].mean(-1);logits=z[r['array']];correct=[]
     for q,v,h in zip(m,vals,logits):
      bs=[];end=False
      for j in range(9):
       if h[j,8]>0:end=True;break
       if j==8:break
       bs.append(int(sum(1<<b for b in range(8) if q[j,b]>0)))
      if v:correct.append(end and bs==v)
     rows.append(dict(pair=seed,life=life,STD=std,view=r['readout'],head_seed=r['head_seed'],cohort=r['pool'],nonempty_exact=float(np.mean(correct))))
overview=[]
for std in (0,1):
 for view in ('MEAN','SEP'):
  d=dict(STD=std,view=view)
  for pool in ('OLD_final','NEW_final','OLD_S_NEW'):d[pool]=float(np.mean([r['nonempty_exact'] for r in rows if r['STD']==std and r['view']==view and r['cohort']==pool]))
  overview.append(d)
(P/'PAYLOAD_BYPASS_POSTHOC.json').write_text(json.dumps(dict(status='POSTHOC_DIAGNOSTIC_NOT_PREREGISTERED_CANDIDATE',training=False,native_calls=False,overview=overview,rows=rows),indent=2)+'\n')
print(json.dumps(overview,indent=2))
