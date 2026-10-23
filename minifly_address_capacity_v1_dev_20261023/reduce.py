"""Artifact-only independent metric reduction; no native imports."""
import json,hashlib,zipfile
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;R=P.parent
rows=[]
for f in (P/'traces').glob('*_outputs.json'):
 d=json.loads(f.read_text());N=int(f.name.split('_')[0])
 for x in d['results']:
  if x['lesion']=='NONE' and x['pool'] in ('OLD_final','NEW_final','OLD_S_NEW','OLD_end'):
   i=d['pools'].index(x['pool']);vals=d['labels'][i];ix=[j for j,v in enumerate(vals) if v]
   exact=np.mean([x['metrics']['outputs'][j]['EOS'] and x['metrics']['outputs'][j]['bytes']==vals[j] for j in ix])
   assert abs(exact-x['metrics']['nonempty_exact'])<1e-12
   rows.append(dict(N=N,pair=d['seed'],life=d['life'],arm=d['arm'],pool=x['pool'],head=x['head'],payload=x['metrics']['payload_exact'],complete=float(exact)))
if rows:
 comparisons=[]
 for pair in sorted({x['pair'] for x in rows}):
  out={'pair':pair}
  for pool in ('OLD_final','NEW_final'):
   for metric in ('payload','complete'):
    nums=[]
    for N in (5177,20708):
     rr=[x for x in rows if x['N']==N and x['pair']==pair and x['pool']==pool and x['arm']=='W' and ((x['head']=='DIRECT')==(metric=='payload'))]
     nums.append(float(np.mean([x[metric] for x in rr])))
    out[f'{pool}_{metric}']=dict(CANON=nums[0],EXPAND=nums[1],delta=nums[1]-nums[0])
  comparisons.append(out)
 summary={'pairs_completed':len(comparisons),'comparisons':comparisons,'storage_screen':len(comparisons)==8 and all(x['OLD_final_payload']['delta']>0 and x['NEW_final_payload']['delta']>=0 for x in comparisons),'integration_screen':len(comparisons)==8 and all(x['OLD_final_complete']['delta']>0 and x['NEW_final_complete']['delta']>=0 for x in comparisons)}
 summary['integration_screen']=summary['storage_screen'] and summary['integration_screen']
 for label in ('OLD_final_payload','NEW_final_payload','OLD_final_complete','NEW_final_complete'):
  a=np.array([x[label]['delta'] for x in comparisons]);mu=float(a.mean());half=float(2.364624251*np.std(a,ddof=1)/np.sqrt(len(a))) if len(a)==8 else None
  summary[label+'_paired_interval']={'mean':mu,'descriptive_95_interval':[mu-half,mu+half] if half is not None else None}
 causal=[]
 for N in (5177,20708):
  for pool in ('OLD_final','NEW_final'):
   means={arm:float(np.mean([x['payload'] for x in rows if x['N']==N and x['arm']==arm and x['pool']==pool and x['head']=='DIRECT'])) for arm in ('W','N','SH')}
   causal.append(dict(N=N,pool=pool,means=means,W_minus_N=means['W']-means['N'],W_minus_SH=means['W']-means['SH']))
 summary['causal_diagnostics']=causal
 summary['attribution_guard']=all(x['W_minus_N']>0 for x in causal if x['N']==20708)
 summary['storage_screen_with_attribution']=summary['storage_screen'] and summary['attribution_guard']
 (P/'PILOT_SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({'reduced_rows':len(rows),'native_rerun':False}))
