import json,gzip
from pathlib import Path
from collections import defaultdict
import numpy as np
R=Path('/Users/pencilbard/Downloads/second_try_copy/DIGITALBABY_CLAUDE_B_D4E4578_20260929')
S=(1.4911274663291492,1.3452365735750882)
stats=defaultdict(list); counts=defaultdict(int)
def features(p,b):
 v=np.array(p[b],float)/S[('shared','private').index(b)]
 sort=np.sort(v)
 q=np.exp(v-v.max());q/=q.sum()
 return dict(gap=float(sort[-1]-sort[-2]),spread=float(np.std(v)),range=float(np.ptp(v)),maxprob=float(q.max()),mean=float(v.mean()),winner=int(v.argmax()))
def add(source,assay,branch,checkpoint,group,p,teacher=None):
 for bank in ('shared','private'):
  f=features(p,bank)
  if teacher is not None:f['correct']=float(f['winner']==teacher-48)
  stats[(source,assay,branch,checkpoint,group,bank)].append(f)
files=list(sorted((R/'next_core_budget_v2_20261006/science/confirm/receipts').glob('*_ERROR_*.json.gz')))
files+=list(sorted((R/'earned_authority_v1_20261007/cycles/cycle3/full').glob('*/receipts/*.json.gz')))
for path in files:
 d=json.load(gzip.open(path,'rt')); source='DEV_one_world' if d['development'] else 'confirmation48'; counts[(source,d['assay'])]+=1
 for branch,bd in d['branches'].items():
  for rec in bd['records']:
   if rec['learn']:
    add(source,d['assay'],branch,'prequential_training',rec['stage'],rec['prediction'],rec['write']['outcome'])
  for probe in bd['probes']:
   if probe['name']!='final':continue
   for row in probe['rows']:
    for p in row.get('predictions',[row.get('prediction')]):
     if p is not None:add(source,d['assay'],branch,'final',row['stage'],p)
out=[]
for key,fs in sorted(stats.items()):
 d=dict(zip(('source','assay','branch','checkpoint','group','bank'),key));d['n_contexts']=len(fs)
 d['metrics']={name:{'mean':float(np.mean([f[name] for f in fs])), 'q10':float(np.quantile([f[name] for f in fs],.1)), 'median':float(np.median([f[name] for f in fs])), 'q90':float(np.quantile([f[name] for f in fs],.9))} for name in fs[0] if name!='winner'}
 d['winner_counts']={str(w):sum(f['winner']==w for f in fs) for w in range(4)};out.append(d)
Path('/private/tmp/authority_evidence_census_20261007.json').write_text(json.dumps({'no_new_learning':True,'no_threshold_search':True,'files_by_scope':{str(k):v for k,v in counts.items()},'summary':out},indent=2))
for d in out:
 if d['branch']=='W' and (d['checkpoint']=='final' or d['group']=='old'):
  m=d['metrics'];print(d['source'],d['assay'],d['checkpoint'],d['group'],d['bank'],'n',d['n_contexts'],'gap',tuple(round(m['gap'][s],3) for s in ('q10','median','q90')),'spread',round(m['spread']['mean'],3),'maxp',round(m['maxprob']['mean'],3),'correct',round(m.get('correct',{}).get('mean',-1),3))
