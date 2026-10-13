"""Fixed 82-contrast reducer: paired worlds, normalized log-CPU domains."""
import json,math
from pathlib import Path
import numpy as np
from scipy.stats import t
from core import CONFIGS,CANDIDATES,BASELINE
from worlds import OPS,CHECKPOINTS
FAMILY=82;N=48
SIMPLE=(('VALUE-ADD','BYTE-ADD'),('VALUE-PROD','BYTE-PROD'),('BYTE-PROD','BYTE-ADD'),('VALUE-PROD','VALUE-ADD'))
def j_curve(v):
 if len(v)!=6:raise ValueError('six-point curve required')
 return float(np.trapz(v,np.log2(CHECKPOINTS))/5)
def curve(r,history='W',op=None):
 rows=r['histories'][history]['checkpoints']
 return [x['scores'][op] if op else np.mean([x['scores'][o] for o in OPS]) for x in rows],[x['cpu'] for x in rows]
def final_score(r,history='W',op=None):
 a=r['histories'][history]['final'];scores={o:np.mean([x['correct'] for x in a if x['operation']==o]) for o in OPS}
 return float(scores[op] if op else np.mean(list(scores.values())))
def cpu_scores(rs):
 curves=[curve(r) for r in rs];xs=[np.array(x,float) for _,x in curves]
 if any(len(x)!=6 or not np.isfinite(x).all() or x[0]<=0 or np.any(np.diff(x)<=0) for x in xs):return [None]*len(rs)
 lo=max(x[0] for x in xs);hi=min(x[-1] for x in xs)
 if hi<=lo:return [None]*len(rs)
 knots=np.unique(np.r_[lo,hi,*[x[(x>lo)&(x<hi)] for x in xs]]);logs=np.log(knots)
 return [float(np.trapz(np.interp(logs,np.log(x),v),logs)/(np.log(hi)-np.log(lo))) for (v,_),x in zip(curves,xs)]
def cpu_pair(a,b):
 q=cpu_scores([a,b]);return None if q[0] is None else q[0]-q[1]
def paired_interval(v,interaction=False,technical=False):
 unavailable={'mean':None,'lower':None,'upper':None,'status':'UNAVAILABLE','family_size':FAMILY,'n_worlds':len(v)}
 if (not technical and len(v)!=N) or len(v)<2 or any(x is None or not math.isfinite(x) for x in v):return unavailable
 a=np.array(v,float);mean=float(a.mean());sd=float(a.std(ddof=1));bound=2 if interaction else 1
 if sd==0:
  radius=bound*np.sqrt(2*np.log(2*FAMILY/.05)/len(v));status='BOUNDED_HOEFFDING'
 else:radius=float(t.ppf(1-.05/(2*FAMILY),len(v)-1)*sd/np.sqrt(len(v)));status='ESTIMATED'
 return {'mean':mean,'lower':max(-bound,mean-radius),'upper':min(bound,mean+radius),
  'status':status,'family_size':FAMILY,'n_worlds':len(v),'technical':technical}
def contrasts(worlds):
 out={}
 for c in CANDIDATES:
  out[f'{c}:vs_LEGACY-B:J_samples']=[j_curve(curve(w[c])[0])-j_curve(curve(w[BASELINE])[0]) for w in worlds]
  out[f'{c}:vs_LEGACY-B:J_cpu_pair_legacy']=[cpu_pair(w[c],w[BASELINE]) for w in worlds]
 common=[dict(zip(CANDIDATES,cpu_scores([w[c] for c in CANDIDATES]))) for w in worlds]
 for a,b in SIMPLE:
  out[f'{a}_minus_{b}:J_samples']=[j_curve(curve(w[a])[0])-j_curve(curve(w[b])[0]) for w in worlds]
  out[f'{a}_minus_{b}:J_cpu_four_arm_common']=[None if q[a] is None else q[a]-q[b] for q in common]
 def interaction(q):return q['VALUE-PROD']-q['BYTE-PROD']-q['VALUE-ADD']+q['BYTE-ADD']
 out['value_x_product_interaction:J_samples']=[interaction({c:j_curve(curve(w[c])[0]) for c in CANDIDATES}) for w in worlds]
 out['value_x_product_interaction:J_cpu_four_arm_common']=[None if any(v is None for v in q.values()) else interaction(q) for q in common]
 for c in CANDIDATES:
  for op in OPS:
   for h in ('N_ALL','SHUFFLE'):out[f'{c}:final_W_minus_{h}:{op}']=[final_score(w[c],op=op)-final_score(w[c],h,op) for w in worlds]
   out[f'{c}:J_samples_vs_LEGACY-B:{op}']=[j_curve(curve(w[c],op=op)[0])-j_curve(curve(w[BASELINE],op=op)[0]) for w in worlds]
   out[f'{c}:final_accuracy_vs_LEGACY-B:{op}']=[final_score(w[c],op=op)-final_score(w[BASELINE],op=op) for w in worlds]
 spec=json.loads((Path(__file__).parent/'design/DESIGN.json').read_text())
 assert set(out)==set(spec['statistics']['contrasts']) and len(out)==FAMILY
 return out
def taught_covers(candidate,baseline):
 # Empirical coverage only, never infer population equivalence from zero errors.
 c={x['prompt_hex']:x['correct'] for x in candidate['histories']['W']['checkpoints'][-1]['taught']}
 b={x['prompt_hex']:x['correct'] for x in baseline['histories']['W']['checkpoints'][-1]['taught']}
 if c.keys()!=b.keys():raise ValueError('taught roster mismatch')
 return all(c[k] for k,v in b.items() if v)
def direct_interference(r,op,kind='held'):
 h=r['histories']['W'];deltas=[];no_intervening=[]
 for ci,cp in enumerate(h['checkpoints']):
  for bi,b in enumerate(h['blocks'][4*ci:4*ci+4]):
   if b['operation']!=op:continue
   if kind=='held':delta=cp['scores'][op]-b['scores'][op]
   else:
    earlier=[x for x in b['taught'] if x['operation']==op];keys={x['prompt_hex'] for x in earlier}
    end=[x for x in cp['taught'] if x['operation']==op and x['prompt_hex'] in keys]
    delta=np.mean([x['correct'] for x in end])-np.mean([x['correct'] for x in earlier])
   (no_intervening if bi==3 else deltas).append(float(delta))
 return {'mean':float(np.mean(deltas)) if deltas else None,'interference_points':deltas,'last_block_no_intervening_points':no_intervening}
def reduce(worlds,technical=False):
 raw=contrasts(worlds);stats={k:paired_interval(v,k.startswith('value_x_product'),technical) for k,v in raw.items()}
 profiles={};eligible=[]
 for c in CANDIDATES:
  op_evidence={op:all(stats[f'{c}:final_W_minus_{h}:{op}']['lower'] is not None and stats[f'{c}:final_W_minus_{h}:{op}']['lower']>0 for h in ('N_ALL','SHUFFLE')) for op in OPS}
  positive=[f'{c}:vs_LEGACY-B:{m}' for m in ('J_samples','J_cpu_pair_legacy')]+[f'{c}:final_W_minus_{h}:{op}' for op in OPS for h in ('N_ALL','SHUFFLE')]
  guards=[f'{c}:{m}:{op}' for op in OPS for m in ('J_samples_vs_LEGACY-B','final_accuracy_vs_LEGACY-B')]
  cover=all(taught_covers(w[c],w[BASELINE]) for w in worlds)
  qualifies=all(stats[k]['lower'] is not None and stats[k]['lower']>0 for k in positive) and all(stats[k]['lower'] is not None and stats[k]['lower']>=0 for k in guards) and cover
  profiles[c]={'final_E3_operations':op_evidence,'all_four_final_E3':all(op_evidence.values()),'observed_taught_set_covers_legacy':cover,'replacement_recommended':qualifies and not technical}
  if qualifies and not technical:eligible.append(c)
 selected=min(eligible,key=lambda c:(-np.mean([final_score(w[c]) for w in worlds]),np.median([w[c]['histories']['W']['training_core_cpu'] for w in worlds]),worlds[0][c]['histories']['W']['storage']['learned_array_bytes'],c)) if eligible else None
 descriptions={c:{'W_macro_curve':np.mean([curve(w[c])[0] for w in worlds],axis=0).tolist(),
  'per_operation':{op:{'W_curve':np.mean([curve(w[c],op=op)[0] for w in worlds],axis=0).tolist(),
   'final_by_history':{h:float(np.mean([final_score(w[c],h,op) for w in worlds])) for h in ('W','N_ALL','SHUFFLE')},
   'direct_interference_by_world':[direct_interference(w[c],op) for w in worlds],
   'taught_interference_by_world':[direct_interference(w[c],op,'taught') for w in worlds]} for op in OPS},
  'W_training_cpu_by_world':[w[c]['histories']['W']['training_core_cpu'] for w in worlds],
  'resource_by_world':[w[c].get('resource',{}) for w in worlds],'whole_job_wall_by_world':[w[c]['wall_seconds'] for w in worlds],
  'storage':worlds[0][c]['histories']['W']['storage']} for c in CONFIGS}
 return {'technical':technical,'n_worlds':len(worlds),'family_size':FAMILY,'verdict':'TECHNICAL_PIPELINE_ONLY' if technical else ('RECOMMEND_NEW_ENGINEERING_BASELINE' if selected else 'NO_CONFIRMED_REPLACEMENT'),
  'selected':selected,'candidate_profiles':profiles,'contrasts':stats,'paired_world_differences':raw,'descriptive_results':descriptions,
  'evidence_scope':'Technical plumbing only, not E3' if technical else 'Retained four-operation family-held-out exact output; no mixed arithmetic or unbounded precision'}
