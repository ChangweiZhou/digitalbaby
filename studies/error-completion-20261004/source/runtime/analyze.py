# SPDX-License-Identifier: GPL-3.0-or-later
"""Fixed sixteen-contrast world-unit comparison; no optional subsets or stopping."""
import math
from pathlib import Path
import numpy as np
from scipy.stats import t
from survivor_fixture import OFFICIAL_WORLDS
from validate_receipt import load
from durable import create
N=64;M=16;FAMILY_ALPHA=.05;NI_MARGIN=.05
ARMS=('R_center','ERROR')
BASE=('E3','raw_heldout','E1','new_gain','raw_old','raw_new')
CONTRASTS=tuple(a+'_'+b for a in ARMS for b in BASE)+('direct_E3','direct_heldout','NI_old','NI_new')
SCOPE='taught-dependent held-out table completion in the locked relabeled Latin-table fixture only'

def world_values(probes):
 batches={(b['branch'],b['when']):b['rows'] for b in probes}
 def acc(arm,branch,name):
  vals=[r['policies']['combined']['correct'] for r in batches[arm+'__'+branch,'final'] if r['set']==name]
  assert len(vals)=={'heldout':4,'old':12,'new':16}[name]
  return float(np.mean(vals))
 v={}
 for arm in ARMS:
  v[arm+'_E3']=acc(arm,'W','heldout')-acc(arm,'N_old_relation','heldout')
  v[arm+'_E1']=acc(arm,'W','old')-acc(arm,'N_old_relation','old')
  v[arm+'_new_gain']=acc(arm,'W','new')-acc(arm,'N_new','new')
  for name in ('heldout','old','new'):v[arm+'_raw_'+name]=acc(arm,'W',name)-.25
 v['direct_E3']=v['ERROR_E3']-v['R_center_E3']
 v['direct_heldout']=v['ERROR_raw_heldout']-v['R_center_raw_heldout']
 v['NI_old']=v['ERROR_raw_old']-v['R_center_raw_old']
 v['NI_new']=v['ERROR_raw_new']-v['R_center_raw_new']
 return v

def bounds(key):
 if '_raw_' in key:return (-.25,.75)
 if key=='direct_E3':return (-2.,2.)
 return (-1.,1.)

def intervals(rows):
 assert len(rows)==N and all(set(r)==set(CONTRASTS) for r in rows)
 out={};critical=float(t.ppf(1-FAMILY_ALPHA/(2*M),N-1))
 for key in CONTRASTS:
  x=np.asarray([r[key] for r in rows],float);lo,hi=bounds(key)
  assert np.isfinite(x).all() and np.all(x>=lo) and np.all(x<=hi)
  mean=float(x.mean());sd=float(x.std(ddof=1));width=hi-lo
  if np.all(x==x[0]):
   sd=0.;radius=width*math.sqrt(math.log(2*M/FAMILY_ALPHA)/(2*N));method='Hoeffding_zero_variance_fallback'
  else:radius=critical*sd/math.sqrt(N);method='paired_Student_t_Bonferroni_approximate'
  out[key]={'mean':mean,'sd':sd,'lower':max(lo,mean-radius),'upper':min(hi,mean+radius),'method':method,'n':N,'range_width':width}
 return out

def decide(ci):
 positive=lambda key:ci[key]['lower']>0
 arm_gates={a:{'causal_completion':positive(a+'_E3'),'above_chance_completion':positive(a+'_raw_heldout'),
   'causal_old':positive(a+'_E1'),'causal_new':positive(a+'_new_gain'),
   'old_above_chance':positive(a+'_raw_old'),'new_above_chance':positive(a+'_raw_new'),
   'old_mean_at_least_90':ci[a+'_raw_old']['mean']+.25>=.90,
   'new_mean_at_least_90':ci[a+'_raw_new']['mean']+.25>=.90} for a in ARMS}
 preserved=ci['NI_old']['lower']>-NI_MARGIN and ci['NI_new']['lower']>-NI_MARGIN
 success=all(arm_gates['ERROR'].values()) and preserved
 raw_better=positive('direct_heldout');causal_better=positive('direct_E3')
 if success:status='COMPLETION_JOINT_CRITERIA_MET'
 elif raw_better and (ci['ERROR_raw_heldout']['upper']<=0 or ci['ERROR_E3']['upper']<=0):status='REDUCTION_IN_HARM_WITHOUT_COMPLETION'
 elif any(ci[k]['upper']<=0 for k in ('ERROR_E3','ERROR_raw_heldout','ERROR_E1','ERROR_new_gain','ERROR_raw_old','ERROR_raw_new')) or ci['NI_old']['upper']<=-NI_MARGIN or ci['NI_new']['upper']<=-NI_MARGIN:status='JOINT_REPAIR_EXCLUDED_BY_REGISTERED_BOUND'
 else:status='INCONCLUSIVE_JOINT_REPAIR_NOT_ESTABLISHED'
 return {'status':status,'joint_success':success,'arm_gates':arm_gates,'old_new_noninferiority':preserved,
  'direct_raw_superiority':raw_better,'direct_causal_superiority':causal_better,
  'error_heldout_below_chance_established':ci['ERROR_raw_heldout']['upper']<0,
  'error_causal_harm_established':ci['ERROR_E3']['upper']<0,
  'terminal_for_this_question':True,'sample_extension_authorized':False,'later_experiment_authorized':False}

def analyze(receipts,expected_source,dest):
 paths={int(p.name):p for p in Path(receipts).iterdir() if p.is_dir() and p.name.isdigit()}
 assert set(paths)==set(OFFICIAL_WORLDS),'all64 exact worlds required; no replacement/subset inference'
 validated=[load(paths[w],w,expected_source,expected_kind='science') for w in OFFICIAL_WORLDS]
 rows=[world_values(v['probes']) for v in validated];ci=intervals(rows)
 result={'schema':'ERROR-COMPLETION-ANALYSIS-v1','worlds':list(OFFICIAL_WORLDS),'family_m':M,'alpha':FAMILY_ALPHA,
  'input_manifest':{str(w):v['manifest_sha256'] for w,v in zip(OFFICIAL_WORLDS,validated)},
  'world_values':rows,'intervals':ci,'decision':decide(ci),'scope':SCOPE}
 create(dest,result);return result
