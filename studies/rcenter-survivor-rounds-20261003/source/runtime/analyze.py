# SPDX-License-Identifier: GPL-3.0-or-later
"""Frozen eleven-contrast world-unit analysis, no optional stopping/subsets."""
import json,math
from pathlib import Path
import numpy as np
from scipy.stats import t
from survivor_fixture import OFFICIAL_WORLDS
from validate_receipt import load
from durable import create
N=64;M=11;FAMILY_ALPHA=.05
CONTRASTS=('E3_1','E3_0','E3_half','E1_final_1','acquisition_final_1','W_heldout_1_minus_chance','W_heldout_0_minus_chance','W_old_1_minus_chance','W_new_1_minus_chance','old_half_minus_1','new_half_minus_1')
def world_values(probes):
 batches={(b['branch'],b['when']):b['rows'] for b in probes}
 def acc(branch,name,policy):
  vals=[r['policies'][policy]['correct'] for r in batches[branch,'final'] if r['set']==name];assert len(vals)=={'heldout':4,'old':12,'new':16}[name];return float(np.mean(vals))
 v={}
 for tag,policy in [('1','alpha1'),('0','alpha0'),('half','alpha_half')]:v['E3_'+tag]=acc('W','heldout',policy)-acc('N_old_relation','heldout',policy)
 v['E1_final_1']=acc('W','old','alpha1')-acc('N_old_relation','old','alpha1')
 v['acquisition_final_1']=acc('W','new','alpha1')-acc('N_new','new','alpha1')
 for name,tag,policy in [('heldout','1','alpha1'),('heldout','0','alpha0'),('old','1','alpha1'),('new','1','alpha1')]:v[f'W_{name}_{tag}_minus_chance']=acc('W',name,policy)-.25
 v['old_half_minus_1']=acc('W','old','alpha_half')-acc('W','old','alpha1')
 v['new_half_minus_1']=acc('W','new','alpha_half')-acc('W','new','alpha1')
 return v

def intervals(rows):
 assert len(rows)==N and all(set(r)==set(CONTRASTS) for r in rows)
 out={};critical=float(t.ppf(1-FAMILY_ALPHA/(2*M),N-1))
 for k in CONTRASTS:
  x=np.array([r[k] for r in rows]);mean=float(x.mean());sd=float(x.std(ddof=1));raw=k.startswith('W_');width=1. if raw else 2.
  if np.all(x==x[0]):
   sd=0.
   radius=width*math.sqrt(math.log(2*M/FAMILY_ALPHA)/(2*N));method='Hoeffding_zero_variance_fallback'
  else:radius=critical*sd/math.sqrt(N);method='paired_Student_t_Bonferroni_approximate'
  lower_bound,upper_bound=(-.25,.75) if raw else (-1.,1.)
  out[k]={'mean':mean,'sd':sd,'lower':max(lower_bound,mean-radius),'upper':min(upper_bound,mean+radius),'method':method,'n':N,'range_width':width}
 return out

def decide(ci):
 positive=lambda k:ci[k]['lower']>0
 joint=all(positive(k) for k in ('E1_final_1','acquisition_final_1','W_old_1_minus_chance','W_new_1_minus_chance'))
 combined=positive('E3_1') and positive('W_heldout_1_minus_chance')
 shared=positive('E3_0') and positive('W_heldout_0_minus_chance')
 half=positive('E3_half') and ci['old_half_minus_1']['lower']>-.05 and ci['new_half_minus_1']['lower']>-.05
 if not joint:route='STOP_JOINT_RETENTION_ACQUISITION_NOT_ESTABLISHED'
 elif combined:route='A_FRESH_CONFIRMATION_PLAN_ONLY_NO_E_OR_I'
 elif shared:route='B_READ_AUTHORITY_ONLY_PLAN_NOT_PROOF_OF_ROUTE_DIFFERENCE'
 else:route='C_SHARED_COMPLETION_NOT_ESTABLISHED_SOURCE_REVIEW_BEFORE_POSSIBLE_FORMATION_PLAN'
 combined_status='positive' if combined else ('nonpositive_interval' if ci['E3_1']['upper']<=0 else 'inconclusive_or_failed_above_chance_gate')
 return {'combined_status':combined_status,'route':route,'joint':joint,'combined':combined,'shared':shared,'half_secondary_candidate_only':half,'later_round_launch_authorized':False}

def analyze(receipts,expected_source,dest):
 paths={int(p.name):p for p in Path(receipts).iterdir() if p.is_dir() and p.name.isdigit()}
 assert set(paths)==set(OFFICIAL_WORLDS),'all64 exact registered worlds required; no replacement/subset inference'
 validated=[load(paths[w],w,expected_source,expected_kind='science') for w in OFFICIAL_WORLDS]
 rows=[world_values(v['probes']) for v in validated];ci=intervals(rows)
 result={'schema':'RC-SURVIVOR-R1-ANALYSIS-v1','worlds':list(OFFICIAL_WORLDS),'family_m':M,'alpha':FAMILY_ALPHA,
         'input_manifest':{str(w):v['manifest_sha256'] for w,v in zip(OFFICIAL_WORLDS,validated)},'world_values':rows,'intervals':ci,'decision':decide(ci),
         'scope':'taught-dependent held-out table completion in the locked relabeled Latin-table fixture only'}
 create(dest,result);return result
