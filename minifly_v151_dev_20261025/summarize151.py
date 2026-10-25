"""Reporting statistics from audited saved responses; no native calls or additional fitting."""
import json,time
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent
load=lambda p:json.loads(p.read_text())
begin=time.process_time();c=load(H/'COMPACT.json');rows=c['per_site_conditions'];means=c['anchor_means'];code=c['C_factorial']
trained=[r for r in rows if r['anchor'].endswith('FINAL')]
paired=[]
for anchor in sorted({r['anchor'] for r in rows}):
 for site in sorted({r['site'] for r in rows if r['anchor']==anchor}):
  rr={r['arm']:r for r in rows if r['anchor']==anchor and r['site']==site};a=rr['A_COMMON'];b=rr['B_MATCHED']
  paired.append({'anchor':anchor,'site':site,'matched_minus_A_BPS':b['delta_BPS'][0]-a['delta_BPS'][0],
                 'matched_over_A_displacement':b['displacement_norms'][0]/a['displacement_norms'][0]})
ops=[load(p) for p in H.glob('OP_*.json')];primary=[]
for o in ops:
 if o['phase']!='SCORED':continue
 G=-np.array(o['K_half'])[0];diag=np.diag(G)
 primary.append({'anchor':o['anchor'],'site':o['site'],'G_self_diagonal_min':float(diag.min()),'G_self_diagonal_max':float(diag.max()),
                 'G_self_diagonal_spread':float(np.ptp(diag)),'active_KCs':o['active_KCs'],'KC_Jaccard':o['KC_Jaccard_with_immediate_source']})
aggregate={}
for arm in ['A_COMMON','B_COMMON','B_MATCHED']:
 r=[x for x in trained if x['arm']==arm]
 aggregate[arm]={'sites':len(r),'mean_delta_BPS':np.mean([x['delta_BPS'] for x in r],axis=0).tolist(),
                 'self_improved':sum(x['delta_BPS'][0]<0 for x in r),
                 'different_target_harmed':sum(x['delta_BPS'][3]>0 for x in r),
                 'same_target_improved':sum(x['delta_BPS'][2]<0 for x in r),
                 'max_linear_loss_residual':max(abs(v) for x in r for v in x['loss_prediction_residual'])}
factorial=[]
for anchor in sorted({x['anchor'] for x in code}):
 r=[x for x in code if x['anchor']==anchor]
 factorial.append({'anchor':anchor,'sites':len(r),'mean_code_step_norm':float(np.mean([x['code_step_norm'] for x in r])),
                    'max_projection_difference_norm':max(x['projection_difference_norm'] for x in r),
                    'mean_self_delta_BPS':{arm:float(np.mean([x['effects'][arm][0] for x in r])) for arm in ['V_ONLY','C_ONLY','JOINT']},
                    'mean_self_interaction_BPS':float(np.mean([x['interaction_delta_BPS'][0] for x in r]))})
pt=[x for x in paired if x['anchor'].endswith('FINAL')]
summary={'version':'V151','scope':'Descriptive paired statistics, no independent-seed inference',
         'trained_anchors_equal_source_mean':aggregate,'paired_matched_vs_A':paired,
         'trained_matched_better_than_A_sites':sum(x['matched_minus_A_BPS']<0 for x in pt),
         'trained_sites':len(pt),'trained_displacement_ratio_range':[min(x['matched_over_A_displacement'] for x in pt),max(x['matched_over_A_displacement'] for x in pt)],
         'scored_operator_details':primary,'C_factorial_aggregate':factorial,
         'max_value_linear_residual':max(x['prediction_residual_max'] for x in rows),'CPU':time.process_time()-begin}
(H/'ANALYSIS.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ['paired_matched_vs_A','scored_operator_details']},indent=2))
