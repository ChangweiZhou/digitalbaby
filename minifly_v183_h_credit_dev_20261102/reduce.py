"""Frozen signed decision; per-site reconstruction is distinct from a formal seal."""
import bootstrap as B
import metrics as M
import math
np=B.np

def rebuild(s):
 y=s['y'];eos=y==256;p=s['prob_EOS'];b=-s['logp_true']/math.log(2)
 return {'targets':len(y),'bits':float(b.mean()),'payload_bits':float(b[~eos].mean()),'accuracy':float((s['pred_y']==y).mean()),'EOS_NLL':float((-s['logp_EOS'][eos]/math.log(2)).mean()),'EOS_Brier':float(((p-eos)**2).mean()),'premature_EOS_rate':float((s['pred_y'][~eos]==256).mean())}
def decide(arms,mean_forecast):
 c=arms['HPC'];normal=arms['NORMAL'];sh=arms['SHUFFLED'];held=lambda a,st,p:a[st][p]['bits'];part=lambda a,st,p:a[st]['EDIT_held']['partitions'][p]['bits'];checks={}
 checks['A2_vs_NORMAL']=held(normal,'A2','A_held')-held(c,'A2','A_held')>=.05;checks['A2_vs_SHUFFLED']=held(sh,'A2','A_held')-held(c,'A2','A_held')>=.05;checks['A2_vs_unigram']=4.403066543210908-held(c,'A2','A_held')>=.05;checks['A2_vs_own_mean']=mean_forecast-held(c,'A2','A_held')>=.05
 for arm in ('NORMAL','SHUFFLED'):
  other=arms[arm]
  for st,p in [('B3','B_held'),('EDIT2','B_held')]:checks[f'{st}_{p}_guard_{arm}']=held(c,st,p)-held(other,st,p)<=.1
  for p in ('SAME_KC','CHANGED'):checks[f'EDIT2_{p}_guard_{arm}']=part(c,'EDIT2',p)-part(other,'EDIT2',p)<=.1
  for st,panels in [('A2',('A_held','B_held')),('B3',('A_held','B_held')),('EDIT2',('EDIT_held','B_held'))]:
   for p in panels:checks[f'{st}_{p}_EOS_guard_{arm}']=c[st][p]['EOS_NLL']-other[st][p]['EOS_NLL']<=.25
 checks['pre_C_A_retention']=held(c,'B3','A_held')-held(c,'A2','A_held')<=.1;checks['post_C_still_valid_SAME_KC']=part(c,'EDIT2','SAME_KC')-part(c,'A2','SAME_KC')<=.15;checks['post_C_B_retention']=held(c,'EDIT2','B_held')-held(c,'B3','B_held')<=.1;checks['correction_feedback_specificity']=part(c,'NOEDIT2','CHANGED')-part(c,'EDIT2','CHANGED')>=.1
 relative=checks['A2_vs_NORMAL'] and checks['A2_vs_SHUFFLED'];verdict='HPC_DEV_GAIN' if all(checks.values()) else 'HPC_RELATIVE_GAIN_ONLY' if relative else 'HPC_NOT_SUPPORTED'
 noassign=held(normal,'A2','A_held')-held(sh,'A2','A_held')>=.05 and held(sh,'A2','A_held')-held(c,'A2','A_held')<.05
 return {'verdict':verdict,'checks':checks,'NO_INFORMATION_ASSIGNMENT_ADVANTAGE':noassign,'A2':{'NORMAL':held(normal,'A2','A_held'),'SHUFFLED':held(sh,'A2','A_held'),'HPC':held(c,'A2','A_held'),'HPC_own_mean':mean_forecast,'unigram':4.403066543210908},'original_A_after_C_diagnostic_only':True,'E3_or_adoption':False}
