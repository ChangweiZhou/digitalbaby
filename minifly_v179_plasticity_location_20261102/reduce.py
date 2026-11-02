"""Saved-only paired causal-package reduction; no model imports or retraining."""
import util as U
import sys,json,time
sys.path.insert(0,str(U.ROOT/'minifly_v178_recurrent_failure_localization_dev_20261102/local/compat'))
import numpy as np

def main():
 result=U.load('RESULTS.json');assert result['status']=='EXECUTION_COMPLETE';rows=result['checkpoints'];q={(r['world'],r['checkpoint']):r for r in rows};parent={(r['world'],r['checkpoint']):r for r in U.load('PARENT_RESULTS.json')['checkpoints']};checks={};raw={};normal_matches=[];out={};pairs={};tau=1e-4
 checks['source_lock']=all(U.sha(U.ROOT/p)==h for p,h in U.load('SOURCE_LOCK.json')['files'].items());checks['all_checkpoints']=len(rows)==30
 for r in rows:
  assert U.sha(U.H/r['checkpoint_file'])==r['checkpoint_sha256']
  if r['checkpoint']=='A_START' or r['checkpoint'].startswith('NORMAL_'):
   ck='AFTER_A' if r['checkpoint']=='A_START' else r['checkpoint'][7:];ck='KEEP_'+ck if ck.startswith('B') else ck;p=parent[r['world'],ck];assert p['state_hash']==r['state_hash'];assert all(abs(p['scores'][k]['bits']-r['scores'][k]['bits'])<2e-6 for k in ('A_held','B_held','EDITED_held'));normal_matches.append([r['world'],r['checkpoint']])
  for name,s in r['scores'].items():
   path=U.H/s['panel'];assert U.sha(path)==s['sha256'];z=np.load(path);assert z['y'].min()>=0 and z['y'].max()<=256 and z['n'].sum()==len(z['y']);assert np.all(z['y'][np.cumsum(z['n'])-1]==256);assert abs(float(-z['target_log2p'].mean())-s['bits'])<2e-5;assert s['max_probability_normalization_error']<1e-9;assert np.all((z['EOS_mass']>=0)&(z['EOS_mass']<=1)) and np.all(z['entropy']>=0);raw[s['panel']]=s['sha256']
  for g in r['generation'].values():assert U.sha(U.H/g['raw'])==g['sha256'];raw[g['raw']]=g['sha256']
 for w in ('W1','W2'):
  out[w]=dict(B={},C={});pairs[w]={}
  for ep in (1,2,3):
   aa,bb=[q[w,arm+'_B'+str(ep)] for arm in ('NORMAL','CLAMP')];out[w]['B'][str(ep)]=dict(normal_train=aa['scores']['B_train']['bits'],clamp_train=bb['scores']['B_train']['bits'],normal_held=aa['scores']['B_held']['bits'],clamp_held=bb['scores']['B_held']['bits'],clamp_minus_normal_held=bb['scores']['B_held']['bits']-aa['scores']['B_held']['bits'],normal_stable_A_change=aa['subsets']['A_stable_bits']-q[w,'A_START']['subsets']['A_stable_bits'],clamp_stable_A_change=bb['subsets']['A_stable_bits']-q[w,'A_START']['subsets']['A_stable_bits'])
   n=np.array(aa['scores']['B_held']['record_targets']);delta=(np.array(bb['scores']['B_held']['record_bit_sums'])-np.array(aa['scores']['B_held']['record_bit_sums']))/n;pairs[w]['B'+str(ep)]=dict(record_targets=n.tolist(),clamp_minus_normal_bits=delta.tolist(),target_weighted=float((delta*n).sum()/n.sum()))
  for arm in ('NORMAL','CLAMP'):
   t1,t3=[q[w,arm+'_B'+str(e)] for e in (1,3)];dt=t3['scores']['B_train']['bits']-t1['scores']['B_train']['bits'];dh=t3['scores']['B_held']['bits']-t1['scores']['B_held']['bits'];out[w]['B'][arm+'_trajectory']=dict(delta_train_B3_minus_B1=dt,delta_held_B3_minus_B1=dh,category='OVERFITTING_LIKE' if dt < -tau and dh>tau else ('BOTH_WORSEN' if dt>tau and dh>tau else 'MIXED_OR_FLAT'))
   e,no=[q[w,arm+'_'+f+'_C2'] for f in ('EDITED','NOEDIT')];baseline=q[w,arm+'_B3'];margin=e['subsets']['changed_margin']-no['subsets']['changed_margin'];benefit=no['subsets']['changed_new_bits']-e['subsets']['changed_new_bits'];retB=e['scores']['B_held']['bits']-baseline['scores']['B_held']['bits'];retA=e['subsets']['A_stable_bits']-q[w,'A_START']['subsets']['A_stable_bits'];afterB=baseline['subsets']['A_stable_bits']-q[w,'A_START']['subsets']['A_stable_bits']
   out[w]['C'][arm]=dict(edited_changed_loss=e['subsets']['changed_new_bits'],noedit_changed_loss=no['subsets']['changed_new_bits'],edited_minus_noedit_margin_advantage=margin,noedit_minus_edited_loss_advantage=benefit,revision_screen=margin>=.2 and benefit>=.1,stable_A_after_B_minus_A=afterB,stable_A_after_C_minus_A=retA,B_after_C_minus_B=retB,retention_screen=afterB<=.1 and retA<=.15 and retB<=.1,final_A_loss=e['scores']['A_held']['bits'],final_B_loss=e['scores']['B_held']['bits'])
   sub=np.load(U.H/f'local/{w}_subsets.npz');ch=sub['changed'];lp=[]
   for rr in (e,no):
    z=np.load(U.H/rr['scores']['EDITED_held']['panel']);lp.append((z['target_log2p'],z['old_log2p'],z['n']))
   ends=np.cumsum(np.r_[0,lp[0][2]]);record=[]
   for i in range(len(lp[0][2])):
    sl=slice(ends[i],ends[i+1]);mask=ch[sl];loss=(lp[0][0]-lp[1][0])[sl];mar=((lp[0][0]-lp[0][1])-(lp[1][0]-lp[1][1]))[sl];record.append(dict(record=i,changed_targets=int(mask.sum()),loss_advantage_sum=float(loss[mask].sum()),margin_advantage_sum=float(mar[mask].sum())))
   pairs[w][arm+'_revision']=record
 freeze=U.load('FREEZE_AUDIT.json');checks['all_six_frozen_B_boundaries']=len(freeze)==6 and all(x['identical_to_A'] for x in freeze)
 forks={}
 for f in U.load('FORK_AUDIT.json'):forks[f['world'],f['branch'],f['phase']]=f
 checks['B_forks_identical']=all(forks[w,'NORMAL','B']['state_hash']==forks[w,'CLAMP','B']['state_hash'] for w in ('W1','W2'));checks['C_feedback_forks_identical']=all(forks[w,a,'EDITED']['state_hash']==forks[w,a,'NOEDIT']['state_hash'] for w in ('W1','W2') for a in ('NORMAL','CLAMP'))
 trace=[json.loads(x) for x in (U.H/'UPDATE_TRACE.jsonl').read_text().splitlines()];counts=U.load('COUNTS.json');led=U.load('TRAINING_LEDGER.json');checks['full_record_dose']=sum(x['record_presentations'] for x in led)==2980;checks['full_target_dose']=sum(x['targets'] for x in trace)==sum(x['target_presentations'] for x in led)==187606;checks['all_updates_logged']=len(trace)==counts['optimizer_steps'];checks['total_count']=counts['records']==2980+U.load('TECHNICAL_COUNTS.json')['records'] and counts['targets']==187606+U.load('TECHNICAL_COUNTS.json')['targets'];checks['within_caps']=counts['records']<=3100 and counts['targets']<=200000
 frozen=[x for x in trace if x['freeze_hidden']];activeC=[x for x in trace if x['context']['phase']=='C' and x['context']['branch'].startswith('CLAMP')];checks['zero_actual_hidden_writing_B']=all(x['layers'][k]['update_sq']==0 for x in frozen for k in ('H','T'));checks['nonzero_actual_hidden_writing_C']=len(activeC)>0 and all(x['layers'][k]['update_sq']>0 for x in activeC for k in ('H','T'));checks['normal_matches_parent']=len(normal_matches)==16;checks['no_failure']=not (U.H/'FAILURE.json').exists();assert all(checks.values()),checks
 energy={}
 for w in ('W1','W2'):
  energy[w]={}
  for a in ('NORMAL','CLAMP'):
   rr=[x for x in trace if x['context']['world']==w and x['context']['branch']==a and x['context']['phase']=='B'];energy[w][a]={layer:dict(sum_update_squared=sum(x['layers'][layer]['update_sq'] for x in rr),sum_update_norm_path_length=sum(x['layers'][layer]['update_norm'] for x in rr)) for layer in ('H','T','O')}
 signs=[out[w]['B']['3']['clamp_minus_normal_held'] for w in ('W1','W2')];interpretation='CLAMP_HELD_BETTER_BOTH_OPENED_WORLDS' if all(v < -tau for v in signs) else ('NORMAL_HELD_BETTER_BOTH_OPENED_WORLDS' if all(v>tau for v in signs) else 'MIXED_OR_NUMERICAL_TIE')
 U.save('SUMMARY.json',dict(status='COMPLETE',identity='V179-PLASTICITY-1',interpretation=interpretation,worlds=out,actual_B_update_energy=energy,normal_reproduced_checkpoints=len(normal_matches),counts=counts,inherited_A_records=4086,inherited_A_targets=246108,new_main_records=2980,new_main_targets=187606,not_independent_confirmation=True,package_level_causality_only=True,no_adoption=True));U.save('PAIRED_RECORD_EFFECTS.json',pairs);U.save('RAW_MANIFEST.json',dict(files=raw,raw_local_only=True,no_external_reconstruction=True));U.save('FINAL_AUDIT.json',dict(status='ACCEPTED',checks=checks,separate_saved_only_process=True,no_external_reconstruction=True,no_new_training_for_reporting=True));U.finish_process('saved_only_paired_reduction');U.save('STATUS.json',dict(status='COMPLETED',verdict=interpretation,official_run=False,core_adopted=False));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
