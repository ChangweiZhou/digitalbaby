"""Prospective saved-only diagnostic reduction; never imports a learner."""
import util as U
import sys,json,time,math
sys.path.insert(0,str(U.ROOT/'minifly_v178_recurrent_failure_localization_dev_20261102/local/compat'))
import numpy as np

def main():
 r=U.load('RESULTS.json');assert r['status']=='EXECUTION_COMPLETE';rows=r['checkpoints'];parent={(a['world'],a['checkpoint']):a for a in U.load('PARENT_RESULTS.json')['checkpoints']};q={(a['world'],a['checkpoint']):a for a in rows};checks={};checks['sources_unchanged']=all(U.sha(U.ROOT/p)==h for p,h in U.load('SOURCE_LOCK.json')['files'].items());raw={};same=[]
 for a in rows:
  assert U.sha(U.H/a['checkpoint_file'])==a['checkpoint_sha256']
  pa=parent.get((a['world'],a['checkpoint']))
  if pa:
   assert pa['state_hash']==a['state_hash'];assert all(abs(pa['scores'][k]['bits']-a['scores'][k]['bits'])<2e-6 for k in ('A_held','B_held','EDITED_held'));same.append((a['world'],a['checkpoint']))
  for k,s in a['scores'].items():
   path=U.H/s['panel'];assert U.sha(path)==s['sha256'];z=np.load(path);assert len(z['y'])==s['targets'] and int(z['n'].sum())==s['targets'];assert abs(float(-z['target_log2p'].mean())-s['bits'])<2e-5;assert z['y'].min()>=0 and z['y'].max()<=256;assert np.all(z['y'][np.cumsum(z['n'])-1]==256);assert s['max_probability_normalization_error']<1e-9;raw[s['panel']]=s['sha256']
  for g in a['generation'].values():assert U.sha(U.H/g['raw'])==g['sha256'];raw[g['raw']]=g['sha256']
 checks['parent_common_states_identical']=len(same)==len(parent);checks['all_passes_saved']=len(rows)==34;out={};contrasts={};tau=1e-4
 for w in ('W1','W2'):
  out[w]={}
  for arm in ('KEEP','RESET_OPT','B_COLD'):
   a,b=[q[w,arm+'_B'+str(e)] for e in (1,3)];dt=b['scores']['B_train']['bits']-a['scores']['B_train']['bits'];dh=b['scores']['B_held']['bits']-a['scores']['B_held']['bits'];category='OVERFITTING_LIKE_DIVERGENCE' if dt < -tau and dh>tau else ('BOTH_DETERIORATE' if dt>tau and dh>tau else 'MIXED_OR_FLAT')
   out[w][arm]=dict(train_B1=a['scores']['B_train']['bits'],train_B3=b['scores']['B_train']['bits'],held_B1=a['scores']['B_held']['bits'],held_B3=b['scores']['B_held']['bits'],delta_train=dt,delta_held=dh,category=category,numerical_tolerance=tau,train_quartiles_B1=a['scores']['B_train']['chronological_target_quartile_bits'],train_quartiles_B3=b['scores']['B_train']['chronological_target_quartile_bits'],entropy_B1=a['scores']['B_held']['entropy_bits'],entropy_B3=b['scores']['B_held']['entropy_bits'],EOS_mass_B1=a['scores']['B_held']['EOS_mass_mean'],EOS_mass_B3=b['scores']['B_held']['EOS_mass_mean'],unused_payload_mass_B1=a['scores']['B_held']['all_absent_mass'],unused_payload_mass_B3=b['scores']['B_held']['all_absent_mass'])
   for split in ('B_train','B_held'):
    aa,bb=[a['scores'][split],b['scores'][split]];nn=np.array(aa['record_targets']);delta=(np.array(bb['record_bit_sums'])-np.array(aa['record_bit_sums']))/nn;contrasts[w+'_'+arm+'_'+split]=dict(delta_bits_per_record_target=delta.tolist(),record_targets=nn.tolist(),weighted_delta=float(np.sum(delta*nn)/nn.sum()))
 tr=[json.loads(l) for l in (U.H/'UPDATE_TRACE.jsonl').read_text().splitlines()];groups={}
 for x in tr:groups.setdefault((x['context']['world'],x['context']['branch'],x['context']['phase'],x['context']['pass_index']),[]).append(x)
 stats=[]
 for key,v in groups.items():
  n=sum(x['targets'] for x in v);z=dict(world=key[0],branch=key[1],phase=key[2],pass_index=key[3],updates=len(v),targets=n,activations={k:sum(x['targets']*x['activations'][k] for x in v)/n for k in v[0]['activations']},layers={})
  for layer in ('H','T','O'):
   z['layers'][layer]={k:sum(x['targets']*x['layers'][layer][k] for x in v)/n for k in ['weight_norm','gradient_norm','update_norm','update_weight_ratio','moment_m_RMS','moment_v_RMS']};z['layers'][layer]['cumulative_update_norm_path_length']=sum(x['layers'][layer]['update_norm'] for x in v)
  stats.append(z)
 counts=U.load('COUNTS.json');led=U.load('TRAINING_LEDGER.json');checks['every_update_logged']=len(tr)==counts['optimizer_steps']==1832;checks['logged_learning_targets']=sum(x['targets'] for x in tr)==463277;checks['full_main_dose']=sum(x['record_presentations'] for x in led)==7532 and sum(x['target_presentations'] for x in led)==463277;checks['within_total_caps']=counts['records']<=7600 and counts['targets']<=510000
 fork={}
 for f in U.load('FORK_AUDIT.json'):fork.setdefault(f['world'],{})[f['arm']]=f
 checks['forks_preserved']=all(f['KEEP']['weights_hash']==f['RESET_OPT']['weights_hash'] and f['RESET_OPT']['optimizer_entries']==0 and f['EDITED']['state_hash']==f['NOEDIT']['state_hash'] for f in fork.values());checks['no_failure']=not (U.H/'FAILURE.json').exists();assert all(checks.values()),checks
 life={};cnt=U.load('COUNT_CONTROLS.json')['worlds']
 for w in ('W1','W2'):
  g=lambda ck,sp:q[w,ck]['scores'][sp]['bits'];s=lambda ck,k:q[w,ck]['subsets'][k]
  acq=g('BIRTH','A_held')-g('AFTER_A','A_held')>=.2 and cnt[w]['A_unigram']['bits']-g('AFTER_A','A_held')>=.05
  reuse=all(g('B_COLD_B'+str(e),'B_held')-g('RESET_OPT_B'+str(e),'B_held')>=.05 for e in (1,3));ret=s('KEEP_B3','A_stable_bits')-s('AFTER_A','A_stable_bits')<=.1 and s('EDITED_C2','A_stable_bits')-s('AFTER_A','A_stable_bits')<=.15 and g('EDITED_C2','B_held')-g('KEEP_B3','B_held')<=.1;rev=s('EDITED_C2','changed_margin')-s('NOEDIT_C2','changed_margin')>=.2 and s('NOEDIT_C2','changed_new_bits')-s('EDITED_C2','changed_new_bits')>=.1
  life[w]=dict(acquisition=acq,reuse=reuse,retention=ret,revision=rev)
 U.save('DIAGNOSTIC_SUMMARY.json',dict(status='COMPLETE',contrasts=out,continuous_lifetime_screens=life,parent_common_checkpoint_count=len(same),all_common_trajectories_bit_identical=True,counts=counts,classification_is_descriptive_not_statistical=True,no_new_mechanism_or_promotion=True))
 U.save('PAIRED_RECORD_EFFECTS.json',contrasts);U.save('UPDATE_SUMMARIES.json',stats);U.save('RAW_MANIFEST.json',dict(files=raw,raw_local_only=True,full257_distributions_not_duplicated=True,external_reconstruction=False));U.save('FINAL_AUDIT.json',dict(status='ACCEPTED',checks=checks,separate_saved_only_process=True,no_training_rerun_for_reporting=True,no_external_reconstruction=True));U.finish_process('saved_only_reduction');U.save('STATUS.json',dict(status='COMPLETED',verdict='DIAGNOSTIC_COMPLETE_NO_PROMOTION',adopted_core_unchanged=True));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
