"""Saved-output-only reduction: no learner imports, training or fresh model scoring."""
import util as U
import sys
sys.path.insert(0,str(U.ROOT/'minifly_v178_recurrent_failure_localization_dev_20261102/local/compat'))
import numpy as np
import json,time,zipfile

def main():
 result=U.load('RESULTS.json');assert result['status']=='EXECUTION_COMPLETE';rows=result['checkpoints'];checks={};out={};panels={};d=U.load('DATA_AUDIT.json');count=U.load('COUNT_CONTROLS.json');ledger=U.load('TRAINING_LEDGER.json');forks=U.load('FORK_AUDIT.json')
 checks['source_lock']=all(U.sha(U.ROOT/p)==h for p,h in U.load('SOURCE_LOCK.json')['files'].items());checks['all_checkpoint_hashes']=all(not r['checkpoint_file'] or U.sha(U.H/r['checkpoint_file'])==r['checkpoint_sha256'] for r in rows)
 rawmanifest={};panels_ok=True
 for r in rows:
  for name,s in r['scores'].items():
   f=U.H/s['panel'];panels_ok &= U.sha(f)==s['sha256'];z=np.load(f);y=z['y'];lp=z['log2p'];bits=-lp[np.arange(len(y)),y];panels_ok &= abs(float(bits.mean())-s['bits'])<2e-5 and lp.shape==(len(y),257) and y.min()>=0 and y.max()<=256 and np.allclose(np.exp2(lp).sum(1),1,atol=1e-5);ends=np.cumsum(z['n']);panels_ok &= bool(np.all(y[ends-1]==256)) and int(z['n'].sum())==len(y);rawmanifest[s['panel']]=s['sha256']
  for s in r['generation'].values():panels_ok &= U.sha(U.H/s['raw'])==s['sha256'];rawmanifest[s['raw']]=s['sha256']
 checks['raw_panels_valid']=bool(panels_ok)
 for w in ('W1','W2'):
  q={r['checkpoint']:r for r in rows if r['world']==w};required=['BIRTH','AFTER_A']+[a+'_B'+str(e) for a in ('KEEP','RESET_OPT','B_COLD') for e in (1,2,3)]+[a+'_C'+str(e) for a in ('EDITED','NOEDIT') for e in (1,2)];checks[w+'_complete_roster']=set(q)==set(required)
  g=lambda ck,sp:q[ck]['scores'][sp]['bits'];s=lambda ck,n:q[ck]['subsets'][n]
  acq=dict(birth_A=g('BIRTH','A_held'),after_A=g('AFTER_A','A_held'),unigram=count['worlds'][w]['A_unigram']['bits']);acq['gain']=acq['birth_A']-acq['after_A'];acq['beyond_unigram']=acq['unigram']-acq['after_A'];acq['pass']=acq['gain']>=.2 and acq['beyond_unigram']>=.05
  reuse={};pairs={}
  for ep in (1,2,3):
   gap=g('B_COLD_B'+str(ep),'B_held')-g('RESET_OPT_B'+str(ep),'B_held');reuse[str(ep)]=dict(cold=g('B_COLD_B'+str(ep),'B_held'),reset_opt=g('RESET_OPT_B'+str(ep),'B_held'),keep=g('KEEP_B'+str(ep),'B_held'),cold_minus_reset=gap,cold_minus_keep=g('B_COLD_B'+str(ep),'B_held')-g('KEEP_B'+str(ep),'B_held'))
   cold=q['B_COLD_B'+str(ep)]['scores']['B_held'];warm=q['RESET_OPT_B'+str(ep)]['scores']['B_held'];n=np.array(cold['record_targets']);delta=(np.array(cold['record_bit_sums'])-np.array(warm['record_bit_sums']))/n;pairs['reuse_B'+str(ep)]=dict(record_differences=delta.tolist(),records=len(n),target_weighted=float((delta*n).sum()/n.sum()),unit='bits per record target; no token-level significance test')
  reuse['pass']=reuse['1']['cold_minus_reset']>=.05 and reuse['3']['cold_minus_reset']>=.05
  ret=dict(A_stable_after_B_minus_after_A=s('KEEP_B3','A_stable_bits')-s('AFTER_A','A_stable_bits'),A_stable_after_C_minus_after_A=s('EDITED_C2','A_stable_bits')-s('AFTER_A','A_stable_bits'),B_after_C_minus_after_B=g('EDITED_C2','B_held')-g('KEEP_B3','B_held'))
  ret['pass']=ret['A_stable_after_B_minus_after_A']<=.1 and ret['A_stable_after_C_minus_after_A']<=.15 and ret['B_after_C_minus_after_B']<=.1
  revision=dict(margin_edited=s('EDITED_C2','changed_margin'),margin_noedit=s('NOEDIT_C2','changed_margin'),margin_before=s('KEEP_B3','changed_margin'),loss_edited=s('EDITED_C2','changed_new_bits'),loss_noedit=s('NOEDIT_C2','changed_new_bits'),changed_positions=d['edits'][w]['changed_held'])
  revision['extra_margin_gain']=revision['margin_edited']-revision['margin_noedit'];revision['loss_advantage']=revision['loss_noedit']-revision['loss_edited'];revision['pass']=revision['extra_margin_gain']>=.2 and revision['loss_advantage']>=.1
  subset=np.load(U.H/f'local/{w}_subsets.npz');stable=subset['stable'];old=subset['old'];ss={}
  for ck in ['AFTER_A','KEEP_B3','EDITED_C2']:
   z=np.load(U.H/q[ck]['scores']['A_held']['panel']);ss[ck]=-z['log2p'][np.arange(len(old)),old]
  ret['stable_positions_degraded_after_B']=int(((ss['KEEP_B3']>ss['AFTER_A'])&stable).sum());ret['stable_positions_degraded_after_C']=int(((ss['EDITED_C2']>ss['AFTER_A'])&stable).sum());ret['stable_positions']=int(stable.sum())
  for later in ['KEEP_B3','EDITED_C2']:
   n=q[later]['scores']['A_held']['record_targets'];e=np.cumsum(np.r_[0,n]);diff=ss[later]-ss['AFTER_A'];tot=[];den=[]
   for i in range(len(n)):
    v=stable[e[i]:e[i+1]];den.append(int(v.sum()));tot.append(float(diff[e[i]:e[i+1]][v].sum()))
   pairs['retention_'+later]=dict(record_bit_differences=tot,stable_targets=den,target_weighted=sum(tot)/sum(den))
  fk={f['arm']:f for f in forks if f['world']==w};checks[w+'_forks']=fk['KEEP']['weights_hash']==fk['RESET_OPT']['weights_hash'] and fk['RESET_OPT']['optimizer_entries']==0 and fk['KEEP']['state_hash']!=fk['RESET_OPT']['state_hash'] and fk['EDITED']['state_hash']==fk['NOEDIT']['state_hash'] and fk['EDITED']['dropout_hash']==fk['NOEDIT']['dropout_hash']
  out[w]=dict(acquisition=acq,reuse=reuse,retention=ret,revision=revision,all_four_pass=all(x['pass'] for x in (acq,reuse,ret,revision)),count_gap_after_B=g('KEEP_B3','B_held')-min(count['worlds'][w]['B_PPM']['bits'],count['worlds'][w]['B_IKN']['bits']))
  panels[w]=pairs
 counts=U.load('COUNTS.json');checks['main_record_count']=sum(l['record_presentations'] for l in ledger)==7532;checks['total_count']=counts['records']==7532+U.load('TECHNICAL_COUNTS.json')['records'] and counts['targets']==sum(l['target_presentations'] for l in ledger)+U.load('TECHNICAL_COUNTS.json')['targets'];checks['within_dose']=counts['records']<=7600 and counts['targets']<=510000;checks['no_failure']=not (U.H/'FAILURE.json').exists()
 assert all(checks.values()),checks
 verdict='PROMISING_LIFETIME_DEV' if all(v['all_four_pass'] for v in out.values()) else 'LIFETIME_DEV_GAPS';U.save('PAIRED_RECORD_EFFECTS.json',panels);U.save('RAW_PANEL_MANIFEST.json',dict(files=rawmanifest,raw_local_only=True,external_reconstruction_performed=False));U.save('SUMMARY.json',dict(verdict=verdict,worlds=out,counts=counts,main_records=7532,main_targets=sum(l['target_presentations'] for l in ledger),no_core_adoption=True,evidence='bounded continual predictive-learning development proxy; no E2/E3 or arithmetic claim',CPU_processes=U.load('PROCESS_LEDGER.json')))
 U.save('FINAL_AUDIT.json',dict(status='ACCEPTED',checks=checks,independence='separate saved-output-only reducer process; not an external independent reconstruction',raw_panel_publication=False,technical_tests=U.load('QUALIFICATION.json')['checks'],no_scientific_rerun=True))
 U.finish_process('saved_only_reduction');U.save('STATUS.json',dict(status='COMPLETED',verdict=verdict,all_registered_lives_complete=True,core_adopted=False));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
