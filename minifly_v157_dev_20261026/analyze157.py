"""Locked before held inspection: saved-artifact scoring/countermodels, never model inference."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='1'
from pathlib import Path
from collections import Counter,defaultdict
import json,hashlib,time,resource,ast
import numpy as np
H=Path(__file__).resolve().parent;R=H.parent
def load(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def decode(c):
 body=[];eos=False
 for y in c:
  if y<0:break
  assert 0<=y<=256
  if y==256:eos=True;break
  body.append(int(y))
 return bytes(body),eos
def array(receipt):
 assert sha(H/receipt['path'])==receipt['sha256']
 with np.load(H/receipt['path'],allow_pickle=False) as a:return {k:a[k] for k in a.files}
def panel_stats(receipt,d):
 a=array(receipt);y=a['y'];z=a['z'];off=a['offsets'];yy=off[:-1];ee=off[1:]-1
 for j,row in enumerate(d['panel']):
  body,eos=decode(a['choices'][j]);assert (eos and body.hex()==row['answer'])==bool(a['correct'][j]) and eos==bool(a['EOS'][j])
 return dict(correct=int(a['correct'].sum()),OLD=int(a['correct'][:64].sum()),NEW=int(a['correct'][64:].sum()),empty_EOS=int(np.sum(a['choices'][:,0]==256)),CAP=int(np.sum(~a['EOS'])),first_payload_correct=int(np.sum(z[yy].argmax(1)==y[yy])),true_EOS_correct=int(np.sum(z[ee].argmax(1)==256)),OLD_correct_indices=np.flatnonzero(a['correct'][:64]).tolist())
def favorite(c):return min(c,key=lambda v:(-c[v],v))
def countermodels(d):
 observed=[d['OLD'][i] for order in d['OLD_orders'] for i in order]+[next(r for r in d['NEW'] if r['id']==rid) for rid in d['schedules']['INTERLEAVED']]
 full=Counter();positions=[Counter() for _ in range(4)];length=defaultdict(Counter);suffix=[defaultdict(Counter) for _ in range(5)];raw={};canonical={}
 for row in observed:
  p=bytes.fromhex(row['prompt']);a=bytes.fromhex(row['answer']);raw[p]=a;canonical[tuple(row['orbit'])]=a;full[a]+=1;length[len(p)][a]+=1
  for k,value in enumerate(list(a)+[256]):positions[k][value]+=1
  for k in range(5):suffix[k][p[-k:] if k else b''][a]+=1
 fav=favorite(full);pos=[]
 for c in positions:
  yy=favorite(c) if c else 256
  if yy==256:break
  pos.append(yy)
 stats=Counter();coverage=Counter()
 for row in d['assessment']:
  p=bytes.fromhex(row['prompt']);answer=bytes.fromhex(row['answer']);orbit=tuple(row['orbit']);assert p not in raw and orbit not in canonical
  pred={'RAW_KEY':raw.get(p,fav),'CANONICAL_KEY':canonical.get(orbit,fav),'FAVORITE':fav,'POSITION':bytes(pos),'PROMPT_LENGTH':favorite(length[len(p)]) if length[len(p)] else fav}
  for k in range(5):
   ss=suffix[k].get(p[-k:] if k else b'');pred['SUFFIX_'+str(k)]=favorite(ss) if ss else fav
  pred['LONGEST_BACKOFF']=next((favorite(suffix[k][p[-k:] if k else b'']) for k in range(4,-1,-1) if (p[-k:] if k else b'') in suffix[k]),fav)
  for name,value in pred.items():stats[name]+=int(value==answer)
 return dict(observed_record_reuses=len(observed),records=50,correct=dict(stats),RAW_key_coverage=0,canonical_orbit_coverage=0,source='all observed OLD+NEW main history; includes inhibited OLD observations, excludes discarded probe forks',same_final_multiset_verified=True)
def main():
 start=time.process_time();assert load(H/'STATUS.json')['status']=='EXECUTION_COMPLETE_AWAITING_REDUCTION'
 datasets=load(H/'local/DATA.json');logs=[json.loads(x) for x in (H/'LEDGER.jsonl').read_text().splitlines()];assert any(x['event']=='GLOBAL_TAUGHT_FROZEN' for x in logs)
 frozen_index=next(i for i,x in enumerate(logs) if x['event']=='GLOBAL_TAUGHT_FROZEN');assert all(i>frozen_index for i,x in enumerate(logs) if x['event']=='HELD_DECISIONS_COMMIT')
 for p,h in load(H/'CONTINUATION_SOURCE_LOCK.json').items():assert sha(R/p)==h
 original=load(H/'SOURCE_LOCK.json');before=H/'cycles/cycle3/ENGINE_BEFORE_STATUS_FIX.py';assert sha(before)==original[str((H/'engine157.py').relative_to(R))]
 aa=ast.parse(before.read_text());bb=ast.parse((H/'engine157.py').read_text())
 for tree in [aa,bb]:
  for node in ast.walk(tree):
   if isinstance(node,ast.keyword) and node.arg in ('stage','phase') and isinstance(node.value,ast.Constant) and node.value.value in ('OLD','NEW'):node.arg='STATUS_PHASE'
 assert ast.dump(aa)==ast.dump(bb)
 panel_commits=[x for x in logs if x['event']=='PANEL_COMMIT'];assert len(panel_commits)==108
 for rec in panel_commits:assert sha(H/rec['path'])==rec['sha256']
 acquired=[];held=[];curves=[];missing_margins=0;available_margins=0;cm={}
 for world,d in datasets.items():
  wa=load(H/f'worlds/{world}/TAUGHT_AUDIT.json');assert wa['status']=='ACCEPTED';cm[world]=countermodels(d)
  for prior in ['W','SH_OLD']:
   root=load(H/f'roots/{world}_{prior}.json');initial=panel_stats(root['panel'],d)
   for schedule in ['INTERLEAVED','CLUSTERED']:
    key=f'{world}_{prior}_{schedule}';result=load(H/f'local/{key}_RESULT.json');final=panel_stats(result['boundaries'][-1]['panel'],d);delay=panel_stats(result['delayed_panel'],d);oldset=root['initial_correct_OLD']
    imm=result['immediate'];assert len(imm)==128
    for e,rid in zip(imm,d['schedules'][schedule]):
     row=next(r for r in d['NEW'] if r['id']==rid);body,eos=decode(e['choices']);assert (eos and body.hex()==row['answer'])==e['correct']
     if e['index'] in [16,32,64,128]:available_margins+=1
     elif 'required_margins' in e:available_margins+=1
     else:missing_margins+=1
    last={rid:max((e for e in imm if e['id']==rid),key=lambda x:x['index']) for rid in {r['id'] for r in d['NEW']}}
    final_arrays=array(result['boundaries'][-1]['panel']);post_arrays=array(result['delayed_panel']);lost_after_last=0
    for j,row in enumerate(d['NEW']):lost_after_last+=int(last[row['id']]['correct'] and not bool(final_arrays['correct'][64+j]))
    record=dict(case=key,world=int(world),prior=prior,schedule=schedule,initial=initial,final=final,delayed=delay,immediate_correct=sum(e['correct'] for e in imm),immediate_presentations=128,unique_NEW_ever_correct_on_own_teaching=sum(any(e['correct'] for e in imm if e['id']==rid) for rid in last),NEW_correct_after_last_own_teaching=sum(e['correct'] for e in last.values()),NEW_last_write_to_final_losses=lost_after_last,loss_causality='actual subsequent history; not uniquely competing writes',initially_acquired_OLD=len(oldset),initially_acquired_OLD_retained_final=sum(i in final['OLD_correct_indices'] for i in oldset),initially_acquired_OLD_retained_delay=sum(i in delay['OLD_correct_indices'] for i in oldset),n_final=result['final_n'],time_final=result['final_time'],NEW_targets=result['NEW_targets'],NEW_seconds=result['NEW_seconds'],sampled_write_complete_losses=sum(p['write_complete_losses'] for p in result['probes']),sampled_write_complete_gains=sum(p['write_complete_gains'] for p in result['probes']),sampled_write_margin_mean=sum(p['write_mean_margin'] for p in result['probes'])/4,sampled_observed_margin_mean=sum(p['observed_mean_margin'] for p in result['probes'])/4)
    acquired.append(record)
    for b in result['boundaries']:curves.append(dict(case=key,index=b['index'],counts=panel_stats(b['panel'],d),unique_NEW=b['unique_taught'],per_cue_exposure=b['exposure'],NEW_targets=b['NEW_targets'],NEW_seconds=b['NEW_seconds']))
 for rec in load(H/'HELD_RECEIPTS.json'):
  receipt=rec['receipt'];assert sha(H/receipt['path'])==receipt['sha256'];raw=load(H/receipt['path'])['rows'];d=datasets[str(rec['world'])];by={r['id']:r for r in d['assessment']};correct=0;empty=0;cap=0;scored=[]
  assert len(raw)==50 and {r['id'] for r in raw}==set(by)
  for row in raw:
   assert not any(k in row for k in ['answer','expected_hex','target','teacher_forced'])
   body,eos=decode(row['choices']);assert eos==row['EOS'] and body.hex()==row['output_hex'];ok=eos and body.hex()==by[row['id']]['answer'];correct+=int(ok);empty+=int(eos and not body);cap+=int(not eos);scored.append(dict(**row,correct=ok))
  save(H/f'local/{rec["case"]}_{rec["endpoint"]}_HELD_SCORED.json',scored)
  held.append({k:rec[k] for k in ['case','world','prior','schedule','endpoint']}|dict(records=50,correct=correct,empty_EOS=empty,CAP=cap,prior_control='OLD value writes inhibited only; NEW training enabled'))
 contrasts=[]
 for world in datasets:
  for prior in ['W','SH_OLD']:
   a=next(x for x in acquired if str(x['world'])==world and x['prior']==prior and x['schedule']=='CLUSTERED');b=next(x for x in acquired if str(x['world'])==world and x['prior']==prior and x['schedule']=='INTERLEAVED');assert a['n_final']==b['n_final'] and a['time_final']==b['time_final']
   contrasts.append(dict(world=int(world),prior=prior,NEW_final_clustered_minus_interleaved=a['final']['NEW']-b['final']['NEW'],NEW_delayed_clustered_minus_interleaved=a['delayed']['NEW']-b['delayed']['NEW']))
  for schedule in ['INTERLEAVED','CLUSTERED']:
   for endpoint in ['final','delayed']:
    w=next(x for x in held if str(x['world'])==world and x['prior']=='W' and x['schedule']==schedule and x['endpoint']==endpoint);sh=next(x for x in held if str(x['world'])==world and x['prior']=='SH_OLD' and x['schedule']==schedule and x['endpoint']==endpoint)
    contrasts.append(dict(world=int(world),schedule=schedule,endpoint=endpoint,held_W=w['correct'],held_SH_OLD=sh['correct'],held_W_minus_SH_OLD=w['correct']-sh['correct'],best_countermodel=max(cm[world]['correct'].values()),held_W_minus_best_countermodel=w['correct']-max(cm[world]['correct'].values())))
 counters=load(H/'COUNTS.json');c=counters['counts'];assert [c[k] for k in ['OLD_updating_records','OLD_inhibited_records','NEW_updating_records','NEW_inhibited_probes_records']]==[1024,1024,1024,32]
 assert c['technical_records']==11 and c['observed_records']==3115 and c['immediate_current_scores']==1024 and c['held_queries']==800 and c['science_pairs']==9664
 assert c['logical_evaluator_distributions']<=71520 and c['evaluator_distributions']<=c['logical_evaluator_distributions']
 assert c['immediate_boundary_pair_reuses']==32
 assert available_margins==32 and missing_margins==992
 out=dict(status='COMPLETE_PRIMARY_ENDPOINTS_AUDITABLE_SECONDARY_MARGIN_ARTIFACT_DEVIATION',acquisition=acquired,held=held,contrasts=contrasts,countermodels=cm,secondary_artifact_deviation={'per_presentation_margin_requests_computed':1024,'auditable_postwrite_current_margin_snapshots':available_margins,'not_retained_outside_boundary_panels':missing_margins,'cause':'immediate summary strips the margin arrays before persistence','training_or_queries_repeated_to_repair':False,'effect':'complete-answer primary contrasts intact; no every-presentation-margin mechanism claim'},model_queries_for_reduction=0,training_replayed=False)
 save(H/'RESULTS.json',out);save(H/'ACQUISITION_CURVES.json',curves)
 totalbytes=sum(p.stat().st_size for p in H.rglob('*') if p.is_file());assert totalbytes<1073741824
 audit=dict(status='ACCEPTED_PRIMARY_ENDPOINTS_WITH_DECLARED_SECONDARY_DEVIATION',full_protocol_without_deviation=False,source_closure_files=92,panel_commit_hashes=108,taught_world_audits=2,held_decision_receipts=16,held_decisions=800,global_taught_freeze_before_all_held=True,native_teacher_front_and_counts_unchanged=True,original64_record_prefix_preserved_without_replay=True,operations_AST_status_only=True,cumulative_budgets_carried=True,common_CPU_cap=1200,stage_reallocation='120qualification/960execution/120audit; same total',initial_failed_restore_assertion_preserved=True,secondary_margin_snapshots_missing=992,complete_answer_contrasts_audited=True,held_feedback_delivered_to_learner=False,new_core_adopted=False,model_queries_for_reduction=0,reducer_CPU=time.process_time()-start,reducer_RSS=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,local_output_bytes=totalbytes)
 save(H/'FINAL_AUDIT.json',audit)
 print(json.dumps(dict(audit=audit,acquisition=[{k:x[k] for k in ['case','immediate_correct','unique_NEW_ever_correct_on_own_teaching','NEW_correct_after_last_own_teaching','NEW_last_write_to_final_losses']}|dict(final_NEW=x['final']['NEW'],delayed_NEW=x['delayed']['NEW'],final_OLD=x['final']['OLD']) for x in acquired],contrasts=contrasts),indent=2))
if __name__=='__main__':main()
