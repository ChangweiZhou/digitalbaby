"""Independent saved-artifact audit. No native models, generators or trajectories run."""
import json,time
from pathlib import Path
from collections import Counter
import numpy as np
import support171 as s
s.STAGE='final_audit'
def main():
 t0=s.cpu();result=json.loads((s.H/'CALIBRATION_RESULTS.json').read_text());data=json.loads((s.H/'local/TRAIN_DATA.json').read_text());lock=json.loads((s.H/'CALIBRATION_SOURCE_LOCK.json').read_text());source_bad=[p for p,h in lock.items() if s.sha(s.ROOT/p)!=h];assert not source_bad,source_bad
 words=json.loads((s.ROOT/'minifly_cls_dev_20261017/OUTPUT_CODEBOOK.json').read_text())['words_by_class'];C=np.array([[1. if w&(1<<j) else -1. for j in range(16)] for w in words]);total_sites=0;tables=[];arrays=0;panel_rows=0
 for world in s.SPEC['data']['calibration_worlds']:
  w=data[str(world)];wr=result['worlds'][str(world)];allrows={x['id']:x for k in ['OLD','NEW'] for x in w[k]};oldids={x['id'] for x in w['OLD']};newids={x['id'] for x in w['NEW']};N=0
  records=[json.loads(x) for x in (s.H/'local'/f'cal_{world}_RECORDS.jsonl').read_text().splitlines()];assert len(records)==1280
  record_counts=Counter(x['id'] for x in records);assert all(record_counts[k]==16 for k in allrows) and set(record_counts)==set(allrows)
  for phase in ['OLD','NEW']:
   for p in range(1,17):
    group=[x for x in records if x['phase']==phase and x['pass_no']==p];wanted=[w[phase][int(i)]['id'] for i in w['orders'][(p-1)%8]] if phase=='OLD' else [x['id'] for x in w['NEW']];assert [x['id'] for x in group]==wanted
    native_v=None;native_p=None;native_meta=None
    for role in ['NATIVE','EXACT3','EXACT_RECORD']:
     f=s.H/'local'/f'cal_{world}_{phase}_pass{p:02d}_{role}.npz'
     receipt=next(x for x in wr['array_receipts'] if x['path']==str(f.relative_to(s.H)));assert s.sha(f)==receipt['sha256']
     with np.load(f,allow_pickle=False) as z:
      md=json.loads(z['metadata'].tobytes());P=z['p'];P0=z['p0'];V=z['v'];assert len(md)==len(P) and P.shape[1]==257
      assert np.allclose(P.sum(1),1,rtol=0,atol=1e-12) and np.isfinite(P).all()
      if role=='NATIVE':native_v=V.copy();native_p=P0.copy();native_meta=md
      else:
       assert np.array_equal(V,native_v) and np.array_equal(P0,native_p)
       assert [(x['id'],x['site'],x['n'],x['native_time'],x['y']) for x in md]==[(x['id'],x['site'],x['n'],x['native_time'],x['y']) for x in native_meta]
       assert np.allclose(z['g'],1+.5*np.tanh(z['a']),rtol=0,atol=1e-14)
       assert np.allclose(z['u'],np.tanh((V-V.mean(1,keepdims=True))/4),rtol=0,atol=1e-14)
       E=P.copy();E[np.arange(len(md)),[x['y'] for x in md]]-=1;dv=E@C/4;dv-=dv.mean(1,keepdims=True);expected=.5*(1-np.tanh(z['a'])**2)*V*dv
       assert np.allclose(z['direct_error'],expected,rtol=1e-10,atol=1e-12)
      for i,x in enumerate(md):
       ys=list(bytes.fromhex(allrows[x['id']]['answer']))+[256];assert x['y']==ys[x['site']]
       expected_w=1/16+15/16*min(x['n']/(1024 if role=='NATIVE' else 2048),1);assert abs(x['warmup']-expected_w)<1e-14
       y=x['y'];mask=P[i].copy();mask[y]=-np.inf;r=int(mask.argmax());assert x['argmax']==int(P[i].argmax()) and x['rival']==r
       assert abs(x['lp']-np.log(P[i,y]))<1e-10 and abs(x['margin']-(np.log(P[i,y])-np.log(P[i,r])))<1e-10
       assert len(x['ages'])<=3 if role=='EXACT3' else True
      total_sites+=len(md);arrays+=1
  for tag,roles in wr['panels'].items():
   for role,rec in roles.items():
    pth=s.H/rec['path'];assert s.sha(pth)==rec['sha256'];rows=json.loads(pth.read_text())['rows'];correct=[]
    for x in rows:
     expected=allrows[x['id']]['answer'];actual=bytes(a for a in x['choices'] if a!=256).hex();assert actual==x['output_hex']
     assert x['EOS']==(bool(x['choices']) and x['choices'][-1]==256);assert x['CAP']==(not x['EOS']);assert x['correct']==(x['EOS'] and expected==x['output_hex'])
     assert [a['y'] for a in x['teacher_forced']]==list(bytes.fromhex(expected))+[256]
     if x['correct']:correct.append(x['id'])
     panel_rows+=1
    assert correct==rec['correct_ids'] and len(correct)==rec['correct']
  for role in ['NATIVE','EXACT3','EXACT_RECORD']:
   ini=set(wr['panels']['OLD16'][role]['correct_ids'])&oldids;final=set(wr['panels']['final_OLD'][role]['correct_ids']);delay=set(wr['panels']['delayed'][role]['correct_ids'])
   tables.append(dict(world=world,role=role,initial_OLD=len(ini),final_OLD=len(final),initial_OLD_retained=len(ini&final),initial_OLD_lost=len(ini-final),final_NEW=wr['panels']['NEW256'][role]['correct'],delayed_OLD=len(delay&oldids),delayed_NEW=len(delay&newids),final_NEW_BPS=wr['panels']['NEW256'][role]['TF_negative_log_likelihood']/wr['panels']['NEW256'][role]['TF_targets']/np.log(2),delayed_total=len(delay)))
 # Record applications and target sites are deliberately separate denominators.
 expected_sites=sum(3*16*sum(len(bytes.fromhex(row['answer']))+1 for group in ('OLD','NEW') for row in data[str(world)][group]) for world in s.SPEC['data']['calibration_worlds'])
 assert total_sites==expected_sites==s.COUNTS['arriving_targets']-75
 assert s.COUNTS['logical_records']==7711 and s.COUNTS['calibration_records']==7680 and s.COUNTS['technical_records']==31
 assert s.COUNTS['paired_queries']==2208+16 and panel_rows==2208
 assert not result['qualified'] and not result['main_started'] and result['assessment_decisions']==0
 output=sum(p.stat().st_size for p in s.H.rglob('*') if p.is_file());assert output<1073741824
 s.save('RESULTS.json',dict(version='V171',origin_design='V169',verdict='REFERENCE_GATE_NOT_PASSED',table=tables,logical_records=7711,technical_records=31,calibration_records=7680,arriving_targets=s.COUNTS['arriving_targets'],main_local_addition_records=0,main_assessment_decisions=0,calibration_gate_unchanged=True,new_core_adopted=False))
 s.save('FINAL_AUDIT.json',dict(status='ACCEPTED_CALIBRATION_RESULTS_WITH_DECLARED_TECHNICAL_DEVIATIONS',full_protocol_without_deviation=False,source_hashes_checked=len(lock),source_mismatches=source_bad,training_array_files=arrays,training_target_sites_verified=total_sites,own_TF_panel_records_verified=panel_rows,worlds=2,logical_histories=6,physical_calibration_native_histories=2,native_writer_shared_without_information_change=True,candidate_rules_local_by_source_audit=True,nonlocal_training_only_declared_references=True,assessment_opened=False,main_gate_obeyed=True,technical_records=31,first_failed_technical_9_records_arrays_not_retained=True,technical_prefix_replayed=False,postimport_source_closure_fix_disclosed=True,operations_failures_preserved=True,source_model_equations_changed_after_science_started=False,model_queries_for_audit=0,output_bytes_before_package=output,audit_CPU=s.cpu()-t0,new_core_adopted=False))
 s.COUNTS['audit_saved_target_verifications']+=total_sites;s.finish('THREE_CYCLES_COMPLETE_REFERENCE_NOT_QUALIFIED')
 print(json.dumps(tables))
if __name__=='__main__':main()
