"""Fixed V169 calibration gate, published as V171. No main assessment labels opened."""
import os,sys,time,json,pickle,traceback,subprocess
from pathlib import Path
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
sys.dont_write_bytecode=True
import numpy as np
import support171 as s
s.STAGE='calibration'
def main():
 q=json.loads((s.H/'QUALIFICATION.json').read_text());assert q['status']=='PASS_TECHNICAL_WITH_PRESERVED_REVISIONS'
 assert not (s.H/'CALIBRATION_STARTED.json').exists(),'No automatic restart/replay'
 from controller171 import Controller
 import native171 as n
 import data171
 # An upper interval contains the charged long-context record, plus other work.
 forecast=q['record_benchmark_conservative_interval_CPU']*(2*1280)+q['pair_benchmark_conservative_CPU']*(6*368)+600
 s.budget(reserve=forecast);assert q['technical_records']==31
 s.save('CALIBRATION_ADMISSION.json',dict(accepted=True,forecast_CPU_upper=forecast,remaining_CPU=14400-s.cpu(),RSS_qualification=json.loads((s.H/'COUNTS.json').read_text())['RSS'],records_planned=7680,pairs_planned=2208,record_and_pair_caps_unchanged=True,main_admission_not_implied=True))
 lock=s.source_lock();s.save('CALIBRATION_SOURCE_LOCK.json',lock);s.save('CALIBRATION_STARTED.json',dict(CPU=s.cpu(),wall=s.wall(),source_lock_sha256=s.sha(s.H/'CALIBRATION_SOURCE_LOCK.json')))
 data=data171.make();results={}
 awake=subprocess.Popen(['caffeinate','-i','-w',str(os.getpid())],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
 s.save('LAUNCH.json',dict(pid=os.getpid(),caffeinate_pid=awake.pid,stage='calibration'))
 try:
  for world in s.SPEC['data']['calibration_worlds']:
   w=data[str(world)];m=n.birth();cs=[Controller(world,k,n.C) for k in ['EXACT3','EXACT_RECORD']];mapping={'NATIVE':None,**{c.role:c for c in cs}};panels={};receipts=[];records=0;train_cpu=0.
   s.log('CALIBRATION_WORLD_START',world=world)
   for phase in ['OLD','NEW']:
    rows=w[phase]
    for p in range(1,17):
     order=[rows[int(i)] for i in w['orders'][(p-1)%8]] if phase=='OLD' else rows
     tapes={role:s.Tape(f'cal_{world}_{phase}_pass{p:02d}_{role}') for role in mapping}
     t0=s.cpu()
     with (s.H/'local'/f'cal_{world}_RECORDS.jsonl').open('a') as f:
      for row in order:
       meta=dict(id=row['id'],phase=phase,pass_no=p,world=world)
       out=n.train_record(m,cs,bytes.fromhex(row['prompt']),bytes.fromhex(row['answer']),tapes,meta,write=True)
       assert all(not c.h.any() and not c.frames for c in cs)
       f.write(json.dumps(dict(meta,**out),allow_nan=False)+'\n');records+=1
     train_cpu+=s.cpu()-t0
     for tape in tapes.values():receipts.append(tape.flush())
     s.log('CALIBRATION_PASS_COMMIT',world=world,phase=phase,pass_no=p,physical_records=records,n=m.n,time=m.time)
     if phase=='OLD' and p in (8,16):
      panels[f'OLD{p}']=n.panel(m,mapping,w['OLD']+w['NEW'],f'cal_{world}_OLD{p}')
      s.checkpoint((m,cs),f'cal_{world}_OLD{p}.pkl')
     if phase=='NEW' and p in (2,4,8,16):panels[f'NEW{16*p}']=n.panel(m,mapping,w['NEW'],f'cal_{world}_NEW{16*p}')
     if phase=='NEW' and p==16:
      panels['final_OLD']=n.panel(m,mapping,w['OLD'],f'cal_{world}_final_OLD');s.checkpoint((m,cs),f'cal_{world}_final.pkl')
     if p%4==0:s.checkpoint((m,cs,phase,p,records),f'cal_{world}_working.pkl')
     s.save('ACTIVE.json',dict(stage='calibration',world=world,phase=phase,pass_no=p,physical_records=records,CPU=s.cpu(),wall=s.wall(),logical_records=s.COUNTS['logical_records']))
     s.finish('CALIBRATION_RUNNING');s.budget()
     output=sum(p.stat().st_size for p in (s.H/'local').rglob('*') if p.is_file())
     if output>1073741824:raise s.Stop('PAUSED_OUTPUT_BUDGET')
   delayed=m.clone();delayed.rest(128);panels['delayed']=n.panel(delayed,mapping,w['OLD']+w['NEW'],f'cal_{world}_delayed');s.checkpoint((delayed,cs),f'cal_{world}_delayed.pkl')
   rules={}
   native_new=panels['NEW256']['NATIVE']['correct'];native_old_initial=set(panels['OLD16']['NATIVE']['correct_ids'])&{x['id'] for x in w['OLD']};native_old_final=set(panels['final_OLD']['NATIVE']['correct_ids'])
   for role in ['EXACT3','EXACT_RECORD']:
    init=set(panels['OLD16'][role]['correct_ids'])&{x['id'] for x in w['OLD']};common=init&native_old_initial;final=set(panels['final_OLD'][role]['correct_ids']);loss_native=len(common-native_old_final);loss_ref=len(common-final)
    rules[role]=dict(native_NEW=native_new,reference_NEW=panels['NEW256'][role]['correct'],strict_NEW_gain=panels['NEW256'][role]['correct']>native_new,shared_initial_OLD=len(common),native_shared_OLD_lost=loss_native,reference_shared_OLD_lost=loss_ref,retention_measurable=bool(common),retention_guard=loss_ref<=loss_native)
    rules[role]['pass']=rules[role]['strict_NEW_gain'] and rules[role]['retention_guard']
   results[str(world)]=dict(world=world,physical_records=records,logical_records=3*records,n=m.n,native_time=m.time,training_CPU=train_cpu,panels=panels,rules=rules,pass_all=all(x['pass'] for x in rules.values()),array_receipts=receipts)
   s.save(f'calibration/{world}/RESULTS.json',results[str(world)]);s.log('CALIBRATION_WORLD_COMPLETE',world=world,rules=rules)
  qualified=all(x['pass_all'] for x in results.values())
  out=dict(version='V171',origin_design='V169',worlds=results,qualified=qualified,verdict='QUALIFIED_FOR_MAIN' if qualified else 'REFERENCE_GATE_NOT_PASSED',assessment_decisions=0,assessment_labels_joined=False,main_started=False,all_counts=dict(s.COUNTS),CPU=s.cpu(),wall=s.wall())
  s.save('CALIBRATION_RESULTS.json',out);s.finish('REFERENCE_QUALIFIED_AWAITING_MAIN' if qualified else 'STOPPED_REFERENCE_NOT_QUALIFIED');s.log('CALIBRATION_COMPLETE',qualified=qualified)
 finally:
  awake.terminate();awake.wait(timeout=5)
if __name__=='__main__':
 try:main()
 except Exception as e:
  s.save('CALIBRATION_FAILURE.json',dict(error=str(e),type=type(e).__name__,traceback=traceback.format_exc(),counts=dict(s.COUNTS),CPU=s.cpu(),wall=s.wall()));s.finish('CALIBRATION_FAILED_NO_RETRY');raise
