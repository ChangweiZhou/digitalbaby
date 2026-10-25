"""Single finite V152 exercise controller; no automatic retry or grid extension."""
import os,sys,json,time,pickle,traceback,subprocess,resource,platform
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='1'
sys.dont_write_bytecode=True
import support152 as s
import models152 as m
from harness152 import group,readiness
import preflight152,credit152


def source_lock():
 old=json.loads((s.ROOT/'minifly_v151_dev_20261025/SOURCE_LOCK.json').read_text());paths={p for p in old if p.startswith(('minifly_a_v3/','minifly_cls_dev_20261017/'))}
 paths|={str(p.relative_to(s.ROOT)) for p in s.H.glob('*.py')};paths|={str(p.relative_to(s.ROOT)) for p in s.DESIGN.glob('*') if p.is_file() and p.suffix in ['.json','.md']}
 paths|={str((s.H/f).relative_to(s.ROOT)) for f in ['AUTHORIZATION.json','PRE_FIT_CLARIFICATIONS.json']}
 for p in old:
  if p in paths:assert s.sha(s.ROOT/p)==old[p],p
 s.save('SOURCE_LOCK.json',{p:s.sha(s.ROOT/p) for p in sorted(paths)})

def main():
 assert not (s.H/'LEDGER.jsonl').exists(),'Existing exercise: do not automatically rerun'
 s.save('AUTHORIZATION.json',dict(user_request='Do the 3-cycle exercise on V152: design, audit, test run, revise. Keep gradual warmup, local candidate updates, no arithmetic-specific structure, and own-byte/EOS unseen-addition generalization per sample and compute.',date='2026-10-25',implementation=True,bounded_three_cycle_exercise=True,formal_confirmation=False,github_push=False,BPTT='independent R_BP diagnostic and evaluator only'))
 s.save('PRE_FIT_CLARIFICATIONS.json',dict(unchanged_scientific_roster=True,unseen_curve='Only final sealed assessment; checkpoint cost curves are acquisition/retention',gradient_implementation='Exact analytic reverse differentiation in R_BP; analytic forward eligibility in local candidates; no Torch dependency',EOS_timing='Predict from state after last payload; reveal then teacher-force only actual observed bytes; EOS triggers update without payload feed',output_accounting='812294 bound is evaluation distributions only; training distributions counted additionally',resume='No automatic replay; interrupted record excluded, current in-memory progress serialized on exception',readiness_failure='Stop assessment; finish independent audit and revision without more learning'))
 source_lock();s.log('CYCLE1_BEGIN');s.save('STATUS.json',dict(status='CYCLE1_QUALIFICATION',training_started=False))
 checks=preflight152.run();s.log('CYCLE1_COMPLETE',checks=len(checks));s.save('cycles/CYCLE1_REVISED.json',dict(status='QUALIFIED',revisions=['Explicit single-final unseen scoring, not unseen checkpoint curve','Separated evaluation distributions from training distributions','Analytic BPTT comparator avoids a new Torch dependency'],scientific_settings_changed=False))
 # Lock all data before calibration, with no model seeing assessment labels.
 manifest={str(w):s.dataset(w) for w in [202610255201,202610255211,202610255212,202610255213]};s.save('local_states/DATA_MANIFEST.json',manifest);s.save('DATA_LOCK.json',dict(sha256=s.sha(s.H/'local_states/DATA_MANIFEST.json'),worlds=list(manifest),recorded_before_feedback=True,assessment_scorer_admitted=False))
 s.log('CYCLE2_BEGIN');s.save('STATUS.json',dict(status='CYCLE2_CALIBRATION',training_started=True));cal=[]
 for eta in [.01,.03]:
  for arm in ['R_BP','R_LE','R_LE0']:
   s.budget();cal.append(group(arm,202610255201,eta,True))
 gate=readiness(cal);s.save('CALIBRATION_LOCK.json',gate);s.log('CYCLE2_COMPLETE',**gate)
 s.save('cycles/CYCLE2_REVISED.json',dict(status='COMPLETE',readiness=gate,no_grid_extension=True,no_rate_selection_per_method=True,assessment_admitted=gate['status']=='READY'))
 s.log('CYCLE3_BEGIN');assessment=[]
 if gate['status']=='READY':
  for world in [202610255211,202610255212,202610255213]:
   for arm in ['N_A','N_2B','N_2B_Q','Q_ONLY','R_BP','R_LE','R_LE0']:
    s.budget();assessment.append(group(arm,world,gate['selected_eta'],False))
   credit152.run(world,gate['selected_eta'])
 verdict='REFERENCE_NOT_READY' if gate['status']!='READY' else 'DEVELOPMENT_COMPLETE_REQUIRES_INTERPRETATION'
 summary=dict(version='V152',verdict=verdict,readiness=gate,completed_calibration_groups=len(cal),completed_assessment_groups=len(assessment),calibration_correlated_histories=3*len(cal),assessment_correlated_histories=3*len(assessment),physical_main_records=sum(x['physical_records'] for x in cal+assessment),observed_main_targets=sum(x['observed_target_applications'] for x in cal+assessment),assessment_scores_opened=bool(assessment),new_core_adopted=False,three_cycles_completed=True)
 s.save('SUMMARY.json',summary)
 s.save('COUNTS.json',dict(counts=dict(s.COUNTS),timings=dict(s.TIMES),CPU=s.cpu(),wall_seconds=time.monotonic()-s.START,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
 # Independent reducer is a separate process and only reads saved responses.
 s.budget();p=subprocess.run([sys.executable,str(s.H/'audit152.py')],cwd=s.ROOT,capture_output=True,text=True)
 (s.H/'audit.log').write_text(p.stdout+p.stderr);assert p.returncode==0,p.stderr
 s.log('CYCLE3_COMPLETE',verdict=verdict);s.save('cycles/CYCLE3_REVISED.json',dict(status='COMPLETE',training_rerun=False,assessment_launched=bool(assessment),audit='FINAL_AUDIT.json',revision='REVISED_DESIGN.md'))
 s.save('STATUS.json',dict(status='THREE_CYCLES_COMPLETE_'+verdict,verdict=verdict,assessment_scores_opened=bool(assessment),new_core_adopted=False,package_complete=False))
 return summary

if __name__=='__main__':
 awake=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())]);s.save('LAUNCH.json',dict(pid=os.getpid(),caffeinate_pid=awake.pid,wall_start=s.WALL_START,workers=1,numerical_threads=1,python=sys.executable,numpy=m.np.__version__,platform=platform.platform()))
 try:result=main();print(json.dumps(result),flush=True)
 except Exception as e:
  active=getattr(s,'ACTIVE_STATE',None)
  if active:
   with (s.H/'local_states/INTERRUPTED_PROGRESS.pkl').open('wb') as f:pickle.dump(active,f,protocol=5)
  s.save('FAILURE.json',dict(error=repr(e),traceback=traceback.format_exc(),budget_pause=isinstance(e,s.BudgetStop),counts=dict(s.COUNTS),CPU=s.cpu(),no_automatic_retry=True))
  s.save('STATUS.json',dict(status=str(e) if isinstance(e,s.BudgetStop) else 'EXECUTION_FAILURE',error=repr(e),preserved=True))
  print(traceback.format_exc(),flush=True);sys.exit(1)
 finally:
  s.save('COUNTS.json',dict(counts=dict(s.COUNTS),timings=dict(s.TIMES),CPU=s.cpu(),wall_seconds=time.monotonic()-s.START,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
  if awake.poll() is None:awake.terminate()
