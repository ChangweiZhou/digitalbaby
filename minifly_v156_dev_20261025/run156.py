"""One authorized three-cycle exercise; each registered case is executed once."""
import os,sys,time,traceback,subprocess
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
import support156 as s
import native156 as n
import feasibility156 as f
import qualify156 as q
import audit156 as a

def run():
 s.log('ATTEMPT_START',attempt=1,scope='three cycles, four fixed cases, no main confirmation')
 s.save('LAUNCH.json',dict(pid=os.getpid(),python=sys.executable,started_unix=time.time(),worker_count=1,numerical_threads=1))
 awake=subprocess.Popen(['/usr/bin/caffeinate','-w',str(os.getpid())])
 try:
  # Only previously saved OLD/NEW data enter the proposed work.
  inputs=s.inputs();s.save('local/INPUTS.json',inputs);s.save('INPUT_LOCK.json',dict(path='local/INPUTS.json',sha256=s.sha(s.H/'local/INPUTS.json'),selected_ids={w:[r['id'] for r in d['selected_NEW']] for w,d in inputs.items()},no_validation_or_assessment=True))
  for x in s.load(s.D/'ASSETS.json')['assets']:
   p=s.ROOT/x['path'];assert p.exists() and s.sha(p)==x['sha256']
  for p,h in s.load(s.ROOT/'minifly_v154_dev_20261025/SOURCE_LOCK.json').items():assert s.sha(s.ROOT/p)==h
  s.stage('qualification');s.save('STATUS.json',dict(status='RUNNING',cycle=1,stage='qualification'));q.run()
  s.save('cycles/cycle1/AUDIT.json',dict(status='ACCEPTED',scope='formula, IO, passive/idle and certificate fixtures',cases_retrained=0))
  s.save('cycles/cycle1/REVISION.json',dict(scope='technical qualification',scientific_parameters_changed=False,resolutions=['Idle uses base seconds API','Native margins use exact inherited read equations','Evaluator witnesses never supply native answers','No approximate infeasibility accepted']))
  s.log('CYCLE_COMMIT',cycle=1,status='QUALIFIED')
  for cycle,world in [(2,202610255401),(3,202610255402)]:
   s.save('STATUS.json',dict(status='RUNNING',cycle=cycle,stage='F',world=world))
   for front in ['I_FAST','P_SLOW']:
    s.stage('F');f.case(front,world)
   s.save('STATUS.json',dict(status='RUNNING',cycle=cycle,stage='M',world=world))
   for front in ['I_FAST','P_SLOW']:
    s.stage('M');n.continuation(front,world)
   s.stage('audit');audits=[a.audit_case(front,world,n.C) for front in ['I_FAST','P_SLOW']]
   s.save(f'cycles/cycle{cycle}/AUDIT.json',dict(status='ACCEPTED',cases=audits,world=world,reused_development_world=True,not_fresh_confirmation=True))
   s.save(f'cycles/cycle{cycle}/REVISION.json',dict(status='EVIDENCE_RECORDED',scientific_parameters_changed=False,world=world,rule='No outcome-driven adjustment to the other registered world; final proposal uses all four cases'))
   s.log('CYCLE_COMMIT',cycle=cycle,status='ACCEPTED');s.save('COUNTS.json',s.snapshot_counts())
  s.stage('audit');s.save('COUNTS.json',s.snapshot_counts());s.save('STATUS.json',dict(status='THREE_CYCLES_EXECUTED_AWAITING_REDUCTION',cycles=3,main_experiment_started=False,assessment_scored=False,new_core_adopted=False));s.log('EXECUTION_COMPLETE',cycles=3)
 except BaseException as exc:
  state='PAUSED_RESOURCE_LIMIT' if isinstance(exc,s.BudgetStop) else 'FAILED'
  if s.ACTIVE_MODEL is not None:s.save('PARTIAL_STATE.json',dict(case=s.ACTIVE_CASE,checkpoint=s.checkpoint(s.ACTIVE_MODEL,'PRESERVED_PARTIAL_NATIVE.pkl'),not_committed_case=True,no_automatic_resume=True))
  s.save('FAILURE.json',dict(status=state,error=repr(exc),traceback=traceback.format_exc(),counters=s.snapshot_counts(),no_automatic_retry=True));s.save('COUNTS.json',s.snapshot_counts());s.save('STATUS.json',dict(status=state,error=repr(exc)));s.log('ATTEMPT_STOP',status=state,error=repr(exc));raise
 finally:
  awake.terminate()
if __name__=='__main__':run()
