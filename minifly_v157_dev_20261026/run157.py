"""Cycle-three fixed A1 roster. Never resumes/replays an already-started science run."""
import os,sys,time,subprocess,traceback
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
sys.dont_write_bytecode=True
import support157 as s
from carry157 import load_prior
load_prior(2)
# Explicit pre-science operational revision: common1200CPU and scientific controls unchanged.
s.CAP.update(qualification=120.,execution=960.,audit=120.)
import native157 as n
import data157 as data
import engine157 as e
import audit157 as audit
def run():
 assert not (s.H/'SCIENCE_STARTED.json').exists(),'No automatic scientific restart or replay'
 assert s.load(s.H/'cycles/cycle2/QUALIFICATION.json')['status']=='PASS'
 assert s.load(s.H/'ADMISSION_R3.json')['status']=='ADMITTED'
 for p,h in s.load(s.H/'SOURCE_LOCK.json').items():assert s.sha(s.ROOT/p)==h
 awake=subprocess.Popen(['/usr/bin/caffeinate','-w',str(os.getpid())]);s.save('LAUNCH.json',dict(pid=os.getpid(),caffeinate_pid=awake.pid,python=sys.executable,workers=1,threads=1))
 results=[];audits=[]
 try:
  s.stage('execution');datasets=data.make();s.save('local/DATA.json',datasets);s.save('DATA_LOCK.json',dict(path='local/DATA.json',sha256=s.sha(s.H/'local/DATA.json'),worlds=list(datasets),NEW_ids={w:[r['id'] for r in d['NEW']] for w,d in datasets.items()},assessment_feedback_forbidden=True))
  s.save('SCIENCE_STARTED.json',dict(scope='fixed A1 only',attempt=1,worlds=list(datasets),never_replay=True));s.log('CYCLE3_SCIENCE_START',worlds=list(datasets))
  for world in ['202610265701','202610265702']:
   s.save('STATUS.json',dict(status='RUNNING_A1',cycle=3,world=world,stage='taught_histories'));s.stage('execution');rr=e.run_world(datasets[world]);results+=rr;s.stage('audit');audits.append(audit.world(datasets[world],rr));s.log('WORLD_TAUGHT_AUDIT_COMMIT',world=world,status='ACCEPTED');s.save('COUNTS.json',s.state())
  assert s.COUNTS['OLD_updating_records']==1024 and s.COUNTS['OLD_inhibited_records']==1024 and s.COUNTS['NEW_updating_records']==1024 and s.COUNTS['NEW_inhibited_probes_records']==32
  endpoints={r['case']:{x:r[x] for x in ['final','delayed']} for r in results};s.save('TAUGHT_ENDPOINT_LOCK.json',dict(status='ALL_TAUGHT_ENDPOINTS_GLOBALLY_FROZEN',endpoints=endpoints,audits=audits,data_sha256=s.sha(s.H/'local/DATA.json'),source_sha256=s.sha(s.H/'SOURCE_LOCK.json'),counts=s.state()))
  s.log('GLOBAL_TAUGHT_FROZEN',branches=len(results),held_scores_opened=False);s.stage('execution');s.save('STATUS.json',dict(status='RUNNING_A1',cycle=3,stage='held_own_only'))
  held=[]
  for r in results:
   prompts=[dict(id=x['id'],prompt=x['prompt']) for x in datasets[str(r['world'])]['assessment']]
   for endpoint in ['final','delayed']:
    m=s.restore(r[endpoint]);held.append(dict(case=r['case'],world=r['world'],prior=r['prior'],schedule=r['schedule'],endpoint=endpoint,receipt=n.held(m,prompts,r['case']+'_'+endpoint+'_HELD')));del m
  s.save('HELD_RECEIPTS.json',held);s.save('cycles/cycle3/AUDIT.json',dict(status='TAUGHT_AUDITS_ACCEPTED_HELD_AWAITING_EXTERNAL_REDUCTION',cases=audits,negative_fixtures=audit.negative_fixtures(),no_old_run_replayed=True));s.log('CYCLE3_EXECUTION_COMPLETE',branches=8,held_queries=s.COUNTS['held_queries']);s.save('STATUS.json',dict(status='EXECUTION_COMPLETE_AWAITING_REDUCTION',cycles=3,records=s.COUNTS['observed_records'],assessment_feedback_given=False,B_resumed=False,C_implemented=False));s.save('COUNTS.json',s.state())
 except BaseException as exc:
  partial=s.checkpoint(s.ACTIVE,'PRESERVED_PARTIAL.pkl') if s.ACTIVE is not None else None
  s.save('EXECUTION_FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc(),active=s.ACTIVE_INFO,partial_state=partial,counts=s.state(),no_automatic_retry=True));s.save('STATUS.json',dict(status='PAUSED_RESOURCE_CAP' if isinstance(exc,s.Limit) else 'FAILED_EXECUTION',error=repr(exc)));s.save('COUNTS.json',s.state());raise
 finally:
  awake.terminate();awake.wait(timeout=5)
if __name__=='__main__':run()
