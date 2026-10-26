"""One explicit status-only operations continuation from the saved64-record boundary.

No birth replacement, scientific replay, dataset regeneration or budget reset.
The original controller failure and source lock remain immutable.
"""
import os,sys,time,json,resource,subprocess,traceback
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
sys.dont_write_bytecode=True
import support157 as s
failure=s.load(s.H/'EXECUTION_FAILURE.json');prior=failure['counts']
s.COUNTS.update(prior['counts']);s.TIMES.update(prior['timings']);s.STAGE_USED.update(prior['stage_CPU']);s.STAGE=prior['stage'];s.STAGE_START=prior['CPU']
def cumulative_cpu():
 c=resource.getrusage(resource.RUSAGE_CHILDREN);return prior['CPU']+time.process_time()+c.ru_utime+c.ru_stime
s.cpu=cumulative_cpu
start=s.load(s.H/'START.json');first=json.loads((s.H/'LEDGER.jsonl').read_text().splitlines()[0]);origin=start['unix']-first['wall'];s.START=time.monotonic()-(time.time()-origin)
s.CAP.update(qualification=120.,execution=960.,audit=120.)
import native157 as n
import engine157 as e
import audit157 as audit
original_old=e.old
def actual_preserved_old(d,which):
 if d['world']!=202610265701 or which!='W':return original_old(d,which)
 key='202610265701_W';trace=f'local/{key}_OLD_TRAIN.jsonl';m=s.restore(failure['partial_state']);rr=[json.loads(x) for x in (s.H/trace).read_text().splitlines()]
 assert len(rr)==64 and [r['id'] for r in rr]==[d['OLD'][i]['id'] for i in d['OLD_orders'][0]]
 assert m.n==rr[-1]['n'] and m.time==rr[-1]['time'] and m.records==64
 s.ACTIVE=m;s.ACTIVE_INFO=dict(stage='OLD',case=key);count=64
 for pass_no in range(2,9):
  for ix in d['OLD_orders'][pass_no-1]:
   s.budget();row=d['OLD'][ix];sites=e.train(m,row,'OLD_updating',True);count+=1;e.write_trace(trace,dict(id=row['id'],pass_no=pass_no,n=m.n,time=m.time,sites=sites,write=True))
  s.log('OLD_PASS_COMMIT',case=key,pass_no=pass_no,records=count,n=m.n);s.save('ACTIVE.json',dict(case=key,phase='OLD',next_record=count,**s.state()))
 assert count==512 and m.n>=1024
 panel=n.evaluate(m,d['panel'],key+'_initial');cp=s.checkpoint(m,key+'_OLD.pkl')
 result=dict(world=d['world'],prior=which,records=512,targets=m.n,n=m.n,time=m.time,checkpoint=cp,panel=panel['receipt'],trace=trace,initial_correct_OLD=np_flat(panel['correct'][:64]))
 s.save(f'roots/{key}.json',result);s.ACTIVE=None;s.ACTIVE_INFO=None;return result
def np_flat(a):return [i for i,value in enumerate(a) if value]
e.old=actual_preserved_old
def run():
 assert (s.H/'SCIENCE_STARTED.json').exists() and not (s.H/'CONTINUATION_STARTED.json').exists()
 assert failure['active']==dict(stage='OLD',case='202610265701_W') and s.COUNTS['OLD_updating_records']==64
 for p,h in s.load(s.H/'CONTINUATION_SOURCE_LOCK.json').items():assert s.sha(s.ROOT/p)==h
 lock=s.load(s.H/'DATA_LOCK.json');assert s.sha(s.H/lock['path'])==lock['sha256'];datasets=s.load(s.H/lock['path'])
 # Pure metadata smoke: reserved fields cannot collide; no model is queried.
 s.save('ops/ACTIVE_FIXTURE.json',dict(case='synthetic',phase='OLD',next_record=0,**s.state()));s.log('LOGGER_METADATA_SMOKE',case='synthetic',phase='OLD')
 s.save('CONTINUATION_STARTED.json',dict(physical_prefix_records=64,original_partial=failure['partial_state'],counters_not_reset=True,wall_not_reset=True,science_replayed=False));s.log('OPERATIONS_CONTINUATION_START',from_records=64)
 awake=subprocess.Popen(['/usr/bin/caffeinate','-w',str(os.getpid())]);s.save('LAUNCH.json',dict(pid=os.getpid(),caffeinate_pid=awake.pid,python=sys.executable,workers=1,threads=1,controller='ops_continue157.py'))
 results=[];audits=[]
 try:
  for world in ['202610265701','202610265702']:
   s.save('STATUS.json',dict(status='RUNNING_A1',cycle=3,world=world,stage='taught_histories',operations_failure_preserved=True));s.stage('execution');rr=e.run_world(datasets[world]);results+=rr;s.stage('audit');audits.append(audit.world(datasets[world],rr));s.log('WORLD_TAUGHT_AUDIT_COMMIT',world=world,status='ACCEPTED');s.save('COUNTS.json',s.state())
  assert [s.COUNTS[k] for k in ['OLD_updating_records','OLD_inhibited_records','NEW_updating_records','NEW_inhibited_probes_records']]==[1024,1024,1024,32]
  endpoints={r['case']:{x:r[x] for x in ['final','delayed']} for r in results};s.save('TAUGHT_ENDPOINT_LOCK.json',dict(status='ALL_TAUGHT_ENDPOINTS_GLOBALLY_FROZEN',endpoints=endpoints,audits=audits,data_sha256=s.sha(s.H/'local/DATA.json'),source_sha256=s.sha(s.H/'CONTINUATION_SOURCE_LOCK.json'),counts=s.state()))
  s.log('GLOBAL_TAUGHT_FROZEN',branches=len(results),held_scores_opened=False);s.stage('execution');s.save('STATUS.json',dict(status='RUNNING_A1',cycle=3,stage='held_own_only'))
  held=[]
  for r in results:
   prompts=[dict(id=x['id'],prompt=x['prompt']) for x in datasets[str(r['world'])]['assessment']]
   for endpoint in ['final','delayed']:
    m=s.restore(r[endpoint]);held.append(dict(case=r['case'],world=r['world'],prior=r['prior'],schedule=r['schedule'],endpoint=endpoint,receipt=n.held(m,prompts,r['case']+'_'+endpoint+'_HELD')));del m
  s.save('HELD_RECEIPTS.json',held);s.save('cycles/cycle3/AUDIT.json',dict(status='TAUGHT_AUDITS_ACCEPTED_HELD_AWAITING_EXTERNAL_REDUCTION',cases=audits,negative_fixtures=audit.negative_fixtures(),original_controller_failure_preserved=True,science_replayed=False));s.log('CYCLE3_EXECUTION_COMPLETE',branches=8,held_queries=s.COUNTS['held_queries']);s.save('STATUS.json',dict(status='EXECUTION_COMPLETE_AWAITING_REDUCTION',cycles=3,records=s.COUNTS['observed_records'],B_resumed=False,C_implemented=False));s.save('COUNTS.json',s.state())
 except BaseException as exc:
  partial=s.checkpoint(s.ACTIVE,'PRESERVED_CONTINUATION_PARTIAL.pkl') if s.ACTIVE is not None else None;s.save('CONTINUATION_FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc(),partial_state=partial,active=s.ACTIVE_INFO,counts=s.state(),no_automatic_retry=True));s.save('STATUS.json',dict(status='PAUSED_RESOURCE_CAP' if isinstance(exc,s.Limit) else 'FAILED_EXECUTION',error=repr(exc)));s.save('COUNTS.json',s.state());raise
 finally:awake.terminate();awake.wait(timeout=5)
if __name__=='__main__':run()
