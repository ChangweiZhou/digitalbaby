"""E13 controller: exact receipts, atomic status, part failure isolation, fixed settings."""
import fcntl,hashlib,json,os,subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
from e9_campaign import read,save
R=Path(__file__).resolve().parent;O=R/'exp45';SEEDS=(73129,93683,116089)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def scientific(v):
 if isinstance(v,dict):return {k:scientific(x) for k,x in v.items() if k!='cpu'}
 if isinstance(v,list):return [scientific(x) for x in v]
 return v

def plan():
 jobs=[]
 for s in SEEDS:
  for a,b,w,lr in [('A1',64,.25,.001),('A2',256,4.,.001),('A3',256,1.,.002),('A4',1024,1.,.001)]:
   tag='' if b==64 else f'_b{b}'
   jobs.append(dict(key=f'{a}_{s}',part='A',command=['exp8_life.py','C',str(s)],env=dict(E8_RUN_PREFIX=f'e13_{a}_',E8_BATCH=str(b),E8_WD=str(w),E8_LR=str(lr)),receipt=f'e13_{a}_e8_life_C{tag}_{s}.json',stages=10))
  jobs.append(dict(key=f'B_{s}',part='B',command=['exp7_decon.py','SCR',str(s)],env=dict(EXP45_HIDDEN='512',E7_DOSE='16384',E7_BATCH='256',E7_RUN_PREFIX='e13_B_'),receipt=f'e13_B_e7_life_SCR_h512_d16384_b256_{s}.json',last_step=4096))
  jobs.append(dict(key=f'C_{s}',part='C',command=['exp8_life.py','D',str(s)],env=dict(E8_RUN_PREFIX='e13_C_',E8_BATCH='256'),receipt=f'e13_C_e8_life_D_b256_{s}.json',stages=10))
 return jobs

def validate(job):
 p=O/job['receipt'];d=read(p)
 if 'stages' in job:
  if len(d['stages'])!=job['stages'] or d['stages'][-1]['task']!='compose':raise ValueError(f'Incomplete exact receipt {p}')
  for st in d['stages']:
   for k,v in st['scores'].items():
    if k=='compose':continue
    assert 0<=v['exact']<=v['n'] and 0<=v['final']<=v['n']
 else:
  assert d['curve'][-1]['step']==job['last_step'],str(p)
  assert all(0<=v['exact']<=v['n'] and 0<=v['final']<=v['n'] for v in d['curve'][-1]['compose'].values())
 return sha(p)

def run(job):
 log=O/'logs'/f'e13_{job["key"]}.log';log.parent.mkdir(exist_ok=True);start=time.monotonic()
 with log.open('w') as f:
  p=subprocess.Popen([sys.executable,'-u',*job['command']],cwd=R,env={**os.environ,**job['env']},stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  save(O/f'E13_JOB_{job["key"]}.json',dict(**job,status='running',pid=p.pid,log=str(log.relative_to(R))))
  code=p.wait()
 row=dict(**job,status='complete' if code==0 else 'failed',returncode=code,wall_seconds=time.monotonic()-start,log=str(log.relative_to(R)))
 save(O/f'E13_JOB_{job["key"]}.json',row);return row

def queue(jobs,worker,validator,on_status,workers=6):
 waiting=list(jobs);pending={};done=[];failed=set()
 with ThreadPoolExecutor(max_workers=workers) as pool:
  while waiting or pending:
   while waiting and len(pending)<workers:
    j=waiting.pop(0)
    if j['part'] not in failed:pending[pool.submit(worker,j)]=j
   on_status(done,[j['key'] for j in pending.values()],[j['key'] for j in waiting],failed)
   if not pending:break
   f=next(as_completed(pending));j=pending.pop(f)
   try:
    row=f.result()
    if row['status']=='complete':row['receipt_sha256']=validator(j)
   except Exception as exc:row=dict(**j,status='failed',error=repr(exc))
   done.append(row)
   if row['status']=='failed':
    failed.add(j['part'])
    for other in pending.values():
     if other['part']==j['part']:
      p=O/f'E13_JOB_{other["key"]}.json'
      if p.exists():
       rec=read(p)
       if rec.get('status')=='running':
        try:os.killpg(rec['pid'],15)
        except ProcessLookupError:pass
 return done,failed

def main():
 g=(O/'E13_CONTROLLER.lock').open('a');fcntl.flock(g,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (O/'E13_STATUS.json').exists():raise RuntimeError('Existing campaign: inspect, do not duplicate')
 lock=read(R/'E13_LOCK.json')
 for n,h in lock['files'].items():assert sha(R/n)==h,n
 checks=read(R/'E13_CYCLES.json');assert checks['all_passed']
 start=time.monotonic();profiles=[]
 save(O/'E13_STATUS.json',dict(status='profiling',pid=os.getpid(),completed=[]))
 specs=[dict(key='profile_E8',part='profile',command=['exp8_life.py','C','73129'],env=dict(E8_RUN_PREFIX='e13_profile_'),receipt='e13_profile_e8_life_C_73129.json',stages=10,reference='e9_archived_C_73129.json'),dict(key='profile_E7',part='profile',command=['exp7_decon.py','SCR','73129'],env=dict(EXP45_HIDDEN='512',E7_DOSE='16384',E7_RUN_PREFIX='e13_profile_'),receipt='e13_profile_e7_life_SCR_h512_d16384_73129.json',last_step=16384,reference='e7_life_SCR_h512_d16384_73129.json')]
 for j in specs:
  row=run(j);profiles.append(row)
  if row['status']!='complete' or scientific(read(O/j['receipt']))!=scientific(read(O/j['reference'])):
   save(O/'E13_STATUS.json',dict(status='stopped_profile_or_parity_failure',completed=profiles));return
  validate(j)
  if j['key']=='profile_E8':save(R/'E13_PROFILE.json',dict(profile=row,forecast_note='18jobs include mixed batch/update costs; preliminary8CPUh brief budget is unverified, shared CPU availability varies.',measured_cpu_seconds=read(O/j['receipt'])['stages'][-1]['cpu'],forecast_cpu_seconds=18*read(O/j['receipt'])['stages'][-1]['cpu'],default_E8_scientific_parity=True))
 def status(done,active,queued,failed):
  save(O/'E13_STATUS.json',dict(status='running',pid=os.getpid(),completed=profiles+done,active=active,queued=queued,failed_parts=sorted(failed),elapsed_seconds=time.monotonic()-start,part_D='See PI_FOLLOWUP_STATUS.json; required separately'))
 done,failed=queue(lock['jobs'],run,validate,status)
 save(O/'E13_STATUS.json',dict(status='ABC_complete' if len(done)==18 and not failed else 'partial_failed',completed=profiles+done,failed_parts=sorted(failed),elapsed_seconds=time.monotonic()-start,part_D='See PI_FOLLOWUP_STATUS.json; overall completion requires D'))
if __name__=='__main__':main()
