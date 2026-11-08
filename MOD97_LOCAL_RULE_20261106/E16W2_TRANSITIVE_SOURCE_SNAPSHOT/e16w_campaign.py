"""Fixed E16W compute campaign. Never restart or launch a duplicate."""
import os,sys,json,time,hashlib,subprocess,fcntl,signal
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
from datetime import datetime
from zoneinfo import ZoneInfo
R=Path(__file__).resolve().parent;O=R/'exp45'
def read(p):return json.loads(p.read_text())
def save(p,d):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def clean(d):
 if isinstance(d,dict):return {k:clean(v) for k,v in d.items() if k!='cpu'}
 if isinstance(d,list):return [clean(v) for v in d]
 return d
def plan():
 jobs=[]
 def add(key,part,seed,arm,wd,schedule,sampling,profile=False):
  tag='' if schedule=='blocked' else '_rp1of2_late_family'+('_balanced' if sampling=='balanced' else '')
  jobs.append(dict(key=key,part=part,seed=seed,arm=arm,profile=profile,deps=[] if profile else ['profile_C_73129'],command=['e14_acceleration_worker.py','exp8_life.py',arm,str(seed)],env=dict(EXP45_HIDDEN='512',E8_BATCH='64',E8_WD=str(wd),E8_LR='.001',E8_SCHEDULE=schedule,E8_REPLAY_SHARE='1/2',E8_BUFFER_CAP='1024',E8_REPLAY_SAMPLING=sampling,E8_SAVE_STAGES='0',E8_DIAGNOSTICS='0',E8_COMPOSE_CHECKPOINTS='0',E8_RUN_PREFIX=f'e16w_{key}_'),receipt=f'e16w_{key}_e8_life_{arm}{tag}_{seed}.json'))
 add('profile_C_73129','profile',73129,'C',.25,'blocked','uniform',True)
 for seed in (73129,93683,116089):
  for part,arm,wd,sampling in [('C_uni','C',.25,'uniform'),('C_bal','C',.25,'balanced'),('D','D',.3,'uniform')]:add(f'{part}_{seed}',part,seed,arm,wd,'late_family',sampling)
 return jobs
def validate(j,short=False):
 d=read(O/j['receipt']);assert d['seed']==j['seed'] and d['arm']==j['arm']
 assert len(d['stages'])==10
 expected=['add97','sub97','max97','min97','add73','sub73','max73','min73','compose'] if j['profile'] else ['add97','sub97','max97','min97','compose','add73','sub73','max73','min73']
 assert [x['task'] for x in d['stages'][1:]]==expected
 if j['profile']:
  assert clean(d)==clean(read(O/'e13_A1_e8_life_C_73129.json')),'Full default receipt mismatch (cpu only exclusion)'
 elif not short:
  for st in d['stages']:
   sc=st['scores'];assert len(sc)==13 and set(sc['compose'])=={'+-','-+','><','<>','within'}
   for v in sc['compose'].values():assert v['n']==600 and all(0<=v[k]<=600 for k in ('exact','final','intermediate'))
   for k,v in sc.items():
    if k!='compose':assert v['n']==(300 if k.endswith('89') else 600) and 0<=v['exact']<=v['n']
  if j['arm']=='C':
   ad=read((O/j['receipt']).with_suffix('.buffer.json'));assert len(ad)==9
   for i,a in enumerate(ad):
    assert a['stage']==i+1 and a['task']==expected[i] and a['n']==1024 and a['all_presented']
    assert len(a['prompts'])==1024 and sha_prompts(a['prompts'])==a['sha256']
   assert all(p[0]==99 for p in ad[4]['prompts'])
 return sha(O/j['receipt'])
def sha_prompts(p):return hashlib.sha256(json.dumps(p,separators=(',',':')).encode()).hexdigest()
def run(j):
 for n,h in read(R/'E16W_LOCK.json')['files'].items():assert sha(R/n)==h,n
 log=O/'logs'/f'e16w_{j["key"]}.log';t=time.monotonic()
 with log.open('w') as h:
  p=subprocess.Popen([sys.executable,'-u',*j['command']],cwd=R,env={**os.environ,**j['env']},stdout=h,stderr=subprocess.STDOUT,start_new_session=True)
  save(O/f'E16W_JOB_{j["key"]}.json',{**j,'status':'running','pid':p.pid,'log':str(log.relative_to(R))})
  print('START',j['key'],p.pid,flush=True);code=p.wait()
 row={**j,'status':'complete' if code==0 else 'failed','returncode':code,'wall_seconds':time.monotonic()-t,'log':str(log.relative_to(R))}
 if code==0:
  try:row['receipt_sha256']=validate(j)
  except Exception as e:row.update(status='failed',error=repr(e))
 save(O/f'E16W_JOB_{j["key"]}.json',row);print('END',j['key'],row['status'],row['wall_seconds'],row.get('error',''),flush=True);return row
def main():
 guard=(O/'E16W_CONTROLLER.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (O/'E16W_STATUS.json').exists(),'Existing campaign; do not duplicate'
 waiting=plan();done=[];active={};failed=set();start=time.monotonic()
 with ThreadPoolExecutor(max_workers=6) as pool:
  while waiting or active:
   success={x['key'] for x in done if x['status']=='complete'}
   now=datetime.now(ZoneInfo('America/New_York'));limit=3 if now.hour<13 or (now.hour==13 and now.minute<5) else 6
   for j in list(waiting):
    if j['part'] in failed or 'profile' in failed:waiting.remove(j);done.append({**j,'status':'NOT_RUN','reason':'dependency/part technical failure'});continue
    if len(active)<limit and set(j['deps'])<=success:active[pool.submit(run,j)]=j;waiting.remove(j)
   save(O/'E16W_STATUS.json',dict(status='running',pid=os.getpid(),completed=done,active=[j['key'] for j in active.values()],queued=[j['key'] for j in waiting],failed_parts=sorted(failed),elapsed_seconds=time.monotonic()-start,planned=10,scientific_lives=9,worker_limit=limit))
   if not active:break
   ready,_=wait(active,timeout=30,return_when=FIRST_COMPLETED)
   for f in ready:
    j=active.pop(f)
    try:row=f.result()
    except Exception as e:row={**j,'status':'failed','error':repr(e)}
    done.append(row)
    if row['status']=='failed':
     failed.add(j['part'])
     for other in active.values():
      if other['part']==j['part']:
       path=O/f'E16W_JOB_{other["key"]}.json'
       if path.exists():
        rec=read(path)
        if rec.get('status')=='running':
         try:os.killpg(rec['pid'],signal.SIGTERM)
         except ProcessLookupError:pass
 save(O/'E16W_STATUS.json',dict(status='complete' if len(done)==10 and not failed else 'partial_failed',completed=done,failed_parts=sorted(failed),elapsed_seconds=time.monotonic()-start,planned=10,scientific_lives=9))
if __name__=='__main__':main()
