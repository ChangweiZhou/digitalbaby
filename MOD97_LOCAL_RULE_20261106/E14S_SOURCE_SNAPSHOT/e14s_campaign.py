"""Fixed E14S dependency-aware campaign; no training on evaluation panels."""
import os,sys,json,time,hashlib,subprocess,fcntl
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
R=Path(__file__).resolve().parent;O=R/'exp45'
def read(p):return json.loads(p.read_text())
def save(p,d):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def clean(d):
 if isinstance(d,dict):return {k:clean(v) for k,v in d.items() if k not in ('cpu','compose_checkpoints')}
 if isinstance(d,list):return [clean(v) for v in d]
 return d
def plan():
 jobs=[]
 def life(key,part,seed,arm,wd,logged=False,deps=[]):
  jobs.append(dict(key=key,part=part,seed=seed,kind='life',deps=deps,command=['e14_acceleration_worker.py','exp8_life.py',arm,str(seed)],env=dict(EXP45_HIDDEN='512',E8_BATCH='64',E8_WD=str(wd),E8_LR='.001',E8_SCHEDULE='blocked',E8_REPLAY_SHARE='1/2',E8_BUFFER_CAP='1024',E8_REPLAY_SAMPLING='uniform',E8_SAVE_STAGES='0',E8_DIAGNOSTICS='0',E8_COMPOSE_CHECKPOINTS='1024' if logged else '0',E8_RUN_PREFIX=f'e14s_{key}_'),receipt=f'e14s_{key}_e8_life_{arm}_{seed}.json'))
 life('C_400009','C',400009,'C',.25,True)
 for seed in (500011,600013,700027):jobs.append(dict(key=f'birth_{seed}',part='birth',seed=seed,kind='birth',deps=[],command=['e14_acceleration_worker.py','exp8_life.py','dev',str(seed)],env=dict(EXP45_HIDDEN='512',E8_BATCH='64',E8_COMPOSE_CHECKPOINTS='0'),receipt=f'e8_dev_{seed}.json',checkpoint=f'e8_dev_{seed}.pt'))
 for seed in (73129,93683,116089):life(f'A_{seed}','A',seed,'D',1.)
 for seed in (500011,600013,700027):
  life(f'B_C_{seed}','B',seed,'C',.25,True,[f'birth_{seed}','C_400009'])
  life(f'B_D_{seed}','B',seed,'D',.3,False,[f'birth_{seed}'])
 return jobs
def validate(j):
 d=read(O/j['receipt']);assert d['seed']==j['seed']
 if j['kind']=='birth':
  import torch
  st=torch.load(O/j['checkpoint'],weights_only=True,map_location='cpu');assert sum(v.numel() for v in st.values())==470801 and all(torch.isfinite(v).all() for v in st.values());assert d['curve'][-1]['step']==24576
 else:
  assert len(d['stages'])==10 and d['stages'][-1]['task']=='compose'
  if j['env']['E8_COMPOSE_CHECKPOINTS']!='0':assert [x['update'] for x in d['compose_checkpoints']]==list(range(1024,8193,1024))
  if j['key']=='C_400009':assert clean(d)==clean(read(O/'e14_A_C_e8_life_C_400009.json')),'Full logged parity failed'
 return sha(O/j['receipt'])
def run(j):
 for n,h in read(R/'E14S_LOCK.json')['files'].items():assert sha(R/n)==h,n
 log=O/'logs'/f'e14s_{j["key"]}.log';t=time.monotonic()
 with log.open('w') as h:
  p=subprocess.Popen([sys.executable,'-u',*j['command']],cwd=R,env={**os.environ,**j['env']},stdout=h,stderr=subprocess.STDOUT,start_new_session=True)
  save(O/f'E14S_JOB_{j["key"]}.json',{**j,'status':'running','pid':p.pid,'log':str(log.relative_to(R))})
  code=p.wait()
 row={**j,'status':'complete' if code==0 else 'failed','returncode':code,'wall_seconds':time.monotonic()-t,'log':str(log.relative_to(R))}
 if code==0:
  try:row['receipt_sha256']=validate(j)
  except Exception as e:row.update(status='failed',error=repr(e))
 save(O/f'E14S_JOB_{j["key"]}.json',row);return row
def main():
 guard=(O/'E14S_CONTROLLER.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (O/'E14S_STATUS.json').exists(),'Existing campaign; do not duplicate'
 waiting=plan();done=[];active={};failed=set();start=time.monotonic()
 with ThreadPoolExecutor(max_workers=6) as pool:
  while waiting or active:
   successful={x['key'] for x in done if x['status']=='complete'}
   for j in list(waiting):
    if j['part'] in failed or (j['part']=='B' and 'C' in failed):waiting.remove(j);done.append({**j,'status':'NOT_RUN','reason':'dependency/part failed'});continue
    if len(active)<6 and set(j['deps'])<=successful:active[pool.submit(run,j)]=j;waiting.remove(j)
   save(O/'E14S_STATUS.json',dict(status='running',pid=os.getpid(),completed=done,active=[j['key'] for j in active.values()],queued=[j['key'] for j in waiting],failed_parts=sorted(failed),elapsed_seconds=time.monotonic()-start,planned=13))
   if not active:break
   ready,_=wait(active,return_when=FIRST_COMPLETED)
   for f in ready:
    j=active.pop(f)
    try:row=f.result()
    except Exception as e:row={**j,'status':'failed','error':repr(e)}
    done.append(row)
    if row['status']=='failed':
     failed.add(j['part'])
     for other in active.values():
      if other['part']==j['part'] or (j['part']=='C' and other['part']=='B'):
       path=O/f'E14S_JOB_{other["key"]}.json'
       if path.exists():
        rec=read(path)
        if rec.get('status')=='running':
         try:os.killpg(rec['pid'],15)
         except ProcessLookupError:pass
 save(O/'E14S_STATUS.json',dict(status='complete' if len(done)==13 and not failed else 'partial_failed',completed=done,failed_parts=sorted(failed),elapsed_seconds=time.monotonic()-start,planned=13))
if __name__=='__main__':main()
