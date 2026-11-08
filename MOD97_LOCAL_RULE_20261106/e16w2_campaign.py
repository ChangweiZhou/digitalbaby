"""Amended campaign; exact local-batch BP depends on completed teacher stream."""
import os,sys,time,subprocess,fcntl,signal,json
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import e16w_campaign as H
R,O=H.R,H.O
save,read,sha=H.save,H.read,H.sha
def plan():
 out=[]
 for seed in (73129,93683,116089):
  for role in ('C_uni','C_share','D_ceiling','D_matched'):
   sampling='share' if role=='C_share' else 'uniform';arm='D' if role=='D_ceiling' else 'C'
   prefix=f'e16w2_{role}_{seed}_';tag='_rp1of2_late_family'+('_share' if sampling=='share' else '')
   j=dict(key=f'{role}_{seed}',role=role,seed=seed,deps=[f'C_uni_{seed}'] if role=='D_matched' else [],env=dict(EXP45_HIDDEN='512',E8_BATCH='64',E8_WD='.3' if role=='D_ceiling' else '.25',E8_LR='.001',E8_SCHEDULE='late_family',E8_REPLAY_SHARE='1/2',E8_BUFFER_CAP='1024',E8_REPLAY_SAMPLING=sampling,E8_SAVE_STAGES='0',E8_DIAGNOSTICS='0',E8_COMPOSE_CHECKPOINTS='0',E8_RUN_PREFIX=prefix),receipt=f'{prefix}e8_life_{arm}{tag}_{seed}.json',adopt=(role=='D_ceiling' and seed==73129))
   if j['adopt']:j['receipt']='e16w_D_73129_e8_life_D_rp1of2_late_family_73129.json'
   out.append(j)
 return out
def validate(j):
 d=read(O/j['receipt']);assert d['seed']==j['seed'] and len(d['stages'])==10
 assert [x['task'] for x in d['stages'][1:]]==['add97','sub97','max97','min97','compose','add73','sub73','max73','min73']
 for st in d['stages']:
  assert len(st['scores'])==13
  for v in st['scores']['compose'].values():assert v['n']==600 and all(0<=v[k]<=600 for k in ('exact','final','intermediate'))
 if not j['adopt']:
  m=read(O/f'E16W2_BATCHES_{j["role"]}_{j["seed"]}.json');assert m['training_batches']==57344
  if j['role']=='C_share':assert all(x['composition']==16*x['updates'] for x in m['exposures'][5:])
  if j['role']=='D_matched':
   teacher=read(O/f'E16W2_BATCHES_C_uni_{j["seed"]}.json');assert m['batch_sha256']==teacher['batch_sha256'] and m['teacher_stream_sha256']==teacher['teacher_stream_sha256']
 return sha(O/j['receipt'])
def run(j):
 for n,h in read(R/'E16W2_LOCK.json')['files'].items():assert sha(R/n)==h,n
 t=time.monotonic()
 if j['adopt']:
  old=read(R/'E16W_AMENDMENT_HANDOVER.json')['retained'][0]
  save(O/f'E16W2_JOB_{j["key"]}.json',{**j,'status':'running','pid':old['pid'],'provenance':'ADOPTED_EXISTING_CEILING'})
  while True:
   d=read(O/j['receipt']) if (O/j['receipt']).exists() else None
   if d and len(d['stages'])==10:break
   try:os.kill(old['pid'],0)
   except ProcessLookupError:raise RuntimeError('Retained ceiling exited before complete receipt')
   time.sleep(15)
  row={**j,'status':'complete','returncode':None,'exitcode_note':'Adopted non-child; numeric exit status not captured. Full receipt independently verified.','adoption_wait_seconds':time.monotonic()-t,'receipt_sha256':validate(j)}
 else:
  log=O/'logs'/f'e16w2_{j["key"]}.log'
  with log.open('w') as h:
   p=subprocess.Popen([sys.executable,'-u','e16w2_worker.py',j['role'],str(j['seed'])],cwd=R,env={**os.environ,**j['env']},stdout=h,stderr=subprocess.STDOUT,start_new_session=True)
   save(O/f'E16W2_JOB_{j["key"]}.json',{**j,'status':'running','pid':p.pid,'log':str(log.relative_to(R))});code=p.wait()
  row={**j,'status':'complete' if code==0 else 'failed','returncode':code,'wall_seconds':time.monotonic()-t,'log':str(log.relative_to(R))}
  if code==0:
   try:row['receipt_sha256']=validate(j)
   except Exception as e:row.update(status='failed',error=repr(e))
 save(O/f'E16W2_JOB_{j["key"]}.json',row);print('END',j['key'],row['status'],flush=True);return row
def main():
 guard=(O/'E16W2_CONTROLLER.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (O/'E16W2_STATUS.json').exists()
 pending=plan();done=[];active={};failed=set();start=time.monotonic()
 with ThreadPoolExecutor(max_workers=6) as pool:
  while pending or active:
   ok={j['key'] for j in done if j['status']=='complete'};now=datetime.now(ZoneInfo('America/New_York'));limit=3 if (now.hour,now.minute)<(13,5) else 6
   for j in list(pending):
    if j['role'] in failed or (j['role']=='D_matched' and 'C_uni' in failed):pending.remove(j);done.append({**j,'status':'NOT_RUN','reason':'technical dependency failure'});continue
    if len(active)<limit and set(j['deps'])<=ok:active[pool.submit(run,j)]=j;pending.remove(j);print('START',j['key'],flush=True)
   save(O/'E16W2_STATUS.json',dict(status='running',pid=os.getpid(),completed=done,active=[j['key'] for j in active.values()],queued=[j['key'] for j in pending],failed_parts=sorted(failed),elapsed_seconds=time.monotonic()-start,planned=12))
   if not active:break
   ready,_=wait(active,timeout=30,return_when=FIRST_COMPLETED)
   for f in ready:
    j=active.pop(f)
    try:row=f.result()
    except Exception as e:row={**j,'status':'failed','error':repr(e)}
    done.append(row)
    if row['status']=='failed':
     failed.add(j['role'])
     for other in active.values():
      if other['role']==j['role'] or (j['role']=='C_uni' and other['role']=='D_matched'):
       p=O/f'E16W2_JOB_{other["key"]}.json'
       if p.exists() and not other['adopt']:
        try:os.killpg(read(p)['pid'],signal.SIGTERM)
        except ProcessLookupError:pass
 save(O/'E16W2_STATUS.json',dict(status='complete' if len(done)==12 and not failed else 'partial_failed',completed=done,failed_parts=sorted(failed),elapsed_seconds=time.monotonic()-start,planned=12))
if __name__=='__main__':main()
