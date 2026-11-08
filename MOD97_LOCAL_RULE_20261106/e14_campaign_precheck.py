"""E14: dependency-aware fixed campaign with exact receipt validation."""
import fcntl,hashlib,json,os,subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
from e9_campaign import read,save
R=Path(__file__).resolve().parent;O=R/'exp45';STANDING=(73129,93683,116089);FRESH=(200003,300007,400009)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def scientific(v):
 if isinstance(v,dict):return {k:scientific(x) for k,x in v.items() if k!='cpu'}
 if isinstance(v,list):return [scientific(x) for x in v]
 return v

def e8(key,part,seed,arm='C',wd=.25,logged=False):
 env=dict(EXP45_HIDDEN='512',E8_BATCH='64',E8_LR='.001',E8_WD=str(wd),E8_SCHEDULE='blocked',E8_REPLAY_SHARE='1/2',E8_BUFFER_CAP='1024',E8_REPLAY_SAMPLING='uniform',E8_SAVE_STAGES='0',E8_DIAGNOSTICS='1' if logged else '0',E8_RUN_PREFIX=f'e14_{key}_')
 j=dict(key=f'{key}_{seed}',part=part,kind='E8',seed=seed,command=['exp8_life.py',arm,str(seed)],env=env,receipt=f'e14_{key}_e8_life_{arm}_{seed}.json',stages=10)
 if logged:
  j['diagnostic']=j['receipt'].replace('.json','.diagnostics.jsonl')
  j['reference']=f'e8_life_D_{seed}.json' if arm=='D' else f'e13_A1_e8_life_C_{seed}.json' if wd==.25 else f'e8_life_C_{seed}.json'
 return j

def e7(key,part,seed,arm='SCR',wd=1.,batch=64):
 env=dict(EXP45_HIDDEN='512',E7_DOSE='16384',E7_BATCH=str(batch),E7_WD=str(wd),E7_RUN_PREFIX=f'e14_{key}_')
 return dict(key=f'{key}_{seed}',part=part,kind='E7',seed=seed,command=['exp7_decon.py',arm,str(seed)],env=env,receipt=f'e14_{key}_e7_life_{arm}_h512_d16384'+(f'_b{batch}' if batch!=64 else '')+f'_{seed}.json',last_step=16384*64//batch)

def plan():
 births=[];jobs=[]
 for s in FRESH:
  for harness in ('E8','E7'):
   prefix='e8_dev' if harness=='E8' else 'e7_dev_h512'
   births.append(dict(key=f'birth_{harness}_{s}',part='A',kind='birth',harness=harness,seed=s,command=['exp8_life.py' if harness=='E8' else 'exp7_decon.py','dev',str(s)],env=dict(EXP45_HIDDEN='512',E7_BATCH='64',E8_BATCH='64',E8_DIAGNOSTICS='0'),receipt=f'{prefix}_{s}.pt',metadata=f'{prefix}_{s}.json',last_step=24576))
  jobs.extend([e8('A_C','A',s),e8('A_D','A',s,'D',.3),e7('A_batch256','A',s,batch=256),e7('A_decay025','A',s,wd=.25),e7('A_BP','A',s,'SCRbp',.3)])
 for s in STANDING:
  for wd,tag in ((.1,'010'),(.5,'050')):jobs.append(e8(f'B_E8_wd{tag}','B',s,wd=wd))
  for wd,tag in ((.1,'010'),(.25,'025'),(.5,'050')):jobs.append(e7(f'B_E7_wd{tag}','B',s,wd=wd))
  jobs.extend([e8('C_wd1','C',s,wd=1.,logged=True),e8('C_wd025','C',s,logged=True),e8('C_BP','C',s,'D',.3,logged=True)])
 extra=e7('profile_E7','profile',STANDING[0]);extra['reference']='e7_life_SCR_h512_d16384_73129.json'
 return births,jobs,extra

def validate(j):
 p=O/j['receipt']
 if j['kind']=='birth':
  import torch
  d=torch.load(p,weights_only=True,map_location='cpu');ref=torch.load(O/('e8_dev_73129.pt' if j['harness']=='E8' else 'e7_dev_h512_73129.pt'),weights_only=True,map_location='cpu')
  assert d.keys()==ref.keys() and all(d[k].shape==ref[k].shape and torch.isfinite(d[k]).all() for k in d)
  curve=read(O/j['metadata'])['curve'];assert curve[-1]['step']==j['last_step']
 elif j['kind']=='E8':
  d=read(p);assert d['seed']==j['seed'] and len(d['stages'])==j['stages'] and d['stages'][-1]['task']=='compose'
  for st in d['stages']:
   for k,v in st['scores'].items():
    if k=='compose':assert set(v)=={'+-','-+','><','<>','within'};continue
    assert 0<=v['exact']<=v['n'] and 0<=v['final']<=v['n']
  if 'diagnostic' in j:
   rows=[json.loads(line) for line in (O/j['diagnostic']).read_text().splitlines()]
   assert len(rows)==112 and [r['global_update'] for r in rows]==list(range(512,57345,512))
   assert all(set(r['parameters'])=={f'{layer}.{kind}' for layer in ('l1','l2','l3') for kind in ('weight','bias')} for r in rows)
   assert scientific(d)==scientific(read(O/j['reference'])),f'Logged parity failed: {j["key"]}'
 else:
  d=read(p);assert d['seed']==j['seed'] and d['curve'][-1]['step']==j['last_step']
  assert all(0<=v['exact']<=v['n'] and 0<=v['final']<=v['n'] for v in d['curve'][-1]['compose'].values())
  if 'reference' in j:assert scientific(d)==scientific(read(O/j['reference'])),j['key']
 return sha(p)

def run(j):
 log=O/'logs'/f'e14_{j["key"]}.log';log.parent.mkdir(exist_ok=True);start=time.monotonic()
 with log.open('w') as f:
  p=subprocess.Popen([sys.executable,'-u',*j['command']],cwd=R,env={**os.environ,**j['env']},stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  save(O/f'E14_JOB_{j["key"]}.json',dict(**j,status='running',pid=p.pid,log=str(log.relative_to(R))))
  code=p.wait()
 row=dict(**j,status='complete' if code==0 else 'failed',returncode=code,wall_seconds=time.monotonic()-start,log=str(log.relative_to(R)))
 save(O/f'E14_JOB_{j["key"]}.json',row);return row

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
      p=O/f'E14_JOB_{other["key"]}.json'
      if p.exists():
       rec=read(p)
       if rec.get('status')=='running':
        try:os.killpg(rec['pid'],15)
        except ProcessLookupError:pass
 return done,failed

def main():
 guard=(O/'E14_CONTROLLER.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (O/'E14_STATUS.json').exists():raise RuntimeError('Existing E14 campaign: inspect explicitly, no duplicate launch')
 lock=read(R/'E14_LOCK.json')
 for n,h in lock['files'].items():assert sha(R/n)==h,n
 for n,h in lock.get('technical_artifacts',{}).items():assert sha(R/n)==h,n
 assert read(R/'E14_CYCLES.json')['all_passed']
 start=time.monotonic();completed=[]
 births,jobs,profile=plan();first=next(j for j in jobs if j['key']=='C_wd1_73129')
 for j in (first,profile):
  save(O/'E14_STATUS.json',dict(status='profiling',pid=os.getpid(),completed=completed,active=[j['key']],planned_subprocesses=46))
  row=run(j);completed.append(row)
  try:
   if row['status']!='complete':raise RuntimeError(row)
   validate(j)
  except Exception as exc:
   save(O/'E14_STATUS.json',dict(status='stopped_profile_or_parity_failure',completed=completed,error=repr(exc)));return
  if j==first:
   cpu=read(O/j['receipt'])['stages'][-1]['cpu'];save(R/'E14_PROFILE.json',dict(logged_E8_profile=row,cpu_seconds=cpu,provisional_E8_life_forecast_cpu_seconds=21*cpu,other_costs='18 E7 candidate lives +1 E7 profile +6 births additional; update after second profile',scientific_logged_unlogged_parity=True))
 save(R/'E14_PROFILE.json',dict(logged_E8_profile=completed[0],E7_profile=completed[1],E8_cpu_seconds=read(O/first['receipt'])['stages'][-1]['cpu'],E7_cpu_seconds=read(O/profile['receipt'])['curve'][-1]['cpu'],forecast_note='21 E8 lives,19 E7 lives including profile,6 fresh births. Birth estimate uses standing-seed metadata where available; forecast is not measured total.'))
 def update(phase,base):
  def callback(done,active,queued,failed):save(O/'E14_STATUS.json',dict(status=phase,pid=os.getpid(),completed=base+done,active=active,queued=queued,failed_parts=sorted(failed),planned_subprocesses=46,elapsed_seconds=time.monotonic()-start))
  return callback
 born,failed_births=queue(births,run,validate,update('births',completed));completed+=born
 verified={}
 if not failed_births:
  for j in births:verified[j['key']]={n:sha(O/n) for n in (j['receipt'],j['metadata'])}
 save(R/'E14_BIRTHS.json',dict(status='complete' if not failed_births else 'failed',files=verified))
 roster=[j for j in jobs if j['key']!=first['key'] and not(j['part']=='A' and failed_births)]
 done,failed=queue(roster,run,validate,update('running',completed));completed+=done
 save(O/'E14_STATUS.json',dict(status='complete' if len(completed)==46 and not failed and not failed_births else 'partial_failed',completed=completed,failed_parts=sorted(failed|failed_births),planned_subprocesses=46,elapsed_seconds=time.monotonic()-start))
if __name__=='__main__':main()
