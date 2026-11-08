"""Three technical cycles before E14; no scientific result is inferred from fixtures."""
import os
os.environ['EXP45_HIDDEN']='512'
import importlib.util,json,tempfile,hashlib,fcntl
from pathlib import Path
import exp8_life as E
import exp7_decon as Q
import e14_diagnostics as D
import e14_campaign as C
R=Path(__file__).resolve().parent;X=E.X;T=E.T

def module(name,file):
 s=importlib.util.spec_from_file_location(name,R/file);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def digest(v):
 if T.is_tensor(v):return [str(v.dtype),list(v.shape),hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest()]
 if isinstance(v,dict):return {str(k):digest(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [digest(x) for x in v]
 return v

def main():
 root=Path(tempfile.mkdtemp(prefix='e14-checks-'));out={};births,jobs,extra=C.plan();assert len(births)==6 and len(jobs)==39
 alljobs=births+jobs+[extra];assert len({j['key'] for j in alljobs})==46 and len({j['receipt'] for j in alljobs})==46
 assert not set(C.FRESH)&set(C.STANDING) and not C.read(R/'E14_SEED_CHECK.json')['collisions']
 assert Q.LIFE_BATCH==64 and Q.DOSE==8192 and E.BATCH==64
 for j in jobs:
  if j['kind']=='E7':assert j['last_step']*int(j['env']['E7_BATCH'])==1048576
 # Direct observer qualification uses actual training X.update, separately on local/BP maps.
 fixture=[E.render(E.LIFE[0],a,b) for a,b in E.split(E.LIFE[0],.25,73129)[0][:8]]
 diagnostic_checks=[]
 for local in (True,False):
  model=X.new_net(73129);opt=T.optim.AdamW(model.parameters(),lr=.001,weight_decay=1.,betas=(.9,.98));logger=D.Logger(root/f'direct_{local}.jsonl',1)
  logger.update(model,opt,fixture,local,None,'fixture',1)
  row=json.loads(logger.path.read_text())
  assert abs(row['parameters']['l3.weight']['raw_gradient_cosine_to_true']-1)<1e-12
  if not local:
   assert all(abs(v['raw_gradient_cosine_to_true']-1)<1e-12 for v in row['parameters'].values())
  assert all(v['weight_norm_before']>0 and v['adam_step_over_weight_norm']>=0 for v in row['parameters'].values())
  diagnostic_checks.append(dict(local=local,output_raw_cosine=row['parameters']['l3.weight']['raw_gradient_cosine_to_true'],BP_all_raw_cosines_one=not local))
 assert D.cosine(T.zeros(2),T.ones(2)) is None
 out['cycle1']=dict(passed=True,subprocesses=46,fresh_seeds_collision_free=True,budgets_and_unique_receipts=True,diagnostic_analytic_checks=diagnostic_checks,zero_norm_cosine_null=True)
 original=X.update;originalout=X.OUT;real_logger=D.Logger;records=[]
 def run(m,arm,folder,kind,logged=False,wd=None):
  folder.mkdir();m.OUT=folder;X.OUT=folder;states=[]
  for src in ('e8_dev_73129.pt','e7_dev_h512_73129.pt'):(folder/src).symlink_to(originalout/src)
  old_flag=os.environ.get('E8_DIAGNOSTICS');os.environ['E8_DIAGNOSTICS']='1' if logged else '0'
  old_wd=os.environ.get('E7_WD')
  if wd is not None:os.environ['E7_WD']=str(wd)
  D.Logger=lambda path:real_logger(path,interval=1)
  def update(net,opt,items,local,store=None):
   value=original(net,opt,items,local,store)
   states.append(digest([net.state_dict(),opt.state_dict(),T.get_rng_state(),value,store.mem if store else None,store.order if store else None]));return value
  X.update=update
  if kind=='E8':m.STAGE_UPDATES=2;m.COMPOSE_UPDATES=2
  else:
   X.BLOCK=2;m.DOSE=16
   if hasattr(m,'LIFE_BLOCK'):m.LIFE_BLOCK=2;m.REPORT_INTERVAL=2048
  try:m.life(arm,73129)
  finally:
   X.update=original;D.Logger=real_logger
   if old_flag is None:os.environ.pop('E8_DIAGNOSTICS',None)
   else:os.environ['E8_DIAGNOSTICS']=old_flag
   if old_wd is None:os.environ.pop('E7_WD',None)
   else:os.environ['E7_WD']=old_wd
  p=next(folder.glob('*life*.json'));receipt=C.scientific(json.loads(p.read_text()))
  if logged:assert len(p.with_suffix('.diagnostics.jsonl').read_text().splitlines())==len(states)
  return states,receipt
 old=module('E13old8','exp8_e13_frozen.py')
 for arm in ('C','D'):
  a=run(old,arm,root/f'old8{arm}','E8');b=run(E,arm,root/f'new8{arm}','E8');c=run(E,arm,root/f'logged8{arm}','E8',True)
  assert a==b==c,arm
  records.append(dict(script='E8',arm=arm,updates=len(a[0]),default_and_every_update_logging_parity=True,model_optimizer_torch_rng_loss_store_and_scores_exact=True))
 old7=module('E13old7','exp7_e13_frozen.py');a=run(old7,'SCR',root/'old7','E7');b=run(Q,'SCR',root/'new7','E7');assert a==b
 records.append(dict(script='E7',updates=len(a[0]),default_parity=True))
 out['cycle2']=dict(passed=True,records=records,scope='Short technical histories; full logged/unlogged and E7 receipts still required before campaign.')
 # Rehearse the production subprocess launcher and scheduler while sibling output is incomplete.
 oldR,oldO=C.R,C.O;C.R=root;C.O=root/'controller';C.O.mkdir();(C.O/'logs').mkdir()
 def job(key,part,delay=0.,exitcode=0):
  path=C.O/f'{key}.json';code=f"import pathlib,json,time,sys;p=pathlib.Path({str(path)!r});p.write_text(json.dumps(dict(seed=73129,curve=[])));time.sleep({delay});p.write_text(json.dumps(dict(seed=73129,curve=[dict(step=1,compose=dict(within=dict(n=1,exact=1,final=1)))])));sys.exit({exitcode})"
  return dict(key=key,part=part,kind='E7',seed=73129,receipt=path.name,last_step=1,command=['-c',code],env={})
 fast,slow=job('same_fast','A',.2),job('same_slow','A',1.)
 done,failed=C.queue([fast,slow],C.run,C.validate,lambda *a:None,2);assert len(done)==2 and not failed
 incomplete=job('incomplete','T');(C.O/incomplete['receipt']).write_text(json.dumps(dict(seed=73129,curve=[])))
 try:C.validate(incomplete)
 except (AssertionError,IndexError):pass
 else:raise AssertionError('Incomplete exact receipt accepted')
 done,failed=C.queue([job('fail','F',0,7),job('skip','F'),job('keep','G')],C.run,C.validate,lambda *a:None,1)
 assert failed=={'F'} and len(done)==2 and not (C.O/'E14_JOB_skip.json').exists()
 with (root/'duplicate.lock').open('a') as one,(root/'duplicate.lock').open('a') as two:
  fcntl.flock(one,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:fcntl.flock(two,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:pass
  else:raise AssertionError('Duplicate launch not blocked')
 C.R,C.O=oldR,oldO
 out['cycle3']=dict(passed=True,production_queue_and_subprocess_used=True,unfinished_sibling_ignored=True,incomplete_exact_receipt_rejected=True,failure_isolates_part_and_skips_queued_sibling=True,other_part_completed=True,duplicate_launch_rejected=True)
 out['all_passed']=True;(R/'E14_CYCLES.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
if __name__=='__main__':main()
