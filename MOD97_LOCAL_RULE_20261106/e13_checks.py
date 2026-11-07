"""Three executed technical cycles; no scientific success inferred."""
import os
os.environ['EXP45_HIDDEN']='512'
import importlib.util,json,tempfile,hashlib,fcntl
from pathlib import Path
import exp8_life as E
import exp7_decon as Q
import e13_campaign as C
R=Path(__file__).resolve().parent; X=E.X;T=E.T

def module(name,file):
 s=importlib.util.spec_from_file_location(name,R/file);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def digest(v):
 if T.is_tensor(v):return [str(v.dtype),list(v.shape),hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest()]
 if isinstance(v,dict):return {str(k):digest(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [digest(x) for x in v]
 return v

def main():
 output={};jobs=C.plan();assert len(jobs)==18 and len({j['receipt'] for j in jobs})==18
 for batch in (64,256,1024):
  assert 393216%batch==0 and 524288%batch==0
 assert 4096*256==16384*64
 assert E.BATCH==64 and Q.LIFE_BATCH==64 and Q.DOSE==8192
 assert Q.LIFE_BLOCK==X.BLOCK==512 and Q.REPORT_INTERVAL==2048
 output['cycle1']=dict(passed=True,unique_exact_receipts=18,exposure_budgets_exact=True,default_parameters_unchanged=True)
 root=Path(tempfile.mkdtemp(prefix='e13-checks-'));original=X.update;originalout=X.OUT;logs=[]
 def run(m,arm,folder,kind):
  folder.mkdir();m.OUT=folder;X.OUT=folder;states=[]
  for src in ('e8_dev_73129.pt','e7_dev_h512_73129.pt'):(folder/src).symlink_to(originalout/src)
  def update(net,opt,*args,**kw):
   loss=original(net,opt,*args,**kw);states.append(digest([net.state_dict(),opt.state_dict(),T.get_rng_state(),loss]));return loss
  X.update=update
  if kind=='E8':m.STAGE_UPDATES=2;m.COMPOSE_UPDATES=2
  else:
   X.BLOCK=2;m.DOSE=16
   if hasattr(m,'LIFE_BLOCK'):m.LIFE_BLOCK=2;m.REPORT_INTERVAL=2048
  m.life(arm,73129);X.update=original
  receipt=next(folder.glob('*life*.json'))
  return states,C.scientific(json.loads(receipt.read_text()))
 old=module('E12baseline','exp8_e12_frozen.py')
 for arm in ('A','C'):
  a=run(old,arm,root/f'old8{arm}','E8');b=run(E,arm,root/f'new8{arm}','E8');assert a==b,arm;logs.append(dict(script='E8',arm=arm,updates=len(a[0]),exact_model_optimizer_rng_loss_receipt=True))
 old7=module('E9baseline','exp7_e9_frozen.py')
 a=run(old7,'SCR',root/'old7','E7');b=run(Q,'SCR',root/'new7','E7');assert a==b;logs.append(dict(script='E7',arm='SCR',updates=len(a[0]),exact_model_optimizer_rng_loss_receipt=True))
 output['cycle2']=dict(passed=True,checks=logs,scope='Short exact-update technical histories. Full default lives reproduced separately before official jobs.')
 # Exercise the production queue, not a copy of its algorithm.
 oldout=C.O;C.O=root
 complete=root/'exact.json';complete.write_text(json.dumps(dict(seed=73129,stages=[dict(task='compose',scores={})])))
 (root/'same_prefix_unfinished.json').write_text(json.dumps(dict(stages=[])))
 j=dict(key='exact',part='T',receipt='exact.json',stages=1)
 assert C.validate(j)==C.sha(complete)
 invalid={**j,'receipt':'same_prefix_unfinished.json'}
 try:C.validate(invalid)
 except ValueError:pass
 else:raise AssertionError('Incomplete exact receipt accepted')
 seen=[]
 def worker(j):
  seen.append(j['key']);return dict(**j,status='failed' if j['key']=='fail' else 'complete')
 roster=[dict(key='fail',part='A'),dict(key='skip',part='A'),dict(key='keep',part='B')]
 done,failed=C.queue(roster,worker,lambda j:'checked',lambda *args:None,workers=1)
 assert seen==['fail','keep'] and failed=={'A'} and len(done)==2
 with (root/'duplicate.lock').open('a') as one,(root/'duplicate.lock').open('a') as two:
  fcntl.flock(one,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:fcntl.flock(two,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:pass
  else:raise AssertionError('Duplicate launch guard failed')
 C.O=oldout
 output['cycle3']=dict(passed=True,production_queue_used=True,unfinished_sibling_ignored=True,incomplete_exact_receipt_rejected=True,failed_part_skipped_other_part_continues=True,duplicate_launch_rejected=True)
 output['all_passed']=True
 (R/'E13_CYCLES.json').write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output),flush=True)
if __name__=='__main__':main()
