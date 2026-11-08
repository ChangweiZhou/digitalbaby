"""Three technical cycles only; short receipts are not scientific evidence."""
import os
os.environ['EXP45_HIDDEN']='512'
import importlib.util,json,hashlib,copy
from pathlib import Path
import exp8_life as E
import e14s_campaign as H
X,T=E.X,E.T
R=Path(__file__).resolve().parent;O=R/'exp45';F=O/'e16w_technical';F.mkdir(exist_ok=True)
def digest(x):
 if T.is_tensor(x):return hashlib.sha256(x.detach().cpu().numpy().tobytes()).hexdigest()
 if isinstance(x,dict):return {str(k):digest(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [digest(v) for v in x]
 return x
def main():
 sp=importlib.util.spec_from_file_location('old','exp8_e16w_frozen.py');old=importlib.util.module_from_spec(sp);sp.loader.exec_module(old)
 original=X.update;score=X.score;X.score=lambda m,it,store=None:score(m,it[:8],store)
 trajectories=[];cycles=[]
 for tag,mod in [('old',old),('default',E)]:
  folder=F/tag;folder.mkdir(exist_ok=True);(folder/'e8_dev_73129.pt').symlink_to(O/'e8_dev_73129.pt')
  mod.OUT=folder;mod.STAGE_UPDATES=2;mod.COMPOSE_UPDATES=4;mod.SAVE_STAGES=True
  trace=[]
  def update(m,opt,it,local,store=None):
   loss=original(m,opt,it,local,store);trace.append(digest([m.state_dict(),opt.state_dict(),T.get_rng_state(),loss,store.mem if store else None,store.order if store else None]));return loss
  X.update=update;mod.life('C',73129);X.update=original
  d=json.loads(next(folder.glob('*life*.json')).read_text());d=H.clean(d)
  # Saved state includes Python replay RNG, buffer and Adam state.
  states=[digest(T.load(p,weights_only=False)) for p in sorted(folder.glob('*state*.pt'))]
  trajectories.append((trace,d,states))
 assert trajectories[0]==trajectories[1]
 cycles.append(dict(cycle=1,status='PASS',old_default_steps=len(trajectories[0][0]),model_optimizer_torch_rng_python_rng_store_buffer_loss_scores_exact=True))
 dry=[]
 for tag,arm,sampling in [('C_uni','C','uniform'),('C_bal','C','balanced'),('D','D','uniform')]:
  folder=F/tag;folder.mkdir(exist_ok=True);(folder/'e8_dev_73129.pt').symlink_to(O/'e8_dev_73129.pt')
  E.OUT=folder;E.SCHEDULE='late_family';E.REPLAY_SAMPLING=sampling;E.BUFFER_CAP=8;E.SAVE_STAGES=False
  captured=[]
  def update(m,opt,it,local,store=None):
   for p,t in it:
    assert len(t)==(6 if p[0]==ord('c') else 3)
    if p[0]==ord('c'):assert t[2]==59 and t[-1]==E.EOS
   captured.extend((p[0],len(t)) for p,t in it)
   return original(m,opt,it,local,store)
  X.update=update;E.life(arm,73129);X.update=original
  d=json.loads(next(folder.glob('*life*.json')).read_text())
  assert [x['task'] for x in d['stages'][1:]]==['add97','sub97','max97','min97','compose','add73','sub73','max73','min73']
  assert all('compose' in x['scores'] for x in d['stages'])
  if arm=='C':
   ad=json.loads(next(folder.glob('*.buffer.json')).read_text());assert len(ad)==9 and all(a['n']==8 and a['all_presented'] for a in ad)
   assert ad[4]['task']=='compose' and all(p[0]==ord('c') for p in ad[4]['prompts'])
   assert sum(x['replay_draws_by_prior_stage'][4] for x in ad[5:])>0
  dry.append(dict(arm=tag,training_updates=len(d['stages'])-1,buffer_compose_replayed=arm=='C',six_byte_targets_checked=True))
 cycles.append(dict(cycle=2,status='PASS',full_schedule_short_loops=dry,training_steps_per_life=20,panel_items_per_score=8,buffer_cap=8))
 # Controller guards: isolated receipt validation rejects partial file and excludes siblings.
 import e16w_campaign as C
 j=C.plan()[1]
 tmp=F/'validator';tmp.mkdir(exist_ok=True)
 valid=copy.deepcopy(d);valid['arm']='C';valid['seed']=73129
 (tmp/j['receipt']).write_text(json.dumps(valid))
 (tmp/'unfinished_sibling.json').write_text('{')
 original_o=C.O;C.O=tmp
 C.validate(j,short=True)
 bad=copy.deepcopy(valid);bad['stages']=bad['stages'][:-1];(tmp/j['receipt']).write_text(json.dumps(bad))
 try:C.validate(j,short=True)
 except AssertionError:pass
 else:raise AssertionError('Partial receipt accepted')
 C.O=original_o
 cycles.append(dict(cycle=3,status='PASS',exact_job_only=True,incomplete_receipt_rejected=True,unfinished_sibling_ignored=True,full_profile_gate='REQUIRED_BEFORE_SCIENTIFIC_JOBS'))
 H.save(R/'E16W_CHECKS.json',dict(status='PASS',cycles=cycles,technical_root=str(F),scope='Short technical checks only; full default parity still pending.'))
 print(json.dumps(cycles,indent=2))
if __name__=='__main__':main()
