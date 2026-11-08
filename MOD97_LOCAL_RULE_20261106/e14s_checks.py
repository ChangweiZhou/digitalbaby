"""Short default/logged trajectory parity; no scientific score claims."""
import os
os.environ['EXP45_HIDDEN']='512'
from pathlib import Path
import importlib.util,tempfile,json,hashlib
import exp8_life as E
import e14s_campaign as C
X=E.X;T=E.T
def digest(x):
 if T.is_tensor(x):return hashlib.sha256(x.detach().numpy().tobytes()).hexdigest()
 if isinstance(x,dict):return {str(k):digest(v) for k,v in x.items()}
 if isinstance(x,(tuple,list)):return [digest(v) for v in x]
 return x
def main():
 root=Path(tempfile.mkdtemp(prefix='e14s-check-'));sp=importlib.util.spec_from_file_location('old','exp8_e14s_frozen.py');old=importlib.util.module_from_spec(sp);sp.loader.exec_module(old)
 original=X.update;score=X.score;records=[]
 X.score=lambda m,it,store=None:score(m,it[:8],store)
 for tag,module,logged in [('old',old,False),('default',E,False),('logged',E,True)]:
  folder=root/tag;folder.mkdir();(folder/'e8_dev_400009.pt').symlink_to(C.O/'e8_dev_400009.pt');module.OUT=folder;module.STAGE_UPDATES=2;module.COMPOSE_UPDATES=4
  if module is E:module.COMPOSE_CHECKPOINTS=2 if logged else 0
  trajectory=[]
  def update(m,opt,it,local,store=None):
   value=original(m,opt,it,local,store);trajectory.append(digest([m.state_dict(),opt.state_dict(),T.get_rng_state(),value,store.mem if store else None,store.order if store else None]));return value
  X.update=update;module.life('C',400009);X.update=original
  f=next(folder.glob('*life*.json'));records.append((trajectory,C.clean(json.loads(f.read_text()))))
 assert records[0]==records[1]==records[2]
 result=dict(passed=True,updates_per_trajectory=len(records[0][0]),default_old_exact=True,logged_old_exact=True,model_optimizer_rng_store_loss_scores_exact=True,fixture_path=str(root),scope='Short technical fixture; full logged C400009 parity still gates new logged B lives')
 C.save(C.R/'E14S_CHECKS.json',result);print(json.dumps(result))
if __name__=='__main__':main()
