import os
os.environ.update(EXP45_HIDDEN='512',E8_WD='.25',E8_SCHEDULE='late_family',E8_REPLAY_SAMPLING='uniform',E8_RUN_PREFIX='e16w2_tech_')
import sys,hashlib,json,runpy
from pathlib import Path
role=sys.argv[1]
if role=="original":
 import exp8_life as E
else:
 import exp8_e16w2 as E
import e14_fast_inference as F
import e14s_campaign as H
root=Path('exp45/e16w2_technical').resolve();root.mkdir(exist_ok=True)
p=root/'e8_dev_73129.pt'
if not p.exists():p.symlink_to(E.OUT/'e8_dev_73129.pt')
E.OUT=root;E.STAGE_UPDATES=2;E.COMPOSE_UPDATES=4;E.BUFFER_CAP=32;E.RUN_PREFIX=f'e16w2_{role}_';E.X.respond=F.respond
score=E.X.score;E.X.score=lambda m,it,store=None:score(m,it[:8],store)
original=E.X.update;trace=[]
def digest(x):
 if E.T.is_tensor(x):return hashlib.sha256(x.detach().numpy().tobytes()).hexdigest()
 if isinstance(x,dict):return {str(k):digest(v) for k,v in x.items()}
 if isinstance(x,(tuple,list)):return [digest(v) for v in x]
 return x
def update(m,opt,it,local,store=None):
 v=original(m,opt,it,local,store);trace.append(digest([m.state_dict(),opt.state_dict(),E.T.get_rng_state(),v,store.mem if store else None,store.order if store else None]));return v
E.X.update=update
if role in ('baseline','original'):E.life('C',73129)
else:
 if role=='C_share':E.REPLAY_SAMPLING='share'
 sys.argv=['e16w2_worker.py',role,'73129'];runpy.run_path('e16w2_worker.py',run_name='__main__')
(root/f'{role}_trajectory.json').write_text(json.dumps(trace))
