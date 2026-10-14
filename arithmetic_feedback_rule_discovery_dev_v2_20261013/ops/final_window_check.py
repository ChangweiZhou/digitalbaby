import sys,json
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
import test_core as t
from core import Core,Config,BEGIN,REPLY,EOS
c=Core(4,Config(width=3,embedding=2,window=4,gradient_mode='final_window'))
c.V[:]=np.random.default_rng(9).normal(0,.1,c.V.shape)
tokens=[BEGIN]+list(range(9))+[REPLY,51,52,EOS];mask=[v not in (BEGIN,REPLY) for v in tokens]
h=np.zeros(3);anchors=[h.copy()]
for token in tokens:
 if token!=EOS:h=t.step(c.theta,h,token)
 anchors.append(h.copy())
start=len(tokens)-1-c.config.window
original=t.objective
def retained_objective(c,tokens,permissions,ignored=None):
 total=0.
 for i,(token,allowed) in enumerate(zip(tokens,permissions)):
  if allowed and i>start:
   h=anchors[start].copy()
   for tok in tokens[start:i]:h=t.step(c.theta,h,tok)
   u=np.concatenate((h,[1.]));u/=np.linalg.norm(u);z=u@c.V
   total+=np.logaddexp.reduce(z)-z[token]
 return total/sum(permissions)
t.objective=retained_objective
error=t.Tests().numeric(c,tokens,mask,keys=list(c.theta))
t.objective=original
assert c.permitted==sum(mask) and len(c.tape)==4 and not c.streaming
(root/'cycles/cycle3/FINAL_WINDOW_GRADIENT.json').write_text(json.dumps({'accepted':True,'max_absolute_numeric_error':error,'encoder_horizon':4,'all_target_denominator':sum(mask),'scientific_jobs':0},indent=2)+'\n')
print('final-window gradient passed',error)
