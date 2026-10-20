"""Targeted third-cycle audit: active-input local derivatives and read-only native lesions."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import bootstrap
import numpy as np
from core import Learner,Limits,BC
from checkpoint import load
from data import split
from evaluate import score
from artifacts import HERE,atomic_json,sha
import json

def run():
 limits=Limits(**json.loads((HERE/'PARAMETER_LOCK.json').read_text())['limits'])
 m=Learner(202610208930,plastic=True,limits=limits);rng=np.random.default_rng(202610208931)
 old=rng.normal(0,.1,64);r=np.asarray(BC.R_TABLE[137]);active=int(np.flatnonzero(r)[0]);row=7
 p={k:v.copy() for k,v in m.params.items()}
 v=np.tanh(p['W']@r+p['U']@old+p['b']);g=1/(1+np.exp(-(p['G']@r+p['H']@old+p['c'])))
 errors={};magnitudes={}
 for name in p:
  column=active if name in ('W','G') else 12
  idx=(row,column) if p[name].ndim==2 else (row,)
  pre=(r[column] if name in ('W','G') else old[column]) if len(idx)==2 else 1.
  factor=g*(1-v*v) if name in ('W','U','b') else g*(1-g)*(v-old)
  expected=factor[row]*pre
  def response(delta):
   q={k:value.copy() for k,value in p.items()};q[name][idx]+=delta
   vv=np.tanh(q['W']@r+q['U']@old+q['b']);gg=1/(1+np.exp(-(q['G']@r+q['H']@old+q['c'])))
   return ((1-gg)*old+gg*vv)[row]
  fd=(response(1e-6)-response(-1e-6))/2e-6
  errors[name]=float(abs(fd-expected));magnitudes[name]=float(abs(expected))
  assert errors[name]<1e-8
 assert magnitudes['W']>1e-5 and magnitudes['G']>1e-5
 m.trace_active=True;m._control(True);trace_errors=[]
 for byte in (0,137,255):
  old=m.h.copy();previous={k:e.copy() for k,e in m.eligibility.items()};r=np.asarray(BC.R_TABLE[byte]);p=m.params
  v=np.tanh(p['W']@r+p['U']@old+p['b']);g=1/(1+np.exp(-np.clip(p['G']@r+p['H']@old+p['c'],-40,40)))
  d=g*(1-v*v);s=g*(1-g)*(v-old)
  direct={'W':d[:,None]*r,'U':d[:,None]*old,'b':d,'G':s[:,None]*r,'H':s[:,None]*old,'c':s}
  expected={}
  for key,e in previous.items():
   x=e*((1-g)[:,None] if e.ndim==2 else (1-g))+direct[key]
   if key in ('U','H'):np.fill_diagonal(x,0.)
   expected[key]=np.clip(x,-limits.eligibility_clip,limits.eligibility_clip)
  m._context(byte)
  error=max(float(abs(m.eligibility[k]-expected[k]).max()) for k in expected);trace_errors.append(error);assert error<1e-12
 assert len({row.tobytes() for row in np.asarray(BC.R_TABLE)})==256
 lesions=[]
 for world in (202610209901,202610209902):
  path=HERE/f'cycles/cycle3/jobs/{world}_P-G/W_FINAL.npz'
  original_hash=sha(path);m=load(path);before=m.digest();sets,_=split(world)
  baseline=score(m,sets['OLD']);lesion=m.clone();lesion.native.fast.fill(0.);lesion.native.slow.fill(0.)
  erased=score(lesion,sets['OLD'])
  assert m.digest()==before and sha(path)==original_hash
  assert erased['accuracy']==0.
  lesions.append(dict(world=world,taught_before=baseline['accuracy'],taught_native_erased=erased['accuracy'],
                      checkpoint_unchanged=True,production_learner_unchanged=True))
 result=dict(accepted=True,record_applications=0,scientific_records=0,
  active_receptor_finite_difference_errors=errors,nonzero_derivative_magnitudes=magnitudes,
  explicit_local_eligibility_recurrence_errors=trace_errors,input_identity_injective=True,
  native_state_readonly_lesions=lesions,
  scope='Read-only taught-cue attribution plus mathematical fixtures; no HELD training or parameter selection')
 atomic_json(HERE/'cycles/cycle3/SUPPLEMENTARY_AUDIT.json',result);print(json.dumps(result))
if __name__=='__main__':run()
