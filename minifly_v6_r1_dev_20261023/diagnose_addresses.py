"""Posthoc E0 localization on OLD prompt bytes only. No native events or fits."""
from core import *
import ops
if True:
 rows=[]
 for seed in (202610236101,202610236102):
  d=json.loads((P/f'manifests/{seed}.json').read_text());f=Fly(seed);born=[f.c.Wv.copy(),f.c.Wg.copy()]
  ck=P/f'checkpoints/{seed}_LEARN_FINAL.npz'
  if not ck.exists():continue
  with np.load(ck) as z:final=[z['c_Wv'].copy(),z['c_Wg'].copy()]
  for name,weights in [('BIRTH',born),('FINAL',final)]:
   f.c.Wv=weights[0].copy();f.c.Wg=weights[1].copy();hs=[];xs=[]
   for i,r in enumerate(d['old'][:8]):
    hh=[];xx=[]
    for tape in range(3):
     f.c.rng=np.random.default_rng(seed+200000+tape);f.c.reset();f.c.step(256,False)
     for b in r['prompt']:f.c.step(b,False)
     f.c.step(257,False);hh.append(f.c.h.copy());xx.append(f.code().copy())
    hs.append(hh);xs.append(xx)
   hs=np.array(hs);xs=np.array(xs)
   def jac(a,b):return float(np.minimum(a,b).sum()/max(1,np.maximum(a,b).sum()))
   noise=float(np.mean([np.linalg.norm(hs[i,0]-hs[i,t]) for i in range(8) for t in (1,2)]));signal=float(np.mean([np.linalg.norm(hs[i,0]-hs[j,0]) for i in range(8) for j in range(i+1,8)]))
   rows.append(dict(seed=seed,stage=name,OLD_prompts_only=True,same_prompt_different_tapes_h_distance=noise,different_prompt_same_tape_h_distance=signal,paired_prompt_difference_over_tape_difference=signal/max(noise,1e-30),same_prompt_different_tapes_KC_Jaccard=float(np.mean([jac(xs[i,0],xs[i,t]) for i in range(8) for t in (1,2)])),different_prompt_same_tape_KC_Jaccard=float(np.mean([jac(xs[i,0],xs[j,0]) for i in range(8) for j in range(i+1,8)])),Wv_payload_column_Frobenius=float(np.linalg.norm(weights[0][:,:256])),Wg_payload_column_Frobenius=float(np.linalg.norm(weights[1][:,:256])),code_activity_min=int(xs.sum(2).min()),code_activity_max=int(xs.sum(2).max())))
 ops.save('results/ADDRESS_DIAGNOSTICS.json',dict(status='POSTHOC_E0_NOT_A_SELECTION_GATE',native_records=0,native_events=0,fits=0,rows=rows,note='Inference from saved/birth recurrent weights, training prompts only, common exogenous tapes to separate byte sensitivity from stochastic state variation; no arithmetic labels, held prompts, native writes or new science trajectories. This is a small distributional diagnostic, not proof of a cause or a new behavioral result.'))

