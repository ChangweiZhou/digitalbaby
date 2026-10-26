"""Unchanged V154/V152 native learner. V156 evaluator, isolated V157 accounting.

Only the identical initial read of two independent query continuations is reused.
This does not change any event, cue, store, class or native teaching equation.
"""
import sys,time
import numpy as np
import support157 as s
sys.path.insert(0,str(s.ROOT/'minifly_v156_dev_20261025'))
import native156 as base
base.s=s;base.n.parent.COUNTS=s.COUNTS;base.n.parent.TIMES=s.TIMES
n=base.n;C=n.C;EOS=256
def birth():
 m=n.Native('N_2B','P_SLOW');assert not m.homeo and m.fresh_adapt and not m.q.any() and len(m.bank)==16;return m
def read(m):
 s.COUNTS['logical_evaluator_distributions']+=1
 return base.read(m)
def pair(m,row,kind='science'):
 s.COUNTS[kind+'_pairs']+=1;s.budget();p=bytes.fromhex(row['prompt']);answer=bytes.fromhex(row['answer'])
 pre=m.clone();pre._prompt(p);own=pre.clone();tf=pre.clone();initial=read(own);choices=[];ended=False;out=[]
 for j in range(4):
  z,lp,cl=initial if j==0 else read(own);y=int(z.argmax());choices.append(y)
  if y==EOS:ended=True;break
  out.append(y);own.step(y)
 zz=[];ll=[];cc=[];ys=list(answer)+[EOS]
 for j,y in enumerate(ys):
  if j==0:
   z,lp,cl=initial;s.COUNTS['logical_evaluator_distributions']+=1;s.COUNTS['identical_first_read_reuses']+=1
  else:z,lp,cl=read(tf)
  zz.append(z);ll.append(lp);cc.append(cl)
  if y!=EOS:tf.step(y)
 z=np.asarray(zz);lp=np.asarray(ll);masked=z.copy();masked[np.arange(len(ys)),ys]=-np.inf;rival=masked.argmax(1);margin=z[np.arange(len(ys)),ys]-z[np.arange(len(ys)),rival]
 return dict(id=row['id'],output_hex=bytes(out).hex(),EOS=ended,choices=choices,correct=ended and bytes(out)==answer,z=z,lp=lp,clipping=np.asarray(cc),y=np.asarray(ys),rival=rival,margin=margin)
def reference_pair(m,row):
 before=s.COUNTS['evaluator_distributions'];out=base.pair(m,row,'technical');s.COUNTS['logical_evaluator_distributions']+=s.COUNTS['evaluator_distributions']-before;return out
def evaluate(m,rows,tag,kind='science'):
 before=s.digest(m);out=[pair(m,r,kind) for r in rows];assert s.digest(m)==before
 offsets=[0]
 for r in out:offsets.append(offsets[-1]+len(r['y']))
 data={k:np.concatenate([r[k] for r in out]) for k in ['z','lp','clipping','y','rival','margin']}
 data.update(offsets=np.asarray(offsets),correct=np.asarray([r['correct'] for r in out]),choices=np.asarray([r['choices']+[-1]*(4-len(r['choices'])) for r in out],np.int16),EOS=np.asarray([r['EOS'] for r in out]),ids=np.asarray([r['id'] for r in out]))
 path=s.H/'local'/f'{tag}.npz';path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**data)
 receipt=dict(tag=tag,path=str(path.relative_to(s.H)),sha256=s.sha(path),records=len(rows),targets=len(data['y']),correct=int(data['correct'].sum()),OLD_correct=int(data['correct'][:64].sum()) if len(rows)==80 else None,NEW_correct=int(data['correct'][64:].sum()) if len(rows)==80 else None,n=int(m.n),time=float(m.time))
 s.log('PANEL_COMMIT',**receipt);return dict(receipt=receipt,**data)
def summary_pair(r):return {k:v for k,v in r.items() if k not in ['z','lp','clipping','y','rival','margin']}
def own(m,prompt,kind='held'):
 """Policy accepts no label, answer prefix, length, record ID or expected class."""
 s.COUNTS[kind+'_queries']+=1;s.budget();view=m.clone();view._prompt(prompt);out=[];choices=[];eos=False
 for _ in range(4):
  z,lp,cl=read(view);y=int(z.argmax());choices.append(y)
  if y==EOS:eos=True;break
  out.append(y);view.step(y)
 return dict(output_hex=bytes(out).hex(),EOS=eos,choices=choices)
def held(m,prompts,tag):
 before=s.digest(m);rows=[]
 for r in prompts:rows.append(dict(id=r['id'],**own(m,bytes.fromhex(r['prompt']))))
 assert s.digest(m)==before
 # Commit decisions before the external reducer joins any expected answers.
 path=f'local/{tag}_OWN.json';s.save(path,dict(tag=tag,rows=rows));receipt=dict(tag=tag,path=path,sha256=s.sha(s.H/path),records=len(rows));s.log('HELD_DECISIONS_COMMIT',**receipt);return receipt
