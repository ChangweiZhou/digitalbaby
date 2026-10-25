"""Evaluator-only world generator; never imported by kernel.py."""
import hashlib
import numpy as np
def seed(world,sub):return int.from_bytes(hashlib.sha256(f'V147|{world}|{sub}'.encode()).digest()[:8],'big')
def world(name):
 assert name.startswith(('V147-D','V147-TECH')), 'Assessment worlds not authorized in technical exercise'
 sigma=np.random.Generator(np.random.PCG64(seed(name,'assignment'))).choice(256,23,replace=False).tolist();r=np.random.Generator(np.random.PCG64(seed(name,'split')));allpairs=[(a,b) for a in range(23) for b in range(a,23)]
 for attempt in range(10000):
  held=[]
  for c in range(23):
   pool=[p for p in allpairs if sum(p)%23==c];held.extend(pool[i] for i in r.choice(12,3,replace=False))
  train=sorted(set(allpairs)-set(held))
  if {a for p in train for a in p}==set(range(23)) and {sum(p)%23 for p in train}==set(range(23)):return dict(name=name,sigma=sigma,train=train,held=sorted(held),split_attempt=attempt+1)
 raise RuntimeError('split coverage rejected')
def oriented(pp):return [q for a,b in pp for q in ([(a,b)] if a==b else [(a,b),(b,a)])]
def prompt(w,p):return bytes(w['sigma'][x] for x in p)
def answer(w,p):return bytes([w['sigma'][sum(p)%23]])
def shuffle(w):
 r=np.random.Generator(np.random.PCG64(seed(w['name'],'shuffle')));pp=w['train']
 for _ in range(10000):
  out={}
  for diag in [True,False]:
   group=[p for p in pp if (p[0]==p[1])==diag];vals=[answer(w,p) for p in group]
   for p,j in zip(group,r.permutation(len(group))):out[p]=vals[j]
  groups={}
  for p in pp:groups.setdefault(answer(w,p),set()).add(out[p])
  if any(len(v)>1 for v in groups.values()):return out
 raise RuntimeError('shuffle only global relabel')
