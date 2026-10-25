"""Evaluator-side token distribution fixtures. No arithmetic/linguistic rule."""
import hashlib
import numpy as np

def seed(fixture,n,stream):return int.from_bytes(hashlib.sha256(f'MINIFLY-OUTPUT-V1|{fixture}|{n}|{stream}'.encode()).digest()[:8],'big')
def rng(fixture,n,stream):return np.random.Generator(np.random.PCG64(seed(fixture,n,stream)))
def make(fixture,n):
 r=rng(fixture,n,'assignment');K=4 if fixture=='four_context_full_support' else 1;q=np.full((K,257),1/257)
 context_bytes=r.choice(256,K,replace=False)
 if fixture=='skewed_full_support':q[:]=.5/257;q[0,int(r.integers(257))]+=.5
 if fixture=='four_context_full_support':
  selected=r.choice(257,16,replace=False).reshape(4,4);q[:]=.1/257
  for i,ss in enumerate(selected):q[i,ss]+=.9/4
 def stream(count,tag):
  context=np.tile(np.arange(K),count//K);rng(fixture,n,tag+'_order').shuffle(context);u=rng(fixture,n,tag+'_targets').random(count);targets=np.array([np.searchsorted(np.cumsum(q[c]),x,side='right') for c,x in zip(context,u)],dtype=int);return context,targets
 tr=stream(4096,'train');va=stream(1024,'validation');return {'name':fixture,'seed':n,'q':q,'context_bytes':context_bytes,'train':tr,'validation':va}
def digest(w):
 return hashlib.sha256(b''.join(np.ascontiguousarray(a).tobytes() for a in [w['q'],w['context_bytes'],*w['train'],*w['validation']])).hexdigest()
