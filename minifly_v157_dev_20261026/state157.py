"""Serialization qualification checks operative values, not pickle memo/alias bytes."""
import numpy as np
from scipy import sparse
def equal(a,b,visited=None,path='root'):
 visited=set() if visited is None else visited
 if (id(a),id(b)) in visited:return []
 visited.add((id(a),id(b)))
 if type(a) is not type(b):return [path+':type']
 if isinstance(a,np.ndarray):return [] if a.shape==b.shape and a.dtype==b.dtype and np.array_equal(a,b) else [path+':array']
 if sparse.issparse(a):return [] if a.shape==b.shape and a.dtype==b.dtype and (a!=b).nnz==0 else [path+':sparse']
 if isinstance(a,dict):
  if a.keys()!=b.keys():return [path+':keys']
  return [d for k in a for d in equal(a[k],b[k],visited,path+'.'+str(k))]
 if isinstance(a,(tuple,list)):
  if len(a)!=len(b):return [path+':length']
  return [d for i,(x,y) in enumerate(zip(a,b)) for d in equal(x,y,visited,path+'.'+str(i))]
 if hasattr(a,'__dict__'):return equal(a.__dict__,b.__dict__,visited,path+'.__dict__')
 try:return [] if bool(a==b) else [path+':value']
 except Exception:return [path+':unsupported']
def independent_mutable_banks(m):
 for k in ['fast','slow','adapt']:
  arrays=[getattr(x.fly.m,k) for x in m.bank]
  assert all(not np.shares_memory(a,b) for i,a in enumerate(arrays) for b in arrays[i+1:])
