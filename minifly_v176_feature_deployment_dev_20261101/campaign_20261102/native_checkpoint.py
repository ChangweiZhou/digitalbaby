"""Mutable-state snapshots with scoped, strongly held immutable asset references."""
import sys,pickle,hashlib
import runtime as C
sys.path.insert(0,str(C.ROOT/'execution_optimization_20261101'));import optimized_native as O
import numpy as np
class NativeModel(O.Model):
 def __getstate__(self):
  state=dict(self.__dict__);state['_encoding_cache']={};return state
class Store:
 def __init__(self):
  self.path=C.R/'local/native_fixed.pkl';self.assets={}
  if self.path.exists():
   with self.path.open('rb') as f:self.assets=pickle.load(f)
 def register(self,model):
  refs={};changed=False
  def add(obj):
   nonlocal changed
   if id(obj) in refs:return
   key=hashlib.sha256(pickle.dumps(obj,protocol=5)).hexdigest();refs[id(obj)]=(obj,key)
   if key not in self.assets:self.assets[key]=obj;changed=True
  for bank in model.bank:
   for name,v in vars(bank.fly.m).items():
    if name not in ('fast','slow','adapt') and (isinstance(v,np.ndarray) or name=='B'):add(v)
  if model.asset is not None:
   for v in model.asset:add(v)
  add(model.bridge);add(model._bt)
  if changed:
   tmp=self.path.with_suffix('.tmp')
   with tmp.open('wb') as f:pickle.dump(self.assets,f,protocol=5)
   tmp.replace(self.path)
  return refs
 def save(self,model,name):
  refs=self.register(model);path=C.R/'local'/name
  class P(pickle.Pickler):
   def persistent_id(self,obj):
    v=refs.get(id(obj));return ('fixed',v[1]) if v is not None and v[0] is obj else None
  with path.open('wb') as f:P(f,protocol=5).dump(model)
  return path
 def load(self,path):
  assets=self.assets
  class U(pickle.Unpickler):
   def persistent_load(self,ref):assert ref[0]=='fixed';return assets[ref[1]]
  with open(path,'rb') as f:return U(f).load()
