"""Pinned runtime and declared platform/path adapters; no learning-state recovery."""
import os,sys,importlib.util,hashlib,json,platform,ctypes
from pathlib import Path
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
H=Path(__file__).resolve().parent;ROOT=H.parent
sys.path.insert(0,'/private/tmp/minifly_v183_runtime')
import numpy as np,torch as T,scipy
assert np.__version__=='2.4.6' and T.__version__.split('+')[0]=='2.14.0'
T.set_num_threads(1)
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def module(name,path,changes=()):
 text=path.read_text()
 for old,new in changes:
  assert old in text,(name,old);text=text.replace(old,new)
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;exec(compile(text,str(path),'exec'),m.__dict__);return m
P=module('parent_learner',H/'assets/v182/parent_learner.py')
sensor=module('sensor',H/'assets/v182/sensor.py',[("ctypes.CDLL('libm.so.6')","ctypes.CDLL(None)")] if sys.platform=='darwin' else [])
sensor.P=ROOT/'minifly_a_v3/package';sensor.ROOT=ROOT
D=module('source_data',H/'assets/v182/data.py',[("minifly_v182_shallow_local_dev_20261102/assets/corpus.py","minifly_v176_feature_deployment_dev_20261101/assets/source/minifly_v148_pretrain_probe_dev_20261025/corpus.py"),("minifly_v182_shallow_local_dev_20261102/assets/corpora","minifly_v180_coherence_plasticity_dev_20261102/local/corpora")])
def statehash(net):
 h=hashlib.sha256()
 for d in (net.p,net.m,net.v):
  for v in d.values():h.update(v.numpy().tobytes())
 h.update(net.rng.get_state().numpy().tobytes());h.update(str(net.clock).encode())
 for name in ('credit_rng','shuffle_rng'):
  if hasattr(net,name):h.update(getattr(net,name).get_state().numpy().tobytes())
 return h.hexdigest()
