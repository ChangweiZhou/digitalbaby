"""Literal source local updates; fixed 257 classes, state-preserving orchestration."""
import util as U
import sys,importlib.util,types,math,copy
sys.path.insert(0,str(U.ROOT/'minifly_v178_recurrent_failure_localization_dev_20261102/local/compat'))
import numpy as np
import torch
import torch.nn.functional as F
torch.set_num_threads(1)
SOURCE=U.H/'local/source'
def imp(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
V=imp('v179_original_v163',SOURCE/'minifly_v163_sslcola_dev_v1_20261026/v163_lib.py')
# Only source-local required dependencies are injected; no historical training loop or class compression is run.
common=types.ModuleType('v167_common');common.V=V;common.SET={'local':{'F_T_seed':22}};sys.modules['v167_common']=common
common165=types.ModuleType('v165_common');common165.V=V;common165.SET={'dfa':{'F_seed':21}};sys.modules['v165_common']=common165
D=imp('v165_dfa',SOURCE/'minifly_v165_dfa_dev_v1_20261026/v165_dfa.py');L=imp('v179_original_v167',SOURCE/'minifly_v167_local_dev_v1_20261026/v167_local.py')
FH=D.feedback(257,512);FT=L.feedback(257,2048,22)
def new():
 return L.make(types.SimpleNamespace(K=257))
def optimizer(net):return torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001,betas=(.9,.999),eps=1e-8)
def state_hash(net,opt=None,rng=None):
 import hashlib
 h=hashlib.sha256()
 for k,v in net.state_dict().items():h.update(k.encode());h.update(v.detach().numpy().tobytes())
 if opt is not None:
  for i,p in enumerate(net.parameters()):
   for k,v in sorted(opt.state.get(p,{}).items()):h.update(f'{i}:{k}'.encode());h.update(v.detach().numpy().tobytes() if torch.is_tensor(v) else str(v).encode())
 if rng is not None:h.update(rng.numpy().tobytes())
 return h.hexdigest()
def snap(net,opt):return dict(model=copy.deepcopy(net.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),rng=torch.get_rng_state().clone())
def restore(st,reset=False):
 net=new();net.load_state_dict(st['model']);opt=optimizer(net)
 if not reset:opt.load_state_dict(copy.deepcopy(st['optimizer']))
 torch.set_rng_state(st['rng'].clone());return net,opt

def step(net,opt,idx,y):
 net.train();opt.zero_grad();L.local_step(net,FH,FT,idx,y);opt.step()
 if not all(torch.isfinite(p).all() for p in net.parameters()):raise RuntimeError('NONFINITE_UPDATE')

def forward(net,idx):return net(idx)
def cached(n):
 z=np.load(U.H/'local'/f'{n}.npz');return {k:(torch.from_numpy(z[k].astype(np.int64)) if k=='idx' else z[k].copy()) for k in z.files}
