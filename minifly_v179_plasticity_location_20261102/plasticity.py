"""H/T learning lesion only: same source forward/teaching; AdamW skips None grads."""
import learner as M
import util as U
T=M.torch

def hidden_named(net):return [(n,p) for n,p in net.named_parameters() if n in ('W','b') or n.startswith('top.first')]
def hidden_digest(net,opt):
 import hashlib
 h=hashlib.sha256()
 for n,p in hidden_named(net):
  h.update(n.encode());h.update(p.detach().numpy().tobytes())
  for k,v in sorted(opt.state.get(p,{}).items()):h.update(k.encode());h.update(v.detach().numpy().tobytes())
 return h.hexdigest()

def step(net,opt,idx,y,freeze=False):
 net.train();opt.zero_grad();M.L.local_step(net,M.FH,M.FT,idx,y);discarded={}
 if freeze:
  for n,p in hidden_named(net):discarded[n]=float(p.grad.square().sum());p.grad=None
 opt.step()
 if not all(T.isfinite(p).all() for p in net.parameters()):raise RuntimeError('NONFINITE_UPDATE')
 return discarded
