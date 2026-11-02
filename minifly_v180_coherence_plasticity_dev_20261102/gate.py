"""Single source-local forward; record-distinct microgradients; post-Adam gate."""
import util as U
import learner as M
import hashlib,math
T=M.torch;F=M.F

def partition(ids):
 unique=sorted(set(ids),key=lambda v:hashlib.sha256(('V180|micro|'+v).encode()).hexdigest());left=set(unique[::2]);mask=T.tensor([v in left for v in ids]);return mask,dict(group_records=[len(unique[::2]),len(unique[1::2])],reliable=min(len(unique[::2]),len(unique[1::2]))>=2)

def coherence(a,c,w,b,axis,reliable):
 dot=(a[w]*c[w]).sum(axis)+a[b]*c[b];na=(a[w].square().sum(axis)+a[b].square()).sqrt();nb=(c[w].square().sum(axis)+c[b].square()).sqrt();co=(dot/(na*nb+1e-12)).clamp(0,1)
 if not reliable:co.zero_()
 return co

def gradients(net,idx,y,ids):
 captured={}
 hook=net.top.first.register_forward_pre_hook(lambda mod,args:captured.update(h=args[0].detach()))
 try:e,drop,dt,dh=M.L.local_step(net,M.FH.to(net.W.dtype),M.FT.to(net.W.dtype),idx,y)
 finally:hook.remove()
 h=captured['h'];mask,info=partition(ids);mask=mask.to(idx.device);n=len(y);micro=[]
 for sel in (mask,~mask):
  ng=int(sel.sum());g={}
  if ng:
   # A local linear surrogate for the embedding sum: no gradient crosses learned layers.
   ww=T.nn.Parameter(T.zeros_like(net.W));a=F.embedding_bag(idx[sel],ww,mode='sum',padding_idx=M.V.PAD);a.backward(dh[sel].detach()*(n/ng));g['W']=ww.grad.detach();g['b']=dh[sel].detach().sum(0)*(n/ng);g['top.first.weight']=dt[sel].detach().T@h[sel]*(n/ng);g['top.first.bias']=dt[sel].detach().sum(0)*(n/ng)
  else:
   g={name:T.zeros_like(p) for name,p in net.named_parameters() if name in ('W','b') or name.startswith('top.first')}
  micro.append(g)
 n1=int(mask.sum());n2=n-n1;errors={}
 for name,p in net.named_parameters():
  if name in micro[0]:
   rec=(n1*micro[0][name]+n2*micro[1][name])/n;errors[name]=float((rec-p.grad).abs().max())
   if not T.allclose(rec,p.grad,rtol=1e-6,atol=1e-7):raise RuntimeError('SPLIT_GRADIENT_PARITY:'+name)
 gains={};cos={}
 with T.no_grad():
  for layer,w,b,axis in [('H','W','b',0),('T','top.first.weight','top.first.bias',1)]:
   a,c=micro;co=coherence(a,c,w,b,axis,info['reliable'])
   cos[layer]=co;gains[layer]=.25+.75*co
 info.update(group_targets=[n1,n2],split_gradient_max_errors=errors,cosine_mean={k:float(v.mean()) for k,v in cos.items()},gain_mean={k:float(v.mean()) for k,v in gains.items()},gain_min={k:float(v.min()) for k,v in gains.items()},gain_max={k:float(v.max()) for k,v in gains.items()})
 return gains,info,micro,drop

def energy(delta):return {'H':delta['W'].square().sum(0)+delta['b'].square(),'T':delta['top.first.weight'].square().sum(1)+delta['top.first.bias'].square()}
def apply_gains(net,before,delta,gains):
 with T.no_grad():
  for name,p in net.named_parameters():
   if name=='W':g=gains['H'][None,:]
   elif name=='b':g=gains['H']
   elif name=='top.first.weight':g=gains['T'][:,None]
   elif name=='top.first.bias':g=gains['T']
   else:continue
   if not T.all(g==1):p.copy_(before[name]+g*delta[name])

def step(net,opt,idx,y,ids,arm,override=None):
 net.train();opt.zero_grad();g,info,micro,drop=gradients(net,idx,y,ids);before={k:p.detach().clone() for k,p in net.named_parameters()};opt.step();delta={k:p.detach()-before[k] for k,p in net.named_parameters()};e=energy(delta);applied={}
 for layer in ('H','T'):
  if override is not None:a=T.full_like(g[layer],override)
  elif arm=='NORMAL':a=T.ones_like(g[layer])
  elif arm=='FIXED-LOW':a=T.full_like(g[layer],.35)
  elif arm=='CGLP':a=g[layer]
  elif arm=='RMS-SHAM':
   val=T.sqrt((g[layer].square()*e[layer]).sum()/e[layer].sum()) if e[layer].sum()>0 else T.tensor(.25,dtype=g[layer].dtype);a=T.ones_like(g[layer])*val
  else:raise ValueError(arm)
  applied[layer]=a
 info.update(arm=arm,proposed_neuron_energy={k:v.detach().clone() for k,v in e.items()},candidate_gains={k:v.detach().clone() for k,v in g.items()},applied_gains={k:v.detach().clone() for k,v in applied.items()},proposed_squared_norm={k:float(v.sum()) for k,v in e.items()},candidate_squared_norm={k:float((g[k].square()*v).sum()) for k,v in e.items()},applied_squared_norm={k:float((applied[k].square()*v).sum()) for k,v in e.items()})
 apply_gains(net,before,delta,applied)
 if not all(T.isfinite(p).all() for p in net.parameters()):raise RuntimeError('NONFINITE_PARAMETERS')
 return info

def public(info):return {k:v for k,v in info.items() if k not in ('proposed_neuron_energy','candidate_gains','applied_gains')}
