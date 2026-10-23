"""External 5801-parameter residual readout. No native/evaluator dependencies."""
import numpy as np
NAMES=('w1','b1','w2','b2')
def initialize(seed):
 r=np.random.default_rng(seed)
 return dict(w1=r.normal(0,1/np.sqrt(171),(171,32)),b1=np.zeros(32),w2=np.zeros((32,9)),b2=np.zeros(9))
def features(D,scale,view):
 q=np.asarray(D)/scale[None,None,:,None]
 if view=='MEAN':q=np.repeat(q.mean(-1,keepdims=True),2,axis=-1)
 B=len(q);x=np.zeros((B,9,171));x[:,:,:162]=q.reshape(B,1,162);x[:,:,162:]=np.eye(9)
 return x,q.mean(-1)
def forward(p,x,base):
 h=np.tanh(x@p['w1']+p['b1']);return h@p['w2']+p['b2']+base,h
def targets(values):
 y=np.zeros((len(values),9,9));w=np.zeros_like(y)
 for i,v in enumerate(values):
  L=len(v)
  for j,t in enumerate(v):y[i,j,:8]=(int(t)>>np.arange(8))&1
  y[i,L,8]=1
  if L:w[i,:L,:8]=.5/(8*L);w[i,:L+1,8]=.5/(L+1)
  else:w[i,0,8]=1
 return y,w
def loss_grad(p,x,base,y,w):
 z,h=forward(p,x,base);B=len(x);loss=float(np.sum(w*(np.logaddexp(0,z)-y*z))/B)
 sig=np.exp(-np.logaddexp(0,-z));g=w*(sig-y)/B
 flat=x.reshape(-1,171);hf=h.reshape(-1,32);gf=g.reshape(-1,9)
 dh=(gf@p['w2'].T)*(1-hf*hf)
 grads=dict(w2=hf.T@gf,b2=gf.sum(0),w1=flat.T@dh,b1=dh.sum(0))
 return loss,grads
def adam(p,g,st,step):
 norm=float(np.sqrt(sum(np.sum(a*a) for a in g.values())));factor=min(1.,1/max(norm,1e-300));lr=.001*min(step/100,1.)
 for k in NAMES:
  v=g[k]*factor;st[k][0]*=.9;st[k][0]+=.1*v;st[k][1]*=.999;st[k][1]+=.001*v*v
  p[k]*=1-lr*.0001;p[k]-=lr*(st[k][0]/(1-.9**step))/(np.sqrt(st[k][1]/(1-.999**step))+1e-8)
 return norm
