"""V177 fixed byte generator, sparse gated graph and matched optimizer."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import json,time,hashlib,resource
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent;AUTH=json.loads((H/'AUTHORIZATION.json').read_text())
def rng(w,split,purpose):return np.random.default_rng(int.from_bytes(hashlib.sha256(f'V177|{w}|{split}|{purpose}'.encode()).digest()[:16],'big'))
def save(n,v):
 p=H/n;p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');q.replace(p)
def topology():
 rows=[]
 for j in range(128):
  m,k=divmod(j,8);rows.append([m*8+i for i in range(8)]+[((m+o)%16)*8+(k+o+d)%8 for o in (-2,-1,1,2) for d in (0,3)])
 return np.array(rows,np.int64)
SRC=topology()
def init(w,bias):
 g=rng(w,'model','birth');p=np.zeros((128,66));p[:,:32]=g.normal(0,.25,(128,32));p[:,33:65]=g.normal(0,.25,(128,32));p[:,65]=bias
 o=np.zeros((257,129));o[:,:128]=g.normal(0,1/128**.5,(257,128));b=rng(w,'model','feedback').normal(0,1/128**.5,(128,257));eg=np.random.default_rng(int.from_bytes(hashlib.sha256(f'V177|{w}|E'.encode()).digest()[:16],'big'));e=eg.choice(np.array([-.25,.25]),(257,16))
 return p,o,e,b
def data(w,split,n):
 alphabet=rng(w,'shared','alphabet').choice(np.array([v for v in range(256) if v!=10]),8,replace=False);g=rng(w,split,'body');x=np.empty((n,97),np.int64)
 for i in range(n):
  x[i,:8]=g.choice(alphabet,8)
  for t in range(8,96):x[i,t]=x[i,t-8] if g.random()<.875 else g.choice(alphabet)
  x[i,96]=10
 y=np.concatenate((x[:,1:],np.full((n,1),256)),axis=1);return x,y,alphabet
def step(p,o,e,h,x,recurrence=True,cross=True):
 if not recurrence:h=np.zeros_like(h)
 neigh=h[:,SRC].copy()
 if not recurrence:neigh[:]=0
 elif not cross:neigh[:,:,8:]=0
 u=e[x];v=u@p[:,:16].T+(neigh*p[None,:,16:32]).sum(2)+p[:,32];q=u@p[:,33:49].T+(neigh*p[None,:,49:65]).sum(2)+p[:,65];a=np.tanh(v);g=1/(1+np.exp(-q));hn=(1-g)*h+g*a;z=hn@o[:,:128].T+o[:,128];z-=z.max(1,keepdims=True);logp=z-np.log(np.exp(z).sum(1,keepdims=True));return hn,logp,a,g
def evaluate(p,o,e,x,y,rollout=False,recurrence=True,cross=True):
 h=np.zeros((len(x),128));lp=[];states=[]
 for t in range(97):h,l,_,_=step(p,o,e,h,x[:,t],recurrence,cross);lp.append(l);states.append(h.copy())
 lp=np.stack(lp,axis=1);pred=lp.argmax(2);nll=-np.take_along_axis(lp,y[:,:,None],2)[:,:,0];out=dict(lag_bits=float(nll[:,7:95].mean()/np.log(2)),lag_accuracy=float((pred[:,7:95]==y[:,7:95]).mean()),initial_bits=float(nll[:,:7].mean()/np.log(2)),newline_accuracy=float((pred[:,95]==10).mean()),EOS_accuracy=float((pred[:,96]==256).mean()),early_EOS=float((pred[:,:95]==256).mean()),evaluated_targets=int(y.size))
 if rollout:
  hits=np.zeros(8);exact=early=0
  for t in range(7,88,8):
   hh=states[t].copy();guess=pred[:,t].copy();good=np.ones(len(x),bool);alive=np.ones(len(x),bool)
   for k in range(8):
    correct=alive&(guess==x[:,t+1+k]);hits[k]+=correct.sum();good&=correct;early+=int(np.sum(alive&(guess==256)));alive&=guess!=256
    if k<7:hh,ll,_,_=step(p,o,e,hh,guess,recurrence,cross);guess=ll.argmax(1)
   exact+=good.sum()
  out.update(rollouts=len(x)*11,rollout_step_accuracy=(hits/(len(x)*11)).tolist(),exact_eight=float(exact/(len(x)*11)),rollout_premature_EOS=early)
 return out
class Adam:
 def __init__(self,p,o):self.params=[p,o];self.m=[np.zeros_like(p),np.zeros_like(o)];self.v=[np.zeros_like(p),np.zeros_like(o)];self.t=0
 def update(self,g,n,targets,lr):
  self.t+=1;w=float((1/16+15/16*np.minimum(np.arange(n,n+targets)/1024,1)).mean());clip=[];norms=[]
  for j,(p,d) in enumerate(zip(self.params,g)):
   d=d/targets;norm=np.linalg.norm(d,axis=1,keepdims=True);clip.append(float((norm>1).mean()));d=d/np.maximum(norm,1);self.m[j]=.9*self.m[j]+.1*d;self.v[j]=.999*self.v[j]+.001*d*d;step=(self.m[j]/(1-.9**self.t))/(np.sqrt(self.v[j]/(1-.999**self.t))+1e-8);delta=lr*w*step;p*=1-lr*w*1e-4;p-=delta;assert np.isfinite(p).all();norms.append(float(np.linalg.norm(delta)))
  return dict(warmup=w,clipped_rows=clip,applied_norms=norms)
def check():
 if time.time()-AUTH['epoch']>AUTH['wall_cap']:raise RuntimeError('WALL_CAP')
 if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>AUTH['rss_cap']:raise RuntimeError('RSS_CAP')
 if float(os.environ.get('V177_PRIOR_CPU','30'))+resource.getrusage(resource.RUSAGE_SELF).ru_utime+resource.getrusage(resource.RUSAGE_SELF).ru_stime>AUTH['cpu_cap']:raise RuntimeError('CPU_CAP')
def charge(records,targets):
 p=H/'COUNTS.json';r=json.loads(p.read_text()) if p.exists() else dict(training_records=24,training_targets=2328)
 r['training_records']+=records;r['training_targets']+=targets;save('COUNTS.json',r)
 if r['training_records']>5000:raise RuntimeError('RECORD_CAP')
 check()
