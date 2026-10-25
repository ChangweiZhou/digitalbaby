"""Generic local surrogate; no task generator, digit test or arithmetic loss."""
import hashlib,copy
import numpy as np
class Map:
 def __init__(self,R,seed):
  r=np.random.default_rng(seed);self.P=r.normal(0,np.sqrt(np.mean(R**2)),(32,88));self.A=np.zeros((32,88));self.a=np.zeros(32);self.n=0;self.R=R.copy()
 def recall(self,x):
  l=self.A@self.R[x]+self.a;q=np.exp(l-l.max());q/=q.sum();return q/np.linalg.norm(q)
 def observe(self,x,u,total=4160):
  sig=4*(1/4)**min(1,self.n/total);j=int(np.argmin(np.sum((self.P-u)**2,axis=1)));w=np.exp(-(np.arange(32)-j)**2/(2*sig**2));t=w/w.sum();l=self.A@self.R[x]+self.a;q=np.exp(l-l.max());q/=q.sum();eta=.01*(1/16+15/16*min(1,self.n/1024));self.P+=eta*w[:,None]*(u-self.P);self.A+=eta*np.outer(t-q,self.R[x]);self.a+=eta*(t-q);self.n+=1
class Local:
 def __init__(self,seed,meta,Vdrive,side,counts,alpha,rho):
  r=np.random.default_rng(seed);self.W=np.c_[r.normal(0,1/np.sqrt(88),(128,88)),r.normal(0,1/np.sqrt(32),(128,32))];self.b=np.zeros(128);self.V=r.normal(0,1/np.sqrt(128),(88,128));self.G=r.normal(0,1/3,(128,9));self.theta=np.zeros((9,len(side)));self.records=0;self.targets=0;self.alpha=alpha;self.rho=rho;self.drive=Vdrive;self.side=side;self.counts=counts;self.meta=meta
 def code(self,p):
  a=self.W@p+self.b;v=self.V@np.maximum(a,0);raw=np.asarray(self.drive.T@v[self.meta]).ravel();ix=[]
  for s in [0,1]:
   inds=np.flatnonzero(self.side==s);k=self.counts[s];cut=np.partition(raw[inds],len(inds)-k)[len(inds)-k];above=inds[raw[inds]>cut];ties=inds[raw[inds]==cut];ix.extend(np.r_[above,ties[:k-len(above)]])
  return a,np.array(sorted(ix))
 def margins(self,p):
  a,ix=self.code(p);return 2*self.theta[:,ix].sum(axis=1)/np.sqrt(len(ix))
 def observe(self,p,y):
  a,ix=self.code(p);m=2*self.theta[:,ix].sum(axis=1)/np.sqrt(len(ix));target=np.array([1 if y&(1<<j) else -1 for j in range(9)]);e=(np.tanh(m/2)-target)/2;s=1/16+15/16*min(1,self.records/1024);d=(self.G@e)*(a>0);self.theta[:,ix]-=self.alpha*s*e[:,None]/np.sqrt(len(ix));self.W-=.003*s*np.outer(d,p);self.b-=.003*s*d;self.targets+=1
 def rest(self):self.W*=1-self.rho;self.b*=1-self.rho;self.theta*=1-self.rho
 def digest(self):return hashlib.sha256(b''.join(a.tobytes() for a in [self.W,self.b,self.V,self.G,self.theta])+str((self.records,self.targets)).encode()).hexdigest()
