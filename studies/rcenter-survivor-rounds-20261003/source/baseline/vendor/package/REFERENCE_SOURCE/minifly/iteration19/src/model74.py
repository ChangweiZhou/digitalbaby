"""Pooled parent dynamics plus declared sensitivity and local contraction factors."""
from common74 import *
from model73 import advance73

def contraction(s,u,f,dt,rho,rule,plastic=True):
    if not plastic or rho==0:return np.zeros_like(s)
    gate=s*u<0
    if rule=='FAST_CONTRACT':gate &= f*u>0
    elif rule!='CONTRACT':raise ValueError(rule)
    # rho is a 30-second hazard, not a free per-function-call erasure.
    fraction=-np.expm1(np.log1p(-rho)*dt/30.)
    return -fraction*gate*s

class Fly(ParentFly):
    def __init__(self,settings=None):
        super().__init__('REF');self.settings=dict(settings or {})
        s=self.settings;self.Q=self.Q.copy()*s.get('q_gain',1.)
        self.ta*=s.get('adapt_tau',1.);self.fw0*=s.get('eta',1.);self.fwd*=s.get('eta',1.)
        self.kernel=self.kernel.copy()
        if 'fraction' in s:self.kernel['fraction']=s['fraction']
        self.kernel['fast_tau']*=s.get('fast_tau',1.)
        self.kernel['slow_tau']*=s.get('slow_tau',1.)
        self.feedback=int(s.get('feedback',0.)!=0);self.strength=s.get('feedback',0.)
        self.diagnostics=[]

    def encode(self,p):
        p=np.asarray(p,float)
        if p.shape!=(88,) or not np.isfinite(p).all() or (p<0).any():raise ValueError('88 finite nonnegative channels required')
        sparsity=self.settings.get('sparsity',.05)
        if sparsity==.05:x=super().encode(p)
        else:
            raw=np.asarray(self.B.T@(p[self.pn_type_index])).ravel();x=np.zeros(len(raw))
            for side in [0,1]:
                ix=np.flatnonzero((self.kc_side==side)&(raw>0));n=min(len(ix),int(np.ceil(sparsity*np.sum(self.kc_side==side))))
                order=np.lexsort((ix,-raw[ix]));x[ix[order[:n]]]=1.
        return x*self.settings.get('activity',1.)

    def clone(self,reset_plastic=False):
        m=super().clone(reset_plastic);m.settings=self.settings.copy();m.diagnostics=[];return m

    def step(self,seconds,pn_activity=None,punishment=0.,extra=None):
        if not np.isfinite(seconds) or seconds<0 or not np.isfinite(punishment):raise ValueError('Invalid event')
        p=np.zeros(88) if pn_activity is None else np.asarray(pn_activity,float)
        x=self.encode(p);k=self.kernel;sf=self.slow.copy();ff=self.fast[:,1].copy()
        r=advance73(x,float(seconds),float(punishment),self.Q,self.T,self.F,self.MM,self.k0,self.fw0,self.fwd,self.ta,self.tg,k['fast_tau'],k['slow_tau'],k['fraction'],
            self.fast,self.slow,self.adapt,0,self.feedback,self.strength,self.Q4,self.weights,self.side,self.ei,self.ej,self.eq,self.share,self.ET,self.R,self.R2,self.mode!='no_learning',1.)
        raw=r[4];v=contraction(sf,raw,ff,seconds,self.settings.get('rho',0.),self.settings.get('rule','CONTRACT'),self.mode!='no_learning')
        if extra is not None:
            if self.mode=='no_learning':raise ValueError('No-write control cannot receive extra write')
            v=np.asarray(extra,float)
            if v.shape!=sf.shape or not np.isfinite(v).all():raise ValueError('Invalid evaluator-only extra')
        self.slow+=v*np.exp(-seconds/k['slow_tau'])
        self.elapsed=np.float64(self.elapsed+seconds);self.event_count+=np.uint64(1);self.presentation_count+=np.uint64(np.any(p>0))
        if self.settings.get('rho',0.) and np.any(x):
            active=(x>0)&np.any(self.T[:,2:]!=0,axis=1);den=max(1,int(active.sum()));mass=float(abs(v).sum());u=float(abs(raw).sum())
            self.diagnostics.append(dict(dt=float(seconds),conflict=float(np.sum(active&(sf*raw<0))/den),
                corroborated=float(np.sum(active&(ff*raw>0))/den),contracted=float(np.sum(active&(v!=0))/den),
                contraction_L1=mass,raw_slow_L1=u,ratio=mass/u if u else 0.,max_contraction=float(abs(v).max()),
                sign_crossings=int(np.sum(sf*self.slow<0))))
        return dict(rates=r[0],payload=r[1],dx=r[2],rawfast=r[3],rawslow=raw,extra=v,pre_slow=sf,pre_fast=ff)

    @classmethod
    def restore(cls,path):
        with np.load(path) as z:
            s=json.loads(str(z['settings'])) if 'settings' in z else {}
            m=cls(s)
            if str(z['model_sha'])!=m.model_sha:raise ValueError('Wrong parent')
            for k in ['fast','slow','adapt']:setattr(m,k,z[k].copy())
            for k in ['elapsed','event_count','presentation_count']:setattr(m,k,z[k][()])
        return m

    def save(self,path):
        np.savez_compressed(path,fast=self.fast,slow=self.slow,adapt=self.adapt,elapsed=self.elapsed,event_count=self.event_count,
            presentation_count=self.presentation_count,model_sha=self.model_sha,settings=json.dumps(self.settings,sort_keys=True))

def probe(m,codes,roles,reader,h=0,unclip=False):
    n=m.clone()
    if h:n.mode='no_learning';n.step(h)
    end=1-(1-n.adapt*np.exp(-.25*codes))*np.exp(-5/n.ta);dx=.5*(n.adapt+end)*codes
    if unclip:
        gamma=np.clip((dx*(n.k0[3]+n.fast[:,0]))@n.Q[:,:2]+35.2,0,71.66)-35.2
        bg=np.clip((dx*n.k0[3])@n.Q[:,:2]+35.2,0,71.66)-35.2
        a=(dx*(n.k0[5]+n.fast[:,1]+n.slow))@n.Q[:,2:]+gamma@n.MM
        base=(dx*n.k0[5])@n.Q[:,2:]+bg@n.MM
    else:a=n.expression(dx);base=n.expression(dx,True)
    obs=reader.predict(dx);d=2*np.asarray(roles)-1
    actual=d*((a-base).mean(1)[::2]-(a-base).mean(1)[1::2]);rd=d*((a-obs).mean(1)[::2]-(a-obs).mean(1)[1::2])
    return actual,rd,(actual>=1)&(rd>=reader.pair_threshold)
