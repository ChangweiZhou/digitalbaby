"""Receiver-only controls and optional unsigned-write bookkeeping; no hidden cue identity."""
from v71_common import *

class HistoryFly(PortFly):
    def __init__(self,path):
        super().__init__(path);self.H=np.zeros_like(self.slow)
        _,self.groups=np.unique(np.column_stack((self.kc_side,self.T[:,2:])),axis=0,return_inverse=True)
    def clone(self,reset_plastic=False):
        m=super().clone(reset_plastic);m.H=self.H.copy()
        if reset_plastic:m.H[:]=0
        return m
    def step(self,dt,pn_activity=None,punishment=0.,space=False):
        s=self.slow.copy();r=PortFly.step(self,dt,pn_activity,punishment);u=r['raw_increment'][:,2].copy()
        if space:
            plan=transport_plan(u,s,self.groups);new=allocate(plan,'PROP',1.,s,self.H)[0]
            self.slow+=(new-u)*np.exp(-dt/self.kernel['slow_tau']);r['raw_increment'][:,2]=new
        # Parent semantics: raw update is applied BEFORE decay.
        self.H=(self.H+abs(r['raw_increment'][:,2]))*np.exp(-dt/self.kernel['slow_tau'])
        assert np.min(self.H-abs(self.slow))>=-1e-8
        return r

def transport_plan(u,s,groups):
    out=[];donor=np.zeros_like(u);active=np.flatnonzero(u!=0)
    for g in np.unique(groups[active]):
        for sign in (-1,1):
            ids=active[(groups[active]==g)&(np.sign(u[active])==sign)];w=abs(u[ids]);c=s[ids]*u[ids]<0
            C=float(w[c].sum());R=float(w[~c].sum());M=min(C,R)
            if M<=0:continue
            dd=ids[c];rr=ids[~c];donor[dd]=sign*M*abs(u[dd])/C
            out.append(dict(group=int(g),sign=sign,donors=dd,receivers=rr,M=M,capacity=R))
    return dict(u=u.copy(),donor=donor,groups=out)

def allocate(plan,arm,fraction,s,H,seed=0):
    u=plan['u'];new=u-fraction*plan['donor'];rows=[];rng=np.random.default_rng(seed)
    for q in plan['groups']:
        ids=q['receivers'];cap=abs(u[ids]);M=fraction*q['M'];sign=q['sign']
        if arm=='PROP':a=M*cap/cap.sum()
        elif arm=='BALANCED':a=waterfill(abs(s[ids]+u[ids]),cap,M)
        elif arm=='OPP' or arm.startswith('SHUFFLE_L'):
            liability=np.maximum((H[ids]-sign*s[ids])/2,0)
            if arm.startswith('SHUFFLE_L'):liability=rng.permutation(liability)
            # A capped priority allocator; L+a is NOT an update to historical opposing mass.
            a=waterfill(liability,cap,M)
        elif arm.startswith('RANDOM'):
            a=np.zeros_like(cap);remaining=M
            for i in rng.permutation(len(ids)):
                z=min(remaining,float(cap[i]));a[i]=z;remaining-=z
        else:raise ValueError(arm)
        new[ids]+=sign*a;rows.append(dict(group=q['group'],sign=sign,M=M,added=float(a.sum()),cap_excess=max(0.,float((a-cap).max())),
          receivers=len(ids),free_mass=float(cap.sum()-M),saturated=bool(abs(cap.sum()-M)<1e-10)))
    verify_transport(plan,new,fraction)
    return new,rows

def verify_transport(plan,new,fraction):
    u=plan['u'];tol=CFG['budget_atol']+CFG['budget_rtol']*float(abs(u).sum())
    assert not np.any(new[u==0]!=0)
    assert np.max(abs(new)-2*abs(u))<=tol and np.min(new*np.sign(u))>=-tol
    for q in plan['groups']:
        d=q['donors'];r=q['receivers'];sign=q['sign'];M=fraction*q['M']
        assert abs(float(((u[d]-new[d])*sign).sum())-M)<=tol
        assert abs(float(((new[r]-u[r])*sign).sum())-M)<=tol

def intervene(parent_after,pre,raw,new,dt):
    m=parent_after.clone();e=np.exp(-dt/m.kernel['slow_tau']);m.slow+=(new-raw)*e;m.H=(pre.H+abs(new))*e
    return m
