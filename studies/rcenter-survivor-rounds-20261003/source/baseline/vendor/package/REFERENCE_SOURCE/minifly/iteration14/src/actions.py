"""Current-write interventions only; frozen parent computes teaching and decay."""
from v69_common import *
import copy

class ActionFly(PortFly):
    @classmethod
    def from_parent(cls,parent,action='PARENT',random_seed=0):
        m=cls.__new__(cls);m.__dict__=parent.__dict__.copy()
        for k in ('fast','slow','adapt'):setattr(m,k,getattr(parent,k).copy())
        m.action=action;m.arm='NO_WRITE' if action=='NO_WRITE' else 'INTACT';m.random_seed=random_seed
        _,m.route_groups=np.unique(np.column_stack((m.kc_side,m.T[:,2:])),axis=0,return_inverse=True)
        return m
    def step(self,seconds,pn_activity=None,punishment=0.):
        s=self.slow.copy();r=super().step(seconds,pn_activity,punishment);raw=r['raw_increment'];u=raw[:,2].copy();v=raw[:,1].copy()
        new=u.copy();add=np.zeros_like(u);conflict=s*u<0;active=np.flatnonzero(u!=0);moved=want=matched=0.
        if self.action=='DROP':new[conflict]=0
        elif self.action=='FAST':new[conflict]=0;add[conflict]=u[conflict]
        elif self.action in ('U_DROP','U_FAST','SPACE') or self.action.startswith('BLIND_SPACE'):
            for group in np.unique(self.route_groups[active]):
                gg=active[self.route_groups[active]==group]
                for sign in (-1,1):
                    ix=gg[np.sign(u[gg])==sign]
                    if not len(ix):continue
                    w=np.abs(u[ix]);c=conflict[ix];C=float(w[c].sum());W=float(w.sum())
                    if not C:continue
                    if self.action in ('U_DROP','U_FAST'):
                        fraction=C/W;new[ix]=u[ix]*(1-fraction)
                        if self.action=='U_FAST':add[ix]=u[ix]*fraction
                        continue
                    R=float(w[~c].sum());M=min(C,R);want+=M
                    if M<=0:continue
                    if self.action=='SPACE':donor=c;receiver=~c;matched+=M
                    else:
                        draw=int(self.action.rsplit('_',1)[1]);rng=np.random.default_rng(self.random_seed+1000003*draw+1009*int(self.event_count)+37*int(group)+(sign+1))
                        donor=receiver=None
                        for attempt in range(16):
                            perm=rng.permutation(len(ix));split=int(np.searchsorted(np.cumsum(w[perm]),M-1e-12))+1
                            dd=np.zeros(len(ix),bool);dd[perm[:split]]=True
                            if w[dd].sum()>=M-1e-10 and w[~dd].sum()>=M-1e-10:
                                donor=dd;receiver=~dd;matched+=M;break
                        if donor is None:continue
                    d=w[donor].sum();t=w[receiver].sum();new[ix[donor]]-=sign*M*w[donor]/d;new[ix[receiver]]+=sign*M*w[receiver]/t;moved+=M
        ef=np.exp(-seconds/self.kernel['fast_tau']);es=np.exp(-seconds/self.kernel['slow_tau'])
        self.fast[:,1]+=add*ef;self.slow+=(new-u)*es
        r['raw_increment'][:,1]=v+add;r['raw_increment'][:,2]=new
        r['budget']=np.array([np.abs(raw[:,0]).sum(),np.abs(v+add).sum(),np.abs(new).sum(),np.abs(u).sum(),np.abs(v).sum(),moved,want,matched])
        r['original_slow_increment']=u;r['original_alpha_fast_increment']=v
        return r
