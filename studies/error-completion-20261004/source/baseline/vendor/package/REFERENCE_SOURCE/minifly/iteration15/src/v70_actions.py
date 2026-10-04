"""Exact net transport. A coordinate cannot both donate and receive in one event.

The random allocator receives raw writes and the external budget, never slow state.
The mixed-integer feasibility solver is part of the fixed algorithm, not a rescue arm.
"""
from v70_common import *
from scipy.optimize import milp,Bounds,LinearConstraint

def tolerance(m):return CFG['budget_atol']+CFG['budget_rtol']*abs(m)

def exact_partition(w,M,rng):
    """Finite meet-in-the-middle subset feasibility, independently checked in original units.

    A randomly selected feasible left subset and right completion provide a seeded
    partition. This is a search method, not uniform sampling of all feasible subsets.
    """
    n=len(w);order=rng.permutation(n);ww=w[order];split=n//2;W=float(w.sum());tol=tolerance(M)
    def sums(a):
        v=np.zeros(1)
        for x in a:v=np.r_[v,v+x]
        return v
    left=sums(ww[:split]);right=sums(ww[split:]);si=np.argsort(right,kind='stable');rs=right[si]
    pad=np.finfo(float).eps*W*max(n,1)*4
    low=np.searchsorted(rs,M-tol-left-pad,side='left');high=np.searchsorted(rs,W-M+tol-left+pad,side='right')
    candidates=np.flatnonzero(high>low)
    if not len(candidates):return None,'infeasible_exact_subset'
    for li in rng.permutation(candidates):
        ri_order=np.arange(low[li],high[li]);rng.shuffle(ri_order)
        for rix in ri_order:
            ri=int(si[rix]);d0=np.r_[(int(li)&(1<<np.arange(split)))!=0,(ri&(1<<np.arange(n-split)))!=0]
            d=np.zeros(n,bool);d[order]=d0
            if w[d].sum()>=M-tol and w[~d].sum()>=M-tol:return d,'exact_subset'
    return None,'infeasible_exact_subset'

def random_partition(w,M,rng):
    W=float(w.sum());n=len(w);tol=tolerance(M)
    if M<=0:return np.zeros(n,bool),'zero'
    if n<2 or W<2*M-2*tol:return None,'infeasible_capacity'
    def valid(d):return w[d].sum()>=M-tol and w[~d].sum()>=M-tol
    for _ in range(CFG['partition_draws']):
        d=rng.random(n)<.5
        if valid(d):return d,'random_partition'
    if n<=36:return exact_partition(w,M,rng)
    # Binary subset feasibility: M <= sum(w_i d_i) <= W-M. Normalize for solver stability.
    cost=rng.uniform(-1,1,n)
    z=milp(cost,integrality=np.ones(n),bounds=Bounds(np.zeros(n),np.ones(n)),
      constraints=LinearConstraint((w/W)[None,:],M/W,1-M/W),options={'time_limit':CFG['solver_seconds']})
    if z.x is not None:
        d=z.x>.5
        if valid(d):return d,'solver_partition'
    return None,'infeasible_partition' if z.status==2 else 'unresolved_solver'

def waterfill(b,cap,M):
    """Capped equalization; the minimum convex occupancy solution, without a fitted parameter."""
    if M==0:return np.zeros_like(cap)
    assert M<=float(cap.sum())+tolerance(M)
    lo=float(b.min());hi=float((b+cap).max())
    for _ in range(100):
        mid=(lo+hi)/2
        if np.minimum(cap,np.maximum(0,mid-b)).sum()<M:lo=mid
        else:hi=mid
    a=np.minimum(cap,np.maximum(0,(lo+hi)/2-b))
    # Only floating-point residual correction; no outcome-dependent rescaling.
    residual=M-float(a.sum())
    if residual>0:
        for i in np.flatnonzero(cap-a>0):
            q=min(residual,float(cap[i]-a[i]));a[i]+=q;residual-=q
            if residual<=0:break
    elif residual<0:
        for i in np.flatnonzero(a>0):
            q=min(-residual,float(a[i]));a[i]-=q;residual+=q
            if residual>=0:break
    assert abs(float(a.sum())-M)<=tolerance(M)
    return a

def allocate(u,s,groups,mode,rng,budget=None):
    new=u.copy();rows=[];active=np.flatnonzero(u!=0)
    keys=set(budget or {})
    keys.update((int(groups[i]),int(np.sign(u[i]))) for i in active)
    for g,sign in sorted(keys):
        ids=active[(groups[active]==g)&(np.sign(u[active])==sign)];w=abs(u[ids]);c=s[ids]*u[ids]<0 if s is not None else None
        if mode=='SPACE':M=min(float(w[c].sum()),float(w[~c].sum()))
        else:M=float(budget.get((g,sign),0.))
        method='zero';feasible=True;delta=np.zeros(len(ids))
        if M>0:
            if mode=='YBLIND':donor,method=random_partition(w,M,rng)
            else:
                donor=c;method='conflict_partition'
                if min(float(w[donor].sum()),float(w[~donor].sum()))<M-tolerance(M):donor=None;method='infeasible_conflict_capacity'
            if donor is None:feasible=False
            else:
                receiver=~donor;delta[donor]=-M*w[donor]/w[donor].sum()
                if mode=='BALANCED':
                    # Count the base increment as occupied before allocating the extra increment.
                    delta[receiver]=waterfill(abs(s[ids[receiver]]+u[ids[receiver]]),w[receiver],M)
                elif mode=='RANDOM_RECEIVER':
                    order=rng.permutation(np.flatnonzero(receiver));remaining=M
                    for i in order:q=min(remaining,float(w[i]));delta[i]=q;remaining-=q
                else:delta[receiver]=M*w[receiver]/w[receiver].sum()
                new[ids]=u[ids]+sign*delta
        removed=float(np.maximum(-delta,0).sum());added=float(np.maximum(delta,0).sum())
        realized=(new[ids]-u[ids])*sign
        net_removed=float(np.maximum(-realized,0).sum());net_added=float(np.maximum(realized,0).sum())
        before=float(u[ids].sum());after=float(new[ids].sum())
        cap=max(0.,float(np.max(abs(new[ids])-2*w))) if len(ids) else 0.
        sign_error=max(0.,float(np.max(-new[ids]*sign))) if len(ids) else 0.
        err=max(abs(net_removed-M),abs(net_added-M),abs(removed-M),abs(added-M))
        row=dict(group=g,sign=sign,requested=M,removed=net_removed,added=net_added,before=before,after=after,
          dose_error=err,conservation_error=abs(after-before),cap_error=cap,sign_error=sign_error,feasible=feasible,method=method)
        if feasible:
            assert err<=tolerance(M),row
            assert abs(after-before)<=tolerance(abs(before)) and cap<=tolerance(M) and sign_error<=tolerance(M),row
        rows.append(row)
    assert not np.any(new[u==0]!=0)
    return new,rows

class TransportFly(ActionFly):
    def step(self,seconds,pn_activity=None,punishment=0.,budget=None):
        s=self.slow.copy();r=PortFly.step(self,seconds,pn_activity,punishment);u=r['raw_increment'][:,2].copy()
        rng=np.random.default_rng(self.random_seed+1009*int(self.event_count))
        mode=self.action
        if mode.startswith('YBLIND'):mode='YBLIND'
        if mode in ('PARENT','NO_WRITE'):new=u;rows=[]
        else:new,rows=allocate(u,None if mode=='YBLIND' else s,self.route_groups,mode,rng,budget)
        feasible=all(q['feasible'] for q in rows)
        # A failed branch is explicitly terminated by the campaign, never scored as a reduced-dose sham.
        if feasible:self.slow+=(new-u)*np.exp(-seconds/self.kernel['slow_tau'])
        r['raw_increment'][:,2]=new;r['original_slow_increment']=u;r['transport']=rows;r['feasible']=feasible
        r['budget']={(q['group'],q['sign']):q['requested'] for q in rows}
        return r
