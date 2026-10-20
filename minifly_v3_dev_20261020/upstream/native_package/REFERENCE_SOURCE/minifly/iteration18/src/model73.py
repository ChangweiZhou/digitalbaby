"""Shared local update at pooled or anatomically supported partner-pair state.

Strengthened current-output feedback is the candidate. Slow-only feedback is a
privileged diagnostic, never a deployable model or promotion candidate.
"""
from v73_common import *

@njit(cache=True)
def advance73(x,dt,pun,Q,T,F,MM,k0,fw0,fwd,ta,tg,tf,ts,fraction,fast,slow,adapt,
              rep,feedback,gain,Q4,weights,side,ei,ej,eq,share,ET,R,R2,plastic,alpha_scale):
    N=len(x);dx=np.empty(N);agg=np.zeros(4);inp=np.zeros(4);cell=np.zeros(4);cellslow=np.zeros(4);collapsed=np.zeros(N)
    if rep==0:collapsed=slow.copy()
    else:
        for e in range(len(ei)):collapsed[ei[e]]+=share[e]*slow[e]
    for i in range(N):
        end=adapt[i]*np.exp(-.05*dt*x[i]);end=1-(1-end)*np.exp(-dt/ta);dx[i]=.5*(adapt[i]+end)*x[i];adapt[i]=end
        for j in range(4):
            agg[j]+=Q[i,j]*dx[i]
            w=fast[i,0] if j<2 else fast[i,1]+collapsed[i]
            inp[j]+=Q[i,j]*dx[i]*(k0[3 if j<2 else 5]+w)
        if rep==2:
            for j in range(4):cell[j]+=Q4[i,j]*dx[i]*(k0[5]+fast[i,1]);cellslow[j]+=Q4[i,j]*dx[i]*k0[5]
    if rep==2:
        for e in range(len(ei)):
            v=eq[e]*dx[ei[e]]*slow[e];cell[ej[e]]+=v;cellslow[ej[e]]+=v
    rm=np.zeros(4);rs=np.zeros(4);rmcell=np.zeros(4)
    for j in range(2):rm[j]=min(max(inp[j]+35.2,0),71.66)-35.2
    if rep<2:
        for j in range(2):
            a=inp[j+2];s=0.
            for i in range(N):s+=Q[i,j+2]*dx[i]*(k0[5]+collapsed[i])
            for h in range(2):
                a+=MM[h,j]*rm[h];s+=MM[h,j]*(min(max(k0[3]*agg[h]+35.2,0),71.66)-35.2)
            rm[j+2]=min(max(a+11.2,0),31.16)-11.2;rs[j]=min(max(s+11.2,0),31.16)-11.2
    else:
        for j in range(4):
            for h in range(2):
                cell[j]+=MM[h,side[j]]*rm[h];cellslow[j]+=MM[h,side[j]]*(min(max(k0[3]*agg[h]+35.2,0),71.66)-35.2)
            rmcell[j]=min(max(cell[j]+11.2,0),31.16)-11.2;rs[j]=min(max(cellslow[j]+11.2,0),31.16)-11.2
            rm[side[j]+2]+=weights[j]*rmcell[j]
    payload=np.zeros(4);rd=np.zeros(4);added=np.zeros(2)
    for d in range(4):
        d0=k0[0 if d<2 else 2]*agg[d]
        for h in range(2):d0+=F[h,d]*rm[h]
        if d<2 or rep<2:
            for h in range(2):d0+=F[h+2,d]*rm[h+2]
        else:
            amp=F[2,d]+F[3,d]
            for j in range(4):d0+=amp*R[j,d-2]*rmcell[j]
        if d>=2 and feedback:
            if rep<2:
                for j in range(2):added[d-2]+=gain*R2[j,d-2]*(rs[j] if feedback==2 else rm[j+2])
            else:
                for j in range(4):added[d-2]+=gain*R[j,d-2]*(rs[j] if feedback==2 else rmcell[j])
            d0+=added[d-2]
        shock=27.85 if d<2 else 11.38;rd[d]=d0+shock*pun;payload[d]=fw0*d0+fwd*shock*pun
    rawfast=np.zeros((N,2));rawslow=np.zeros(len(slow))
    for i in range(N):
        if plastic:
            rawfast[i,0]=(T[i,0]*payload[0]+T[i,1]*payload[1])*dx[i]*dt/90
            alpha=(T[i,2]*payload[2]+T[i,3]*payload[3])*dx[i]*dt/90*alpha_scale
            rawfast[i,1]=(1-fraction)*alpha
            if rep==0:rawslow[i]=fraction*alpha
        fast[i,0]=(fast[i,0]+rawfast[i,0])*np.exp(-dt/tg);fast[i,1]=(fast[i,1]+rawfast[i,1])*np.exp(-dt/tf)
    if rep:
        for e in range(len(ei)):
            if plastic:rawslow[e]=fraction*(ET[e,0]*payload[2]+ET[e,1]*payload[3])*dx[ei[e]]*dt/90*alpha_scale
    for e in range(len(slow)):slow[e]=(slow[e]+rawslow[e])*np.exp(-dt/ts)
    return np.concatenate((rd,rm)),payload,dx,rawfast,rawslow,fw0*added,rmcell

class Fly(SmallFly):
    def __init__(self,arm='REF'):
        super().__init__(MODEL);self.arm=arm;self.rep=0 if arm in ['REF','FB','SLOW_P'] else (1 if arm=='REPOOL' else 2)
        self.feedback=2 if arm.startswith('SLOW') else (1 if arm in ['FB','COUPLED','MAP_LEFT','MAP_RIGHT','MAP_BOTH'] else 0)
        z=np.load(ROOT/'models/anatomy.npz')
        for key,name in [('Q4','Q4'),('cell_weight','weights'),('side','side'),('edge_kc','ei'),('edge_mbon','ej'),('edge_Q','eq'),('edge_share','share'),('edge_T','ET'),('R4','R'),('R2','R2')]:setattr(self,name,z[key].copy())
        self.strength=float(z['strength'])
        if arm.startswith('MAP_'):
            perm=np.arange(4)
            for s,token in [(0,'LEFT'),(1,'RIGHT')]:
                if arm in ['MAP_'+token,'MAP_BOTH']:
                    ids=np.flatnonzero(self.side==s);perm[ids]=ids[::-1]
            self.R=self.R[perm];D=z['dan_mbon_counts'][:,perm]
            for d in range(2):
                denom=np.bincount(self.ei,weights=self.share*D[d,self.ej],minlength=len(self.ids));self.ET[:,d]=self.T[self.ei,d+2]*D[d,self.ej]/denom[self.ei]
        if self.rep:self.slow=np.zeros(len(self.ei))
        for n in ['Q4','weights','side','ei','ej','eq','share','ET','R','R2']:getattr(self,n).flags.writeable=False
    def step(self,seconds,pn_activity=None,punishment=0.,alpha_scale=1.):
        if not np.isfinite(seconds) or seconds<0 or not np.isfinite(punishment):raise ValueError('Invalid event')
        x=self.encode(np.zeros(self.input_channels) if pn_activity is None else pn_activity);k=self.kernel
        r=advance73(x,float(seconds),float(punishment),self.Q,self.T,self.F,self.MM,self.k0,self.fw0,self.fwd,self.ta,self.tg,k['fast_tau'],k['slow_tau'],k['fraction'],
          self.fast,self.slow,self.adapt,self.rep,self.feedback,self.strength,self.Q4,self.weights,self.side,self.ei,self.ej,self.eq,self.share,self.ET,self.R,self.R2,self.mode!='no_learning',alpha_scale)
        self.elapsed=np.float64(self.elapsed+seconds);self.event_count+=np.uint64(1);self.presentation_count+=np.uint64(np.any(x>0))
        return dict(rates=r[0],payload=r[1],dx=r[2],rawfast=r[3],rawslow=r[4],feedback_payload=r[5],individual_MBON=r[6])
    def projected_slow(self):
        return self.slow.copy() if not self.rep else np.bincount(self.ei,weights=self.share*self.slow,minlength=len(self.ids))
    def adopt_parent(self,m):
        self.fast=m.fast.copy();self.slow=m.slow[self.ei].copy() if self.rep else m.slow.copy();self.adapt=m.adapt.copy()
        for k in ['elapsed','event_count','presentation_count']:setattr(self,k,getattr(m,k))
    def save(self,path):
        np.savez_compressed(path,arm=self.arm,fast=self.fast,slow=self.slow,adapt=self.adapt,elapsed=self.elapsed,event_count=self.event_count,presentation_count=self.presentation_count,model_sha=self.model_sha)
    @classmethod
    def restore(cls,path):
        z=np.load(path);m=cls(str(z['arm']));assert str(z['model_sha'])==m.model_sha
        for key in ['fast','slow','adapt']:setattr(m,key,z[key].copy())
        for key in ['elapsed','event_count','presentation_count']:setattr(m,key,z[key][()])
        return m
    def expression(self,dx,zero=False):
        gamma=np.clip((dx*(self.k0[3]+(0 if zero else self.fast[:,0])))@self.Q[:,:2]+35.2,0,71.66)-35.2
        if self.rep<2:
            inp=(dx*(self.k0[5]+(0 if zero else self.fast[:,1]+self.projected_slow())))@self.Q[:,2:]+gamma@self.MM
            return np.clip(inp,-11.2,19.96)
        inp=(dx*(self.k0[5]+(0 if zero else self.fast[:,1])))@self.Q4
        if not zero:
            W=np.zeros_like(self.Q4);W[self.ei,self.ej]=self.eq*self.slow;inp+=dx@W
        inp+=gamma@self.MM[:,self.side];cells=np.clip(inp,-11.2,19.96);pool=np.column_stack([cells[:,self.side==s]@self.weights[self.side==s] for s in [0,1]])
        return pool

def probe(m,codes,roles,reader,h=0):
    n=m.clone()
    if h:n.mode='no_learning';n.step(h);n.mode=m.mode
    end=1-(1-n.adapt*np.exp(-.25*codes))*np.exp(-5/n.ta);dx=.5*(n.adapt+end)*codes
    alpha=n.expression(dx);base=n.expression(dx,True);obs=reader.predict(dx);direction=2*np.asarray(roles)-1
    mem=(alpha-base).mean(1);rd=(alpha-obs).mean(1);a=direction*(mem[::2]-mem[1::2]);r=direction*(rd[::2]-rd[1::2])
    return a,r,(a>=1)&(r>=reader.pair_threshold)
