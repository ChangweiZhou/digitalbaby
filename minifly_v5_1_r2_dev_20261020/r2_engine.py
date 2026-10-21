"""Native event equations, strict allocation intervention and target-free reference."""
from r2_common import np, fresh_bank, query_batch, predict_drive
from numba import njit
import math, copy, hashlib
DT=30/14

@njit(cache=True)
def neutral_query(adapt,x,dt,Q,reader_Q,MM,k0,ta):
    nc,n=adapt.shape;actual=np.empty((nc,2));drive=np.zeros((nc,4))
    da=math.exp(-dt/ta);obs=math.exp(-5/ta)
    for c in range(nc):
        inp=np.zeros(4)
        for i in range(n):
            if x[i]!=0:
                a=1-(1-adapt[c,i])*da
                end=1-(1-a*math.exp(-.25*x[i]))*obs
                dx=.5*(a+end)*x[i]
                for j in range(4):
                    drive[c,j]+=dx*reader_Q[i,j]
                    inp[j]+=dx*Q[i,j]*k0[3 if j<2 else 5]
        gamma=np.empty(2)
        for j in range(2):gamma[j]=min(max(inp[j]+35.2,0),71.66)-35.2
        for j in range(2):
            v=inp[j+2]
            for h in range(2):v+=gamma[h]*MM[h,j]
            actual[c,j]=min(max(v+11.2,0),31.16)-11.2
    return actual,drive

@njit(cache=True)
def event(fast,slow,adapt,x,dt,coeff,Q,T,F,MM,k0,fw0,fwd,ta,tg,tf,ts,
          fraction,alpha_scale,genes,sf,ss,q,q_at,now,gated,replay,signs):
    nc,n=slow.shape
    dg,df,ds,da=math.exp(-dt/tg),math.exp(-dt/tf),math.exp(-dt/ts),math.exp(-dt/ta)
    applied=0.;slow_applied=0.
    for c in range(nc):
        dx=np.empty(n);end=np.empty(n);agg=np.zeros(4);inp=np.zeros(4)
        for i in range(n):
            end[i]=1-(1-adapt[c,i]*math.exp(-.05*dt*x[i]))*da
            dx[i]=.5*(adapt[c,i]+end[i])*x[i]
            for j in range(4):
                agg[j]+=Q[i,j]*dx[i]
                value=fast[c,i,0] if j<2 else fast[c,i,1]+slow[c,i]
                inp[j]+=Q[i,j]*dx[i]*(k0[3 if j<2 else 5]+value)
        rm=np.empty(4)
        for j in range(2):rm[j]=min(max(inp[j]+35.2,0),71.66)-35.2
        for j in range(2):
            v=inp[j+2]
            for h in range(2):v+=MM[h,j]*rm[h]
            rm[j+2]=min(max(v+11.2,0),31.16)-11.2
        p0=np.empty(4);p1=np.empty(4)
        for j in range(4):
            v=k0[0 if j<2 else 2]*agg[j]
            for h in range(4):v+=F[h,j]*rm[h]
            p0[j]=fw0*v;p1[j]=fw0*v+fwd*(27.85 if j<2 else 11.38)
        for i in range(n):
            f0,f1,sl=fast[c,i,0],fast[c,i,1],slow[c,i]
            if coeff[c]!=0 and dx[i]!=0:
                g0=(T[i,0]*p0[0]+T[i,1]*p0[1])*dx[i]*dt/90
                g1=(T[i,0]*p1[0]+T[i,1]*p1[1])*dx[i]*dt/90
                a0=(T[i,2]*p0[2]+T[i,3]*p0[3])*dx[i]*dt/90*alpha_scale
                a1=(T[i,2]*p1[2]+T[i,3]*p1[3])*dx[i]*dt/90*alpha_scale
                fshare0=fraction;fshare1=fraction
                if np.any(genes[:4]!=0):
                    base=math.log(fraction/(1-fraction))+genes[0]+genes[1]*(2*adapt[c,i]-1)
                    influence=genes[2]*math.tanh(f1/sf)+genes[3]*math.tanh(sl/ss)
                    fshare0=1/(1+math.exp(-(base+influence*np.sign(a0))))
                    fshare1=1/(1+math.exp(-(base+influence*np.sign(a1))))
                if gated:
                    z=abs(q[c,i]*math.exp(-(now-q_at[c,i])/86400.))
                    fshare0=0. if z<=3 else .5*(1-math.exp(-(z-3)))
                    fshare1=fshare0
                if replay:fshare0=0.;fshare1=0.
                d=coeff[c]*(a1-a0)
                if abs(d)>1e-14 and not replay:
                    s=1 if d>0 else -1
                    if signs[c,i]==0:signs[c,i]=s
                    elif signs[c,i]!=s:signs[c,i]=2
                fast[c,i,0]=f0*dg+coeff[c]*((f0+g1)*dg-(f0+g0)*dg)
                fast[c,i,1]=f1*df+coeff[c]*((f1+(1-fshare1)*a1)*df-(f1+(1-fshare0)*a0)*df)
                slow[c,i]=sl*ds+coeff[c]*((sl+fshare1*a1)*ds-(sl+fshare0*a0)*ds)
                applied+=abs(d);slow_applied+=abs(coeff[c]*(fshare1*a1-fshare0*a0))
            else:
                fast[c,i,0]=f0*dg;fast[c,i,1]=f1*df;slow[c,i]=sl*ds
            adapt[c,i]=end[i]
    return applied,slow_applied

@njit(cache=True)
def confirm(q,q_at,last_at,ring,ring_pos,signs,prompt_id,now):
    accepted=0
    for c in range(q.shape[0]):
        for i in range(q.shape[1]):
            s=signs[c,i]
            if (s==1 or s==-1) and now-last_at[c,i]>=60:
                repeat=False
                for k in range(8):
                    if ring[c,i,k]==prompt_id:repeat=True
                if not repeat:
                    q[c,i]=q[c,i]*math.exp(-(now-q_at[c,i])/86400.)+s
                    q_at[c,i]=now;last_at[c,i]=now
                    ring[c,i,ring_pos[c,i]]=prompt_id;ring_pos[c,i]=(ring_pos[c,i]+1)%8
                    accepted+=1
    return accepted

class Engine:
    def __init__(self,n,seed,gated=True):
        self.bank=fresh_bank(n,seed);self.gated=gated;self.time=0.;self.events=0
        shape=(9,n);self.q=np.zeros(shape);self.q_at=np.zeros(shape)
        self.last_at=np.full(shape,-1e30);self.ring=np.full(shape+(8,),-1,dtype=np.int64)
        self.ring_pos=np.zeros(shape,dtype=np.int8);self.signs=np.zeros(shape,dtype=np.int8)
        self.write_l1=0.;self.slow_write_l1=0.;self.confirmations=0
    def begin(self):self.bank.adapt.fill(1.);self.signs.fill(0)
    def margins(self,x,dt=DT):
        b=self.bank
        actual,drive=query_batch(b.fast,b.slow,b.adapt,x,dt,b.Q,b.reader_Q,b.MM,b.k0,b.ta,b.tg,b.tf,b.ts)
        neutral,drive0=neutral_query(b.adapt,x,dt,b.Q,b.reader_Q,b.MM,b.k0,b.ta)
        assert np.array_equal(drive,drive0)
        base=predict_drive(drive,b.reader_head)
        z=(actual-base).mean(1);r=(neutral-base).mean(1)
        return z-r,z,r
    def tick(self,x,coeff=None,dt=DT,replay=False):
        b=self.bank;c=np.zeros(9) if coeff is None else np.asarray(coeff,dtype=float)
        if c.shape!=(9,) or not np.isfinite(c).all() or np.max(abs(c))>1+1e-12:raise ValueError('invalid residual')
        a,s=event(b.fast,b.slow,b.adapt,x,dt,c,b.Q,b.T,b.F,b.MM,b.k0,b.fw0,b.fwd,b.ta,b.tg,b.tf,b.ts,b.fraction,b.alpha_scale,b.genes,b.sf,b.ss,self.q,self.q_at,self.time+dt,self.gated,replay,self.signs)
        self.time+=dt;self.events+=1;self.write_l1+=a;self.slow_write_l1+=s
    def end(self,prompt,write=True,replay=False):
        if write and not replay:
            identity=int.from_bytes(hashlib.blake2b(prompt,digest_size=8).digest(),'little')&((1<<63)-1)
            self.confirmations+=confirm(self.q,self.q_at,self.last_at,self.ring,self.ring_pos,self.signs,identity,self.time)
        self.signs.fill(0)
    def rest(self,dt):self.bank.rest(dt);self.time+=dt
    def clone(self):
        c=copy.copy(self);c.bank=self.bank.clone()
        for key in ('q','q_at','last_at','ring','ring_pos','signs'):setattr(c,key,getattr(self,key).copy())
        return c
    def stats(self):
        return dict(native_time_seconds=self.time,native_events=self.events,write_l1=self.write_l1,slow_write_l1=self.slow_write_l1,
            confirmations=self.confirmations,q_max=float(abs(self.q).max()),fast_l1=float(abs(self.bank.fast).sum()),slow_l1=float(abs(self.bank.slow).sum()),
            mutable_bytes=sum(v.nbytes for v in [self.bank.fast,self.bank.slow,self.bank.adapt,self.q,self.q_at,self.last_at,self.ring,self.ring_pos,self.signs]))

def targets(token):
    if not 0<=token<=256:raise ValueError('invalid token')
    t=np.ones(9)*-1
    if token==256:t[8]=1
    else:t[:8]=((token>>np.arange(8))&1)*2-1
    return t
def decode(m):
    return 256 if m[8]>0 else int(np.sum((m[:8]>0).astype(int)*(1<<np.arange(8))))
def residual(m,token,strength):
    s=strength*(np.tanh(m/2)-targets(token))/2
    if token==256:s[:8]=0
    return s
