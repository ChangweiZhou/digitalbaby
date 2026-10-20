"""Generic learned byte identities, context classes and bounded sequence circuitry.

No arithmetic generator, semantic labels or expected next bytes enter this module.
"""
from common import np
from numba import njit
import copy, math
BITS=(((np.arange(256)[:,None]>>np.arange(8))&1)*2-1).astype(float)

@njit(cache=True)
def learn_bytes(data,w,usage,bias,counts,eta_max,gain,start,ramp=50000):
    for t in range(len(data)):
        byte=data[t];b=BITS[byte];best=0;score=-1e30
        for u in range(len(w)):
            v=-bias[u]
            for k in range(8):v-=(b[k]-w[u,k])**2
            if v>score:score=v;best=u
        eta=eta_max*(1/16+(15/16)*min((start+t)/ramp,1))
        for k in range(8):w[best,k]+=eta*(b[k]-w[best,k])
        for u in range(len(w)):
            usage[u]+=.001*((1. if u==best else 0.)-usage[u])
            bias[u]=gain*usage[u]
        counts[byte]+=1

class Codes:
    def __init__(self,seed,eta=.001,gain=0.):
        rng=np.random.default_rng(seed)
        self.w=rng.uniform(-1,1,(64,8));self.usage=np.zeros(64);self.bias=np.zeros(64)
        self.counts=np.zeros(256,dtype=np.int64);self.eta=eta;self.gain=gain;self.bytes=0
        self.left=np.zeros((64,64));self.right=np.zeros((64,64))
        self.class_proto=rng.uniform(0,1,(6,128));self.class_proto/=self.class_proto.sum(1)[:,None]
        self.classes=np.full(64,6,dtype=np.int16);self.context_bytes=0
        self.identity_frozen=False;self.context_frozen=False
    def winner(self,b):return int(np.argmax(-((self.w-BITS[b])**2).sum(1)-self.bias))
    def table(self):return np.argmax(-((BITS[:,None,:]-self.w[None,:,:])**2).sum(2)-self.bias,axis=1)
    def hear(self,data):
        if self.identity_frozen:raise ValueError('identity is frozen')
        a=np.frombuffer(bytes(data),dtype=np.uint8)
        learn_bytes(a,self.w,self.usage,self.bias,self.counts,self.eta,self.gain,self.bytes)
        self.bytes+=len(a)
    def profiles(self):
        l=self.left/np.maximum(1,self.left.sum(1))[:,None]
        r=self.right/np.maximum(1,self.right.sum(1))[:,None]
        return np.c_[l,r]
    def hear_context(self,records):
        if not self.identity_frozen or self.context_frozen:raise ValueError('context phase order')
        for p,a in records:
            seq=[self.winner(int(b)) for b in p+a]
            previous=None
            for u in seq:
                if previous is not None:self.right[previous,u]+=1;self.left[u,previous]+=1
                profile=self.profiles()
                # Update only identities whose context has actually just arrived.
                eta=.001*(1/16+(15/16)*min(self.context_bytes/50000,1))
                for observed in ([u] if previous is None else [previous,u]):
                    c=int(np.argmin(((self.class_proto-profile[observed])**2).sum(1)))
                    self.class_proto[c]+=eta*(profile[observed]-self.class_proto[c])
                self.context_bytes+=1
                previous=u
        profile=self.profiles();seen=(self.left.sum(1)+self.right.sum(1))>0
        self.classes[seen]=np.argmin(((profile[seen,None,:]-self.class_proto[None,:,:])**2).sum(2),axis=1)
    def byte_class(self,b):return int(self.classes[self.winner(b)])
    def diagnostics(self):
        seen=np.flatnonzero(self.counts);tab=self.table();win=tab[seen]
        return dict(bytes=self.bytes,observed_types=seen.tolist(),winner_map={str(int(b)):int(tab[b]) for b in seen},
           alias_count=len(seen)-len(set(win.tolist())),unused_units=64-len(set(win.tolist())),
           type_weighted_reconstruction=float(np.mean((self.w[win]-BITS[seen])**2)) if len(seen) else 0,
           counts=self.counts.tolist(),classes=self.classes.tolist())

class Sequence:
    def __init__(self,codes,segmented=True):
        self.codes=codes;self.segmented=segmented;self.open=[];self.closed=[];self.last=None
        self.last_class=None;self.age=0;self.reply=None;self.overflow=0;self.closures=0
    def feed(self,b):
        if not 0<=int(b)<256:raise ValueError('not payload')
        c=self.codes.byte_class(b)
        if self.segmented and self.last_class is not None and c!=self.last_class:
            self.closed=[list(self.open)]+self.closed[:3];self.open=[];self.age=0;self.closures+=1
        self.open.append(int(b))
        if len(self.open)>16:self.open.pop(0);self.overflow+=1
        self.age+=1;self.last=int(b);self.last_class=c
    def begin_reply(self):self.reply=0
    def feed_output(self,b):self.feed(b);self.reply+=1
    def feature(self):
        # 5 generic registers x 16 offsets x (64 identity, 8 raw, presence),
        # two 17-cell clocks, then previous byte (same 73 channels).
        f=np.zeros(5*16*73+34+73)
        registers=[self.open]+self.closed if self.segmented else [self.open]
        for s,segment in enumerate(registers[:5]):
            for k,b in enumerate(reversed(segment[-16:])):
                off=(s*16+k)*73;f[off+self.codes.winner(b)]=1
                f[off+64:off+72]=BITS[b];f[off+72]=1
        off=5*16*73;f[off+min(self.age,16)]=1
        if self.reply is not None:f[off+17+min(self.reply,16)]=1
        off+=34
        if self.last is not None:
            f[off+self.codes.winner(self.last)]=1;f[off+64:off+72]=BITS[self.last];f[off+72]=1
        return f

class Geometry:
    def __init__(self,n,seed,side,active_fraction,local=True):
        rng=np.random.default_rng(seed);self.n=n;self.side=np.array(side);self.active_fraction=active_fraction
        self.index=np.zeros((n,4),dtype=np.int32);self.weights=np.ones((n,4))*.5
        anchor=rng.integers(0,16,n)
        for j in range(4):
            slot=rng.integers(0,5,n)
            pos=np.clip(anchor+rng.integers(-1,2,n),0,15) if local else rng.integers(0,16,n)
            family=rng.random(n);unit=rng.integers(0,64,n)
            raw=family<.2;presence=family>.9
            unit[raw]=64+rng.integers(0,8,np.sum(raw));unit[presence]=72
            self.index[:,j]=(slot*16+pos)*73+unit
            self.weights[raw,j]*=rng.choice([-1.,1.],np.sum(raw))
        half=n//2;off=5*16*73
        self.index[half:,2]=off+rng.integers(0,34,n-half)
        self.index[half:,3]=off+34+rng.integers(0,73,n-half)
        self.weights[half:,2:]=.5
        self.theta=np.zeros(n);self.frequency=np.full(n,active_fraction);self.frozen=False
    def encode(self,state):
        f=state.feature();raw=(f[self.index]*self.weights).sum(1)-self.theta;x=np.zeros(self.n)
        for s in (0,1):
            ix=np.flatnonzero((self.side==s)&(raw>0));k=min(len(ix),int(np.ceil(self.active_fraction*np.sum(self.side==s))))
            if k:
                order=np.lexsort((ix,-raw[ix]));x[ix[order[:k]]]=1
        return x,raw
    def calibrate(self,state):
        if self.frozen:raise ValueError('thresholds frozen')
        x,raw=self.encode(state);self.frequency+=.01*(x-self.frequency)
        self.theta+=.05*np.maximum(0,self.frequency-5*self.active_fraction)*max(float((raw+self.theta).std()),1e-12)/self.active_fraction*.1

