"""Pinned v5.1 generic sequence and geometry; gated learned identities."""
from r2_common import np
from r2_representation import BITS
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
                off=(s*16+k)*73;f[off+self.codes.winner(b)]=self.codes.byte_gate(b)
                f[off+64:off+72]=BITS[b];f[off+72]=1
        off=5*16*73;f[off+min(self.age,16)]=1
        if self.reply is not None:f[off+17+min(self.reply,16)]=1
        off+=34
        if self.last is not None:
            f[off+self.codes.winner(self.last)]=self.codes.byte_gate(self.last);f[off+64:off+72]=BITS[self.last];f[off+72]=1
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

