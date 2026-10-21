"""Observed-vector competitive learning. No task parser, targets or role labels."""
from r2_common import np
import math
BITS=(((np.arange(256)[:,None]>>np.arange(8))&1)*2-1).astype(float)
TOL=1e-12

class Competitive:
    def __init__(self,capacity,dim,seed,cap):
        self.capacity=capacity;self.dim=dim;self.rng=np.random.default_rng(seed)
        self.w=np.empty((0,dim));self.reservoir=[];self.cap=cap;self.views=0
        self.wins=np.zeros(0,dtype=np.int64);self.block=np.zeros(0,dtype=np.int64)
        self.born=np.zeros(0,dtype=np.int64);self.reseeds=[];self.initial_at=None
    def retain(self,v):
        self.views+=1
        if len(self.reservoir)<self.cap:self.reservoir.append(v.copy())
        else:
            i=int(self.rng.integers(self.views))
            if i<self.cap:self.reservoir[i]=v.copy()
    def initialize(self,t):
        if not self.reservoir:raise ValueError('empty observed reservoir')
        a=np.array(self.reservoir);ids=[int(self.rng.integers(len(a)))];dist=((a-a[ids[0]])**2).mean(1)
        while len(ids)<self.capacity and float(dist.max())>TOL:
            i=int(np.argmax(dist));ids.append(i);dist=np.minimum(dist,((a-a[i])**2).mean(1))
        self.w=a[ids].copy();self.wins=np.zeros(len(ids),dtype=np.int64);self.block=self.wins.copy()
        self.born=np.full(len(ids),t,dtype=np.int64);self.initial_at=t
    def winner(self,v):
        if not len(self.w):raise ValueError('uninitialized competition')
        return int(np.argmin(((self.w-v)**2).mean(1)))
    def gate(self,i,t):return float(min(1,max(0,(t-self.born[i])/1024)))
    def update(self,v,t):
        i=self.winner(v);eta=.001*(1/16+15/16*min(max(0,t-self.initial_at)/4096,1))
        self.w[i]+=eta*(v-self.w[i]);self.wins[i]+=1;self.block[i]+=1
    def checkpoint(self,t,remaining):
        if self.initial_at is None or t-self.initial_at<4096:return
        dead=np.flatnonzero(self.block==0)
        if len(dead) and remaining>=1024:
            a=np.array(self.reservoir);dist=((a[:,None,:]-self.w[None,:,:])**2).mean(2).min(1)
            j=int(np.argmax(dist));i=int(dead[0])
            if dist[j]>TOL:
                self.w[i]=a[j];self.born[i]=t;self.reseeds.append(dict(at=t,unit=i,mse=float(dist[j])))
        self.block.fill(0)
    def report(self):
        return dict(capacity=self.capacity,active_prototypes=len(self.w),wins=self.wins.tolist(),
                    initialization_payload=self.initial_at,reservoir_views=self.views,reservoir_size=len(self.reservoir),reseeds=self.reseeds)

class Codes:
    def __init__(self,seed):
        self.seed=seed;self.m2=Competitive(64,8,seed,256);self.bytes=0
        self.counts=np.zeros(256,dtype=np.int64);self.left=np.zeros((64,64));self.right=self.left.copy()
        self.context_bytes=0;self.candidates={c:Competitive(c,128,seed+100+c,64) for c in range(1,7)}
        self.classes=np.full(64,6,dtype=np.int16);self.identity_frozen=False;self.context_frozen=False
        self.selected=None;self.selection=None;self.profile_views=0
    def hear(self,data,freeze_cap):
        if self.identity_frozen:raise ValueError('identity frozen')
        for b in data:
            v=BITS[b];self.bytes+=1;self.counts[b]+=1;self.m2.retain(v)
            if self.bytes==1024:self.m2.initialize(self.bytes)
            elif len(self.m2.w):self.m2.update(v,self.bytes)
            if self.bytes%4096==0:self.m2.checkpoint(self.bytes,freeze_cap-self.bytes)
    def winner(self,b):return self.m2.winner(BITS[int(b)])
    def byte_gate(self,b):return self.m2.gate(self.winner(b),self.bytes)
    def table(self):return np.array([self.winner(b) for b in range(256)])
    def profiles(self):
        valid=(self.left.sum(1)>0)&(self.right.sum(1)>0)
        a=.5*np.c_[self.left/np.maximum(1,self.left.sum(1))[:,None],self.right/np.maximum(1,self.right.sum(1))[:,None]]
        return a,valid
    def hear_context(self,p,a,freeze_cap):
        if not self.identity_frozen or self.context_frozen:raise ValueError('phase order')
        prev=None
        for b in p+a:
            u=self.winner(b);self.context_bytes+=1
            if prev is not None:self.right[prev,u]+=1;self.left[u,prev]+=1
            profiles,valid=self.profiles()
            for j in ([u] if prev is None else [prev,u]):
                if valid[j]:
                    self.profile_views+=1
                    for m in self.candidates.values():
                        m.retain(profiles[j])
                        if len(m.w):m.update(profiles[j],self.context_bytes)
            if self.context_bytes==4096:
                for m in self.candidates.values():m.initialize(self.context_bytes)
            if self.context_bytes>4096 and (self.context_bytes-4096)%4096==0:
                for m in self.candidates.values():m.checkpoint(self.context_bytes,freeze_cap-self.context_bytes)
            prev=u
    def select(self,validation):
        profile,valid=self.profiles();active=np.flatnonzero((self.left.sum(1)+self.right.sum(1))>0)
        A=len(active);N=int(self.right.sum());rows=[];assignments={}
        for C,m in self.candidates.items():
            cl=np.full(64,-1,dtype=int)
            for u in active:
                if valid[u]:cl[u]=m.winner(profile[u])
            counts=np.ones((len(m.w)+1,64))*.5
            for u in active:counts[cl[u] if cl[u]>=0 else len(m.w)]+=self.right[u]
            probs=counts/np.maximum(1,counts[:,active].sum(1))[:,None]
            nll=0.;trans=0
            for p,a in validation:
                seq=[self.winner(b) for b in p+a]
                for u,v in zip(seq,seq[1:]):
                    nll-=math.log2(probs[cl[u] if cl[u]>=0 else len(m.w),v]);trans+=1
            retained=len(m.w);penalty=.5*(3*retained*(A-1))*math.log2(max(2,N))
            assign=(A+retained)*math.log2(6)
            rows.append(dict(C=C,retained=retained,nll_bits=nll,validation_transitions=trans,parameter_bits=penalty,assignment_bits=assign,total_bits=nll+penalty+assign))
            assignments[C]=cl
        chosen=min(rows,key=lambda r:(r['total_bits'],r['C']))['C'];self.selected=chosen
        self.classes[:]=np.where(assignments[chosen]<0,6,assignments[chosen]);self.context_frozen=True
        identity=np.ones((64,64))*.5+self.right;den=identity[:,active].sum(1);nll=0.
        for p,a in validation:
            seq=[self.winner(b) for b in p+a]
            for u,v in zip(seq,seq[1:]):nll-=math.log2(identity[u,v]/den[u])
        self.selection=dict(rows=rows,selected=chosen,A=A,training_transitions=N,identity_nll_bits=nll,
                            identity_parameter_bits=.5*A*(A-1)*math.log2(max(2,N)),validation_model_views=len(validation)*7)
        return self.selection
    def byte_class(self,b):return int(self.classes[self.winner(b)])
    def diagnostics(self):
        seen=np.flatnonzero(self.counts);tab=self.table();win=tab[seen];p,v=self.profiles()
        return dict(observed_types=seen.tolist(),winner_map={str(int(b)):int(tab[b]) for b in seen},
                    alias_count=len(seen)-len(set(win.tolist())),class_map={str(int(b)):self.byte_class(b) for b in seen},
                    occupied_classes=len(set(self.classes[win].tolist())),profile_sums=p[v].sum(1).tolist(),
                    reconstruction_mse=float(np.mean((self.m2.w[win]-BITS[seen])**2)),payload_bytes=self.bytes,
                    context_payload_bytes=self.context_bytes,profile_views=self.profile_views,
                    M2=self.m2.report(),M3={str(c):m.report() for c,m in self.candidates.items()},selection=self.selection)
