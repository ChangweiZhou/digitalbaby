"""Reference and local learner kernels for bounded engineering checks."""
import copy
from hashlib import sha256
from random import Random

import numpy as np
import torch as T
from torch import nn
from torch.nn.functional import cross_entropy, pad

from fixtures import EOS,SEED
from backend import Backend

T.set_num_threads(1)


def digest(model):
    return sha256(b"".join(x.detach().numpy().tobytes() for x in model.state_dict().values())).hexdigest()


class Sensory:
    def __init__(self, archive):
        self.sensor=Backend(archive).sensor
        self.cache={}
        self.seconds=0
        self.new=0

    def codes(self, prefix):
        import time
        raw=bytes(prefix)
        if raw not in self.cache:
            start=time.process_time()
            self.cache[raw]=np.packbits(self.sensor.codes(raw)>0,axis=1)
            self.seconds+=time.process_time()-start
            self.new+=1
        return T.from_numpy(np.unpackbits(self.cache[raw],axis=1,count=5177).astype(np.float32))

    def batch(self,xs):
        return T.stack([self.codes(row[1:]) for row in xs.tolist()])


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding=nn.Embedding(257,16)
        self.rnn=nn.GRU(16,64,batch_first=True)
        self.head=nn.Linear(64,256)

    def features(self,x):
        return self.rnn(self.embedding(x))[0]


def pretrain(windows):
    T.manual_seed(SEED)
    random=Encoder()
    models={}
    for condition in ("aligned","shuffled"):
        m=copy.deepcopy(random)
        opt=T.optim.Adam(m.parameters(),lr=.003)
        for source in windows:
            x=T.tensor(source)
            ys=x[:,1:]
            shifted=ys.roll(1,0)
            assert T.equal(ys.sort(dim=0).values,shifted.sort(dim=0).values)
            for idx in range(0,len(x),8):
                y=(ys if condition=="aligned" else shifted)[idx:idx+8]
                p=m.head(m.features(x[idx:idx+8,:-1]))
                step(opt,cross_entropy(p.flatten(0,1),y.flatten()))
        models[condition]=m
    models["random"]=random
    for m in models.values():
        m.head=nn.Identity()
        m.requires_grad_(False)
    return models


class Local:
    def __init__(self,encoder,sensor):
        self.encoder,self.sensor=encoder,sensor
        self.w=T.zeros(257,65)
        self.project=T.randn(5177,64,generator=T.Generator().manual_seed(SEED))/64

    @T.no_grad()
    def features(self,prefix):
        if self.encoder is None:
            h=self.sensor.codes(prefix)@self.project
        elif prefix:
            x=self.encoder.features(T.tensor([prefix]))[0]
            h=T.cat([T.zeros(1,64),x])
        else:
            h=T.zeros(1,64)
        h=h/h.norm(dim=1,keepdim=True).clamp_min(1)
        return T.cat([h,T.ones(len(h),1)],dim=1)

    @T.no_grad()
    def train(self,row):
        for h,y in zip(self.features(row[:-1]),row):
            error=-(self.w@h).softmax(0)
            error[y]+=1
            self.w.add_(.03*error[:,None]*h)

    @T.no_grad()
    def generate(self,prompt,cap=8):
        p=list(prompt)
        out=[]
        for _ in range(cap):
            y=int((self.w@self.features(p)[-1]).argmax())
            out.append(y)
            if y==EOS: break
            p.append(y)
        return out


class Modular(nn.Module):
    def __init__(self,sensor):
        super().__init__()
        self.sensor=sensor
        self.book=nn.Embedding(258,8,padding_idx=257)
        self.shared=nn.Sequential(nn.Linear(64,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh(),nn.Linear(128,257))
        self.association=nn.Linear(5177,257,bias=False)

    def forward(self,x):
        u=self.book(pad(x,(7,0),value=257).unfold(1,8,1)).flatten(2)
        return self.shared(u)+self.association(self.sensor.batch(x))

    @T.no_grad()
    def shrink(self,factor):
        for layer in self.shared:
            if isinstance(layer,nn.Linear): layer.weight.mul_(factor)
        self.book.weight.mul_(factor)


def topology(rewired):
    n=32
    initial={tuple(sorted((i,(i+d)%n))) for i in range(n) for d in (1,2)}
    edges=initial.copy()
    if rewired:
        rng=Random(SEED)
        for _ in range(2048):
            x,y=rng.sample(sorted(edges),2)
            a,b=x
            c,d=y
            if len({a,b,c,d})<4: continue
            if rng.randrange(2): c,d=d,c
            new={tuple(sorted(p)) for p in ((a,c),(b,d))}
            if new&edges: continue
            edges.difference_update((x,y))
            edges.update(new)
    adj=T.zeros(n,n)
    for a,b in edges:
        adj[a,b]=adj[b,a]=.25
    assert T.equal(adj.sum(1),T.ones(n))
    reached={0}
    while True:
        now=reached|{i for a in reached for i in adj[a].nonzero().flatten().tolist()}
        if now==reached:break
        reached=now
    assert len(reached)==n
    triangles=int(T.trace((adj*4).matrix_power(3)).item()/6)
    return adj,triangles


class Attractor(nn.Module):
    def __init__(self,sensor,rewired):
        super().__init__()
        self.sensor=sensor
        self.register_buffer("adj",topology(rewired)[0])
        rng=T.Generator().manual_seed(SEED+11)
        codes=T.randn(258,8,generator=rng)
        codes[257].zero_()
        self.register_buffer("book",codes)
        self.gate=nn.Linear(8,32)
        self.drive=nn.Linear(8,32,bias=False)
        self.gains=nn.Parameter(T.tensor([1.25,1.40,2.8]))
        self.head=nn.Linear(32,257)
        self.association=nn.Linear(5177,257,bias=False)

    def transition(self,h,current,gate):
        a,b,c=self.gains.abs()
        proposal=(a*h+b*(h@self.adj)-c*h.mean(1,keepdim=True)+current-.12).clamp(0,1)
        return h+gate*(proposal-h)

    def forward(self,x):
        e=self.book[x]
        h=e.new_zeros(len(x),32)
        trajectory=[]
        for u,g in zip(self.drive(e).unbind(1),self.gate(e).sigmoid().unbind(1)):
            h=self.transition(h,u,g)
            trajectory.append(h)
        return self.head(T.stack(trajectory,dim=1))+self.association(self.sensor.batch(x))

    @T.no_grad()
    def persists(self):
        h=T.zeros(8,32)
        for i in range(8):
            h[i,[(i*4+j)%32 for j in range(3)]]=1
        gate=self.gate(T.zeros(8,8)).sigmoid()
        current=T.zeros_like(h)
        for i in range(80):
            h=self.transition(h,current,gate)
            if i==59: previous=h.clone()
        norms=h.norm(dim=1)
        support=(h>.4).sum(1)
        stable=(h-previous).abs().amax(1)<.01
        eligible=(norms>.5)&(support<=10)&(support>=2)&stable
        return {"stable_local_states":int(eligible.sum()),"distinct_supports":len({tuple(x.tolist()) for x in (h[eligible]>.4)}),
                "amplitudes":[round(float(x),3) for x in h.amax(1)]}


def step(opt,value):
    opt.zero_grad()
    assert T.isfinite(value)
    value.backward()
    assert all(T.isfinite(p.grad).all() for g in opt.param_groups for p in g["params"])
    opt.step()
    return float(value.detach())


def loss(model,rows,mask_after=None):
    data=T.tensor([[257,*r] for r in rows])
    logits=model(data[:,:-1])
    error=cross_entropy(logits.flatten(0,1),data[:,1:].flatten(),reduction="none")
    error=error.reshape(len(rows),-1)
    if mask_after is not None: error[:,mask_after:]=0
    return error.mean()


@T.no_grad()
def generate(model,prompt,cap=5):
    p=[257,*prompt]
    out=[]
    for _ in range(cap):
        y=int(model(T.tensor([p]))[0,-1].argmax())
        out.append(y)
        if y==EOS:break
        p.append(y)
    return out
