"""Portable development screen for specific operations omitted by the first sweep."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

ROSTER = {
    "sparse_fast_slow": ("coupled_sparse", "no_replay", "shuffled_teaching"),
    "separate_wm_gates": ("separate", "tied", "always_open"),
    "apical_feedback": ("adaptive_apical", "fixed_apical", "shuffled_error"),
    "cerebellar_timing": ("localized_ltd", "scrambled_ltd", "linear_ltd"),
    "multi_action_credit": ("trace", "immediate", "shifted"),
    "grid_graph_transfer": ("structured_grid", "shuffled_grid", "reset"),
    "fb_goal_writes": ("aligned", "permuted", "frozen"),
    "mb_apl_reversal": ("apl_addressed", "apl_off", "dan_permuted"),
    "sleep_order_scaling": ("ordered_scaled", "shuffled_scaled", "ordered_unscaled"),
    "thalamic_feedback_gate": ("learned", "constant", "open_loop"),
    "operator_binding": ("attention", "register", "static"),
}
PROFILES = {
    "smoke": ((73129,), 12, 24),
    "pilot": ((73129,), 384, 96),
    "screen": ((73129, 93683, 116089), 2048, 256),
}
NOTES = {
    "sparse_fast_slow": "Sparse parametric fast path teaches a shared slow path on retained prompts. Synthetic sequential binary rules.",
    "separate_wm_gates": "Learned separate input and read gates with a distractor gap. Supervised development.",
    "apical_feedback": "Local hidden mismatch and a trained feedback projection. Small classification proxy.",
    "cerebellar_timing": "Climbing-error-inspired depression/potentiation on temporally localized expansion features.",
    "multi_action_credit": "Scalar outcome after four decisions. Success requires the whole action sequence.",
    "grid_graph_transfer": "Fourier grid transport of learned edge displacement. Synthetic toroidal graph and unseen edge pairs.",
    "fb_goal_writes": "Fan-shaped-body-style columnar goal trace with local writes. Engineered proxy without a FlyWire plasticity mask.",
    "mb_apl_reversal": "Sparse KC-style features, activity-dependent gain, and compartment-specific reversal. Synthetic proxy.",
    "sleep_order_scaling": "Same experiences and replay count. Compare one order ablation and one scaling ablation.",
    "thalamic_feedback_gate": "Three shared computation passes with learned, constant, or disabled feedback. Supervised development.",
    "operator_binding": "Raw-byte rewrite and isolated learned execution, with held directed operator pairs. The harness supplies stage indicators.",
}
IMPLEMENTATION = Path(__file__).resolve()


def tensor(x, dtype=None):
    return torch.as_tensor(x, dtype=dtype)


def fingerprint(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str((a.shape, a.dtype.str)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def result(metric, value, panel, **extra):
    return dict(metric=metric, value=float(value), panel_sha256=fingerprint(*panel), **extra)


def fixed_features(rng, rows, n_in=16, n_fast=64):
    x = rng.normal(size=(rows, n_in))
    projection = rng.normal(size=(n_in, n_fast))
    return x, projection


def sparse_fast_slow(arm, seed, steps, n):
    rng = np.random.default_rng(seed)
    x, proj = fixed_features(rng, 4*(n+48))
    task = np.repeat(np.arange(4), n+48)
    x = np.column_stack((x, np.eye(4)[task]))
    y = ((x[:,:16]*rng.normal(size=(4,16))[task]).sum(1)>0).astype(float)
    fast_input = x[:,:16] @ proj
    keep = fast_input >= np.partition(fast_input, -4, axis=1)[:,-4,None]
    z = keep.astype(float)/2
    f = np.zeros((4,z.shape[1]))
    w = np.zeros(len(x[0]))
    taught = [np.arange(j*(n+48), j*(n+48)+48) for j in range(4)]
    held = np.concatenate([np.arange(j*(n+48)+48,(j+1)*(n+48)) for j in range(4)])
    sampler = np.random.default_rng(seed+2)
    for j in range(4):
        for _ in range(steps//4):
            i = sampler.choice(taught[j])
            q = sampler.choice(np.concatenate(taught[:j])) if j else i
            f[j] += .2*(y[i]-z[i]@f[j])*z[i]
            w += .025*(y[i]-x[i]@w)*x[i]
            if arm!="no_replay" and j:
                target = np.clip(z[q]@f[task[q]],0,1)
                if arm=="shuffled_teaching":
                    other = taught[task[q]][(q % (n+48)+17)%48]
                    target = np.clip(z[other]@f[task[q]],0,1)
                w += .025*(target-x[q]@w)*x[q]
    correct = ((x[held]@w >= .5)==y[held]).reshape(4,n)
    return result("old_rule_accuracy", correct[:3].mean(), (x[held],y[held]),
                  by_rule=correct.mean(1).tolist(), n=n*3)


def fit_torch(model, x, y, steps, seed):
    opt = torch.optim.AdamW(model.parameters(), lr=.003)
    rng = np.random.default_rng(seed)
    curve = []
    for i in range(1,steps+1):
        ix=rng.integers(len(x),size=48)
        out=model(x[ix])
        loss=nn.functional.cross_entropy(out.reshape(-1,out.shape[-1]),y[ix].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i in (1,steps) or i%256==0:
            curve.append([i,float(loss.detach())])
    return curve


class MemoryGates(nn.Module):
    def __init__(self, arm):
        super().__init__()
        self.arm=arm
        self.emb=nn.Embedding(14,24)
        self.write=nn.Linear(48,24)
        self.read=nn.Linear(48,24)
        self.value=nn.Linear(48,24)
        self.out=nn.Linear(24,4)

    def forward(self,x):
        h=torch.zeros(len(x),24)
        e=self.emb(x)
        for v in e.unbind(1):
            z=torch.cat((v,h),dim=1)
            a=self.write(z).sigmoid()
            if self.arm=="always_open": a=torch.ones_like(a)
            h=torch.lerp(h,self.value(z).tanh(),a)
        z=torch.cat((v,h),dim=1)
        b=self.read(z).sigmoid() if self.arm=="separate" else a
        return self.out(b*h)


def separate_wm_gates(arm,seed,steps,n):
    rng=np.random.default_rng(seed)
    def batch(count,gap):
        c=rng.integers(4,size=(count,2))
        q=rng.integers(2,size=count)
        x=np.column_stack((c[:,0],c[:,1]+4,rng.integers(8,12,size=(count,gap)),12+q))
        return x,c[np.arange(count),q]
    x,y=batch(1536,8)
    test,truth=batch(n,16)
    torch.manual_seed(seed)
    model=MemoryGates(arm)
    curve=fit_torch(model,tensor(x),tensor(y),steps,seed)
    with torch.no_grad(): pred=model(tensor(test)).argmax(1).numpy()
    return result("held_long_gap_accuracy",(pred==truth).mean(),(test,truth),n=n,curve=curve)


def apical_feedback(arm,seed,steps,n):
    rng=np.random.default_rng(seed)
    x=rng.normal(size=(1024+n,12))
    teacher=rng.normal(size=(12,4))
    y=np.eye(4)[(np.sin(x@teacher)*.7+x@teacher).argmax(1)]
    w=rng.normal(0,.15,(12,24))
    r=rng.normal(0,.15,(24,4))
    feedback=rng.normal(0,.2,(4,24))
    for _ in range(steps):
        ids=rng.integers(1024,size=32)
        s=x[ids]
        h=np.tanh(s@w)
        logits=h@r
        p=np.exp(logits-logits.max(1,keepdims=True)); p/=p.sum(1,keepdims=True)
        e=y[ids]-p
        teaching=np.roll(e,1,axis=0) if arm=="shuffled_error" else e
        target=np.clip(h+.2*(teaching@feedback),-1,1)
        w+=.12*(s.T@((target-h)*(1-h*h)))/len(ids)
        r+=.12*(h.T@e)/len(ids)
        if arm=="adaptive_apical":
            # Learn to reconstruct perturbations of hidden activity from paired output perturbations.
            noise=rng.normal(0,.1,h.shape)
            observed=noise@r
            feedback+=.5*(observed.T@(noise-observed@feedback))/len(ids)
        else:
            rng.normal(0,.1,h.shape)  # pair sampling across arms
    pred=(np.tanh(x[1024:]@w)@r).argmax(1)
    return result("held_accuracy",(pred==y[1024:].argmax(1)).mean(),(x[1024:],y[1024:]),n=n)


def cerebellar_timing(arm,seed,steps,n):
    rng=np.random.default_rng(seed)
    data=rng.uniform(-1,1,(1200+n,2))
    signal=np.sin(4*data[:,0])+ .45*np.cos(5*data[:,1])+rng.normal(0,.025,len(data))
    centers=rng.uniform(-1,1,(32,2))
    z=np.exp(-((data[:,None]-centers)**2).sum(2)/.18)
    if arm=="scrambled_ltd":
        shuffle=np.random.default_rng(seed+43)
        z=np.concatenate([np.stack([col[shuffle.permutation(len(col))] for col in part.T],axis=1)
                          for part in (z[:1200],z[1200:])])
    if arm=="linear_ltd":
        z=.5+.2*(data@centers.T)
    z/=np.maximum(np.linalg.norm(z,axis=1,keepdims=True),.01)
    w=np.zeros(z.shape[1])
    sampling=np.random.default_rng(seed+17)
    for _ in range(steps*4):
        i=sampling.integers(1200)
        error=signal[i]-z[i]@w
        w+=.14*error*z[i]
    err=np.sqrt(np.mean((signal[1200:]-z[1200:]@w)**2))
    return result("held_timing_rmse",err,(data[1200:],signal[1200:]),n=n,lower_is_better=True)


def multi_action_credit(arm,seed,steps,n):
    rng=np.random.default_rng(seed)
    mapping=rng.integers(2,size=16)
    # Separate cue vocabulary at each decision.
    train=rng.integers(4,size=(steps*6,4))+4*np.arange(4)
    test=rng.integers(4,size=(n,4))+4*np.arange(4)
    draws=rng.random(train.shape)
    w=np.zeros(16)
    baseline=0.
    rewards=0
    for cues,u in zip(train,draws):
        p=1/(1+np.exp(-w[cues]))
        a=(u<p).astype(float)
        r=float(np.all(a==mapping[cues]))
        rewards+=r
        delta=r-baseline
        baseline+=.03*delta
        elig=(a-p)*(.9**np.arange(3,-1,-1))
        if arm=="immediate": elig[:-1]=0
        if arm=="shifted": elig=np.roll(elig,1)
        w[cues]=np.clip(w[cues]+.3*delta*elig,-12,12)
    return result("held_trajectory_success",((w[test]>=0)==mapping[test]).all(1).mean(),
                  (test,mapping[test]),n=n,train_rewards=int(rewards))


def grid_graph_transfer(arm,seed,steps,n):
    rng=np.random.default_rng(seed)
    size=11
    deltas=np.array([[1,0],[0,1],[-1,0],[0,-1]])
    # Taught edges supply observed displacements. Labels name body-frame commands.
    samples=rng.integers(4,size=max(16,steps))
    observed=deltas[samples]+rng.normal(0,.03,(len(samples),2))
    starts=rng.integers(size,size=(n,2))
    # Held test is directed two-edge compositions; training sees individual edges.
    paths=rng.integers(4,size=(n,2))
    truth=(starts+deltas[paths].sum(1))%size
    estimate=np.array([observed[samples==i].mean(0) if (samples==i).any()
                       else np.zeros(2) for i in range(4)])
    if arm=="shuffled_grid": estimate=estimate[[1,0,3,2]]
    state=np.exp(2j*np.pi*starts/size)
    for k in range(2):
        if arm=="reset": state=np.exp(2j*np.pi*starts/size)
        state*=np.exp(2j*np.pi*estimate[paths[:,k]]/size)
    decoded=np.round(np.angle(state)*size/(2*np.pi)).astype(int)%size
    return result("held_two_edge_destination_accuracy",(decoded==truth).all(1).mean(),
                  (starts,paths,truth),n=n)


def fb_goal_writes(arm,seed,steps,n):
    rng=np.random.default_rng(seed)
    angles=np.arange(16)*2*np.pi/16
    goals=rng.integers(16,size=steps+n)
    # Random scalar-reward samples. The hidden target is never an update input.
    actions=rng.integers(16,size=steps)
    kernel=np.exp(3*(np.cos(angles[:,None]-angles)-1))
    w=np.zeros((16,16))
    for i,(context,action) in enumerate(zip(goals[:steps],actions)):
        hidden=(3*context+5)%16
        reward=float(action==hidden)
        if arm=="frozen": continue
        credit=kernel[action if arm=="aligned" else (action+4)%16]
        w[context]+= .35*reward*credit
    test=goals[steps:]
    truth=(3*test+5)%16
    return result("hidden_goal_choice_accuracy",(w[test].argmax(1)==truth).mean(),(test,truth),n=n)


def mb_apl_reversal(arm,seed,steps,n):
    rng=np.random.default_rng(seed)
    x,proj=fixed_features(rng,n+64,12,96)
    drive=np.maximum(x@proj,0)
    drive*=drive>=np.partition(drive,-8,axis=1)[:,-8,None]
    if arm!="apl_off":
        drive/=np.linalg.norm(drive,axis=1,keepdims=True).clip(.1)
    else:
        drive/=np.median(np.linalg.norm(drive,axis=1)).clip(.1)
    ctx=(x[:,0]>0).astype(int)
    target=(x[:,1]+.4*x[:,2]>0).astype(float)
    w=np.zeros((2,96))
    schedule=rng.integers(64,size=2*steps)
    for step,i in enumerate(schedule):
        c=ctx[i]
        desired=1-target[i] if step>=steps and c==0 else target[i]
        p=np.clip(.5+w[c]@drive[i],0,1)
        destination=1-c if arm=="dan_permuted" else c
        w[destination]+=.15*(desired-p)*drive[i]
    test_target=np.where(ctx[64:]==0,1-target[64:],target[64:])
    ok=((w[ctx[64:]]*drive[64:]).sum(1)>=0)==test_target
    return result("held_reversal_and_preservation_accuracy",ok.mean(),(x[64:],test_target),n=n,
                  reversed_accuracy=float(ok[ctx[64:]==0].mean()) if (ctx[64:]==0).any() else None,
                  preserved_accuracy=float(ok[ctx[64:]==1].mean()) if (ctx[64:]==1).any() else None)


def sleep_order_scaling(arm,seed,steps,n):
    rng=np.random.default_rng(seed)
    x=rng.normal(size=(2*(64+n),16))
    lab=np.repeat(np.arange(2),64+n)
    y=(np.sum(x*rng.normal(size=(2,16))[lab],1)>0).astype(float)
    x=np.column_stack((x,np.eye(2)[lab]))
    w=np.zeros(x.shape[1])
    old=np.arange(64)
    new=64+n+np.arange(64)
    schedule=np.r_[rng.choice(old,steps),rng.choice(new,steps)]
    replay=np.tile(old,max(1,steps//128))
    permuted=replay[rng.permutation(len(replay))]
    def update(i):
        nonlocal w
        w+=.02*(y[i]-x[i]@w)*x[i]
    for i in schedule: update(i)
    for i in permuted if arm=="shuffled_scaled" else replay: update(i)
    if arm!="ordered_unscaled": w*=.85
    held=np.r_[64:64+n,128+n:128+2*n]
    # Centered threshold makes scaling meaningful for subsequent writes, not by itself.
    for i in schedule[-max(1,steps//5):]: update(i)
    correct=((x[held]@w>=.5)==y[held]).reshape(2,n).mean(1)
    return result("held_old_rule_accuracy",correct[0],(x[held],y[held]),n=n,
                  new_accuracy=float(correct[1]),replay_updates=len(replay))


class FeedbackGate(nn.Module):
    def __init__(self,arm):
        super().__init__()
        self.arm=arm
        self.enc=nn.Sequential(nn.Linear(10,48),nn.Tanh())
        self.mix=nn.Linear(96,48)
        self.gate=nn.Linear(96,48)
        self.out=nn.Linear(48,8)

    def forward(self,x):
        u=self.enc(x)
        h=torch.zeros_like(u)
        for _ in range(3):
            source=torch.cat((u,h if self.arm!="open_loop" else torch.zeros_like(h)),1)
            gain=self.gate(source).sigmoid() if self.arm=="learned" else .5
            h=(1-gain)*h+gain*self.mix(source).tanh()
        return self.out(h)


def thalamic_feedback_gate(arm,seed,steps,n):
    rng=np.random.default_rng(seed)
    def batch(k,held):
        state=rng.integers(8,size=k)
        o=rng.integers(2,size=k)
        p=1-o if held else o
        x=np.column_stack((np.eye(8)[state],o,p)).astype(np.float32)
        transform=lambda s,g:np.where(g==0,(s+1)%8,(3*s)%8)
        return x,transform(transform(state,o),p)
    x,y=batch(768,False)
    test,truth=batch(n,True)
    # Shared support of single operations makes the held composition meaningful.
    s=rng.integers(8,size=384); o=rng.integers(2,size=len(s))
    single=np.column_stack((np.eye(8)[s],o,np.full(len(s),-1))).astype(np.float32)
    x=np.concatenate((x,single)); y=np.r_[y,np.where(o==0,(s+1)%8,(3*s)%8)]
    torch.manual_seed(seed)
    model=FeedbackGate(arm)
    curve=fit_torch(model,tensor(x),tensor(y),steps,seed)
    with torch.no_grad(): pred=model(tensor(test)).argmax(1).numpy()
    return result("held_cross_order_accuracy",(pred==truth).mean(),(test,truth),n=n,curve=curve,
                  active_feedback=arm!="open_loop")


OPS = "+-@"
CODES = "rtu"
EOS = 256
HELD = ((0, 1), (1, 2), (2, 0))

def op(k, a, b):
    return (a + b if k == 0 else a - b if k == 1 else a + 2*b) % 17

def single(k, a, b):
    return f"{CODES[k]}{a:02d}{OPS[k]}{b:02d}="

def problem(i, j, a, b, c):
    middle = op(i, a, b)
    return (f"c{a:02d}{OPS[i]}{b:02d}{OPS[j]}{c:02d}=",
            single(i, a, b), single(j, middle, c), middle, op(j, middle, c))

def chars(lines):
    return torch.tensor([list(s.encode("ascii")) for s in lines], dtype=torch.long)

def answers(numbers):
    return torch.tensor([list(f"{n:02d}".encode()) + [EOS] for n in numbers])

def fixtures(seed, split, train_n, test_n):
    rng = np.random.default_rng(seed)
    kinds = 2 if split == "strict" else 3
    held = ((0, 1), (1, 0)) if split == "strict" else HELD
    taught = [(i,j) for i in range(kinds) for j in range(kinds) if (i,j) not in held]
    # Fully teach the single-step executor. Novelty is in directed combinations.
    primitive = [(k,a,b) for k in range(kinds) for a in range(17) for b in range(17)]

    def triples(i, j, n, seen):
        rows = []
        while len(rows) < n:
            a,b,c = map(int,rng.integers(17,size=3))
            key = (i,j,a,b,c)
            if key in seen:
                continue
            seen.add(key)
            rows.append(problem(i,j,a,b,c))
        return rows

    seen = set()
    train = sum((triples(i,j,train_n,seen) for i,j in taught), [])
    panels = {OPS[i]+OPS[j]: triples(i,j,test_n,seen) for i,j in taught+list(held)}
    assert len(seen)==len(train)+sum(map(len,panels.values()))
    return primitive, train, panels, [OPS[i]+OPS[j] for i,j in held]

class Solver(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(257,24)
        self.net = nn.Sequential(nn.Linear(7*24,256), nn.Tanh(), nn.Linear(256,256),
                                 nn.Tanh(), nn.Linear(256,3*257))
    def forward(self,x):
        return self.net(self.embed(x).flatten(1)).reshape(-1,3,257)

class Router(nn.Module):
    def __init__(self, arm):
        super().__init__()
        self.arm = arm
        self.embed = nn.Embedding(257,48)
        self.pos = nn.Parameter(torch.randn(13,48)*.15)
        self.keys = nn.Linear(48,48)
        self.values = nn.Linear(48,48)
        self.query = nn.Parameter(torch.randn(7,48)*.15)
        self.read = nn.Sequential(nn.Linear(96,96),nn.Tanh(),nn.Linear(96,257))
    def forward(self,x):
        tok = self.embed(x)+self.pos
        key_input = self.pos.expand_as(tok) if self.arm=="window" else tok
        weights = (self.keys(key_input) @ self.query.T).transpose(1,2) / 48**.5
        v = self.values(tok)
        if self.arm=="gate":
            # Equivalent to sequential register interpolation with an initial zero state.
            g=weights.sigmoid()
            tail=torch.cumprod((1-g).flip(-1),dim=-1).flip(-1)
            w=g*torch.cat((tail[...,1:],torch.ones_like(tail[...,:1])),dim=-1)
        else:
            w=weights.softmax(-1)
        h=w@v
        return self.read(torch.cat((h,self.query.expand(len(x),-1,-1)),dim=-1))

def router_input(rows, phase, middle=None):
    if phase==0:
        return chars(["1"+r[0]+"__" for r in rows])
    prefix=chars(["2"+r[0]+"__" for r in rows])
    if middle is None:
        middle=chars([f"{r[3]:02d}" for r in rows])
    prefix[:,-2:]=middle
    return prefix

@torch.no_grad()
def infer(model,x):
    return torch.cat([model(batch).argmax(-1) for batch in x.split(256)])

@torch.no_grad()
def score(solver,model,rows):
    p1,p2=(chars([r[k] for r in rows]) for k in (1,2))
    y1,y2=(answers([r[k] for r in rows]) for k in (3,4))
    a1=infer(model,router_input(rows,0))
    mid=infer(solver,a1)
    a2=infer(model,router_input(rows,1,mid[:,:2]))
    a2_true=infer(model,router_input(rows,1))
    final=infer(solver,a2)
    oracle1=infer(solver,p1)
    oracle2=infer(solver,p2)
    cnt=lambda a,b:int((a==b).all(1).sum())
    # Fixed-width inputs pass generated bytes straight through, including errors.
    return dict(n=len(rows), rewrite1=cnt(a1,p1), intermediate=cnt(mid,y1),
                rewrite2_true_mid=cnt(a2_true,p2), rewrite2_generated_mid=cnt(a2,p2),
                final=cnt(final,y2), oracle_step1=cnt(oracle1,y1), oracle_step2=cnt(oracle2,y2),
                oracle_both=int(((oracle1==y1).all(1)&(oracle2==y2).all(1)).sum()),
                generated=[dict(first=a,mid=b,second=c,final=d) for a,b,c,d in
                           zip(a1.tolist(),mid.tolist(),a2.tolist(),final.tolist())])

def operator_binding(arm,seed,steps,n):
    primitive, train, panels, held=fixtures(seed,"balanced",max(32,steps),n)
    torch.manual_seed(seed)
    solver=Solver()
    sx=chars([single(*r) for r in primitive])
    sy=answers([op(*r) for r in primitive])
    solver_steps=steps if steps<100 else max(2048,steps*2)
    solver_curve=fit_torch(solver,sx,sy,solver_steps,seed+11)
    torch.manual_seed(seed+101)
    model=Router({"static":"window","register":"gate"}.get(arm,arm))
    x=torch.cat([router_input(train,0),router_input(train,1)])
    y=torch.cat([chars([r[k] for r in train]) for k in (1,2)])
    curve=fit_torch(model,x,y,steps,seed+19)
    rows={kind:score(solver,model,panels[kind]) for kind in held}
    minimum_oracle=min(r["oracle_both"]/r["n"] for r in rows.values())
    sealed=sum([panels[k] for k in held],[])
    return result("held_composed_final_accuracy",sum(r["final"] for r in rows.values())/(len(held)*n),
                  (chars([r[0] for r in sealed]).numpy(),answers([r[4] for r in sealed]).numpy()),
                  n=len(held)*n,panels=rows,curve=curve,solver_curve=solver_curve,solver_updates=solver_steps,
                  oracle_qualified=minimum_oracle>=.85,oracle_min_accuracy=minimum_oracle)


class DiagonalState(nn.Module):
    def __init__(self):
        super().__init__()
        self.drive=nn.Parameter(torch.randn(14,24)*.1)
        self.decay=nn.Parameter(torch.linspace(0,3,24))
        self.head=nn.Sequential(nn.Linear(24,48),nn.Tanh(),nn.Linear(48,4))

    def evolve(self,x):
        z=self.decay.sigmoid()
        h=torch.zeros(len(x),len(z))
        for u in x.unbind(1):
            h=z*h+(1-z)*self.drive[u]
        return h

    @torch.no_grad()
    def traces(self,x,full):
        z=self.decay.sigmoid()
        h=torch.zeros(len(x),len(z))
        e=torch.zeros(len(x),14,len(z))
        q=torch.zeros_like(h)
        batch=torch.arange(len(x))
        for u in x.unbind(1):
            current=self.drive[u]
            e.mul_(z if full else 0)
            e[batch,u]+=1-z
            q=z*q if full else torch.zeros_like(q)
            q+=(h-current)*z*(1-z)
            h=z*h+(1-z)*current
        return h,e,q


def diagonal_trace(arm,seed,steps,n):
    rng=np.random.default_rng(seed)
    def examples(count,gap):
        cue=rng.integers(4,size=count)
        junk=rng.integers(4,12,size=(count,gap))
        return np.column_stack((cue,junk,np.full(count,12))),cue
    train,y=examples(1536,8)
    test,truth=examples(n,16)
    x,y=tensor(train),tensor(y)
    torch.manual_seed(seed)
    m=DiagonalState()
    opt=torch.optim.AdamW(m.parameters(),lr=.004)
    # Qualification: explicit online traces reproduce diagonal temporal
    # derivatives against full backprop when given the same learning signal.
    check=x[:4]
    direction=torch.randn(4,24)
    g=torch.autograd.grad((m.evolve(check)*direction).sum(),(m.drive,m.decay))
    _,e,q=m.traces(check,True)
    assert torch.allclose(g[0],torch.einsum("bh,bvh->vh",direction,e),atol=2e-6)
    assert torch.allclose(g[1],(direction*q).sum(0),atol=2e-6)
    for _ in range(steps):
        ix=rng.integers(len(x),size=48)
        batch=x[ix]
        opt.zero_grad()
        if arm=="bptt":
            loss=nn.functional.cross_entropy(m.head(m.evolve(batch)),y[ix])
            loss.backward()
        else:
            h,e,q=m.traces(batch,arm=="trace")
            h.requires_grad_()
            loss=nn.functional.cross_entropy(m.head(h),y[ix])
            loss.backward()
            m.drive.grad=torch.einsum("bh,bvh->vh",h.grad,e)
            m.decay.grad=(h.grad*q).sum(0)
        opt.step()
    with torch.no_grad(): pred=m.head(m.evolve(tensor(test))).argmax(1).numpy()
    return result("held_long_gap_accuracy",(pred==truth).mean(),(test,truth),n=n,
                  trace_gradient_check=True)


ROSTER["diagonal_trace"]=("trace","no_trace","bptt")
NOTES["diagonal_trace"]="Online eligibility on a diagonal state, against detached state and exact BPTT; delayed cue extrapolation."


def atomic(path,obj):
    temp=path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(obj,indent=2,allow_nan=False)+"\n")
    temp.replace(path)


def run(profile,blocks,out):
    torch.set_num_threads(1)
    if profile not in PROFILES or len(set(blocks))!=len(blocks) or not set(blocks)<=set(ROSTER):
        raise ValueError("Unknown profile or invalid blocks list.")
    blocks=list(blocks or ROSTER)
    seeds,steps,n=PROFILES[profile]
    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    lock=dict(profile=profile,seeds=list(seeds),base_updates=steps,panel_base_n=n,
              roster={b:list(ROSTER[b]) for b in blocks},
              scope={b:NOTES[b] for b in blocks},
              implementation_sha256=hashlib.sha256(IMPLEMENTATION.read_bytes()).hexdigest(),
              runtime=dict(numpy=np.__version__,torch=str(torch.__version__)),
              qualification="Engineering screen. Report floor or saturated comparisons. "
                            "Operator binding needs at least 85% both-step oracle on each held pair to interpret end-to-end.")
    lockfile=out/"LOCK.json"
    if lockfile.exists():
        if json.loads(lockfile.read_text())!=lock:
            raise ValueError("The existing lock differs. Choose a new RUN_ID.")
    else:
        atomic(lockfile,lock)
    lock_hash=hashlib.sha256(lockfile.read_bytes()).hexdigest()
    rows=[]
    for block in blocks:
        for seed in seeds:
            signatures=set()
            for arm in ROSTER[block]:
                dst=out/f"{block}-{seed}-{arm}.json"
                key=dict(block=block,arm=arm,seed=seed,lock_sha256=lock_hash)
                if dst.exists():
                    receipt=json.loads(dst.read_text())
                    if receipt["identity"]!=key:
                        raise ValueError(f"Receipt identity mismatch at {dst}")
                else:
                    t=time.perf_counter()
                    value=globals()[block](arm,seed,steps,n)
                    receipt=dict(identity=key,result=value,wall_seconds=time.perf_counter()-t)
                    atomic(dst,receipt)
                r=receipt["result"]
                signatures.add(r["panel_sha256"])
                rows.append((block,seed,arm,r))
                print(f"{block} · {seed} · {arm} · {r['value']:.4f}",flush=True)
            if len(signatures)!=1:
                raise ValueError(f"Mismatched held panels in {block}, seed {seed}")
    report=["# MiniFly gap sweep", "",
            f"Profile **{profile}**. Completed **{len(rows)}** jobs with verified matched panel hashes.", "",
            "| Block | Seed | Arm | Metric | Score | Interpretation |",
            "| --- | ---: | --- | --- | ---: | --- |"]
    for block,seed,arm,r in rows:
        note="Execution check" if profile=="smoke" else "Development"
        if profile!="smoke" and block=="operator_binding" and not r["oracle_qualified"]:
            note=f"Oracle below gate ({r['oracle_min_accuracy']:.1%})"
        report.append(f"| {block} | {seed} | {arm} | {r['metric']} | {r['value']:.4f} | {note} |")
    report+=["","## Scope",""]+[f"- **{b}**. {NOTES[b]}" for b in blocks]
    report+=["","All scores except timing RMSE are fractions correct. Timing RMSE is lower-is-better.",
             "A successful receipt establishes execution. A mechanism claim requires a capable reference and a paired advantage.",
             ""]
    (out/"SCORECARD.md").write_text("\n".join(report))
    return rows
