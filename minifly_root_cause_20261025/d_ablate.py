# Step D: ablations of the same-answer (relational) rule.
#  REP: '1D-additive' s=u_a+v_b ; '8D-additive' s=U_a+V_b in R^8 ; '32D-nonlinear' s=relu(U_a+V_b+c) in R^32 ;
#       'FE-linear' s=w.x with x the real FE-PERM trace (88-d) ; 'FE-8D' s=Wx in R^8.
#  Rule: pull same-answer taught pairs together, push different-answer pairs apart (contrastive, local).
#  Naming: nearest class-centroid of taught states (no extra training) -> evaluates whether held states land on the right sum.
import sys, numpy as np
sys.path.insert(0,str(__import__('pathlib').Path(__file__).resolve().parents[1]/'minifly_learned_bank_v1_20261024')); import data, lb1
W=['RC-D%d'%i for i in range(1,9)]
ALL=[(a,b) for a in range(10) for b in range(10)]
FE=np.array([lb1.trace_for(data.prompt(a,b)) for a,b in ALL]); FE=(FE-FE.mean(0))/FE.std(0).clip(1e-9); IDX={p:i for i,p in enumerate(ALL)}
def shuf_labels(wd,seed):
    r=np.random.default_rng(seed); orb=wd['train']; lab=[a+b for a,b in orb]
    for _ in range(1000):
        p=r.permutation(len(orb)); new=[lab[i] for i in p]
        if np.mean([x==y for x,y in zip(new,lab)])<0.15: break
    m={o:new[i] for i,o in enumerate(orb)}; return {(a,b):m[(min(a,b),max(a,b))] for a,b in wd['train_pairs']}
class Rep:
    def __init__(s,kind,r):
        s.kind=kind
        if kind=='1D-additive': s.U=r.normal(0,1,(10,1)); s.V=r.normal(0,1,(10,1))
        if kind=='8D-additive': s.U=r.normal(0,1,(10,8)); s.V=r.normal(0,1,(10,8))
        if kind=='32D-nonlinear': s.U=r.normal(0,1,(10,32)); s.V=r.normal(0,1,(10,32)); s.c=r.normal(0,1,32)
        if kind=='FE-linear': s.Wm=r.normal(0,0.1,(88,1))
        if kind=='FE-8D': s.Wm=r.normal(0,0.1,(88,8))
    def state(s,P):
        a=np.array([p[0] for p in P]); b=np.array([p[1] for p in P])
        if s.kind in ('1D-additive','8D-additive'): return s.U[a]+s.V[b]
        if s.kind=='32D-nonlinear': return np.maximum(s.U[a]+s.V[b]+s.c,0)
        return FE[[IDX[p] for p in P]]@s.Wm
    def update(s,P,i,j,g,lr):      # g: d(loss)/d(state_i) for each sampled pair (state_j gets -g)
        a=np.array([p[0] for p in P]); b=np.array([p[1] for p in P])
        if s.kind in ('1D-additive','8D-additive'):
            for idx,sg in ((i,1),(j,-1)): np.add.at(s.U,a[idx],-lr*sg*g); np.add.at(s.V,b[idx],-lr*sg*g)
        elif s.kind=='32D-nonlinear':
            pre=s.U[a]+s.V[b]+s.c
            for idx,sg in ((i,1),(j,-1)):
                gg=sg*g*(pre[idx]>0); np.add.at(s.U,a[idx],-lr*gg); np.add.at(s.V,b[idx],-lr*gg); s.c-=lr*gg.sum(0)/len(idx)
        else:
            X=FE[[IDX[p] for p in P]]
            s.Wm-=lr*(X[i].T@g-X[j].T@g)/len(i)
def run(kind,tr,ytr,he,yte,seed,steps=4000,lr=0.05):
    r=np.random.default_rng(seed); R=Rep(kind,r)
    for t in range(steps):
        S=R.state(tr); i,j=r.integers(0,len(tr),(2,4*len(tr))); same=(ytr[i]==ytr[j])[:,None]; d=S[i]-S[j]
        nrm=np.linalg.norm(d,axis=1,keepdims=True)+1e-9; scale=np.sqrt(np.mean(np.sum((S-S.mean(0))**2,1)))+1e-9
        g=np.where(same,d,-d/nrm*np.exp(-nrm/scale)*0.5)          # pull same, push different (scale-free push)
        R.update(tr,i,j,g,lr)
    S=R.state(tr); cls=np.unique(ytr); cent=np.array([S[ytr==c].mean(0) for c in cls])
    pred=lambda Q: cls[np.argmin(((R.state(Q)[:,None,:]-cent[None])**2).sum(-1),1)]
    return np.mean(pred(tr)==ytr), np.mean(pred(he)==yte)
for kind in ('1D-additive','8D-additive','32D-nonlinear','FE-linear','FE-8D'):
    T=[];S=[]
    for w in W:
        wd=data.make_world(w); tr,he=wd['train_pairs'],wd['held_pairs']; yte=np.array([x+y for x,y in he]); sh=shuf_labels(wd,data.seed_of('rc-shuf:'+w))
        for s_ in range(2):
            T.append(run(kind,tr,np.array([x+y for x,y in tr]),he,yte,s_)); S.append(run(kind,tr,np.array([sh[p] for p in tr]),he,yte,s_))
    print(f"{kind:14s} TRUE taught {np.mean([x[0] for x in T]):.2f} held {np.mean([x[1] for x in T]):.2f} | SHUF taught {np.mean([x[0] for x in S]):.2f}",flush=True)
