# Step E: does FE-PERM's max-normalisation (read() = p / max(p)) damage the additive structure the rule needs?
import sys, numpy as np
sys.path.insert(0,str(__import__('pathlib').Path(__file__).resolve().parents[1]/'minifly_learned_bank_v1_20261024')); import data, lb1
FM=lb1.fastmem; ALL=[(a,b) for a in range(10) for b in range(10)]
def trace(prompt, raw):
    fe=FM.FEPerm(); t=0.; fe.t=t
    for b in bytes(prompt): t+=lb1.DELTA; fe.advance(t); fe.feed(b,t)
    t+=2*lb1.DELTA; fe.advance(t); return fe.p.copy() if raw else fe.read()
OH=np.array([np.r_[np.eye(10)[a],np.eye(10)[b],1] for a,b in ALL])
for raw in (False,True):
    X=np.array([trace(data.prompt(a,b),raw) for a,b in ALL]); c,_,_,_=np.linalg.lstsq(OH,X,rcond=None)
    print("raw p" if raw else "read() (max-normalised)", " affine-in-operands residual: %.4f"%(np.linalg.norm(OH@c-X)/np.linalg.norm(X-X.mean(0))))
# rerun the relational rule through each version of the trace
src=open('d_ablate.py').read()
pre=src[:src.index("for kind in")]
for raw in (False,True):
    ns={'__file__':__file__}; exec(pre,ns)
    X=np.array([trace(data.prompt(a,b),raw) for a,b in ALL]); ns['FE'][:]=(X-X.mean(0))/X.std(0).clip(1e-9)
    T=[];S=[]
    for w in ns['W']:
        wd=data.make_world(w); tr,he=wd['train_pairs'],wd['held_pairs']; yte=np.array([x+y for x,y in he]); sh=ns['shuf_labels'](wd,data.seed_of('rc-shuf:'+w))
        for s_ in range(2):
            T.append(ns['run']('FE-linear',tr,np.array([x+y for x,y in tr]),he,yte,s_)); S.append(ns['run']('FE-linear',tr,np.array([sh[p] for p in tr]),he,yte,s_))
    print(("raw p    " if raw else "read()   "), "relational rule, linear map: TRUE taught %.2f held %.2f | SHUF taught %.2f"%(np.mean([x[0] for x in T]),np.mean([x[1] for x in T]),np.mean([x[0] for x in S])),flush=True)
