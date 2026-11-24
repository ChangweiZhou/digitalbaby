import json, numpy as np
from scipy.stats import t as tdist
exec(open("analysis/ana3.py").read().split("MECH=")[0])
MECH=["tag","tautag","cap","gate","gth","turn","boost","otr","hubf","hubm"]
for r in P: L=r["ladder"]; r["s2"]=L[2]-L[1]
def design(G,keep):
    F=[feats(r) for r in G]; cols=[]; names=[]
    for k in F[0]:
        if not keep(k): continue
        vals=sorted({str(f[k]) for f in F})
        if len(vals)<2: continue
        for v in vals[1:]:   # first value = baseline
            c=np.array([str(f[k])==v for f in F],float)
            if c.std()>0: cols.append(c); names.append(f"{k}={v} (vs {vals[0]})")
    return np.column_stack(cols), names
def bh(p):
    o=np.argsort(p); q=np.empty_like(p); m=len(p); prev=1
    for i in range(m-1,-1,-1): j=o[i]; prev=min(prev,p[j]*m/(i+1)); q[j]=prev
    return q
def ridge_cv(X,Y,k=10,seed=0):
    rng=np.random.default_rng(seed); idx=rng.permutation(len(Y)); folds=np.array_split(idx,k); best=-9
    for lam in np.logspace(-1,3,13):
        pr=np.empty_like(Y)
        for f in folds:
            tr=np.setdiff1d(idx,f); mu=X[tr].mean(0); ym=Y[tr].mean(); A=X[tr]-mu
            b=np.linalg.solve(A.T@A+lam*np.eye(X.shape[1]),A.T@(Y[tr]-ym)); pr[f]=ym+(X[f]-mu)@b
        best=max(best,1-((Y-pr)**2).sum()/((Y-Y.mean())**2).sum())
    return best
def fit(G,name,y,keep):
    X,names=design(G,keep); Y=np.array([r[y] for r in G]); A=np.column_stack([np.ones(len(Y)),X])
    b,*_=np.linalg.lstsq(A,Y,rcond=None); res=Y-A@b; n,p=A.shape; s2=res@res/(n-p)
    se=np.sqrt(np.diag(s2*np.linalg.pinv(A.T@A))); tv=b/se; pv=2*tdist.sf(abs(tv),n-p)
    R2=1-res@res/((Y-Y.mean())**2).sum(); aR2=1-(1-R2)*(n-1)/(n-p); cv=ridge_cv(X,Y)
    q=bh(pv[1:])
    print(f"\n[{name}] y={y} N={n} params={p-1} R2={R2:.2f} adjR2={aR2:.2f} CV-R2(ridge)={cv:.2f}")
    for i in np.argsort(pv[1:])[:6]: print(f"   {names[i]:34s} coef {b[i+1]:+.4f}  p {pv[i+1]:.4f}  FDR q {q[i]:.2f}")
R1=[r for r in P if r["round"]==1]; R23=[r for r in P if r["round"]>1]; R3=[r for r in P if r["round"]==3]
for y in ["slope","s2"]:
    fit(R1,"R1",y,lambda k:True); fit(R23,"R2+3 structure",y,lambda k:k not in MECH); fit(R3,"R3 mechanisms only",y,lambda k:k in MECH)
