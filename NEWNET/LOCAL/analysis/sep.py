import numpy as np
from scipy.stats import t as tdist, spearmanr
exec(open("analysis/reg.py").read().split("R1=[r for r")[0])
def cells(r): return r["opts"]["R"]*r["opts"]["C"]*r["opts"]["M"] if r["round"]==1 else sum(p["n"] for p in r["cand"]["pops"])
SIZE={"R","C","M","n","maxn","rd_n"}
def rows(G,keep):
    X,names=design(G,keep); R=[];Y=[];cid=[]
    for i,r in enumerate(G):
        for j,s in enumerate((1,2,4)):
            R.append((i,np.log2(cells(r)*s),np.log2(32*s))); Y.append(r["ladder"][j]); cid.append(i)
    return X,names,np.array(R),np.array(Y),np.array(cid)
def ols_cl(A,Y,cid):
    b=np.linalg.lstsq(A,Y,rcond=None)[0]; e=Y-A@b; Bi=np.linalg.pinv(A.T@A); M=np.zeros((A.shape[1],)*2)
    for c in np.unique(cid): s=A[cid==c].T@e[cid==c]; M+=np.outer(s,s)
    G=len(np.unique(cid)); V=Bi@M@Bi*G/(G-1); se=np.sqrt(np.diag(V)); return b,se,2*tdist.sf(abs(b/se),G-1),e
for gname,G,keep in [("R1",[r for r in P if r["round"]==1],lambda k:k not in SIZE),
                     ("R2+3",[r for r in P if r["round"]>1],lambda k:k not in SIZE and k not in MECH)]:
    X,names,R,Y,cid=rows(G,keep); Xr=X[R[:,0].astype(int)]
    lc=R[:,1]-R[:,1].mean(); ld=R[:,2]-R[:,2].mean()
    A=np.column_stack([np.ones(len(Y)),lc,ld,Xr]); b,se,p,e=ols_cl(A,Y,cid)
    print(f"\n[{gname}] {len(G)} candidates x 3 points; cell counts span 2^{R[:,1].min():.0f}..2^{R[:,1].max():.0f}")
    print(f"   per doubling of cells : {b[1]:+.4f} (se {se[1]:.4f}, p {p[1]:.2g})")
    print(f"   per doubling of data  : {b[2]:+.4f} (se {se[2]:.4f}, p {p[2]:.2g})   R2 {1-e@e/((Y-Y.mean())**2).sum():.2f}")
    # cells x data interaction
    A2=np.column_stack([A,lc*ld]); b2,se2,p2,_=ols_cl(A2,Y,cid); print(f"   cells x data interaction: {b2[-1]:+.4f} (p {p2[-1]:.2g})")
    # per-dimension change of size exponent, one dimension at a time (other dims as main effects)
    dims=sorted({n.split("=")[0] for n in names}); hits=[]
    for d in dims:
        idx=[i for i,n in enumerate(names) if n.split("=")[0]==d]
        A3=np.column_stack([A]+[Xr[:,i]*lc for i in idx]+[Xr[:,i]*ld for i in idx]); b3,se3,p3,_=ols_cl(A3,Y,cid)
        k=A.shape[1]
        for m,i in enumerate(idx): hits.append((p3[k+m],names[i],b3[k+m],"cells")); hits.append((p3[k+len(idx)+m],names[i],b3[k+len(idx)+m],"data"))
    ps=np.array([h[0] for h in hits]); q=bh(ps)
    print("   dimension values that change the per-doubling effect (FDR q<0.2 shown, else top 5):")
    order=np.argsort(ps); shown=[i for i in order if q[i]<0.2] or list(order[:5])
    for i in shown: pv,n,c,w=hits[i]; print(f"     {w:5s} exponent change {c:+.4f} for {n:34s} p {pv:.2g} q {q[i]:.2f}")
print("\n=== holdout: fit on R2+R3 points seen before, predict the 30+ new R3 points")
G=[r for r in P if r["round"]>1]; R3=[r for r in P if r["round"]==3]; newnames={r["name"] for r in R3[94:]}
X,names,R,Y,cid=rows(G,lambda k:k not in SIZE and k not in MECH); Xr=X[R[:,0].astype(int)]
isnew=np.array([G[int(i)]["name"] in newnames for i in R[:,0]])
lc=R[:,1]-R[:,1].mean(); ld=R[:,2]-R[:,2].mean()
for nm,A in [("cells+data only",np.column_stack([np.ones(len(Y)),lc,ld])),("+ structure dims",np.column_stack([np.ones(len(Y)),lc,ld,Xr])),("+ cells x data",np.column_stack([np.ones(len(Y)),lc,ld,lc*ld,Xr]))]:
    tr=~isnew; mu=A[tr].mean(0); mu[0]=0
    b=np.linalg.solve(A[tr].T@A[tr]+1.0*np.diag([0]+[1]*(A.shape[1]-1)),A[tr].T@Y[tr]); pr=A[isnew]@b; yt=Y[isnew]
    print(f"  {nm:18s}: new-point R2 {1-((yt-pr)**2).sum()/((yt-yt.mean())**2).sum():+.2f}  spearman {spearmanr(pr,yt)[0]:+.2f}  (n points {isnew.sum()})")
# refit on new points alone: cells and data exponents
A=np.column_stack([np.ones(isnew.sum()),lc[isnew],ld[isnew]]); b,se,p,_=ols_cl(A,Y[isnew],cid[isnew])
print(f"  new points alone: cells {b[1]:+.4f} (p {p[1]:.2g}), data {b[2]:+.4f} (p {p[2]:.2g})")
