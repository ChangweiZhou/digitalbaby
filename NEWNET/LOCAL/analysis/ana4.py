import json, numpy as np, sys
from scipy.stats import spearmanr, hypergeom
sys.path.insert(0,"analysis")
exec(open("analysis/ana3.py").read().split("MECH=")[0])
for r in P: L=r["ladder"]; r["s1"]=L[1]-L[0]; r["s2"]=L[2]-L[1]
print("spearman slope vs ladder[0]:",round(spearmanr([r['slope'] for r in P],[r['ladder'][0] for r in P])[0],2),
      " slope vs ladder[2]:",round(spearmanr([r['slope'] for r in P],[r['ladder'][2] for r in P])[0],2),
      " s1 vs s2:",round(spearmanr([r['s1'] for r in P],[r['s2'] for r in P])[0],2))
print("share with s2 < s1 (curve flattens):",round(np.mean([r['s2']>r['s1'] for r in P]),2), " share s2>=0 (no gain 2x->4x):",round(np.mean([r['s2']>=0 for r in P]),2))
print("best s2 (2x->4x step):"); [print(f"   {r['name']} s1 {r['s1']:+.3f} s2 {r['s2']:+.3f} ladder {r['ladder']}") for r in sorted(P,key=lambda r:r['s2'])[:8]]
MECH=["tag","tautag","cap","gate","gth","turn","boost","otr","hubf","hubm"]
groups=[("R1",[r for r in P if r["round"]==1],None),("R2+3",[r for r in P if r["round"]>1],lambda k:k not in MECH),("R3 mech",[r for r in P if r["round"]==3],lambda k:k in MECH)]
for metric in ["s2"]:
    print(f"## enrichment in top 10% by {metric}")
    for gname,G,keep in groups:
        N=len(G); top=set(id(r) for r in sorted(G,key=lambda r:r[metric])[:max(5,N//10)]); n=len(top); F=[feats(r) for r in G]; T=0; hits=[]
        for k in F[0]:
            if keep and not keep(k): continue
            vals={}
            for r,f in zip(G,F): vals.setdefault(f[k],[]).append(r)
            for v,rs in vals.items():
                T+=1; K=len(rs); x=sum(id(r) in top for r in rs); p=hypergeom.sf(x-1,N,K,n)
                if p<0.05: hits.append((p,k,v,x,K,n,N))
        print(f"  [{gname}] tests {T}, chance hits ~{T*0.05:.1f}")
        for p,k,v,x,K,n_,N_ in sorted(hits): print(f"     {k}={v}: {x}/{n_} vs expected {K*n_/N_:.1f}, p={p:.4f}")
