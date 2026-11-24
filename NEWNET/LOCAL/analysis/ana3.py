import json, numpy as np
from scipy.stats import hypergeom
D="./"
P=[]
for f,rd in [("round1.jsonl",1),("round2_pop.jsonl",2),("round3_mech.jsonl",3)]:
    for l in open(D+f):
        r=json.loads(l)
        if "slope" in r: r["round"]=rd; r["m4"]=r["ladder"][2]; P.append(r)
def feats(r):
    if r["round"]==1: return {k:v for k,v in r["opts"].items()}
    c=r["cand"]; g=dict(c["g"]); ps=c["pops"]
    f={k:v for k,v in g.items()}
    f.update(K=len(ps), maxn=max(p["n"] for p in ps), anyFD=int(any(p["FD"] for p in ps)), longtau=int(any(p["thi"]>=64 for p in ps)),
             nrd=sum(p["rd"] for p in ps), anylp=int(any(p["lp"] for p in ps)), anyld=int(any(p["ld"] for p in ps)), anyad=int(any(p["ad"] for p in ps)),
             maxa=max(p["a"] for p in ps), rd_a=max(p["a"] for p in ps if p["rd"]), rd_n=sum(p["n"] for p in ps if p["rd"]), rd_tau=max(p["thi"] for p in ps if p["rd"]),
             rd_FD=max(p["FD"] for p in ps if p["rd"]), rd_np=max(p["np"] for p in ps if p["rd"]))
    if r["round"]==2: [f.pop(k,None) for k in []]
    return f
MECH=["tag","tautag","cap","gate","gth","turn","boost","otr","hubf","hubm"]
groups=[("R1",[r for r in P if r["round"]==1],None),("R2+3",[r for r in P if r["round"]>1],lambda k:k not in MECH),("R3 mech",[r for r in P if r["round"]==3],lambda k:k in MECH)]
for metric in ["cl","m4","slope"]:
    print(f"\n##### metric {metric}: values over-represented in top 10% (hypergeometric p<0.05), with best value reached")
    for gname,G,keep in groups:
        N=len(G); top=set(id(r) for r in sorted(G,key=lambda r:r[metric])[:max(5,N//10)]); n=len(top)
        F=[feats(r) for r in G]; hits=[]
        for k in F[0]:
            if keep and not keep(k): continue
            vals={}
            for r,f in zip(G,F): vals.setdefault(f[k],[]).append(r)
            for v,rs in vals.items():
                K=len(rs); x=sum(id(r) in top for r in rs)
                p=hypergeom.sf(x-1,N,K,n)
                if p<0.05: hits.append((p,k,v,x,K,n,N,min(r[metric] for r in rs)))
        print(f"  [{gname}] N={N} top={n}")
        for p,k,v,x,K,n_,N_,b in sorted(hits): print(f"     {k}={v}: {x}/{n_} in top vs {K}/{N_} overall (expected {K*n_/N_:.1f}), p={p:.4f}, best {metric} {b:.3f}")
        if not hits: print("     none")
print("\nexpected false hits: ~5% of all value-tests; count tests:")
