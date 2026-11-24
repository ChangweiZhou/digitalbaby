import json, numpy as np
from scipy.stats import spearmanr
D="./"
P=[]
for f,rd in [("round1.jsonl",1),("round2_pop.jsonl",2),("round3_mech.jsonl",3)]:
    for l in open(D+f):
        r=json.loads(l)
        if "slope" not in r: continue
        r["round"]=rd
        if rd==1:
            o=r["opts"]; r["cells"]=o["R"]*o["C"]*o["M"]
        else:
            c=r["cand"]; r["cells"]=sum(p["n"] for p in c["pops"]); r["act"]=sum(p["n"]*p["a"] for p in c["pops"])
        r["m4"]=r["ladder"][2]; r["fg"]=np.mean(list(r["forget"].values()))
        # first-visit score of each stream and end score
        m=np.array(r["mat"]); r["first"]=np.mean([m[0,0],m[1,1],m[2,2],m[3,3]])
        P.append(r)
print("N",len(P))
def q(x): x=np.array(x); return f"{np.min(x):.3f} {np.percentile(x,10):.3f} {np.median(x):.3f} {np.percentile(x,90):.3f} {np.max(x):.3f}"
for rd in (1,2,3,0):
    S=[r for r in P if rd==0 or r["round"]==rd]
    print(f"round {rd or 'all'} n={len(S)}")
    for k in ["cl","slope","m4","fg","first","secs"]: print(f"   {k:6s} min/p10/med/p90/max", q([r[k] for r in S]))
    print("   over16min",sum(r["secs"]>960 for r in S))
# correlations
A=P
for a,b in [("cl","slope"),("cl","m4"),("slope","m4"),("cl","secs"),("slope","secs"),("m4","secs"),("cl","cells"),("slope","cells"),("m4","cells"),("cl","fg"),("cl","first")]:
    print(f"spearman {a}-{b}: {spearmanr([r[a] for r in A],[r[b] for r in A])[0]:+.2f}")
S=[r for r in P if r["round"]>1]
for a in ["cl","slope","m4","secs"]: print(f"r2+3 spearman act-{a}: {spearmanr([r['act'] for r in S],[r[a] for r in S])[0]:+.2f}")
# slope reliability
mono=sum(r["ladder"][0]>=r["ladder"][1]>=r["ladder"][2] for r in P); print("monotone ladders",mono,"/",len(P))
d1=np.array([r["ladder"][1]-r["ladder"][0] for r in P]); d2=np.array([r["ladder"][2]-r["ladder"][1] for r in P])
print("step1 median",np.median(d1),"step2 median",np.median(d2),"spearman step1-step2",spearmanr(d1,d2)[0])
# pareto on cl and slope
F=[r for r in P if not any((o["cl"]<=r["cl"] and o["slope"]<=r["slope"] and (o["cl"]<r["cl"] or o["slope"]<r["slope"])) for o in P)]
print("PARETO cl/slope")
for r in sorted(F,key=lambda r:r["cl"]): print(f"   {r['name']:8s} R{r['round']} cl {r['cl']:.3f} slope {r['slope']:+.3f} ladder {r['ladder']} forget {r['fg']:.3f} cells {r['cells']} secs {r['secs']}")
print("TOP10 cl"); [print(f"   {r['name']:8s} cl {r['cl']:.3f} slope {r['slope']:+.3f} m4 {r['m4']:.3f} cells {r['cells']} secs {r['secs']}") for r in sorted(P,key=lambda r:r["cl"])[:10]]
print("TOP10 m4"); [print(f"   {r['name']:8s} cl {r['cl']:.3f} slope {r['slope']:+.3f} m4 {r['m4']:.3f} cells {r['cells']} secs {r['secs']}") for r in sorted(P,key=lambda r:r["m4"])[:10]]
print("TOP10 slope"); [print(f"   {r['name']:8s} cl {r['cl']:.3f} slope {r['slope']:+.3f} ladder {r['ladder']} cells {r['cells']} secs {r['secs']}") for r in sorted(P,key=lambda r:r["slope"])[:10]]
# per-dimension effects: r1 opts; r2+3 globals + pop aggregates; r3 mech
def eff(rows,get,name):
    vals={}
    for r in rows:
        v=get(r); vals.setdefault(v,[]).append(r)
    if len(vals)<2: return None
    cl={v:np.median([r["cl"] for r in rs]) for v,rs in vals.items()}; sl={v:np.median([r["slope"] for r in rs]) for v,rs in vals.items()}
    rng_cl=max(cl.values())-min(cl.values()); rng_sl=max(sl.values())-min(sl.values())
    return (rng_cl,rng_sl,name," ".join(f"{v}:{cl[v]:.3f}/{sl[v]:+.3f}(n{len(vals[v])})" for v in sorted(vals,key=str)))
out=[]
R1=[r for r in P if r["round"]==1]
for k in R1[0]["opts"]: out.append(("R1",eff(R1,lambda r,k=k:r["opts"][k],k)))
for k in S[0]["cand"]["g"]:
    if k in ["tag","tautag","cap","gate","gth","turn","boost","otr","hubf","hubm"]: continue
    out.append(("R2+3",eff(S,lambda r,k=k:r["cand"]["g"][k],k)))
pk=["n","FP","FD","B","a","lp","ld","ad","np","rho","sth","dg"]
out.append(("R2+3",eff(S,lambda r:len(r["cand"]["pops"]),"K")))
for k in pk: out.append(("R2+3",eff(S,lambda r,k=k:r["cand"]["pops"][0][k],"pop0."+k)))
out.append(("R2+3",eff(S,lambda r:tuple(r["cand"]["pops"][0][x] for x in ("tlo","thi")),"pop0.tau")))
out.append(("R2+3",eff(S,lambda r:max(p["n"] for p in r["cand"]["pops"]),"max n")))
out.append(("R2+3",eff(S,lambda r:int(np.log2(r["cells"])),"log2 cells")))
out.append(("R2+3",eff(S,lambda r:int(np.log2(max(r["act"],1))),"log2 active/byte")))
out.append(("R2+3",eff(S,lambda r:int(any(p["FD"] for p in r["cand"]["pops"])),"any distal")))
out.append(("R2+3",eff(S,lambda r:sum(p["rd"] for p in r["cand"]["pops"]),"# readout pops")))
out.append(("R2+3",eff(S,lambda r:int(any(p["thi"]>=64 for p in r["cand"]["pops"])),"any long tau")))
out.append(("R2+3",eff(S,lambda r:r["round"],"round (lt2 vs lt3)")))
R3=[r for r in P if r["round"]==3]
for k in ["tag","tautag","cap","gate","gth","turn","boost","otr","hubf","hubm"]: out.append(("R3",eff(R3,lambda r,k=k:r["cand"]["g"][k],k)))
out=[(s,e) for s,e in out if e]
print("DIMENSIONS sorted by spread of median cl (value: median cl / median slope)")
for s,e in sorted(out,key=lambda x:-x[1][0]): print(f"  [{s}] {e[2]:18s} dcl {e[0]:.3f} dslope {e[1]:.3f} | {e[3]}")
# output-side settings co-occurrence among top 20 cl in r2+3
T=sorted(S,key=lambda r:r["cl"])[:20]
for k in ["orep","eta_o","NO","rho_o","eta"]:
    from collections import Counter; print("top20 cl r2+3",k,Counter(r["cand"]["g"][k] for r in T), "| all",Counter(r["cand"]["g"][k] for r in S))
# matrix: mean learning curves
M=np.array([r["mat"] for r in P]); print("mean mat over all (rows=after shak,emma,code,moby,shak; cols shak emma code moby)"); print(np.round(M.mean(0),3))
print("=== clustering check")
for lo,hi in [(3.340,3.352),(3.30,3.37)]:
    for rd in (1,2,3): print(f" round {rd} share cl in [{lo},{hi}]:", round(np.mean([lo<=r['cl']<=hi for r in P if r['round']==rd]),2))
import sys; sys.path.insert(0,D)
from ltb import S, held
def H(p,x):
    c=np.bincount(np.frombuffer(bytes(x),np.uint8) if not isinstance(x,np.ndarray) else x,minlength=256)
    return float(-(c*np.log(p)).sum()/c.sum())
def dist(x,a=0.5):
    x=np.frombuffer(bytes(x),np.uint8) if not isinstance(x,np.ndarray) else x
    c=np.bincount(x,minlength=256)+a; return c/c.sum()
names=["shak","emma","code","moby"]
last=dist(S["shak"][64000:128000])
print(" unigram from last shak 64k -> held:",{k:round(H(last,held(k,10000)),3) for k in names}, "mean",round(np.mean([H(last,held(k,10000)) for k in names]),3))
for k in names:
    pk=dist(S[k][:64000]); print(f" unigram fit on {k} -> held:",{j:round(H(pk,held(j,10000)),3) for j in names})
# how many candidates are within 0.03 of the shak-unigram mean on every column of final row
u={k:H(last,held(k,10000)) for k in names}
near=[r for r in P if all(abs(r["mat"][-1][i]-u[k])<0.05 for i,k in enumerate(names))]
print(" candidates whose final row is within 0.05 of shak-unigram on all 4:",len(near),"/",len(P))
print(" moby 4x vs moby unigram:",round(H(dist(S['moby'][:128000]),held('moby',10000)),3))
