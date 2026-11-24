import json, numpy as np, sys
D="./"; sys.path.insert(0,D)
from ltb import S, held
P=[]
for f,rd in [("round1.jsonl",1),("round2_pop.jsonl",2),("round3_mech.jsonl",3)]:
    for l in open(D+f):
        r=json.loads(l)
        if "slope" in r: r["round"]=rd; P.append(r)
def bigram(tr,te,a=0.5):
    tr=np.asarray(tr,np.int64); te=np.asarray(te,np.int64)
    c=np.zeros((256,256))+a; np.add.at(c,(tr[:-1],tr[1:]),1); p=c/c.sum(1,keepdims=True)
    return float(-np.log(p[te[:-1],te[1:]]).mean())
for n in (32000,64000,128000): print("moby bigram reference, trained on",n,"->",round(bigram(S["moby"][:n],held("moby",10000)),3))
print("secs bin -> n, median cl, median moby4x, best moby4x, median slope")
b=[0,30,100,300,960,3000,1e9]
for lo,hi in zip(b,b[1:]):
    s=[r for r in P if lo<=r["secs"]<hi]
    if s: print(f"  {lo:>5}-{hi:<6} n={len(s):3d} cl {np.median([r['cl'] for r in s]):.3f} m4 {np.median([r['ladder'][2] for r in s]):.3f} best m4 {min(r['ladder'][2] for r in s):.3f} slope {np.median([r['slope'] for r in s]):+.3f}")
print("moby4x distribution: share below 3.0:",round(np.mean([r['ladder'][2]<3.0 for r in P]),2),"below 2.8:",round(np.mean([r['ladder'][2]<2.8 for r in P]),2))
