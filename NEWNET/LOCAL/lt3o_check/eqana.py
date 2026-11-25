import json, collections, numpy as np
R=collections.defaultdict(dict)
for l in open("eq.jsonl"): r=json.loads(l); R[r["name"]][(r["code"],r["seed"])]=r
print(f"{'cand':7s} {'metric':6s} {'old s1':>7s} {'old s2':>7s} {'new s1':>7s} {'new s2':>7s} {'seed gap old':>12s} {'new-old (mean)':>14s}   time old/new")
for n,d in R.items():
    for m in ["cl","m4","slope"]:
        g=lambda k: (d[k][m] if m!="m4" else d[k]["ladder"][2]) if k in d else float('nan')
        o1,o2,n1,n2=g(("old",1)),g(("old",2)),g(("new",1)),g(("new",2))
        t=[d[k]["secs"] if k in d else float('nan') for k in [("old",1),("new",1)]]
        print(f"{n:7s} {m:6s} {o1:7.3f} {o2:7.3f} {n1:7.3f} {n2:7.3f} {abs(o1-o2):12.3f} {np.nanmean([n1,n2])-np.nanmean([o1,o2]):+14.3f}   {t[0]:.0f}s/{t[1]:.0f}s" if m=="cl" else f"{'':7s} {m:6s} {o1:7.3f} {o2:7.3f} {n1:7.3f} {n2:7.3f} {abs(o1-o2):12.3f} {np.nanmean([n1,n2])-np.nanmean([o1,o2]):+14.3f}")
