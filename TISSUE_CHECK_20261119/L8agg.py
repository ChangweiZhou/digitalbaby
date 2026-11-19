"""Part 3: six lives a seed (reference + five changed by 1e-12 at byte 0). For each of the nine entries and lags 0, 1, 2 near the output
lines (enlarged lists): smallest and largest S, and whether every life is above levels (a) and (b)."""
import glob, json, subprocess, sys
res = {}
for seed in (73129, 93683, 116089):
    tags = [f"L8_{seed}_ref"] + sorted(p[4:-6] for p in glob.glob(f"out/L8_{seed}_p*.stats"))
    tags = [t for t in tags if glob.glob(f"out/{t}.stats")]
    R = {}
    for t in tags:
        if not glob.glob(f"out/{t}_L8read.json"):
            subprocess.run([sys.executable, "L8read.py", t, str(seed)], check=True, capture_output=True)
        R[t] = json.load(open(f"out/{t}_L8read.json"))
    ent = {}
    for k in R[tags[0]]["nine"]:
        v = [R[t]["nine"][k] for t in tags]
        ent[k] = dict(min=min(x["S"] for x in v), max=max(x["S"] for x in v), a=all(x["a"] for x in v), b=all(x["b"] for x in v))
    for L in (0, 1, 2):
        v = [R[t]["near"]["enlarged_lists"][f"L{L}"] for t in tags]
        ent[f"near L{L}"] = dict(min=min(x["S"] for x in v), max=max(x["S"] for x in v), a=all(x["a"] for x in v), b=all(x["b"] for x in v),
                                 levels=[round(x["level"], 4) for x in v])
    res[seed] = dict(lives=len(tags), entries=ent)
    print(seed, len(tags), "lives:", "; ".join(f"{k} {e['min']:.3f}-{e['max']:.3f}{'' if e['a'] and e['b'] else (' [b fails]' if e['a'] else ' [BELOW a]')}" for k, e in ent.items()), flush=True)
json.dump(res, open("out/L8_part3.json", "w"), indent=1)
