"""Summary of search2 results: one row per candidate (full runs only), dev and held-out means, sorted by dev.
Usage: python3 search2_summary.py [search2_results.tsv ...]"""
import sys, collections, re, os
MODE = "quick" if os.environ.get("QUICK") == "1" else "full"        # QUICK=1: rows of the quick battery only
def depth(opts):
    """serial depth of the context tissue: regions on the longest source path (the per-byte serial steps; owner, 23 Nov: real-time
    reaction allows only a few serial steps, so the search widens and records this)"""
    try:
        from net7 import Tissue
        kv = dict(re.findall(r"(\w+)=(\S+)", opts)); g = lambda k, d: int(float(kv.get(k, d)))
        L = g("L", 16); t = Tissue(seed=g("seed", 1) + 300, L=L, C=8, M=2, k_win=2, n_values=128, ctx=2, hold=0,
                                   mesh=g("mesh", 1), mesh_base=g("mesh_base", 0), mesh_dag=g("mesh_dag", 5))
        dp = [1] * L
        for d in range(L):
            for s_ in t.msrc[d]: dp[d] = max(dp[d], dp[int(s_)] + 1)
        return max(dp)
    except Exception: return "?"
DEV = ["moby", "keep", "keep4", "keep2", "mem", "life23", "enzh"]; HELD = ["code", "life27", "zhen"]
R = collections.defaultdict(dict); OPT = {}
for f in (sys.argv[1:] or ["search2_results.tsv"]):
    for line in open(f):
        p = line.rstrip("\n").split("\t")
        if len(p) < 7 or p[6] != MODE: continue
        R[p[0]][p[3]] = float(p[4]); OPT[p[0]] = f"net={p[1]} {p[2]}"
def m(d, ks):
    v = [d[k] for k in ks if k in d and d[k] == d[k]]; return (sum(v) / len(v), len(v)) if v else (float("nan"), 0)
rows = sorted(R, key=lambda n: -m(R[n], DEV)[0])
print("| cand | " + " | ".join(DEV + HELD) + " | dev | held | depth | options |\n|---|" + "---|" * (len(DEV) + len(HELD) + 4))
for n in rows:
    d = R[n]; dv, nd = m(d, DEV); hv, nh = m(d, HELD)
    print(f"| {n} | " + " | ".join(f"{d[k]:.3f}" if k in d else "-" for k in DEV + HELD) + f" | {dv:.4f} ({nd}) | {hv:.4f} ({nh}) | {depth(OPT[n])} | {OPT[n]} |")
