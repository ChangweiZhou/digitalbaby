"""Summary of search2 results: one row per candidate (full runs only), dev and held-out means, sorted by dev.
Usage: python3 search2_summary.py [search2_results.tsv ...]"""
import sys, collections
DEV = ["moby", "keep", "keep4", "keep2", "mem", "life23", "enzh"]; HELD = ["code", "life27", "zhen"]
R = collections.defaultdict(dict); OPT = {}
for f in (sys.argv[1:] or ["search2_results.tsv"]):
    for line in open(f):
        p = line.rstrip("\n").split("\t")
        if len(p) < 7 or p[6] != "full": continue
        R[p[0]][p[3]] = float(p[4]); OPT[p[0]] = f"net={p[1]} {p[2]}"
def m(d, ks):
    v = [d[k] for k in ks if k in d and d[k] == d[k]]; return (sum(v) / len(v), len(v)) if v else (float("nan"), 0)
rows = sorted(R, key=lambda n: -m(R[n], DEV)[0])
print("| cand | " + " | ".join(DEV + HELD) + " | dev | held | options |\n|---|" + "---|" * (len(DEV) + len(HELD) + 3))
for n in rows:
    d = R[n]; dv, nd = m(d, DEV); hv, nh = m(d, HELD)
    print(f"| {n} | " + " | ".join(f"{d[k]:.3f}" if k in d else "-" for k in DEV + HELD) + f" | {dv:.4f} ({nd}) | {hv:.4f} ({nh}) | {OPT[n]} |")
