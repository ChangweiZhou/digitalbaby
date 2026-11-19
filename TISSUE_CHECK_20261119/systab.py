"""Table of sysmini.py runs: last quarter per rule, condition, seed: median and range over the five orders; the copy rule by quarter."""
import json, glob, os, numpy as np
O = "/home/user/tissue_check/out/sys/"
def get(rule, i, s, L):
    v = []
    for g in range(5):
        f = O + f"{rule}_x{i}_s{s}_g{g}_{L}.json"
        if os.path.exists(f): v.append(100 * json.load(open(f))["quarters"][3])
    return v
for L in ("64k", "256k"):
    for rule in ("copy", "copytext", "novel", "errgate", "tag", "tagH1"):
        rows = []
        for s in (73129, 406364, 545353):
            cells = []
            for i in (1, 2, 3):
                v = get(rule, i, s, L)
                cells.append(f"({'i' * i}) {np.median(v):5.2f} [{min(v):.2f}-{max(v):.2f}] n{len(v)}" if v else f"({'i' * i}) -")
            rows.append(f"  s{s}: " + "  ".join(cells))
        if any("n" in r for r in rows): print(f"== {rule} {L}, last quarter %, median [range] over orders"); print("\n".join(rows))
refs = {}
for f in glob.glob(O + "copy_x1_s*_g0_*.json"):
    d = json.load(open(f)); refs[os.path.basename(f)] = {k: round(100 * v, 2) for k, v in d["refs_last_quarter"].items()}
print("refs", refs)
