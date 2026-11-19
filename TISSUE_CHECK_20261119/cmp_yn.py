"""Compare own_yn.py results (out/yn/c<label>.json) with the main worker's res/<label>.json: rule by rule, every window as floats."""
import json, os, glob
for f in sorted(glob.glob("/home/user/tissue_check/ref_yn/*.json")):
    lab = os.path.basename(f)[:-5]; mine = f"/home/user/tissue_check/out/yn/c{lab}.json"
    if not os.path.exists(mine): print(lab, "not run"); continue
    R = json.load(open(f))["rules"]; M = json.load(open(mine))["rules"]
    diff = [k for k in R if k not in M or R[k]["windows"] != M[k]["windows"] or R[k]["last_quarter"] != M[k]["last_quarter"]]
    print(lab, "EQUAL in every window of all", len(R), "rules" if not diff else f"DIFFER: {diff}")
