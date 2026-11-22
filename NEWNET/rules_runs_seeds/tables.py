# builds the tables of RULES_SEEDS_REPORT.md from rules_runs/res.jsonl (seed 1) and rules_runs_seeds/res.jsonl
import json, numpy as np
R = [json.loads(l) for f in ("rules_runs/res.jsonl", "rules_runs_seeds/res.jsonl") for l in open(f)]
def name(r):
    if r["net"] == "store": return "storage"
    o = r["opts"]
    if r["net"] == 10: return "K8 (net10)" if r["struct"] == "K8" else "N1 (net10)"
    n = "net12 direct" + (" + relay" if o.get("relay") else "")
    if o.get("ctxg"): n += " + ctxg=%d err=%g" % (o["ctxg"], o.get("ctxg_err", 0))
    return n
L = ["0b", "0c", "1a", "1b", "1c", "2b", "2c"]
print("## per run\n\n| run | seed | " + " | ".join(L) + " |\n|---|---|" + "---|" * len(L))
for r in sorted(R, key=lambda r: (name(r), r["seed"])):
    print("| %s | %d | " % (name(r), r["seed"]) + " | ".join("%.2f (%d)" % (r["layers"][k]["right"], r["layers"][k]["n"]) for k in L) + " |")
print("\n## after_change\n")
for r in sorted(R, key=lambda r: (name(r), r["seed"])):
    a = r["after_change"]; print("| %s | %d | %s | %s |" % (name(r), r["seed"], " ".join("%.2f" % x for x in a["1"]), " ".join("%.2f" % x for x in a["2"])))
print("\n## mean ± sd over seeds\n\n| run | n seeds | " + " | ".join(L) + " |\n|---|---|" + "---|" * len(L))
G = {}
for r in R: G.setdefault(name(r), []).append(r)
for g, rs in G.items():
    print("| %s | %d (%s) | " % (g, len(rs), ",".join(str(r["seed"]) for r in rs)) + " | ".join(
        "%.2f ± %.2f" % (np.mean(v), np.std(v, ddof=1) if len(v) > 1 else 0) for v in ([r["layers"][k]["right"] for r in rs] for k in L)) + " |")
print("\n## after_change mean over seeds (window: mean)\n")
for g, rs in G.items():
    for ph in "12":
        m = np.mean([r["after_change"][ph] for r in rs], 0)
        print("| %s | %s | %s |" % (g, ph, " ".join("%.2f" % x for x in m)))
