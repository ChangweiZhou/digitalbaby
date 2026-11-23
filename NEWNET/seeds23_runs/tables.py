# tables for SEEDS23_REPORT.md: seed 1 from life_rules_runs/, add_runs/, rules_runs/; seeds 2,3 from seeds23_runs/
import json, statistics as st
def rd(f):
    try: return [json.loads(l) for l in open(f)]
    except FileNotFoundError: return []
def key(d): return (d["net"], json.dumps(d.get("opts", {}), sort_keys=True))
def ms(v):
    v = [x for x in v if x is not None]
    if not v: return "-"
    return "%.2f ± %.2f" % (st.mean(v), st.pstdev(v)) if len(v) > 1 else "%.2f" % v[0]
def msi(v): return "{:,.0f} ± {:,.0f}".format(st.mean(v), st.pstdev(v)) if len(v) > 1 else "{:,}".format(v[0])
NAMES = {("store", "{}"): "storage", (10, "{}"): "K8 (net10)", (12, '{"lagro": 1, "relay": 1}'): "net12 relay",
         (12, '{"lagro": 1, "relay": 2}'): "net12 relay=2"}
def probe(d, line, k):
    for p in d["probes"]:
        if p["line"] == line: return p[k]
# t23
t23 = [d for d in rd("life_rules_runs/res.jsonl") if d["cue"] == 1 and key(d) in NAMES] + rd("seeds23_runs/res_t23.jsonl")
print("## t23 (cue=1)\n\n| net | seed | bits total | P0 copy b / c | P1 rev b | P3 succ b | P4 copy b | P5 rev b | text 17k -> 29k | succ 23k -> 29k | copy / rev at 29k |\n|---|---|---|---|---|---|---|---|---|---|---|")
g = {}
for d in sorted(t23, key=lambda d: (list(NAMES).index(key(d)), d["seed"])):
    f = d["first_met"]; r = lambda k: f[k]["right"] if k in f else None
    row = dict(bits=d["bits"]["total"], P0b=r("P0_copy_b"), P0c=r("P0_copy_c"), P1b=r("P1_rev_b"), P3b=r("P3_succ_b"), P4b=r("P4_copy_b"),
               P5b=r("P5_rev_b"), t17=probe(d, 17000, "text"), t29=probe(d, 29000, "text"), s23=probe(d, 23000, "succ"),
               s29=probe(d, 29000, "succ"), c29=probe(d, 29000, "copy"), r29=probe(d, 29000, "rev"))
    g.setdefault(NAMES[key(d)], []).append(row)
    print("| %s | %d | {:,} | %.2f / %.2f | %.2f | %.2f | %.2f | %.2f | %.2f -> %.2f | %.2f -> %.2f | %.2f / %.2f |".format(row["bits"]) % (
        NAMES[key(d)], d["seed"], row["P0b"], row["P0c"], row["P1b"], row["P3b"], row["P4b"], row["P5b"], row["t17"], row["t29"], row["s23"], row["s29"], row["c29"], row["r29"]))
print("\nMean ± spread (population sd) over seeds:\n\n| net | n | bits total | P0 copy b | P0 copy c | P1 rev b | P3 succ b | P4 copy b | P5 rev b | text 17k | text 29k | succ 23k | succ 29k | copy 29k | rev 29k |\n|" + "---|" * 15)
for n, rows in g.items():
    c = lambda k: [r[k] for r in rows]
    print("| %s | %d | %s | " % (n, len(rows), msi(c("bits"))) + " | ".join(ms(c(k)) for k in ["P0b", "P0c", "P1b", "P3b", "P4b", "P5b", "t17", "t29", "s23", "s29", "c29", "r29"]) + " |")
# t24
t24 = [d for d in rd("add_runs/res.jsonl") if key(d) in NAMES] + rd("seeds23_runs/res_t24.jsonl")
print("\n## t24\n\n| net | fmt | seed | lines right P1 old | P1 new | free 8 digits (last test, line) |\n|---|---|---|---|---|---|")
g = {}
for d in sorted(t24, key=lambda d: (d["fmt"], list(NAMES).index(key(d)), d["seed"])):
    L = d["lines"]; fr = d["free"][-1]
    g.setdefault((NAMES[key(d)], d["fmt"]), []).append((L["P1_old"]["right"], L["P1_new"]["right"], fr["8"]))
    print("| %s | %s | %d | %.2f | %.2f | %.2f (%d) |" % (NAMES[key(d)], d["fmt"], d["seed"], L["P1_old"]["right"], L["P1_new"]["right"], fr["8"], fr["line"]))
print("\n| net | fmt | n | P1 old | P1 new | free 8 |\n|---|---|---|---|---|---|")
for (n, f), rows in g.items():
    print("| %s | %s | %d | %s | %s | %s |" % (n, f, len(rows), ms([r[0] for r in rows]), ms([r[1] for r in rows]), ms([r[2] for r in rows])))
# t22
t22 = [d for d in rd("rules_runs/res.jsonl") if d["opts"].get("relay") == 2] + rd("seeds23_runs/res_t22.jsonl")
LK = ["0b", "0c", "0n", "1a", "1b", "1c", "1n", "2a", "2b", "2c", "2n"]
print("\n## t22 net12 relay=2 (right share, bytes in brackets)\n\n| seed | " + " | ".join(LK) + " |\n|" + "---|" * (len(LK) + 1))
for d in sorted(t22, key=lambda d: d["seed"]):
    print("| %d | " % d["seed"] + " | ".join("%.2f (%d)" % (d["layers"][k]["right"], d["layers"][k]["n"]) if k in d["layers"] else "-" for k in LK) + " |")
print("| mean ± sd | " + " | ".join(ms([d["layers"][k]["right"] for d in t22 if k in d["layers"]]) for k in LK) + " |")
print("\nAfter a change of rule (250-line windows):\n")
for d in sorted(t22, key=lambda d: d["seed"]):
    print("- seed %d: copy->rev %s; rev->copy %s" % (d["seed"], d["after_change"]["1"], d["after_change"]["2"]))
