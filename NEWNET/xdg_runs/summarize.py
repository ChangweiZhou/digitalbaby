"""Tables for XDG_REPORT.md from xdg_runs/res.jsonl and the matching baselines in life_rules_runs/res.jsonl."""
import json
def load(f):
    try: return [json.loads(l) for l in open(f)]
    except FileNotFoundError: return []
def name(d):
    o = d["opts"]; s = "+".join("%s=%s" % kv for kv in o.items()) or d["net"]
    return "%s cue=%d s%d" % (s, d["cue"], d["seed"])
base = [d for d in load("life_rules_runs/res.jsonl") if d["net"] == 12 and d["opts"] == {"lagro": 1, "relay": 1}]
runs = base + load("xdg_runs/res.jsonl")
END = [6000, 12000, 17000, 23000, 26000, 29000]
print("## Bits of the whole life\n\n| run | total | input | answer | text | answer b/byte | text b/byte |\n|---|---|---|---|---|---|---|")
for d in runs:
    b = d["bits"]; print("| %s | %d | %d | %d | %d | %.3f | %.3f |" % (name(d), b["total"], b["input"], b["answer"], b["text"], b["per_byte"]["answer"], b["per_byte"]["text"]))
keys = sorted({k for d in runs for k in d["first_met"] if k.split("_")[-1] in "bc"})
print("\n## First-met right shares (layers b, c; n in brackets)\n\n| run | " + " | ".join(keys) + " |\n|---|" + "---|" * len(keys))
for d in runs:
    f = d["first_met"]; print("| %s | " % name(d) + " | ".join("%.3f (%d)" % (f[k]["right"], f[k]["n"]) if k in f else "-" for k in keys) + " |")
print("\n## First windows (250 lines) of P4 (copy again) and P5 (rev again), with P0 / P1 for comparison\n\n| run | P0 first 4 | P4 first 4 | P1 first 4 | P5 first 4 |\n|---|---|---|---|---|")
for d in runs:
    w = d["windows"]; print("| %s | %s | %s | %s | %s |" % ((name(d),) + tuple(" ".join("%.2f" % x for x in w[p][:4]) for p in ("P0", "P4", "P1", "P5"))))
print("\n## Probes (learning off) at the end of each phase: copy / rev / succ / text\n\n| run | " + " | ".join(str(l) for l in END) + " |\n|---|" + "---|" * len(END))
for d in runs:
    P = {p["line"]: p for p in d["probes"]}
    print("| %s | " % name(d) + " | ".join("%.2f/%.2f/%.2f/%.3f" % (P[l]["copy"], P[l]["rev"], P[l]["succ"], P[l]["text"]) if l in P else "-" for l in END) + " |")
print("\n## Kept: text probe at 17000 vs 29000; succ probe at 23000 vs 29000; copy at 6000 vs 29000; rev at 12000 vs 29000\n\n| run | text 17k | text 29k | succ 23k | succ 29k | copy 6k | copy 29k | rev 12k | rev 29k |\n|---|---|---|---|---|---|---|---|---|")
for d in runs:
    P = {p["line"]: p for p in d["probes"]}; g = lambda l, k: ("%.3f" % P[l][k]) if l in P else "-"
    print("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (name(d), g(17000, "text"), g(29000, "text"), g(23000, "succ"), g(29000, "succ"), g(6000, "copy"), g(29000, "copy"), g(12000, "rev"), g(29000, "rev")))
