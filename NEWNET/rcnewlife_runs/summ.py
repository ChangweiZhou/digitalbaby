import json, sys
rs = [json.loads(l) for l in open("rcnewlife_runs/res.jsonl")]
def name(r): return ("rc" if "rc" in r["opts"] else "base") if r["net"] == 12 else "K8" if r["net"] == 10 else "store"
for r in sorted(rs, key=lambda r: (-r["cue"], r["seed"], name(r))):
    b = r["bits"]; P = {p["line"]: p for p in r["probes"]}; e = P[29000]
    print(f"{name(r):5s} cue={r['cue']} s={r['seed']} bits {b['total']:,} (in {b['input']:,} ans {b['answer']:,} txt {b['text']:,}) {r['s']}s rec={r.get('nrecruit')}")
    print("   end:", {k: e[k] for k in ("copy","swap","rot","skip","map","text")})
    print("   map probes:", [P[l]["map"] for l in sorted(P)])
    print("   text probes:", [P[l]["text"] for l in sorted(P)])
    for k in ("rot", "swap", "skip", "copy"): print(f"   {k}:", [P[l][k] for l in sorted(P)])
    print("   first-met:", {k: (v["right"], v["n"]) for k, v in r["first_met"].items()})
    print("   windows:", r["windows"])
