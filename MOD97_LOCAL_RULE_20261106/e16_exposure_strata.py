"""Second audit of 10 November, point 2: was the scored relation already supplied? The SAME registered held items (k = 12, 400 per
operation) of the fifteen corrected-panel drift seeds, scored once from the saved final weights without the store, and split by the
equivalent supervision each item's pair has anywhere in the life (exp16.exposure_flags). Pooled over seeds; absolute accuracy for the
learner and the reference in each stratum, with counts. Scoring only. Writes exp45/E16_EXPOSURE_STRATA.md and .json."""
import json, sys
import torch as T
import exp16_integrated as G
import exp11_cell as E

SEEDS = (200003, 300007, 400009, 500011, 600013, 700027, 800011, 900007, 1000003, 1100009, 1200007, 1300021, 1400017, 1500007, 1600033)
agg = {}
for s in SEEDS:
    src = G.exposure_sources(s); P, _ = G.panels(s); models = {}
    for arm, f in (("T", f"e16_life_T_raw65536_drift_online_{s}.pt"), ("D", f"e16_life_D_drift_{s}.pt")):
        m = E.new_net(s); m.load_state_dict(T.load(G.OUT / f)); models[arm] = m
    for t in G.F1 + G.F3:
        name = E.NAME(*t); items = P[name][12]
        outs = {arm: E.respond(m, [p for p, _ in items], None, cap=G.CAP) for arm, m in models.items()}
        for i, (p, tgt) in enumerate(items):
            a = int(bytes(p[1:3])); b = int(bytes(p[-3:-1]))
            fl = G.exposure_flags(t, a, b, src); key = " + ".join(fl) if fl else "none (clean)"
            e = agg.setdefault((f"mod {t[1]} {t[0]}", key), dict(n=0, T=0, D=0))
            e["n"] += 1; e["T"] += outs["T"][i] == tgt; e["D"] += outs["D"][i] == tgt
            e2 = agg.setdefault((f"mod {t[1]} {t[0]}", "ALL (registered panel)"), dict(n=0, T=0, D=0))
            e2["n"] += 1; e2["T"] += outs["T"][i] == tgt; e2["D"] += outs["D"][i] == tgt
    print(s, "done", flush=True)
L = ["# E16 drift seeds: the registered held items split by equivalent supervision (fifteen corrected-panel seeds pooled, k = 12, no store)", "",
     "Same items as the registered panels; nothing is resampled. A pair can be supplied by its reversal being taught (commutative operations), by being a step of a taught composition in either orientation (mod 97), or, for max and min, by being taught under another modulus.", "",
     "| Operation | Equivalent supervision | Items | Learner correct | Reference correct | Ratio |", "|---|---|---:|---:|---:|---:|"]
for (opn, key), e in sorted(agg.items()):
    L.append(f"| {opn} | {key} | {e['n']} | {100 * e['T'] / e['n']:.1f} | {100 * e['D'] / e['n']:.1f} | {(e['T'] / e['D']) if e['D'] else float('nan'):.3f} |")
(G.OUT / "E16_EXPOSURE_STRATA.md").write_text("\n".join(L) + "\n")
(G.OUT / "E16_EXPOSURE_STRATA.json").write_text(json.dumps({f"{k[0]} | {k[1]}": v for k, v in agg.items()}, indent=1))
print("\n".join(L))
