"""Rule Q on exact inputs: last quarter of the single-rule lives (generator default_rng(seed)) and range over five generators;
weight-row norms over 262,144 bytes; windows late in the life."""
import json, os
O = "/home/user/tissue_check/out/yn/"
J = lambda f: json.load(open(O + f + ".json")) if os.path.exists(O + f + ".json") else None
R = [("Q_1-4", "Q eta=0.25"), ("Q_1-16", "Q eta=0.0625"), ("Q_1-64", "Q eta=0.015625"), ("D_1-16", "D eta=0.0625"), ("D_1-64", "D eta=0.015625"), ("B_1-16", "B eps=0.0625")]
for K in ("64k", "256k"):
    print(f"== {K}: last quarter %, generator default_rng(seed): Q 1/4 | Q 1/16 | Q 1/64 | D 1/16 | D 1/64 | B ;  five generators B / Q1/4 / Q1/16 / Q1/64")
    for s in (93683, 73129, 116089):
        for i in (1, 2, 3):
            row = []
            for t, nm in R:
                d = J(f"q_{t}_x{i}_s{s}_{K}"); row.append(f"{100*d['rules'][nm]['last_quarter']:5.2f}" if d else "  -  ")
            g = [J(f"qg{k}_x{i}_s{s}_{K}") for k in range(1, 6)]
            rg = " / ".join(f"{min(100*x['rules'][nm]['last_quarter'] for x in g):.2f}-{max(100*x['rules'][nm]['last_quarter'] for x in g):.2f}" for nm in ("B eps=0.0625", "Q eta=0.25", "Q eta=0.0625", "Q eta=0.015625")) if all(g) else "-"
            print(f"  s{s} {i} byte{'s' if i > 1 else ' '}: " + " | ".join(row) + "  ;  " + rg)
print("== weight-row norms (largest, median) at 16k / 64k / 128k / 256k and windows % at the same bytes, single-rule lives, 262,144 bytes")
for s in (93683, 73129, 116089):
    for i in (1, 2, 3):
        for t, nm in R:
            d = J(f"q_{t}_x{i}_s{s}_256k")
            if not d: continue
            v = d["rules"][nm]; nr = v["norms"]; w = v["windows"]
            print(f"  s{s} {i} {nm:16s} max " + " ".join(f"{nr[j]['max']:.3g}" for j in (3, 15, 31, 63)) + "  median " + " ".join(f"{nr[j]['median']:.3g}" for j in (3, 15, 31, 63)) +
                  "  windows " + " ".join(f"{100*w[j]:.1f}" for j in (3, 15, 31, 63)) + f"  best {100*max(w):.1f}")
