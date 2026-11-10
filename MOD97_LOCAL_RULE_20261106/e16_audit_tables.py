"""Tables answering the 10 November source audit for the eighteen drift seeds (window learner, raw reservoir 65,536, online admission).

Part 1 (receipts only): each seed's panel version and its three gates (family means as pre-registered; `+-`/`-+` alone; whole worked answer),
        with totals by panel version. The pre-registered gate is not rewritten; the strata are shown beside it.
Part 2 (saved final weights, scoring only, no training): the registered held panel and the strict held panel (no reversal taught, no taught
        sub-step; exp16.strict_panels) at k = 12, both WITHOUT the score-time store (the store is not saved with the weights), for the learner
        and its reference. Also the receipt's own with-store score, so the store's contribution at k = 12 is visible.
Writes exp45/E16_AUDIT_TABLES.md and .json."""
import json, os, sys
from pathlib import Path
import torch as T
import exp16_integrated as G
import exp11_cell as E

OUT = G.OUT
SEEDS = (73129, 93683, 116089, 200003, 300007, 400009, 500011, 600013, 700027, 800011, 900007, 1000003, 1100009, 1200007, 1300021, 1400017, 1500007, 1600033)
F1n, F3n, H = [E.NAME(*t) for t in G.F1], [E.NAME(*t) for t in G.F3], ("+-", "-+", "><", "<>")
pct = lambda d: 100 * d["exact"] / d["n"]
rows, res = [], {}
for s in SEEDS:
    cp = (OUT / f"e16_life_T_raw65536_drift_online_cp_{s}.json").exists() and "--original" not in sys.argv     # repeat on the corrected panels, if one exists
    Tr = json.load(open(OUT / f"e16_life_T_raw65536_drift_online{'_cp' if cp else ''}_{s}.json"))["stages"]; Dr = json.load(open(OUT / f"e16_life_D_drift{'_cp' if cp else ''}_{s}.json"))["stages"]
    st = [x["settings"] for x in Tr if "settings" in x][-1]
    ver = ("corrected" if "keyed by split" in st.get("panels", "") else "original (pre-fix)") + (" (repeat, owner's machine)" if cp else "")
    t, d = Tr[6]["scores"], Dr[6]["scores"]
    fam = lambda sc: (sum(pct(sc[n]["12"]) for n in F1n) / 4, sum(100 * sc["compose"][h]["final"] / sc["compose"][h]["n"] for h in H) / 4, sum(pct(sc[n]["12"]) for n in F3n) / 4)
    r = [a / b for a, b in zip(fam(t), fam(d))]
    hard = (sum(t["compose"][h]["final"] for h in H[:2]) / sum(d["compose"][h]["final"] for h in H[:2]))
    worked = (sum(t["compose"][h]["exact"] for h in H) / sum(d["compose"][h]["exact"] for h in H))
    res[s] = dict(panel=ver, ratios=r, family_gate=all(v >= .8 for v in r), hard_pair_ratio=hard, hard_pair_gate=hard >= .8, worked_ratio=worked, worked_gate=worked >= .8,
                  receipt_T_with_store={n: pct(t[n]["12"]) for n in F1n + F3n})
L = ["# E16 drift seeds: gates by panel version, and a strict held-out re-score (answer to the 10 November source audit)", "",
     "## 1. Gates by panel version (from the receipts)", "",
     "| Seed | Panel | Ratios to reference (mod-97 · held pairs · mod-73) | Family gate | `+-`/`-+` ratio | Hard-pair gate | Worked-answer ratio | Worked-answer gate |", "|---:|---|---|---|---:|---|---:|---|"]
for s in SEEDS:
    v = res[s]; L.append(f"| {s} | {v['panel']} | {v['ratios'][0]:.2f} · {v['ratios'][1]:.2f} · {v['ratios'][2]:.2f} | {'pass' if v['family_gate'] else '**fail**'} | {v['hard_pair_ratio']:.2f} | {'pass' if v['hard_pair_gate'] else 'fail'} | {v['worked_ratio']:.2f} | {'pass' if v['worked_gate'] else 'fail'} |")
L += ["", "| Stratum | Seeds | Family gate | Hard-pair gate | Worked-answer gate |", "|---|---:|---:|---:|---:|"]
for name, sel in (("corrected panels, cloud seeds (fifteen later seeds)", lambda v: v["panel"] == "corrected"), ("corrected panels, repeats of the three development seeds", lambda v: "repeat" in v["panel"]),
                  ("original panels (pre-fix)", lambda v: v["panel"].startswith("original")), ("all seeds in this table", lambda v: True)):
    vs = [v for v in res.values() if sel(v)]
    L.append(f"| {name} | {len(vs)} | {sum(v['family_gate'] for v in vs)} | {sum(v['hard_pair_gate'] for v in vs)} | {sum(v['worked_gate'] for v in vs)} |")
L += ["", "The earlier part 2 of this file (a re-score on a resampled \"strict\" panel) is withdrawn; see `E16_EXPOSURE_STRATA.md`, which splits the same registered items by exposure."]
if "--strict-resample" in sys.argv:
    L += ["", "## 2. Registered against strict held panels, final weights, k = 12, no score-time store", "",
          "Strict: a held pair is dropped if the operation is commutative and its reversal is taught, or (mod 97) if it is supervised as a step of a taught composition.", "",
          "| Seed | Learner mod-97: receipt (with store) | registered, no store | strict, no store | Reference mod-97: registered | strict | Ratio registered | Ratio strict | mod-73 ratio registered | strict |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s in SEEDS:
        pt, pd = OUT / f"e16_life_T_raw65536_drift_online_{s}.pt", OUT / f"e16_life_D_drift_{s}.pt"
        if not (pt.exists() and pd.exists()):
            L.append(f"| {s} | final weights not on file | | | | | | | | |"); res[s]["rescore"] = None; continue
        P, comp = G.panels(s); SP, counts = G.strict_panels(s); sc = {}
        for arm, p in (("T", pt), ("D", pd)):
            m = E.new_net(s); m.load_state_dict(T.load(p))
            sc[arm] = dict(reg={n: pct(G.score(m, P[n][12], None)) for n in F1n + F3n}, strict={n: pct(G.score(m, SP[n], None)) for n in F1n + F3n})
        mean = lambda d, names: sum(d[n] for n in names) / len(names)
        v = dict(counts=counts, **sc)
        v["mod97"] = dict(T_receipt=mean(res[s]["receipt_T_with_store"], F1n), T_reg=mean(sc["T"]["reg"], F1n), T_strict=mean(sc["T"]["strict"], F1n), D_reg=mean(sc["D"]["reg"], F1n), D_strict=mean(sc["D"]["strict"], F1n))
        v["mod73"] = dict(T_reg=mean(sc["T"]["reg"], F3n), T_strict=mean(sc["T"]["strict"], F3n), D_reg=mean(sc["D"]["reg"], F3n), D_strict=mean(sc["D"]["strict"], F3n))
        res[s]["rescore"] = v; a, b = v["mod97"], v["mod73"]
        L.append(f"| {s} | {a['T_receipt']:.1f} | {a['T_reg']:.1f} | {a['T_strict']:.1f} | {a['D_reg']:.1f} | {a['D_strict']:.1f} | {a['T_reg'] / a['D_reg']:.3f} | {a['T_strict'] / a['D_strict']:.3f} | {b['T_reg'] / b['D_reg']:.3f} | {b['T_strict'] / b['D_strict']:.3f} |")
        print(s, "done", flush=True)
(OUT / "E16_AUDIT_TABLES.md").write_text("\n".join(L) + "\n"); (OUT / "E16_AUDIT_TABLES.json").write_text(json.dumps({str(k): v for k, v in res.items()}, indent=1))
print("\n".join(L))
