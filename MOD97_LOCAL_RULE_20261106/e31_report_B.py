"""E31, worker B: the combined end-of-life table and the outcomes of every prediction (brief OPUS_TASK_E31_ONE_SWITCH_LIVES_20261113.md,
"Done", with addenda 2 and 3). Reads receipts only; writes exp45/e31/E31_TABLE.json and prints the table.

    python3 e31_report_B.py [A_COMMIT]     worker A's receipts (G, H, P2b) are read from that commit (default a930ed95) with git show;
                                           worker B's (K, P2a, P2c) from exp45/e31/ here.

Measures as exp45/e26/verify_lives.py defines them, exactly as e31_table_A.py (worker A) computes them, written here independently:
percent whole answers (exact); within = comp["within"]; plus-minus = "+-" and "-+" pooled; F1 / F3 = family means at k = 12. End of life =
stage 6. Floor = exp45/e26/e26_life_S_<seed>.json; ceiling = exp45/e26x/e26x_life_SB_<seed>.json. U = the end-of-life undecided share on
the plus-minus panel (E30's definition), as each receipt records it.
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
E31 = HERE / "exp45" / "e31"
SEEDS = (73129, 93683, 116089)
F1 = ("add97", "sub97", "max97", "min97"); F3 = ("add73", "sub73", "max73", "min73")
pct = lambda d: 100.0 * d["exact"] / d["n"]
A_ARMS, B_ARMS = ("G", "H", "P2b"), ("K", "P2a", "P2c")
M = "MOD97_LOCAL_RULE_20261106"


def load(path=None, commit=None):
    if commit:
        return json.loads(subprocess.run(["git", "show", f"{commit}:{M}/exp45/e31/{path}"], cwd=HERE, check=True, capture_output=True, text=True).stdout)
    return json.loads(Path(path).read_text())


def end_row(r, name):
    assert isinstance(r.get("complete"), dict), f"{name}: not a complete life"
    assert len(r["stages"]) == 7, f"{name}: {len(r['stages'])} stages"
    st = r["stages"][-1]; sc, cp = st["scores"], st["scores"]["compose"]
    row = dict(within=pct(cp["within"]), plus_minus=100.0 * (cp["+-"]["exact"] + cp["-+"]["exact"]) / (cp["+-"]["n"] + cp["-+"]["n"]),
               F1=sum(pct(sc[n]["12"]) for n in F1) / 4, F3=sum(pct(sc[n]["12"]) for n in F3) / 4,
               F1_stage2=None, tensors_sha256=r["complete"]["tensors_sha256"], cpu_hours=r["complete"]["cpu_total"] / 3600, file=name)
    e = st.get("e31") or {}
    for k in ("U_percent", "U_plus_minus"):
        if k in e:
            row["U"] = e[k]
    if "cos_Bhh_W" in e:
        row["cos_Bhh_W"] = e["cos_Bhh_W"]
    for key in ("lam", "gate_push"):                      # worker A records lam under "lam", worker B under "gate_push"
        v = (r.get("e31") or {}).get(key)
        if isinstance(v, dict) and isinstance(v.get("lam"), (int, float)):
            row["lam"] = v["lam"]
    return row


def main():
    acommit = sys.argv[1] if len(sys.argv) > 1 else "a930ed95"
    T = {"floor": {s: end_row(load(HERE / f"exp45/e26/e26_life_S_{s}.json"), f"e26_life_S_{s}") for s in SEEDS},
         "ceiling": {s: end_row(load(HERE / f"exp45/e26x/e26x_life_SB_{s}.json"), f"e26x_life_SB_{s}") for s in SEEDS}}
    for a in A_ARMS:
        T[a] = {s: end_row(load(f"e31_life_{a}_{s}.json", acommit), f"{acommit}:e31_life_{a}_{s}") for s in SEEDS}
    for a in B_ARMS:
        T[a] = {s: end_row(load(E31 / f"e31_life_{a}_{s}.json"), f"e31_life_{a}_{s}") for s in SEEDS if (E31 / f"e31_life_{a}_{s}.json").exists()}
    pm = lambda a, s: T[a][s]["plus_minus"]
    gap = lambda a, s: pm(a, s) - pm("floor", s) >= 0.5 * (pm("ceiling", s) - pm("floor", s))
    O = {}
    d = [pm("G", s) - pm("floor", s) for s in SEEDS]
    O["P31.1"] = dict(rule="G at least 15 points above the floor in at least two of three seeds", G_minus_floor=d, holds=sum(x >= 15 for x in d) >= 2)
    d = [pm("ceiling", s) - pm("H", s) for s in SEEDS]
    O["P31.2"] = dict(rule="H within 10 points of the ceiling in at least two of three seeds", ceiling_minus_H=d, holds=sum(abs(x) <= 10 for x in d) >= 2)
    if len(T["K"]) == 3:
        d = [pm("K", s) - pm("H", s) for s in SEEDS]
        O["P31.3"] = dict(rule="K within 5 points of H in all three seeds", K_minus_H=d, holds=all(abs(x) <= 5 for x in d))
        c = [T["K"][s]["cos_Bhh_W"] for s in SEEDS]
        O["P31.4"] = dict(rule="K's final cosine between Bhh and the forward matrix above 0.99 in all three", cosine=c, holds=all(x > .99 for x in c))
    O["P31.5"] = O["P31.6"] = "withdrawn unjudged with arm P (addendum 2, ruling 3)"
    U = [T["P2b"][s]["U"] for s in SEEDS]
    O["P31.7"] = dict(rule="at strength 0.05 the end-of-life U lies between 35% and 65% in all three seeds", U_P2b=U, holds=all(35 <= u <= 65 for u in U))
    if all(len(T[a]) == 3 for a in ("P2a", "P2c")):
        up = {a: [pm(a, s) - pm("floor", s) for s in SEEDS] for a in ("P2a", "P2b", "P2c")}
        O["P31.8"] = dict(rule="at least one strength 10 points or more above the floor on plus-minus in at least two of three seeds", above_floor=up,
                          per_strength={a: sum(x >= 10 for x in v) >= 2 for a, v in up.items()}, holds=any(sum(x >= 10 for x in v) >= 2 for v in up.values()))
        best = [max(("P2a", "P2b", "P2c"), key=lambda a: pm(a, s)) for s in SEEDS]
        O["P31.9"] = dict(rule="the strongest push (P2a) is not the best of the three in at least two of three seeds", best_by_seed=best,
                          holds=sum(b != "P2a" for b in best) >= 2)
    O["P31.10"] = "unjudged: arm Ks was not run at full length (the user's instruction to worker B: do not run arm Ks at full length)"
    O["closes_half_the_gap"] = {a: dict(per_seed=[gap(a, s) for s in SEEDS], in_two_of_three=sum(gap(a, s) for s in SEEDS) >= 2)
                                for a in ("G", "H") + B_ARMS + ("P2b",) if len(T[a]) == 3}
    out = dict(worker_A_receipts=acommit, table=T, outcomes=O)
    (E31 / "E31_TABLE.json").write_text(json.dumps(out, indent=1))
    for a, rows in T.items():
        for s, r in rows.items():
            print(f"{a:8s} {s:6d} within {r['within']:5.1f}  pm {r['plus_minus']:5.1f}  F1 {r['F1']:5.1f}  F3 {r['F3']:5.1f}"
                  + (f"  U {r['U']:.1f}" if "U" in r else "") + (f"  cos {r['cos_Bhh_W']:.7f}" if "cos_Bhh_W" in r else ""))
    print(json.dumps(O, indent=1))


if __name__ == "__main__":
    main()
