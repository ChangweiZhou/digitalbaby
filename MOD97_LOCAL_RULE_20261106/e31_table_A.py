"""E31, worker A: the end-of-life table (within, plus-minus, F1, F3) per arm and seed beside the floor and the ceiling, and the outcomes of
the predictions that rest on worker A's arms only (P31.1 G, P31.2 H, P31.7 P2b). Reads receipts only; writes exp45/e31/E31_TABLE_A.json.

Measures as exp45/e26/verify_lives.py defines them: percent whole answers (exact); within = comp["within"] (600); plus-minus = "+-" and "-+"
pooled (1,200); F1 / F3 = the family means at k = 12 (add97 sub97 max97 min97 / add73 sub73 max73 min73). End of life = stage 6.
Floor = exp45/e26/e26_life_S_<seed>.json, ceiling = exp45/e26x/e26x_life_SB_<seed>.json (the brief's table)."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
E31 = HERE / "exp45" / "e31"
SEEDS = (73129, 93683, 116089)
F1 = ("add97", "sub97", "max97", "min97"); F3 = ("add73", "sub73", "max73", "min73")
pct = lambda d: 100.0 * d["exact"] / d["n"]


def end_row(path):
    r = json.loads(Path(path).read_text())
    assert isinstance(r.get("complete"), dict) and r["complete"]["stages"] == 7, f"{path}: not a complete life"
    st = r["stages"][-1]; sc, cp = st["scores"], st["scores"]["compose"]
    row = dict(within=pct(cp["within"]), plus_minus=100.0 * (cp["+-"]["exact"] + cp["-+"]["exact"]) / (cp["+-"]["n"] + cp["-+"]["n"]),
               F1=sum(pct(sc[n]["12"]) for n in F1) / 4, F3=sum(pct(sc[n]["12"]) for n in F3) / 4,
               tensors_sha256=r["complete"]["tensors_sha256"], cpu_hours=r["complete"]["cpu_total"] / 3600, file=str(Path(path).relative_to(HERE)))
    if "e31" in st:
        row["e31_stage6"] = st["e31"]
    if r.get("e31"):
        row["lam"] = (r["e31"].get("lam") or {}).get("lam")
    return row


def main():
    T = {"floor": {s: end_row(HERE / f"exp45/e26/e26_life_S_{s}.json") for s in SEEDS},
         "ceiling": {s: end_row(HERE / f"exp45/e26x/e26x_life_SB_{s}.json") for s in SEEDS}}
    for arm in ("G", "H", "P2b"):
        T[arm] = {s: end_row(E31 / f"e31_life_{arm}_{s}.json") for s in SEEDS if (E31 / f"e31_life_{arm}_{s}.json").exists()}
    pm = lambda a, s: T[a][s]["plus_minus"]
    out = dict(table=T, outcomes={})
    if len(T["G"]) == 3:
        d = [pm("G", s) - pm("floor", s) for s in SEEDS]
        out["outcomes"]["P31.1"] = dict(G_minus_floor=d, holds=sum(x >= 15 for x in d) >= 2, rule="G at least 15 points above the floor in at least two of three seeds")
        half = [pm("G", s) - pm("floor", s) >= 0.5 * (pm("ceiling", s) - pm("floor", s)) for s in SEEDS]
        out["outcomes"]["G_closes_half_the_gap"] = dict(per_seed=half, in_two_of_three=sum(half) >= 2)
    if len(T["H"]) == 3:
        d = [pm("ceiling", s) - pm("H", s) for s in SEEDS]
        out["outcomes"]["P31.2"] = dict(ceiling_minus_H=d, holds=sum(abs(x) <= 10 for x in d) >= 2, rule="H within 10 points of the ceiling in at least two of three seeds")
        half = [pm("H", s) - pm("floor", s) >= 0.5 * (pm("ceiling", s) - pm("floor", s)) for s in SEEDS]
        out["outcomes"]["H_closes_half_the_gap"] = dict(per_seed=half, in_two_of_three=sum(half) >= 2)
    if len(T["P2b"]) == 3:
        U = [T["P2b"][s]["e31_stage6"]["U_percent"] for s in SEEDS]
        out["outcomes"]["P31.7"] = dict(U_end_P2b=U, holds=all(35 <= u <= 65 for u in U), rule="at strength 0.05 the end-of-life U lies between 35% and 65% in all three seeds")
        out["outcomes"]["P2b_minus_floor"] = [pm("P2b", s) - pm("floor", s) for s in SEEDS]
    out["not_judged_here"] = "P31.3, P31.4 (arm K), P31.8, P31.9 (need P2a and P2c), P31.10 (Ks): worker B's lives"
    (E31 / "E31_TABLE_A.json").write_text(json.dumps(out, indent=1))
    for a, rows in T.items():
        for s, r in rows.items():
            print(f"{a:8s} {s:6d} within {r['within']:5.1f}  pm {r['plus_minus']:5.1f}  F1 {r['F1']:5.1f}  F3 {r['F3']:5.1f}" + (f"  U {r['e31_stage6']['U_percent']:.1f}" if 'e31_stage6' in r and 'U_percent' in r['e31_stage6'] else ""))
    print(json.dumps(out["outcomes"], indent=1))


if __name__ == "__main__":
    main()
