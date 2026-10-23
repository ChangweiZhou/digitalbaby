"""Aggregate results/transfer/*.json into results/TRANSFER_SUMMARY.json and print a table."""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ARMS = ("FRESH", "MEM", "WD", "GEN", "SHUF")
WORLDS = (202610143901, 202610143902, 202610143903)
SEEDS = (0, 1, 2)
PAIRS = (("GEN", "SHUF"), ("GEN", "MEM"), ("GEN", "FRESH"), ("WD", "MEM"), ("MEM", "FRESH"), ("SHUF", "FRESH"))


def nz(v):
    # A criterion never reached counts as "slower than any run" (10**9 steps); 0 is a real value.
    return 10 ** 9 if v is None else v


def at(curve, step, key):
    return next(acc[key] for s, acc in curve if s == step)


def held_parts(r, b):
    # Split final unseen-multiplication answers: an operand 0 or 1 (copy / zero rule) vs both operands >= 2.
    triv, hard, rel = [], [], []
    for (a, c) in r["held_pairs"]:
        got, want = b["mul_held_answers"][f"{a},{c}"], str(a * c)
        (triv if min(a, c) <= 1 else hard).append(got == want)
        if min(a, c) > 1 and got.isdigit():
            rel.append(abs(int(got) - a * c) / (a * c))
    return float(np.mean(triv)), float(np.mean(hard)), float(np.median(rel)) if rel else None


def main():
    runs = {(r["arm"], r["world"], r["seed"]): r for r in
            (json.loads(p.read_text()) for p in sorted((HERE / "results" / "transfer").glob("*.json")))}
    doc = dict(schema="ADDITION_MULTIPLICATION_TRANSFER_SUMMARY_V1", runs=len(runs), arms={}, paired={})
    for bwd in ("wd0", "wd1"):
        for arm in ARMS:
            rs = [runs[(arm, w, s)] for w in WORLDS for s in SEEDS if (arm, w, s) in runs]
            if not rs:
                continue
            bs = [r["phase_b"][bwd] for r in rs]
            parts = [held_parts(r, b) for r, b in zip(rs, bs)]
            doc["arms"][f"{arm}_{bwd}"] = dict(
                n=len(rs),
                phase_a_add_taught=None if rs[0]["phase_a"] is None else float(np.mean([r["phase_a"]["add_taught"] for r in rs])),
                phase_a_add_held=None if rs[0]["phase_a"] is None else float(np.mean([r["phase_a"]["add_held"] for r in rs])),
                phase_a_matches_stored=None if rs[0]["phase_a"] is None else
                [r["phase_a"].get("matches_stored_run") for r in rs],
                steps_to_mul_taught_100=[b["steps_to_mul_taught_100"] for b in bs],
                steps_to_mul_taught_100_median=float(np.median([nz(b["steps_to_mul_taught_100"]) for b in bs])),
                mul_taught_at_100=float(np.mean([at(b["curve"], 100, "mul_taught") for b in bs])),
                mul_held_final=float(np.mean([b["mul_held_final"] for b in bs])),
                mul_held_final_operand_0_1=float(np.mean([p[0] for p in parts])),
                mul_held_final_both_ge_2=float(np.mean([p[1] for p in parts])),
                mul_held_median_rel_error_both_ge_2=float(np.median([p[2] for p in parts if p[2] is not None])),
                mul_held_commutative_agreement=float(np.mean([b["mul_held_commutative_agreement"] for b in bs])),
                add_taught_after={str(s): float(np.mean([at(b["curve"], s, "add_taught") for b in bs]))
                                  for s in (0, 10, 50, 200, 1000, 20000)},
                add_held_after={str(s): float(np.mean([at(b["curve"], s, "add_held") for b in bs]))
                                for s in (0, 10, 50, 200, 1000, 20000)},
                cpu_s_mean=float(np.mean([r["cpu_s"] for r in rs])))
        for x, y in PAIRS:
            keys = [(w, s) for w in WORLDS for s in SEEDS if (x, w, s) in runs and (y, w, s) in runs]
            if not keys:
                continue
            out = {}
            for metric, sign in (("steps_to_mul_taught_100", -1), ("mul_held_final", 1), ("add_taught_final", 1)):
                d = [nz(runs[(x, w, s)]["phase_b"][bwd][metric]) - nz(runs[(y, w, s)]["phase_b"][bwd][metric])
                     for w, s in keys]
                out[metric] = dict(mean_diff=float(np.mean(d)), x_better=sum(sign * v > 0 for v in d),
                                   x_worse=sum(sign * v < 0 for v in d), tied=sum(v == 0 for v in d))
            doc["paired"][f"{x}_vs_{y}_{bwd}"] = out
    (HERE / "results" / "TRANSFER_SUMMARY.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(f"{'arm':12} {'n':>2} {'A:+held':>7} {'steps100':>9} {'x@100':>6} {'xheld':>6} {'x01':>5} {'x>=2':>5} "
          f"{'relerr':>6} {'comm':>5} {'+t@10':>6} {'+t@50':>6} {'+t end':>6} {'+h@10':>6} {'+h end':>6}")
    for k, v in doc["arms"].items():
        ah = "-" if v["phase_a_add_held"] is None else f"{v['phase_a_add_held']:.2f}"
        print(f"{k:12} {v['n']:>2} {ah:>7} {v['steps_to_mul_taught_100_median']:9.0f} {v['mul_taught_at_100']:6.2f} "
              f"{v['mul_held_final']:6.2f} {v['mul_held_final_operand_0_1']:5.2f} {v['mul_held_final_both_ge_2']:5.2f} "
              f"{v['mul_held_median_rel_error_both_ge_2']:6.2f} {v['mul_held_commutative_agreement']:5.2f} "
              f"{v['add_taught_after']['10']:6.2f} {v['add_taught_after']['50']:6.2f} {v['add_taught_after']['20000']:6.2f} "
              f"{v['add_held_after']['10']:6.2f} {v['add_held_after']['20000']:6.2f}")
    for k, v in doc["paired"].items():
        print(k, {m: f"{x['mean_diff']:+.3g} ({x['x_better']}b/{x['x_worse']}w/{x['tied']}t)" for m, x in v.items()})


if __name__ == "__main__":
    main()
