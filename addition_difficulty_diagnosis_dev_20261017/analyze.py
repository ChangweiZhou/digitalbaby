"""Derived checks behind REPORT.md -> results/ANALYSIS.json (reads results/runs only; no training)."""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.argv = sys.argv[:1]
import diag  # noqa: E402
from summarize import value  # noqa: E402

HERE = Path(__file__).resolve().parent
WORLDS = (202610143901, 202610143902, 202610143903)
SEEDS = (0, 1, 2)


def load():
    runs = {}
    for p in (HERE / "results" / "runs").glob("*.json"):
        r = json.loads(p.read_text())
        key = (r["regime"], r["wd"], r["inp"], r["out"], r.get("aux", "none"), r.get("width", 64))
        runs[key + (r["world"], r["seed"])] = r
    return runs


def identifiability():
    # Fit answer = u[a] + v[b] to the 80 taught facts only, then round its prediction for the 20 HELD facts.
    out = {}
    for w in WORLDS:
        sp = diag.split(w)
        taught = diag.ordered(sp["old"]) + diag.ordered(sp["new"])
        held = diag.ordered(sp["held"])
        A = np.zeros((len(taught), 20))
        for i, (a, b) in enumerate(taught):
            A[i, a] = A[i, 10 + b] = 1
        x = np.linalg.lstsq(A, np.array([a + b for a, b in taught], float), rcond=None)[0]
        out[str(w)] = dict(rank=int(np.linalg.matrix_rank(A)),
                           held_exact=float(np.mean([round(x[a] + x[10 + b]) == a + b for a, b in held])))
    return out


def paired(runs, k1, k2):
    d = [runs[k1 + (w, s)]["held_final"] - runs[k2 + (w, s)]["held_final"] for w in WORLDS for s in SEEDS]
    return dict(mean_diff=float(np.mean(d)), better=sum(x > 0 for x in d), worse=sum(x < 0 for x in d),
                tied=sum(x == 0 for x in d))


def errors(runs, cond):
    rs = [runs[cond + (w, s)] for w in WORLDS for s in SEEDS]
    signed, malformed, comm = [], 0, []
    for r in rs:
        got = {tuple(h["pair"]): h["got"] for h in r["held"]}
        for h in r["held"]:
            if h["got"] == h["want"]:
                continue
            g = value(h["got"], r["out"])
            if g is None:
                malformed += 1
            else:
                signed.append(g - value(h["want"], r["out"]))
        comm += [got[(a, b)] == got[(b, a)] for a, b in got if a < b]
    return dict(wrong=len(signed) + malformed, off_by_one=sum(abs(e) == 1 for e in signed),
                off_by_two=sum(abs(e) == 2 for e in signed), larger=sum(abs(e) > 2 for e in signed),
                malformed=malformed, over=sum(e > 0 for e in signed), under=sum(e < 0 for e in signed),
                histogram={str(k): v for k, v in sorted(Counter(signed).items())},
                commutative_agreement=float(np.mean(comm)))


def main():
    runs = load()
    base = ("batch", 0.0, "digit", "dec", "none", 64)
    online = ("online", 0.0, "digit", "dec", "none", 64)
    wd1 = ("batch", 1.0, "digit", "dec", "none", 64)
    doc = dict(
        schema="ADDITION_DIFFICULTY_DIAGNOSIS_ANALYSIS_V1",
        runs=len(runs),
        identifiability_additive_fit_on_taught_facts=identifiability(),
        paired_vs_registered_format=dict(
            unary_input=paired(runs, ("batch", 0.0, "unary", "dec", "none", 64), base),
            weight_decay_1=paired(runs, wd1, base),
            sym_minus_dec_output_wd0=paired(runs, ("batch", 0.0, "digit", "sym", "none", 64), base),
            dec_minus_sym_output_wd1=paired(runs, wd1, ("batch", 1.0, "digit", "sym", "none", 64)),
            width16=paired(runs, base[:5] + (16,), base),
            width32=paired(runs, base[:5] + (32,), base)),
        counting_experience={
            f"{reg}_wd{wd:g}": dict(
                count_minus_none=paired(runs, (reg, wd, "digit", "dec", "count", 64), (reg, wd, "digit", "dec", "none", 64)),
                count_minus_permuted=paired(runs, (reg, wd, "digit", "dec", "count", 64),
                                            (reg, wd, "digit", "dec", "count_shuf", 64)))
            for reg, wd in (("online", 0.0), ("batch", 0.0), ("batch", 1.0))},
        held_error_structure=dict(online_wd0=errors(runs, online), batch_wd0=errors(runs, base),
                                  batch_wd1=errors(runs, wd1),
                                  batch_wd1_count=errors(runs, ("batch", 1.0, "digit", "dec", "count", 64))))
    (HERE / "results" / "ANALYSIS.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps(doc, indent=1))


if __name__ == "__main__":
    main()
