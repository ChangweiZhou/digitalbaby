"""Read-only: how differently does the frozen native input path (FE0 -> PN -> KC) see byte sequences?

No learning. Uses the newborn native fly from minifly_a_v3 (its src/ and package/ must be present) and the same byte
timing as the lineage comparison (one byte every 30/14 s; one extra step for REPLY before the first answer read).
Writes results/ORDER_PROBE.json.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "minifly_a_v3" / "src"))
import stores  # noqa: E402

FLY = stores.birth("native")[0].fly
BC = stores.bb.bc
ENC = BC.native().model.encode_sparse
DELTA = 30 / 14
PAIRS = [("3+8=", "8+3=", "same bytes, reordered"),
         ("5+3*2=", "5*3+2=", "same bytes, reordered"),
         ("(5+3)*2=", "(5*3)+2=", "same bytes, reordered"),
         ("12+34=", "34+12=", "same bytes, reordered"),
         ("(5+3)*2=", "5+3*2=", "brackets removed"),
         ("2*(5+3)=", "(5+3)*2=", "operands swapped around brackets"),
         ("1+9=", "2+9=", "first number changed"),
         ("3+8=", "4+8=", "first number changed"),
         ("3+8=", "3+9=", "last number changed"),
         ("(5+3)*2=", "(6+3)*2=", "first number changed"),
         ("3+8=1", "8+3=1", "reordered, after one answer byte"),
         ("the cat sat.", "the dog sat.", "subject changed"),
         ("dog bites man.", "dog bites cat.", "last word changed"),
         ("dog bites man.", "man bites dog.", "words swapped")]


def view(seq):
    fe = BC.FE0()
    t = fe.t = 0.0
    for b in seq:
        t += DELTA
        fe.advance(t)
        fe.feed(b, t)
    t += DELTA
    fe.advance(t)
    p = fe.read()
    return p, ENC(FLY.m, p)


def main():
    rows = []
    for a, b, what in PAIRS:
        pa, xa = view(a.encode())
        pb, xb = view(b.encode())
        rows.append(dict(a=a, b=b, differs=what,
                         receptor_cosine=float(pa @ pb / np.linalg.norm(pa) / np.linalg.norm(pb)),
                         kc_jaccard=float((xa * xb).sum() / ((xa + xb) > 0).sum()), active_kcs=int(xa.sum())))
    doc = dict(schema="NATIVE_INPUT_ORDER_PROBE_V1", byte_interval_s=DELTA, fe0_tau_s=BC.FE_PARAMS["FE0"]["tau_p"],
               weight_by_age_newest_first=[1.0] + [math.exp(-DELTA * k / BC.FE_PARAMS["FE0"]["tau_p"])
                                                   for k in range(1, 9)],
               note="Weights are at the moment each byte is fed relative to the newest; the extra REPLY step scales "
                    "all of them equally before the max-normalisation.",
               pairs=rows)
    (HERE / "results" / "ORDER_PROBE.json").write_text(json.dumps(doc, indent=1) + "\n")
    for r in rows:
        print(f"{r['a']:>15} vs {r['b']:<15} {r['differs']:34} cosine {r['receptor_cosine']:.3f}  "
              f"KC Jaccard {r['kc_jaccard']:.2f}")


if __name__ == "__main__":
    main()
