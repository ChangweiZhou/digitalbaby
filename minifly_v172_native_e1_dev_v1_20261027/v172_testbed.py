"""V172 C0 testbed (design section 4): evaluator-only screen on the untaught native code; 0 learner records.
Run through launch.py:  python launch.py run --stage testbed -- /tmp/venv_n/bin/python v172_testbed.py
For the technical and dev worlds' 80 taught items (OLD 64 + NEW 16), along the teacher-forced path of an untaught bank:
  collisions: answer-site pairs with identical KC codes but different targets; Jaccard overlap at the end of the prompt;
  CEILING: ridge map from the KC code to the target's C16 code, fitted on all taught sites, decoded as at recall;
  FLOOR: the most frequent taught byte at each answer position, ignoring the prompt.
Continue if CEILING >= 72/80 and FLOOR <= 8/80 in both worlds. Writes results/testbed/TESTBED.json.
"""
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v172_lib as L  # noqa: E402

STAGE = os.environ.get("MINIFLY_STAGE", "unlaunched")
T = L.SET["testbed"]


def site_codes(bank, item):
    """Binary KC code at each answer site (the code values() reads), along the teacher-forced path."""
    q = bank.clone()
    q._prompt(item["prompt"])
    out = []
    for y in (*item["answer"], L.EOS):
        f = q.fe.clone()
        f.advance(q.time + L.DELTA)
        out.append((q.encode(f.read())[0].astype(np.uint8), int(y)))
        if y != L.EOS:
            q._event(y)
    return out


def screen(world, bank):
    d = L.dataset(world)
    items = d["OLD"] + d["NEW"]
    sites = [(i, k, x, y) for i, it in enumerate(items) for k, (x, y) in enumerate(site_codes(bank, it))]
    X = np.array([s[2] for s in sites], dtype=float)
    Y = np.array([s[3] for s in sites])
    # collisions: identical codes with different targets
    by = {}
    for s in sites:
        by.setdefault(s[2].tobytes(), set()).add(s[3])
    hard = sum(1 for v in by.values() if len(v) > 1)
    first = np.array([s[2] for s in sites if s[1] == 0], dtype=float)
    inter = first @ first.T
    size = first.sum(1)
    jac = inter / (size[:, None] + size[None, :] - inter)
    iu = np.triu_indices(len(first), 1)
    j = jac[iu]
    # CEILING: ridge in dual form, all taught sites, complete answers along the teacher-forced path
    K = X @ X.T
    A = np.linalg.solve(K + T["ridge_lambda"] * np.eye(len(K)), L.C[Y])
    pred = K @ A
    logits = (pred - pred.mean(1, keepdims=True)) @ L.C.T / 4
    ok = logits.argmax(1) == Y
    ceil = sum(all(ok[[n for n, s in enumerate(sites) if s[0] == i]]) for i in range(len(items)))
    # FLOOR: most frequent taught byte per answer position
    maj = {k: Counter(s[3] for s in sites if s[1] == k).most_common(1)[0][0] for k in {s[1] for s in sites}}
    floor = sum(all(s[3] == maj[s[1]] for s in sites if s[0] == i) for i in range(len(items)))
    return {"items": len(items), "sites": len(sites), "kc_active_mean": float(X.sum(1).mean()),
            "distinct_codes": len(by), "codes_with_conflicting_targets": hard,
            "prompt_jaccard": {"mean": float(j.mean()), "q50": float(np.median(j)), "q95": float(np.quantile(j, .95)),
                               "max": float(j.max()), "pairs_over_0.8": int((j > .8).sum()), "pairs": int(len(j))},
            "ceiling_correct": int(ceil), "ceiling_site_acc": float(ok.mean()), "floor_correct": int(floor),
            "floor_majority": {str(k): v for k, v in maj.items()},
            "ceiling_pass": bool(ceil >= T["ceiling_pass"]), "floor_fails": bool(floor <= T["floor_fail_max"])}


def main():
    t0 = time.process_time()
    bank = L.make("N2B")
    res = {w: screen(L.SET["worlds"][w], bank) for w in ("technical", "dev")}
    out = {"stage": STAGE, "learner_records": 0, "results": res,
           "decision": "continue" if all(r["ceiling_pass"] and r["floor_fails"] for r in res.values())
           else "CANNOT_TEST_CLAIM_AS_SPECIFIED", "cpu_s": round(time.process_time() - t0, 1)}
    d = HERE / "results" / "testbed"
    d.mkdir(parents=True, exist_ok=True)
    (d / "TESTBED.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({w: {k: v for k, v in r.items() if k != "floor_majority"} for w, r in res.items()}, indent=1))
    print("decision:", out["decision"])
    import runtime
    runtime.write_closure()


if __name__ == "__main__":
    main()
