"""V174 C0 testbed (design section 3): V172's evaluator screen on the two dev worlds' 80 taught items, untaught bank;
plus a label-free check of the slot draw. 0 learner records.
Run through launch.py:  python launch.py run --stage testbed -- /tmp/venv_n/bin/python v174_testbed.py
Continue if CEILING >= 72/80 and FLOOR <= 8/80 in both worlds. Writes results/testbed/TESTBED.json.
"""
import importlib.util
import inspect
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v174_lib as L  # noqa: E402

STAGE = os.environ.get("MINIFLY_STAGE", "unlaunched")
T = L.SET["testbed"]


def site_codes(bank, item):
    q = bank.clone()
    q._prompt(item["prompt"])
    out = []
    for y in (*item["answer"], L.EOS):
        f = q.fe.clone()
        f.advance(q.time + L.L2.DELTA)
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
    by = {}
    for s in sites:
        by.setdefault(s[2].tobytes(), set()).add(s[3])
    K = X @ X.T
    A = np.linalg.solve(K + T["ridge_lambda"] * np.eye(len(K)), L.L2.C[Y])
    pred = K @ A
    ok = ((pred - pred.mean(1, keepdims=True)) @ L.L2.C.T / 4).argmax(1) == Y
    ceil = sum(all(ok[[n for n, s in enumerate(sites) if s[0] == i]]) for i in range(len(items)))
    maj = {k: Counter(s[3] for s in sites if s[1] == k).most_common(1)[0][0] for k in {s[1] for s in sites}}
    floor = sum(all(s[3] == maj[s[1]] for s in sites if s[0] == i) for i in range(len(items)))
    return {"items": len(items), "sites": len(sites), "codes_with_conflicting_targets": sum(len(v) > 1 for v in by.values()),
            "ceiling_correct": int(ceil), "floor_correct": int(floor),
            "ceiling_pass": bool(ceil >= T["ceiling_pass"]), "floor_fails": bool(floor <= T["floor_fail_max"])}


def main():
    t0 = time.process_time()
    bank = L.make(False)
    res = {w: screen(L.SET["worlds"][w], bank) for w in ("W1", "W2")}
    src = inspect.getsource(L.fork)
    label_free = (list(inspect.signature(L.slot_index).parameters) == ["g", "n"]
                  and "slot_index(g, len(hist))" in src and 'rng(world, "slot")' in src)
    out = {"stage": STAGE, "learner_records": 0, "results": res, "slot_draw_label_free": label_free,
           "decision": "continue" if label_free and all(r["ceiling_pass"] and r["floor_fails"] for r in res.values())
           else "CANNOT_TEST_CLAIM_AS_SPECIFIED", "cpu_s": round(time.process_time() - t0, 1)}
    d = HERE / "results" / "testbed"
    d.mkdir(parents=True, exist_ok=True)
    (d / "TESTBED.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    spec = importlib.util.spec_from_file_location("v174_runtime", HERE / "runtime.py")
    rt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rt)
    rt.write_closure()


if __name__ == "__main__":
    main()
