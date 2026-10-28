"""V174 C1 technical checks (design section 3), technical world. Run through launch.py:
python launch.py run --stage technical -- /tmp/venv_n/bin/python v174_technical.py
  1. The H0 learner equals V172's DELTA arm (values identical after 3 records).
  2. A write-inhibited record leaves every compartment's fast/slow values unchanged, and advances the clock and warmup
     counter exactly as a written record does.
  3. BLOCKED and REPLAY forks draw identical slot records.
  4. A panel leaves the model unchanged.
  5. Training is deterministic.
  6. Cost per record.
Writes results/technical/CHECKS.json.
"""
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v174_lib as L  # noqa: E402

STAGE = os.environ.get("MINIFLY_STAGE", "unlaunched")


def main():
    t0 = time.process_time()
    world = L.SET["worlds"]["technical"]
    d = L.dataset(world)
    three = d["OLD"][:3]
    records, checks = 0, {}

    a, b = L.make(False), L.L2.make("DELTA")
    tr = time.process_time()
    for r in three:
        a.train(r["prompt"], r["answer"])
        b.train(r["prompt"], r["answer"])
    per_record = (time.process_time() - tr) / 6
    records += 6
    checks["1_h0_equals_v172_delta"] = {"pass": bool(np.array_equal(a.values(), b.values()))}

    w, i = a.clone(), a.clone()
    before = L.values_digest(i)
    w.train(d["OLD"][3]["prompt"], d["OLD"][3]["answer"], write=True)
    i.train(d["OLD"][3]["prompt"], d["OLD"][3]["answer"], write=False)
    records += 2
    checks["2_inhibited_slot"] = {"pass": bool(L.values_digest(i) == before and L.values_digest(w) != before
                                               and i.time == w.time and i.n == w.n),
                                  "time": [i.time, w.time], "n": [i.n, w.n]}

    hist = list(three)
    mb, db = L.fork(a, hist, d, world, replay=False, passes=1)
    mr, dr = L.fork(a, hist, d, world, replay=True, passes=1)
    records += 2 * 32
    g = L.rng(world, "slot")
    content_free = [L.slot_index(g, len(hist) + 2 * j + 1) for j in range(len(db))]
    checks["3_identical_slot_draws"] = {"pass": db == dr and [k for k, _ in db] == content_free, "draws": len(db),
                                        "differs_after_replay_writes": L.values_digest(mb) != L.values_digest(mr)}

    dg = L.L2.digest(a)
    tp = time.process_time()
    L.L2.panel(a, d["OLD"][:2])
    per_item = (time.process_time() - tp) / 2
    checks["4_panel_no_mutation"] = {"pass": L.L2.digest(a) == dg}

    c = L.make(False)
    for r in three:
        c.train(r["prompt"], r["answer"])
    records += 3
    checks["5_deterministic"] = {"pass": L.L2.digest(c) == L.L2.digest(a)}
    checks["6_cost"] = {"cpu_s_per_record": round(per_record, 3), "cpu_s_per_panel_item": round(per_item, 3),
                        "c2_forecast_cpu_h": round((4608 * per_record + (64 * 3 + 80 * 6) * per_item) / 3600, 3)}

    out = {"stage": STAGE, "checks": checks, "all_pass": all(v.get("pass", True) for v in checks.values()),
           "learner_records": records, "cpu_s": round(time.process_time() - t0, 1)}
    dd = HERE / "results" / "technical"
    dd.mkdir(parents=True, exist_ok=True)
    (dd / "CHECKS.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    spec = importlib.util.spec_from_file_location("v174_runtime", HERE / "runtime.py")
    rt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rt)
    rt.write_closure()
    sys.exit(0 if out["all_pass"] else 1)


if __name__ == "__main__":
    main()
