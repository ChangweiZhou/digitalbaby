"""V172 C1 technical checks (design section 4), on the technical world. Run through launch.py:
python launch.py run --stage technical -- /tmp/venv_n/bin/python v172_technical.py
  1. N2B equals Codex's models152.Native('N_2B') (vendored from ec38e8be): values and site log-probs after 3 records.
  2. DELTA-H with warmup forced to 1 equals fastmem.FastMemory(homeo=True, teach='delta').train: values and theta.
  3. A panel leaves the model unchanged (state digest).
  4. Training is deterministic (two fresh DELTA-H runs, identical digests).
  5. The warmup counter counts answer and EOS targets.
  6. Homeostasis: off arms keep theta = 0 and the firing estimate unchanged; on arms update the estimate.
  7. Cost per record and per panel item.
Writes results/technical/CHECKS.json.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v172_lib as L  # noqa: E402
sys.path.insert(0, str(HERE / "vendor"))
import models152  # noqa: E402

STAGE = os.environ.get("MINIFLY_STAGE", "unlaunched")


def train(m, items):
    out = []
    for r in items:
        out += m.train(r["prompt"], r["answer"])
    return out


def main():
    t0 = time.process_time()
    d = L.dataset(L.SET["worlds"]["technical"])
    three = d["OLD"][:3]
    records, checks = 0, {}

    a, b = L.make("N2B"), models152.Native("N_2B")
    sa = train(a, three)
    sb = [s for r in three for s in b.train(r["prompt"], r["answer"])]
    records += 6
    va, vb = a.values(), b.values()
    checks["1_n2b_equals_models152"] = {"pass": bool(np.array_equal(va, vb) and [s["lp"] for s in sa] == [s["lp"] for s in sb]),
                                        "max_abs_value_diff": float(np.abs(va - vb).max())}

    c, ref = L.make("DELTA-H", force_warmup=1.0), L.fastmem.FastMemory(homeo=True, teach="delta")
    train(c, three)
    for r in three:
        ref.train(r["prompt"], r["answer"])
    records += 6
    vc, vr = c.values(), ref.values()
    checks["2_delta_h_equals_cls_fastmem"] = {"pass": bool(np.array_equal(vc, vr) and np.array_equal(c.theta, ref.theta)
                                                           and np.array_equal(c.freq, ref.freq)),
                                              "max_abs_value_diff": float(np.abs(vc - vr).max())}

    before = L.digest(c)
    tp = time.process_time()
    L.panel(c, d["OLD"][:2])
    per_item = (time.process_time() - tp) / 2
    checks["3_panel_no_mutation"] = {"pass": L.digest(c) == before}

    c2 = L.make("DELTA-H", force_warmup=1.0)
    train(c2, three)
    records += 3
    checks["4_deterministic"] = {"pass": L.digest(c2) == L.digest(c)}

    targets = sum(len(r["answer"]) + 1 for r in three)
    checks["5_warmup_counter"] = {"pass": a.n == targets == c.n, "n": a.n, "targets": targets}

    h, nh = L.make("N2B-H"), L.make("DELTA")
    f0 = h.freq.copy()
    tr = time.process_time()
    train(h, three)
    train(nh, three)
    per_record = (time.process_time() - tr) / 6
    records += 6
    checks["6_homeostasis"] = {"pass": bool(not nh.theta.any() and np.array_equal(nh.freq, f0) and not a.theta.any()
                                            and not np.array_equal(h.freq, f0)),
                               "on_theta_max": float(h.theta.max()), "on_freq_changed": int((h.freq != f0).sum())}
    checks["7_cost"] = {"cpu_s_per_record": round(per_record, 3), "cpu_s_per_panel_item": round(per_item, 3),
                        "c2_forecast_cpu_h": round(4 * (1152 * per_record + (64 + 64 + 80 + 80) * per_item) / 3600, 3)}

    out = {"stage": STAGE, "checks": checks, "all_pass": all(v.get("pass", True) for v in checks.values()),
           "learner_records": records, "cpu_s": round(time.process_time() - t0, 1)}
    dd = HERE / "results" / "technical"
    dd.mkdir(parents=True, exist_ok=True)
    (dd / "CHECKS.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    import runtime
    runtime.write_closure()
    sys.exit(0 if out["all_pass"] else 1)


if __name__ == "__main__":
    main()
