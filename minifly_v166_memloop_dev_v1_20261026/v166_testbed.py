"""V166 C0 testbed (design section 4): references and qualification only, on technical worlds. No learner records
(every arm is a BPTT evaluator instrument). Run through launch.py:
    python launch.py run --stage testbed -- /tmp/venv_t/bin/python v166_testbed.py
T0-a factored read == explicit M_t q_t (float64); T0-b oracle reads with the true codes; fits (one child process each,
at most SETTINGS.parallel at a time): LOOP and NO-LOOP on ONE-HOP (T0-c), LOOP on MIXED (T0-d), LOOP-NULL on MIXED
(T0-e). Writes results/testbed/TESTBED.json. --smoke: tiny sizes, writes only under cache/ (code check, no outcome).
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import v166_lib as L  # noqa: E402

torch.set_num_threads(1)
STAGE = os.environ.get("MINIFLY_STAGE", "unlaunched")
CACHE = HERE / "cache"
FITS = [("LOOP", "ONE-HOP"), ("NO-LOOP", "ONE-HOP"), ("LOOP", "MIXED"), ("LOOP-NULL", "MIXED")]
SMOKE = {"train": 64, "calib": 32, "test": 64, "epochs": 2}


def worlds(task, smoke):
    return [L.make_world("technical", p, task, SMOKE[p] if smoke else None) for p in ("train", "calib", "test")]


def fit(arm, task, smoke):
    tr, ca, te = worlds(task, smoke)
    tag = f"{'smoke_' if smoke else ''}c0_{arm}_{task}"
    with open(CACHE / f"{tag}.log", "w") as log:
        net, info = L.train(arm, tr, ca, max_epochs=SMOKE["epochs"] if smoke else None, log=log)
    ev = L.evaluate(net, te, arm)
    out = {"training": info, "test": L.summary(ev), "line": {k: ev[k].tolist() for k in ev if k != "all_bits"}}
    (CACHE / f"{tag}.json").write_text(json.dumps(out))
    torch.save(net.state_dict(), CACHE / f"{tag}.pt")
    print(tag, json.dumps(out["test"]), info["best_epoch"], info["epochs_run"], info["cpu_s"], flush=True)


def check_read(smoke):
    w = L.make_world("technical", "train", "MIXED", 32)
    K, V, ws = L.memory_tensors(w, np.arange(32), "LOOP", torch.float64)
    q = torch.rand(32, L.NKC, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    return max(float((L.read_factored(K, V, ws, q, t) - L.read_explicit(K, V, ws, q, t)).abs().max()) for t in range(L.T))


def main():
    smoke = "--smoke" in sys.argv
    if len(sys.argv) > 3 and sys.argv[1] == "--fit":
        fit(sys.argv[2], sys.argv[3], smoke)
        return
    t0 = time.time()
    CACHE.mkdir(exist_ok=True)
    ru = L.SET["rules"]
    read_diff = check_read(smoke)
    orc = {task: L.oracle(worlds(task, smoke)[2]) for task in ("ONE-HOP", "MIXED")}
    print("T0-a read diff", read_diff, "T0-b oracle", orc, flush=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    pending, running, failed = list(FITS), [], []
    while pending or running:
        while pending and len(running) < L.SET["parallel"]:
            arm, task = pending.pop(0)
            cmd = [sys.executable, __file__, "--fit", arm, task] + (["--smoke"] if smoke else [])
            running.append(((arm, task), subprocess.Popen(cmd, env=env)))
        (arm_task, p) = running.pop(0)
        if p.wait():
            failed.append(arm_task)
    if failed:
        raise SystemExit(f"fits failed: {failed}")
    pre = "smoke_" if smoke else ""
    res = {f"{a}|{t}": json.loads((CACHE / f"{pre}c0_{a}_{t}.json").read_text()) for a, t in FITS}
    s = {k: v["test"] for k, v in res.items()}
    checks = {
        "T0-a_read": {"max_abs_diff": read_diff, "pass": read_diff <= ru["read_tol"]},
        "T0-b_oracle": {"acc": orc, "pass": all(v[k] >= ru["oracle_acc"] for v in orc.values() for k in v if k.endswith("acc"))},
        "T0-c_one_hop_sanity": {"LOOP": s["LOOP|ONE-HOP"]["q1_bits"], "NO-LOOP": s["NO-LOOP|ONE-HOP"]["q1_bits"],
                                "pass": max(s["LOOP|ONE-HOP"]["q1_bits"], s["NO-LOOP|ONE-HOP"]["q1_bits"]) <= ru["sanity_bits"]},
        "T0-d_ceiling": {"q2": s["LOOP|MIXED"]["q2_bits"], "q1": s["LOOP|MIXED"]["q1_bits"],
                         "best_epoch_at_edge": res["LOOP|MIXED"]["training"]["cut_off_by_budget"],
                         "pass": s["LOOP|MIXED"]["q2_bits"] <= ru["ceiling_q2"] and s["LOOP|MIXED"]["q1_bits"] <= ru["ceiling_q1"]},
        "T0-e_floor": {"q2": s["LOOP-NULL|MIXED"]["q2_bits"], "pass": s["LOOP-NULL|MIXED"]["q2_bits"] >= ru["floor_q2"]},
    }
    verdict = "CONTINUE" if all(c["pass"] for c in checks.values()) else "CANNOT_TEST_CLAIM_AS_SPECIFIED"
    out = {"stage": STAGE, "smoke": smoke, "settings": L.SET, "checks": checks, "verdict": verdict, "summary": s,
           "training": {k: v["training"] for k, v in res.items()}, "line": {k: v["line"] for k, v in res.items()},
           "learner_records": 0, "evaluator_epochs": {k: v["training"]["epochs_run"] for k, v in res.items()},
           "wall_s": round(time.time() - t0, 1)}
    if smoke:
        (CACHE / "smoke_TESTBED.json").write_text(json.dumps(out, indent=1))
    else:
        d = HERE / "results" / STAGE
        d.mkdir(parents=True, exist_ok=True)
        (d / "TESTBED.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({"checks": checks, "verdict": verdict}, indent=1))
    if not smoke:
        import runtime
        runtime.write_closure()


if __name__ == "__main__":
    main()
