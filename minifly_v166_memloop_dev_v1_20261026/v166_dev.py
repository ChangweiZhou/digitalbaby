"""V166 C2 dev (design section 4): the comparison on fresh dev worlds (seeds 201-203), all four arms trained on MIXED
with the identical schedule (one child process each, at most SETTINGS.parallel at a time). 0 learner records (every arm
is a BPTT evaluator instrument). Run through launch.py:
    python launch.py run --stage dev -- /tmp/venv_t/bin/python v166_dev.py
P1: LOOP beats NO-LOOP on Q2 by >= 0.5 bits and >= 3 paired line-bootstrap SDs. P2: the same against LOOP-SHUF.
Guards: LOOP's and NO-LOOP's Q1 <= 0.5 bits. Verdict SUPPORTED / NOT_SUPPORTED / INVALID. Diagnostics (non-gating):
LOOP-NULL, LOOP-CUT (W_r ignored at test), LOOP read with deranged memory at test, accuracies, query targeting.
Writes results/dev/RESULTS.json.
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
ARMS = ["LOOP", "NO-LOOP", "LOOP-SHUF", "LOOP-NULL"]


def worlds():
    return [L.make_world("dev", p, "MIXED") for p in ("train", "calib", "test")]


def fit(arm):
    tr, ca, te = worlds()
    with open(CACHE / f"c2_{arm}.log", "w") as log:
        net, info = L.train(arm, tr, ca, log=log)
    ev = L.evaluate(net, te, arm)
    out = {"training": info, "test": L.summary(ev), "line": {k: ev[k].tolist() for k in ev if k != "all_bits"}}
    (CACHE / f"c2_{arm}.json").write_text(json.dumps(out))
    torch.save(net.state_dict(), CACHE / f"c2_{arm}.pt")
    print(f"c2_{arm}", json.dumps(out["test"]), info["best_epoch"], info["epochs_run"], info["cpu_s"], flush=True)


def compare(a, b, res, hop=2):
    """b's bits minus a's bits on Q<hop> (positive: a is better), with the paired line-bootstrap SD."""
    la, lb = res[a]["line"], res[b]["line"]
    d = res[b]["test"][f"q{hop}_bits"] - res[a]["test"][f"q{hop}_bits"]
    sd = L.boot_sd(np.array(lb[f"q{hop}_sum"]), np.array(la[f"q{hop}_sum"]), np.array(la[f"q{hop}_n"]))
    return {"gain_bits": d, "boot_sd": sd, "z": d / sd if sd > 0 else None}


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--fit":
        fit(sys.argv[2])
        return
    t0 = time.time()
    CACHE.mkdir(exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    pending, running, failed = list(ARMS), [], []
    while pending or running:
        while pending and len(running) < L.SET["parallel"]:
            arm = pending.pop(0)
            running.append((arm, subprocess.Popen([sys.executable, __file__, "--fit", arm], env=env)))
        arm, p = running.pop(0)
        if p.wait():
            failed.append(arm)
    if failed:
        raise SystemExit(f"fits failed: {failed}")
    ru = L.SET["rules"]
    res = {a: json.loads((CACHE / f"c2_{a}.json").read_text()) for a in ARMS}
    s = {a: res[a]["test"] for a in ARMS}
    p1, p2 = compare("LOOP", "NO-LOOP", res), compare("LOOP", "LOOP-SHUF", res)
    p1["pass"] = bool(p1["gain_bits"] >= ru["pass_bits"] and p1["gain_bits"] >= ru["pass_sd"] * p1["boot_sd"])
    p2["pass"] = bool(p2["gain_bits"] >= ru["pass_bits"] and p2["gain_bits"] >= ru["pass_sd"] * p2["boot_sd"])
    guards = {"LOOP_q1": s["LOOP"]["q1_bits"], "NO-LOOP_q1": s["NO-LOOP"]["q1_bits"],
              "pass": bool(max(s["LOOP"]["q1_bits"], s["NO-LOOP"]["q1_bits"]) <= ru["guard_q1"])}
    verdict = "INVALID" if not guards["pass"] else "SUPPORTED" if p1["pass"] and p2["pass"] else "NOT_SUPPORTED"
    # diagnostics (non-gating)
    te = L.make_world("dev", "test", "MIXED")
    nets = {}
    for a in ("LOOP", "NO-LOOP"):
        nets[a] = L.Net(a)
        nets[a].load_state_dict(torch.load(CACHE / f"c2_{a}.pt"))
    diag = {"LOOP-CUT": L.summary(L.evaluate(nets["LOOP"], te, "LOOP", cut=True)),
            "LOOP_with_deranged_memory_at_test": L.summary(L.evaluate(nets["LOOP"], te, "LOOP-SHUF")),
            "targeting": {a: L.targeting(nets[a], te, "LOOP") for a in ("LOOP", "NO-LOOP")},
            "comparisons_q2": {"LOOP_vs_LOOP-NULL": compare("LOOP", "LOOP-NULL", res),
                               "NO-LOOP_vs_LOOP-NULL": compare("NO-LOOP", "LOOP-NULL", res),
                               "LOOP-SHUF_vs_LOOP-NULL": compare("LOOP-SHUF", "LOOP-NULL", res)},
            "set_floor_q2_bits": float(np.log2(L.LINE["chains"])), "chance_bits": float(np.log2(L.NI))}
    out = {"stage": STAGE, "summary": s, "P1_LOOP_vs_NO-LOOP": p1, "P2_LOOP_vs_LOOP-SHUF": p2, "guards": guards,
           "verdict": verdict, "diagnostics_non_gating": diag,
           "training": {a: res[a]["training"] for a in ARMS}, "line": {a: res[a]["line"] for a in ARMS},
           "learner_records": 0, "evaluator_epochs": {a: res[a]["training"]["epochs_run"] for a in ARMS},
           "wall_s": round(time.time() - t0, 1)}
    d = HERE / "results" / STAGE
    d.mkdir(parents=True, exist_ok=True)
    (d / "RESULTS.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("summary", "P1_LOOP_vs_NO-LOOP", "P2_LOOP_vs_LOOP-SHUF", "guards", "verdict",
                                          "diagnostics_non_gating")}, indent=1))
    import runtime
    runtime.write_closure()


if __name__ == "__main__":
    main()
