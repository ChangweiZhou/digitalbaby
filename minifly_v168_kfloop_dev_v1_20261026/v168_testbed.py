"""V168 C0 testbed (design section 4): references only, on V168's technical worlds; no learner code exists. 0 learner
records (every arm is an evaluator instrument). Run through launch.py:
    python launch.py run --stage testbed -- /tmp/venv_t/bin/python v168_testbed.py
T0-a wiring (V168's reference forward == V166's Net; BPTT gradient == autograd through V166's Net); T0-b oracle reads;
T0-c per budget config: BPTT ceiling and TBPTT-1 floor; TBPTT-2/3 descriptive. Selection: the first config in
SETTINGS.config_order passing both. Fits run as child processes, at most SETTINGS.parallel at a time.
Writes results/testbed/TESTBED.json. --smoke: tiny sizes, writes only under cache/ (code check, no outcome).
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
import torch.nn.functional as F  # noqa: E402

import v168_lib as L  # noqa: E402

torch.set_num_threads(1)
STAGE = os.environ.get("MINIFLY_STAGE", "unlaunched")
CACHE = HERE / "cache"
SMOKE = {"n_train": 16, "calib": 16, "test": 32, "max_epochs": 2}
C1_PLANNED_RECORDS = 184


def cfg_of(name, smoke):
    c = dict(L.SET["configs"][name])
    if smoke:
        c.update(n_train=SMOKE["n_train"], max_epochs=SMOKE["max_epochs"])
    return c


def fit(cname, rule, smoke):
    cfg = cfg_of(cname, smoke)
    tr = L.world("v168_technical", "train", cfg["n_train"])
    ca = L.world("v168_technical", "calib", SMOKE["calib"] if smoke else None)
    te = L.world("v168_technical", "test", SMOKE["test"] if smoke else None)
    tag = f"{'smoke_' if smoke else ''}c0_{cname}_{rule}"
    with open(CACHE / f"{tag}.log", "w") as log:
        net, info = L.train(f"{cname}/{rule}", cfg, tr, ca, L.ref_grad(rule), log=log)
    out = {"training": info, **L.test_record(net, te)}
    (CACHE / f"{tag}.json").write_text(json.dumps(out))
    torch.save(net.state_dict(), CACHE / f"{tag}.pt")
    print(tag, json.dumps(out["test"]), info["best_epoch"], info["epochs_run"], info["cpu_s"], flush=True)


def check_wiring():
    """T0-a: forward equality (float32, bitwise) and BPTT gradient equality (float64) against V166's Net."""
    w = L.world("v168_technical", "train", 8)
    idx = np.arange(8)
    net = L.L166.Net("LOOP")
    x, K, V, ws, tg = L.L166.batch(w, idx, "LOOP")
    with torch.no_grad():
        fwd_equal = bool(torch.equal(L.forward(net, x, K, V, ws), net(x, K, V, ws)))
    net = net.double()
    x, K, V, ws, tg = L.L166.batch(w, idx, "LOOP", torch.float64)
    m = tg >= 0
    net.zero_grad()
    F.cross_entropy(net(x, K, V, ws)[m], tg[m]).backward()
    g166 = torch.cat([p.grad.flatten() for p in net.parameters()])
    net.zero_grad()
    L.ref_grad("BPTT")(net, x, K, V, ws, tg)
    g168 = torch.cat([p.grad.flatten() for p in net.parameters()])
    rel = float((g168 - g166).norm() / g166.norm())
    windows = {r: sorted(L.detach_steps(r)) for r in L.SET["references"]}
    nonoverlap = all(min(L.ANS[i] - L.ANS[i - 1] for i in range(1, len(L.ANS))) >= int(r[-1])
                     for r in L.SET["references"] if r != "BPTT")
    return {"forward_bitwise_equal": fwd_equal, "bptt_grad_rel_diff": rel, "ans_steps": L.ANS,
            "detach_steps": windows, "windows_non_overlapping": nonoverlap,
            "pass": bool(fwd_equal and rel <= L.SET["rules"]["wiring_tol"] and nonoverlap)}


def main():
    smoke = "--smoke" in sys.argv
    if len(sys.argv) > 3 and sys.argv[1] == "--fit":
        fit(sys.argv[2], sys.argv[3], smoke)
        return
    t0 = time.time()
    CACHE.mkdir(exist_ok=True)
    ru = L.SET["rules"]
    wiring = check_wiring()
    orc = L.L166.oracle(L.world("v168_technical", "test", SMOKE["test"] if smoke else None))
    print("T0-a", json.dumps(wiring), "T0-b", orc, flush=True)
    order = L.SET["config_order"]
    fits = [(c, r) for c in reversed(order) for r in L.SET["references"]]     # longest configs first
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    pending, running, failed = list(fits), [], []
    while pending or running:
        while pending and len(running) < L.SET["parallel"]:
            c, r = pending.pop(0)
            cmd = [sys.executable, __file__, "--fit", c, r] + (["--smoke"] if smoke else [])
            running.append(((c, r), subprocess.Popen(cmd, env=env)))
        cr, p = running.pop(0)
        if p.wait():
            failed.append(cr)
    if failed:
        raise SystemExit(f"fits failed: {failed}")
    pre = "smoke_" if smoke else ""
    res = {c: {r: json.loads((CACHE / f"{pre}c0_{c}_{r}.json").read_text()) for r in L.SET["references"]} for c in order}
    per_cfg, selected = {}, None
    for c in order:
        b, t1 = res[c]["BPTT"], res[c]["TBPTT-1"]
        g = L.gap(t1, b)
        ceiling = {"q2": b["test"]["q2_bits"], "q1": b["test"]["q1_bits"],
                   "best_epoch_is_last": b["training"]["best_epoch_is_last"],
                   "pass": b["test"]["q2_bits"] <= ru["ceiling_q2"] and b["test"]["q1_bits"] <= ru["ceiling_q1"]}
        floor = {**g, "pass": bool(g["gain_bits"] >= ru["floor_gap_bits"] and g["gain_bits"] >= ru["floor_gap_sd"] * g["boot_sd"])}
        desc = {r: {**res[c][r]["test"], "best_epoch": res[c][r]["training"]["best_epoch"],
                    "epochs_run": res[c][r]["training"]["epochs_run"],
                    "share_of_gap_q2": (t1["test"]["q2_bits"] - res[c][r]["test"]["q2_bits"]) / g["gain_bits"]
                    if g["gain_bits"] > 0 else None} for r in L.SET["references"]}
        cfg = L.SET["configs"][c]
        per_cfg[c] = {"config": cfg, "ceiling": ceiling, "floor": floor, "qualifies": bool(ceiling["pass"] and floor["pass"]),
                      "references": desc, "c2_learner_records": 2 * cfg["n_train"] * cfg["max_epochs"]}
        if selected is None and per_cfg[c]["qualifies"]:
            selected = c
    checks = {"T0-a_wiring": wiring,
              "T0-b_oracle": {"acc": orc, "pass": all(orc[k] >= ru["oracle_acc"] for k in orc if k.endswith("acc"))},
              "T0-c_budget_grid": {"selected": selected, "pass": selected is not None}}
    verdict = "CONTINUE" if all(c["pass"] for c in checks.values()) else "CANNOT_TEST_CLAIM_AS_SPECIFIED"
    need = None if selected is None else {
        "config": selected, "c1_planned_records": C1_PLANNED_RECORDS,
        "c2_learner_records": per_cfg[selected]["c2_learner_records"],
        "total_records": C1_PLANNED_RECORDS + per_cfg[selected]["c2_learner_records"],
        "fits_default_caps": C1_PLANNED_RECORDS + per_cfg[selected]["c2_learner_records"] <= 5000}
    out = {"stage": STAGE, "smoke": smoke, "settings": L.SET, "checks": checks, "verdict": verdict, "per_config": per_cfg,
           "records_needed": need, "training": {f"{c}|{r}": res[c][r]["training"] for c in order for r in res[c]},
           "line": {f"{c}|{r}": res[c][r]["line"] for c in order for r in res[c]},
           "learner_records": 0, "wall_s": round(time.time() - t0, 1)}
    if smoke:
        (CACHE / "smoke_TESTBED.json").write_text(json.dumps(out, indent=1))
    else:
        d = HERE / "results" / STAGE
        d.mkdir(parents=True, exist_ok=True)
        (d / "TESTBED.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({"checks": checks, "verdict": verdict, "records_needed": need,
                      "per_config": {c: {k: v for k, v in per_cfg[c].items() if k != "config"} for c in per_cfg}}, indent=1))
    if not smoke:
        import runtime
        runtime.write_closure()


if __name__ == "__main__":
    main()
