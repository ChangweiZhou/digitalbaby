"""V170 C2 dev (design section 4): strategy A at the selected budget R_A (results/testbed/TESTBED_2.json) on fresh dev
worlds. Learner arms A, A-SHUF, A-CUT (one-step learner); references LOOP-BPTT and LOOP-TBPTT-1 at R_A's budget
(loop_ref), and LOOP-BPTT at E1 (descriptive order reference). B is NOT_RUN by the selection rule. Run through launch.py:
    python launch.py run --stage dev -- /tmp/venv_t/bin/python v170_dev.py
Writes results/dev/RESULTS.json (learner_records = lines x epochs run, summed over the three learner arms)."""
import importlib.util
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

import v170_lib as L  # noqa: E402
import v170_learn as LR  # noqa: E402

torch.set_num_threads(1)
STAGE = os.environ.get("MINIFLY_STAGE", "unlaunched")
CACHE = HERE / "cache"
CACHE.mkdir(exist_ok=True)
R = L.SET["rules"]
SCRATCH = {"dev": {"train": [990721, 1600], "calib": [990722, 200], "test": [990723, 1000]}}
SMOKE = {"n_train": 12, "calib": 12, "test": 24, "max_epochs": 1}


def selection():
    return json.loads((HERE / "results" / "testbed" / "TESTBED_2.json").read_text())["selection"]


def gate_stats(net, w):
    """Gate decision accuracy at ANS steps (g > 0.5 <=> TWO-HOP), overall and per unordered instruction pair."""
    r, _ = L.a_states(w, "A")
    with torch.no_grad():
        g, *_ = net.parts(torch.from_numpy(w["x"]), torch.from_numpy(r).float())
    g = g.numpy()
    ok, per = [], {}
    for i in range(w["n"]):
        for e, s in enumerate(L.ANS):
            hp = int(w["hop"][i, s])
            c = int((g[i, s] > 0.5) == (hp == 2))
            ok.append(c)
            u, v = int(w["episodes"][i, e, 5]), int(w["episodes"][i, e, 6])
            per.setdefault(f"{min(u, v)}{max(u, v)}", []).append(c)
    return {"acc": float(np.mean(ok)), "per_pair": {k: float(np.mean(v)) for k, v in sorted(per.items())}}


def fit(arm, cname, rule, smoke):
    cfg = dict(L.SET["configs"][cname])
    sd = SCRATCH if smoke else None
    if smoke:
        cfg.update(n_train=SMOKE["n_train"], max_epochs=SMOKE["max_epochs"])
    tr = L.make_world("dev", "train", "T", cfg["n_train"], sd)
    ca = L.make_world("dev", "calib", "T", SMOKE["calib"] if smoke else None, sd)
    grad = LR.a_grad(arm) if rule == "learner" else L.ref_grad(arm, rule)
    tag = f"{'smoke_' if smoke else ''}c2_{arm}_{cname}_{rule}"
    with open(CACHE / f"{tag}.log", "w") as log:
        net, info = L.train(arm, cfg, tr, ca, grad, log=log)
    out = {"training": info, "rule": rule, "config": cname}
    for p in ("T", "H"):
        te = L.make_world("dev", "test", p, SMOKE["test"] if smoke else None, sd)
        out[f"test_{p}"] = L.test_record(net, arm, te)
        if L.arm_kind(arm) == "A":
            out[f"gate_{p}"] = gate_stats(net, te)
            cache = L.a_state_cache(te, L.a_mode(arm))
            out[f"forced_{p}"] = {str(gv): L.summary(L.evaluate(net, arm, te, cache=cache, gate=gv)) for gv in (0, 1)}
    (CACHE / f"{tag}.json").write_text(json.dumps(out))
    print(tag, json.dumps(out["test_T"]["test"]), json.dumps(out["test_H"]["test"]), info["best_epoch"], info["cpu_s"],
          flush=True)


def run(jobs, smoke):
    procs, pending = [], list(jobs)
    while pending or procs:
        while pending and len(procs) < L.SET["parallel"]:
            j = pending.pop(0)
            procs.append((j, subprocess.Popen([sys.executable, __file__, "--fit", *j] + (["--smoke"] if smoke else []), cwd=HERE)))
        for k, (j, p) in enumerate(procs):
            if p.poll() is not None:
                if p.returncode != 0:
                    raise SystemExit(f"fit {j} failed")
                procs.pop(k)
                break
        else:
            time.sleep(0.5)


def main(smoke):
    sel = selection()
    ra = sel["R_A"]
    lref = L.SET["configs"][ra]["loop_ref"]
    jobs = [("A", ra, "learner"), ("A-SHUF", ra, "learner"), ("A-CUT", ra, "learner"),
            ("LOOP", lref, "BPTT"), ("LOOP", lref, "TBPTT-1"), ("LOOP", "E1", "BPTT")]
    run(jobs, smoke)
    res = {j: json.loads((CACHE / f"{'smoke_' if smoke else ''}c2_{j[0]}_{j[1]}_{j[2]}.json").read_text()) for j in jobs}
    A, S, C = res[jobs[0]], res[jobs[1]], res[jobs[2]]
    LB, LT, LE = res[jobs[3]], res[jobs[4]], res[jobs[5]]
    tT = lambda x: x["test_T"]["test"]  # noqa: E731
    tH = lambda x: x["test_H"]["test"]  # noqa: E731

    def sig(g, bits):
        return bool(g["gain_bits"] >= bits and g["z"] is not None and g["z"] >= R["pass_sd"])
    g_loop, g_shuf, g_cut = L.gap(LT["test_T"], A["test_T"]), L.gap(S["test_T"], A["test_T"]), L.gap(C["test_T"], A["test_T"])
    P1 = bool(tT(A)["q2_bits"] <= R["ceiling_q2"] and tT(A)["q1_bits"] <= R["ceiling_q1"] and sig(g_loop, R["floor_gap_bits"]))
    P2 = bool(sig(g_shuf, R["floor_gap_bits"]) and sig(g_cut, R["floor_gap_bits"]))
    P3 = bool(tH(A)["q1_bits"] <= R["ceiling_q1"] and tH(A)["q2_bits"] <= R["ceiling_q2"])
    G1 = bool(tT(S)["q1_bits"] <= R["ceiling_q1"] and tT(C)["q1_bits"] <= R["ceiling_q1"])
    verdict = "INVALID" if not G1 else ("PASS" if P1 and P2 and P3 else "FAIL")
    recs = sum(r["training"]["line_presentations"] for r in (A, S, C))
    out = {"stage": STAGE, "smoke": smoke, "R_A": ra, "loop_ref": lref, "learner_records": recs,
           "arms": {f"{j[0]}/{j[1]}/{j[2]}": {k: v for k, v in res[j].items() if k not in ("test_T", "test_H")}
                    | {"test_T": tT(res[j]), "test_H": tH(res[j])} for j in jobs},
           "rules": {"P1": {"A_test_T": tT(A), "loop_tbptt1_minus_A_q2": g_loop, "pass": P1},
                     "P2": {"shuf_minus_A_q2": g_shuf, "cut_minus_A_q2": g_cut, "pass": P2},
                     "P3": {"A_test_H": tH(A), "pass": P3},
                     "G1": {"shuf_q1": tT(S)["q1_bits"], "cut_q1": tT(C)["q1_bits"], "pass": G1}},
           "verdict_A": verdict, "verdict_B": "NOT_RUN: " + sel["B"],
           "diagnostics_non_gating": {"A_test_H_minus_test_T": {h: L.gap(A["test_H"], A["test_T"], h) for h in (1, 2)},
                                      "loopE1_minus_A_q2_test_T": L.gap(LE["test_T"], A["test_T"])}}
    dest = (CACHE / "smoke_RESULTS.json") if smoke else (HERE / "results" / "dev" / "RESULTS.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1))
    if not smoke:
        spec = importlib.util.spec_from_file_location("v170_runtime", HERE / "runtime.py")
        rt = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rt)
        rt.write_closure()
    print(json.dumps({k: v["pass"] for k, v in out["rules"].items()}), verdict, recs)


if __name__ == "__main__":
    a = sys.argv[1:]
    smoke = "--smoke" in a
    if "--fit" in a:
        i = a.index("--fit")
        fit(a[i + 1], a[i + 2], a[i + 3], smoke)
    else:
        main(smoke)
