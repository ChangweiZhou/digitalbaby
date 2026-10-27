"""V170 C0 testbed (design section 4): references only, on V170's technical worlds; no learner code exists. 0 learner
records (every arm is an evaluator instrument). Run through launch.py:
    python launch.py run --stage testbed -- /tmp/venv_t/bin/python v170_testbed.py
T0-a wiring (V170's LOOP == V166's Net; P, J, k(x) == V166's CODES); T0-b re-entry identity; T0-c oracle on test-T and
test-H; per G config T0-d (A-BPTT ceiling) and T0-e (A-CUT-BPTT and LOOP-TBPTT-1 floors); per G config and E1 T0-f
(LOOP qualification). Selection of R_A and R_B as designed. Fits run as child processes, at most SETTINGS.parallel at a
time. Writes results/testbed/TESTBED.json. --smoke: scratch seeds, tiny sizes, writes only under cache/ (no outcome).
"""
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
import torch.nn.functional as F  # noqa: E402

import v170_lib as L  # noqa: E402

torch.set_num_threads(1)
STAGE = os.environ.get("MINIFLY_STAGE", "unlaunched")
CACHE = HERE / "cache"
CACHE.mkdir(exist_ok=True)
SCRATCH = {"technical": {"train": [990701, 1600], "calib": [990702, 200], "test": [990703, 1000]}}
SMOKE = {"n_train": 12, "calib": 12, "test": 24, "max_epochs": 2}
R = L.SET["rules"]


def seeds(smoke):
    return SCRATCH if smoke else None


def cfg_of(name, smoke):
    c = dict(L.SET["configs"][name])
    if smoke:
        c.update(n_train=SMOKE["n_train"], max_epochs=SMOKE["max_epochs"])
    return c


def fit(cname, arm, rule, smoke):
    cfg = cfg_of(cname, smoke)
    sd = seeds(smoke)
    tr = L.make_world("technical", "train", "T", cfg["n_train"], sd)
    ca = L.make_world("technical", "calib", "T", SMOKE["calib"] if smoke else None, sd)
    tag = f"{'smoke_' if smoke else ''}c0_{cname}_{arm}_{rule}"
    with open(CACHE / f"{tag}.log", "w") as log:
        net, info = L.train(f"{arm}", cfg, tr, ca, L.ref_grad(arm, rule), log=log)
    out = {"training": info, "rule": rule}
    for p in ("T", "H"):
        te = L.make_world("technical", "test", p, SMOKE["test"] if smoke else None, sd)
        out[f"test_{p}"] = L.test_record(net, arm, te)
    (CACHE / f"{tag}.json").write_text(json.dumps(out))
    torch.save(net.state_dict(), CACHE / f"{tag}.pt")
    print(tag, json.dumps(out["test_T"]["test"]), json.dumps(out["test_H"]["test"]), info["best_epoch"],
          info["cpu_s"], flush=True)


def check_wiring():
    """T0-a: V170 LoopNet(d_in=70) vs V166's Net on V166 technical lines; P/J/codes vs V166's CODES."""
    w = L.L166.make_world("technical", "train", "MIXED", 8)
    idx = np.arange(8)
    ref = L.L166.Net("LOOP")
    mine = L.LoopNet(d_in=L.L166.D_IN)
    same_init = all(torch.equal(a, b) for a, b in zip(ref.parameters(), mine.parameters()))
    x, K, V, ws, tg = L.L166.batch(w, idx, "LOOP")
    with torch.no_grad():
        fwd_equal = bool(torch.equal(mine(x, K, V, ws), ref(x, K, V, ws)))
    ref, mine = ref.double(), mine.double()
    x, K, V, ws, tg = L.L166.batch(w, idx, "LOOP", torch.float64)
    m = tg >= 0
    F.cross_entropy(ref(x, K, V, ws)[m], tg[m]).backward()
    F.cross_entropy(mine(x, K, V, ws)[m], tg[m]).backward()
    rel = max(float((a.grad - b.grad).norm() / max(b.grad.norm(), 1e-300)) for a, b in zip(mine.parameters(),
                                                                                           ref.parameters()))
    codes_equal = bool(np.array_equal(L.CODES, L.L166.CODES))
    ok = same_init and fwd_equal and rel <= R["wiring_tol"] and codes_equal
    return {"same_init": same_init, "forward_bitwise_equal": fwd_equal, "bptt_grad_max_rel_diff": rel,
            "codes_equal_v166": codes_equal, "pass": bool(ok)}


def check_reentry():
    """T0-b: K(e_x) = k(x) for all items."""
    K = L.reenter(np.eye(L.NI), L.P_NP)
    eq = bool(np.array_equal(K, L.CODES))
    return {"reentry_equals_input_code_all_items": eq, "pass": eq}


def run_fits(jobs, smoke):
    procs, logs = [], []
    pending = list(jobs)
    par = L.SET["parallel"]
    while pending or procs:
        while pending and len(procs) < par:
            c, a, r = pending.pop(0)
            cmd = [sys.executable, __file__, "--fit", c, a, r] + (["--smoke"] if smoke else [])
            procs.append(((c, a, r), subprocess.Popen(cmd, cwd=HERE)))
        for j, (k, p) in enumerate(procs):
            if p.poll() is not None:
                if p.returncode != 0:
                    raise SystemExit(f"fit {k} failed with {p.returncode}")
                procs.pop(j)
                break
        else:
            time.sleep(0.5)


def load(c, a, r, smoke):
    return json.loads((CACHE / f"{'smoke_' if smoke else ''}c0_{c}_{a}_{r}.json").read_text())


def main(smoke):
    t0 = time.time()
    out = {"stage": STAGE, "smoke": smoke, "learner_records": 0, "T0-a": check_wiring(), "T0-b": check_reentry()}
    sd = seeds(smoke)
    out["T0-c"] = {p: L.oracle(L.make_world("technical", "test", p, SMOKE["test"] if smoke else None, sd))
                   for p in ("T", "H")}
    out["T0-c"]["pass"] = all(out["T0-c"][p][f"q{h}_acc"] >= R["oracle_acc"] for p in ("T", "H") for h in (1, 2))
    jobs = []
    for c in L.SET["a_order"]:
        jobs += [(c, "A", "BPTT"), (c, "A-CUT", "BPTT")]
    for c in L.SET["b_order"]:
        jobs += [(c, "LOOP", "BPTT"), (c, "LOOP", "TBPTT-1")]
    run_fits(jobs, smoke)
    cfgs = {}
    for c in L.SET["b_order"]:
        row = {}
        lb, lt = load(c, "LOOP", "BPTT", smoke), load(c, "LOOP", "TBPTT-1", smoke)
        row["LOOP-BPTT"] = {"test_T": lb["test_T"]["test"], "test_H": lb["test_H"]["test"], "best_epoch": lb["training"]["best_epoch"]}
        row["LOOP-TBPTT-1"] = {"test_T": lt["test_T"]["test"], "test_H": lt["test_H"]["test"], "best_epoch": lt["training"]["best_epoch"]}
        gl = L.gap(lt["test_T"], lb["test_T"])
        row["T0-f"] = {"loop_gap": gl,
                       "pass": bool(lb["test_T"]["test"]["q2_bits"] <= R["ceiling_q2"] and lb["test_T"]["test"]["q1_bits"] <= R["ceiling_q1"]
                                    and gl["gain_bits"] >= R["floor_gap_bits"] and gl["z"] is not None and gl["z"] >= R["pass_sd"])}
        if c in L.SET["a_order"]:
            ab, ac = load(c, "A", "BPTT", smoke), load(c, "A-CUT", "BPTT", smoke)
            row["A-BPTT"] = {"test_T": ab["test_T"]["test"], "test_H": ab["test_H"]["test"], "best_epoch": ab["training"]["best_epoch"]}
            row["A-CUT-BPTT"] = {"test_T": ac["test_T"]["test"], "test_H": ac["test_H"]["test"], "best_epoch": ac["training"]["best_epoch"]}
            g_cut, g_loop = L.gap(ac["test_T"], ab["test_T"]), L.gap(lt["test_T"], ab["test_T"])
            row["T0-d"] = {"pass": bool(ab["test_T"]["test"]["q2_bits"] <= R["ceiling_q2"] and ab["test_T"]["test"]["q1_bits"] <= R["ceiling_q1"])}
            ok = all(g["gain_bits"] >= R["floor_gap_bits"] and g["z"] is not None and g["z"] >= R["pass_sd"] for g in (g_cut, g_loop))
            row["T0-e"] = {"cut_gap": g_cut, "loop_tbptt1_gap": g_loop, "pass": bool(ok)}
        cfgs[c] = row
    out["configs"] = cfgs
    pres = {c: L.SET["configs"][c]["n_train"] * L.SET["configs"][c]["max_epochs"] for c in L.SET["b_order"]}
    ra = next((c for c in L.SET["a_order"] if cfgs[c]["T0-d"]["pass"] and cfgs[c]["T0-e"]["pass"]), None)
    rb = next((c for c in L.SET["b_order"] if cfgs[c]["T0-f"]["pass"]), None)
    rec_a = 3 * pres[ra] if ra else 0
    b_fits = rb is not None and rec_a + 2 * pres[rb] + L.SET["c1_records_cap"] <= L.SET["max_records"]
    base = out["T0-a"]["pass"] and out["T0-b"]["pass"] and out["T0-c"]["pass"]
    out["selection"] = {"R_A": ra, "R_B": rb, "c2_records_A": rec_a, "c2_records_B_if_run": 2 * pres[rb] if rb else None,
                        "B_fits_caps": bool(b_fits),
                        "decision": ("continue" if base and (ra or b_fits) else "CANNOT_TEST_CLAIM_AS_SPECIFIED"),
                        "A": "C2" if base and ra else "CANNOT_TEST_CLAIM_AS_SPECIFIED",
                        "B": ("C2" if base and b_fits else ("NOT_RUN: needs " + (f"{2 * pres[rb]} C2 records at {rb} plus C1" if rb
                                                                                 else "a budget beyond E1 (no qualifying LOOP reference up to 9,600 presentations)")))}
    out["wall_s"] = round(time.time() - t0, 1)
    dest = (CACHE / "smoke_TESTBED.json") if smoke else (HERE / "results" / "testbed" / "TESTBED.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1))
    if not smoke:
        spec = importlib.util.spec_from_file_location("v170_runtime", HERE / "runtime.py")
        rt = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rt)
        rt.write_closure()
    print(json.dumps(out["selection"]))


if __name__ == "__main__":
    a = sys.argv[1:]
    smoke = "--smoke" in a
    if "--fit" in a:
        i = a.index("--fit")
        fit(a[i + 1], a[i + 2], a[i + 3], smoke)
    else:
        main(smoke)
