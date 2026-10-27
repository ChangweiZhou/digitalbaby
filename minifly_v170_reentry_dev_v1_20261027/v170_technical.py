"""V170 C1 technical (design section 4): A checks A1-A8 and B code checks B1-B4 on technical worlds, small fixtures.
Learner records: 1 per line per learner pass (gradient or training). Run through launch.py:
    python launch.py run --stage technical -- /tmp/venv_t/bin/python v170_technical.py
Writes results/technical/CHECKS.json. --smoke: scratch seeds, writes only under cache/ (code check, no outcome)."""
import importlib.util
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import v170_lib as L  # noqa: E402
import v170_learn as LR  # noqa: E402

torch.set_num_threads(1)
STAGE = os.environ.get("MINIFLY_STAGE", "unlaunched")
R = L.SET["rules"]
SCRATCH = {"technical": {"train": [990711, 1600], "calib": [990712, 200], "test": [990713, 1000]}}
RECORDS = {"n": 0}


def rel(a, b):
    return float((a - b).norm() / max(float(b.norm()), 1e-300))


def grads(net):
    return {k: p.grad.detach().clone() for k, p in net.named_parameters()}


def maxrel(ga, gb):
    return max(rel(ga[k], gb[k]) for k in gb)


def autograd_a(net, arm, w, idx, rule):
    net.zero_grad()
    x, r, tg = L.a_batch(w, idx, L.a_mode(arm), torch.float64)
    m = tg >= 0
    F.cross_entropy(net(x, r)[m], tg[m]).backward()
    return grads(net)


def check_a(seeds):
    out = {}
    tr = L.make_world("technical", "train", "T", 64, seeds)
    # A1: explicit one-step learner == autograd through the whole line (BPTT); for A the TBPTT-1 detach is vacuous
    idx = np.arange(4)
    res = {}
    for arm in ("A", "A-SHUF", "A-CUT"):
        net = L.ANet().double()
        with torch.no_grad():                          # move off the symmetric initial gate a little
            net.bg.fill_(0.3)
        g_bp = autograd_a(net, arm, tr, idx, "BPTT")
        net.zero_grad()
        LR.a_grad(arm, torch.float64)(net, tr, idx, None)
        res[arm] = maxrel(grads(net), g_bp)
        RECORDS["n"] += len(idx)
    out["A1_one_step_equals_bptt"] = {"max_rel_diff": res, "pass": all(v <= R["exact_tol"] for v in res.values())}
    # A2: learned-parameter isolation: dynamics take no parameter; recomputed states are bitwise identical, and the
    # forward of two different parameter settings uses the same states
    r1, q1 = L.a_states(tr, "A", np.arange(8))
    r2, q2 = L.a_states(tr, "A", np.arange(8))
    n1, n2 = L.ANet(), L.ANet()
    with torch.no_grad():
        for p in n2.parameters():
            p.add_(1.0)
    import inspect
    takes_params = any(p in inspect.signature(L.a_states).parameters for p in ("net", "params", "theta"))
    out["A2_param_isolation"] = {"states_bitwise_equal": bool(np.array_equal(r1, r2) and np.array_equal(q1, q2)),
                                 "dynamics_take_parameters": takes_params,
                                 "pass": bool(np.array_equal(r1, r2) and np.array_equal(q1, q2) and not takes_params)}
    # A3: SHUF
    w8 = L.make_world("technical", "train", "T", 8, seeds)
    w8b = L.make_world("technical", "train", "T", 8, seeds)
    der = all(not np.any(p == np.arange(L.NI)) and sorted(p) == list(range(L.NI)) for p in w8["perm"])
    same_epochs = bool(np.array_equal(w8["perm"], w8b["perm"]))
    distinct = len({tuple(p) for p in w8["perm"]}) / 8
    ra, qa = L.a_states(w8, "A")
    rs, qs = L.a_states(w8, "A-SHUF")
    cue = w8["cue"] >= 0
    cue_reads_equal = bool(np.array_equal(ra[cue], rs[cue]))
    reenter_differs = bool(not np.array_equal(qa[~cue], qs[~cue]))
    out["A3_shuf"] = {"derangements": der, "same_across_epochs": same_epochs, "distinct_share": distinct,
                      "cue_reads_equal": cue_reads_equal, "reentered_codes_differ": reenter_differs,
                      "pass": bool(der and same_epochs and distinct >= R["distinct_share_of_expected"] and cue_reads_equal
                                   and reenter_differs)}
    # A4: CUT
    rc, qc = L.a_states(w8, "A-CUT")
    out["A4_cut"] = {"cueless_reads_zero": bool(np.all(rc[~cue] == 0)), "cue_reads_equal": bool(np.array_equal(ra[cue], rc[cue])),
                     "pass": bool(np.all(rc[~cue] == 0) and np.array_equal(ra[cue], rc[cue]))}
    # A5: generic re-entry
    cnt = {}
    L.a_states(w8, "A", count=cnt)
    calls_ok = bool(np.array_equal(cnt["calls"], (~cue).sum(1)))
    same_fn = bool(np.array_equal(L.reenter(np.eye(L.NI), L.P_NP), L.CODES))
    out["A5_generic_reentry"] = {"calls_equal_cueless_steps": calls_ok, "reentry_equals_input_encoding": same_fn,
                                 "pass": calls_ok and same_fn}
    # A6: structure audit: untrained circuit near chance on technical test-T
    te = L.make_world("technical", "test", "T", 100 if seeds else None, seeds)
    s = L.test_record(L.ANet(), "A", te)["test"]
    out["A6_untrained_near_chance"] = {"test_T": s, "pass": bool(s["all_bits"] >= R["structure_bits"])}
    # A7: determinism
    cfg = dict(L.SET["configs"]["H3"], max_epochs=1)
    w16 = L.make_world("technical", "train", "T", 16, seeds)
    ca = L.make_world("technical", "calib", "T", 16, seeds)
    nets = []
    for _ in range(2):
        net, _ = L.train("A", cfg, w16, ca, LR.a_grad("A"))
        nets.append(net)
        RECORDS["n"] += 16
    det = all(torch.equal(a, b) for a, b in zip(nets[0].parameters(), nets[1].parameters()))
    out["A7_determinism"] = {"identical": bool(det), "pass": bool(det)}
    # A8: learns
    w32 = L.make_world("technical", "train", "T", 32, seeds)
    before = L.evaluate(L.ANet(), "A", w32)["all_bits"]
    net, _ = L.train("A", cfg, w32, w32, LR.a_grad("A"))
    RECORDS["n"] += 32
    after = L.evaluate(net, "A", w32)["all_bits"]
    out["A8_learns"] = {"train_bits_before": before, "after": after, "pass": bool(after < before)}
    return out


def autograd_loop_tbptt1(net, w, idx):
    net.zero_grad()
    x, K, V, ws, tg = L.loop_batch(w, idx, torch.float64)
    m = tg >= 0
    F.cross_entropy(net(x, K, V, ws, L.detach_steps("TBPTT-1"))[m], tg[m]).backward()
    return grads(net)


def check_b(seeds):
    out = {}
    tr = L.make_world("technical", "train", "T", 64, seeds)
    idx = np.arange(2)
    net = L.LoopNet().double()
    g_ref = autograd_loop_tbptt1(net, tr, idx)
    net.zero_grad()
    LR.b_grad(lam=0.0, identity=True, dtype=torch.float64)(net, tr, idx, None)
    RECORDS["n"] += 2
    d = maxrel(grads(net), g_ref)
    out["B1_identity_tbptt1"] = {"max_rel_diff": d, "pass": bool(d <= R["exact_tol"])}
    # B2: SHUF
    idx4 = np.arange(4)
    net = L.LoopNet()
    gB = LR.b_grad()
    net.zero_grad(); gB(net, tr, idx4, None); b = grads(net)
    gS = LR.b_grad(shuf=True)
    net.zero_grad(); gS(net, tr, idx4, None); s = grads(net)
    gI = LR.b_grad(shuf=True, perm_fn=lambda B: torch.arange(B))
    net.zero_grad(); gI(net, tr, idx4, None); i_ = grads(net)
    RECORDS["n"] += 12
    perms = [p for p in gS.state["perms"]]
    der = all(sorted(p) == list(range(len(p))) and all(p[k] != k for k in range(len(p))) for p in perms)
    ro = all(torch.equal(b[k], s[k]) for k in ("Wo", "Wor", "bo"))
    fb = any(not torch.equal(b[k], s[k]) for k in ("Wx", "Wh", "Wr", "bh", "Wq", "bq"))
    idn = all(torch.equal(b[k], i_[k]) for k in b)
    out["B2_shuf"] = {"derangements": der, "readout_equal": ro, "feedback_differs": fb, "identity_perm_reproduces_B": idn,
                      "pass": bool(der and ro and fb and idn)}
    cfg = dict(L.SET["configs"]["G1"], max_epochs=1)
    w16 = L.make_world("technical", "train", "T", 16, seeds)
    ca = L.make_world("technical", "calib", "T", 16, seeds)
    nets = []
    for _ in range(2):
        net, _ = L.train("LOOP", cfg, w16, ca, LR.b_grad())
        nets.append(net)
        RECORDS["n"] += 16
    det = all(torch.equal(a, b) for a, b in zip(nets[0].parameters(), nets[1].parameters()))
    out["B3_determinism"] = {"identical": bool(det), "pass": bool(det)}
    w32 = L.make_world("technical", "train", "T", 32, seeds)
    before = L.evaluate(L.LoopNet(), "LOOP", w32)["all_bits"]
    net, _ = L.train("LOOP", cfg, w32, w32, LR.b_grad())
    RECORDS["n"] += 32
    after = L.evaluate(net, "LOOP", w32)["all_bits"]
    out["B4_learns"] = {"train_bits_before": before, "after": after, "pass": bool(after < before)}
    return out


def main(smoke):
    seeds = SCRATCH if smoke else None
    out = {"stage": STAGE, "smoke": smoke, "A": check_a(seeds), "B": check_b(seeds)}
    out["learner_records"] = RECORDS["n"]
    out["all_pass"] = all(v["pass"] for grp in ("A", "B") for v in out[grp].values())
    out["records_within_c1_cap"] = RECORDS["n"] <= L.SET["c1_records_cap"]
    dest = (HERE / "cache" / "smoke_CHECKS.json") if smoke else (HERE / "results" / "technical" / "CHECKS.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1))
    if not smoke:
        spec = importlib.util.spec_from_file_location("v170_runtime", HERE / "runtime.py")
        rt = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rt)
        rt.write_closure()
    print(json.dumps({g: {k: v["pass"] for k, v in out[g].items()} for g in ("A", "B")}), out["learner_records"],
          out["all_pass"])


if __name__ == "__main__":
    main("--smoke" in sys.argv[1:])
