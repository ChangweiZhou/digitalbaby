"""V166 C1 technical (design section 4): checks of the comparison code on technical worlds (small instances).
0 learner records (every arm is a BPTT evaluator instrument). Run through launch.py:
    python launch.py run --stage technical -- /tmp/venv_t/bin/python v166_technical.py
1 NO-LOOP isolation; 2 parameter counts; 3 shuffle is a per-line derangement; 4 cut; 5 BPTT gradient vs finite
differences; 6 determinism. Writes results/technical/CHECKS.json.
"""
import copy
import json
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import v166_lib as L  # noqa: E402

torch.set_num_threads(1)
STAGE = os.environ.get("MINIFLY_STAGE", "unlaunched")
C1 = L.SET["c1"]


def states(net, w, idx, arm, cut=False, V_override=None):
    x, K, V, ws, _ = L.batch(w, idx, arm, torch.float64)
    if V_override is not None:
        V = V_override
    return net(x, K, V, ws, collect=True, cut=cut)[1]


def check_isolation(w):
    idx = np.arange(C1["iso_lines"])
    out = {}
    for arm in ("NO-LOOP", "LOOP"):
        net = L.Net(arm).double()
        h_true = states(net, w, idx, "LOOP")
        h_zero = states(net, w, idx, "LOOP-NULL")
        h_shuf = states(net, w, idx, "LOOP-SHUF")
        x, K, V, ws, _ = L.batch(w, idx, "LOOP", torch.float64)
        V = V.clone().requires_grad_(True)
        K = K.clone().requires_grad_(True)
        hs = net(x, K, V, ws, collect=True)[1]
        gV, gK = torch.autograd.grad(hs.sum(), (V, K), allow_unused=True)
        gmax = max(0.0 if g is None else float(g.abs().max()) for g in (gV, gK))
        out[arm] = {"states_equal_zero_mem": bool(torch.equal(h_true, h_zero)),
                    "states_equal_shuf_mem": bool(torch.equal(h_true, h_shuf)), "max_abs_grad_states_wrt_memory": gmax}
    ok = (out["NO-LOOP"]["states_equal_zero_mem"] and out["NO-LOOP"]["states_equal_shuf_mem"]
          and out["NO-LOOP"]["max_abs_grad_states_wrt_memory"] == 0.0
          and out["LOOP"]["max_abs_grad_states_wrt_memory"] > 0.0
          and not out["LOOP"]["states_equal_zero_mem"])
    return {**out, "pass": bool(ok)}


def check_params():
    n = {a: sum(p.numel() for p in L.Net(a).parameters()) for a in ("LOOP", "NO-LOOP", "LOOP-SHUF", "LOOP-NULL")}
    rel = n["NO-LOOP"] / n["LOOP"] - 1
    ok = n["NO-LOOP"] >= n["LOOP"] and rel <= C1["param_tol"] and n["LOOP-SHUF"] == n["LOOP"] == n["LOOP-NULL"]
    return {"params": n, "widths": {a: L.width(a) for a in n}, "no_loop_excess": rel,
            "formula": {"LOOP": L.n_params(L.width("LOOP"), True), "NO-LOOP": L.n_params(L.width("NO-LOOP"), False)},
            "pass": bool(ok)}


def check_shuffle():
    res, ok = {}, True
    for part in ("train", "test"):                     # technical worlds only; dev worlds stay unused until C2
        world = f"technical_{part}"
        w = L.make_world("technical", part, "MIXED")
        w2 = L.make_world("technical", part, "MIXED", 200)
        der = bool(np.all(w["vals_shuf"] != w["vals"]))
        multiset = bool(np.array_equal(np.sort(w["vals_shuf"], 1), np.sort(w["vals"], 1)))
        perms = {tuple(int(np.flatnonzero(w["vals"][i] == v)[0]) for v in w["vals_shuf"][i]) for i in range(w["n"])}
        share = len(perms) / w["n"]
        D = round(math.factorial(L.NPAIR) / math.e)                      # derangements of NPAIR (14,833 for 8)
        expected = D * (1 - (1 - 1 / D) ** w["n"]) / w["n"]                 # birthday expectation, uniform draws
        same = bool(np.array_equal(w2["vals_shuf"], w["vals_shuf"][:200]) and np.array_equal(w2["x"], w["x"][:200]))
        res[world] = {"derangement_every_line": der, "multiset_kept": multiset, "distinct_permutation_share": share,
                      "expected_share_uniform": expected, "regenerated_identically": same}
        ok &= der and multiset and share >= C1["distinct_share_of_expected"] * expected and same
    return {**res, "pass": bool(ok)}


def check_cut(w):
    idx = np.arange(C1["iso_lines"])
    net = L.Net("LOOP").double()
    zr = copy.deepcopy(net)
    with torch.no_grad():
        zr.Wr.zero_()
    a = states(zr, w, idx, "LOOP")
    b = states(net, w, idx, "LOOP", cut=True)
    c = states(net, w, idx, "LOOP")
    return {"wr_zero_equals_cut": bool(torch.equal(a, b)), "cut_changes_states": bool(not torch.equal(b, c)),
            "pass": bool(torch.equal(a, b) and not torch.equal(b, c))}


def check_grad(w):
    idx = np.arange(C1["fd_lines"])
    net = L.Net("LOOP").double()
    x, K, V, ws, tg = L.batch(w, idx, "LOOP", torch.float64)
    m = tg >= 0

    def loss():
        return F.cross_entropy(net(x, K, V, ws)[m], tg[m])
    net.zero_grad()
    loss().backward()
    rng = np.random.default_rng(3)
    ga, gf, eps = [], [], C1["fd_eps"]
    with torch.no_grad():
        for name, p in net.named_parameters():
            if p.dim() < 2:
                continue
            for _ in range(C1["fd_entries_per_param"]):
                ij = tuple(int(rng.integers(s)) for s in p.shape)
                old = p[ij].item()
                p[ij] = old + eps
                lp = loss().item()
                p[ij] = old - eps
                lm = loss().item()
                p[ij] = old
                gf.append((lp - lm) / (2 * eps))
                ga.append(p.grad[ij].item())
    ga, gf = np.array(ga), np.array(gf)
    rel = float(np.linalg.norm(ga - gf) / np.linalg.norm(ga))
    return {"entries": len(ga), "relative_error": rel, "pass": rel <= C1["fd_tol"]}


def check_determinism():
    tr = L.make_world("technical", "train", "MIXED", C1["det_train"])
    ca = L.make_world("technical", "calib", "MIXED", C1["det_calib"])
    a, ia = L.train("LOOP", tr, ca, max_epochs=1)
    b, ib = L.train("LOOP", tr, ca, max_epochs=1)
    same = all(torch.equal(a.state_dict()[k], b.state_dict()[k]) for k in a.state_dict())
    return {"identical_parameters": bool(same), "calib_bits": [ia["calib_bits"], ib["calib_bits"]], "pass": bool(same)}


def main():
    t0 = time.process_time()
    w = L.make_world("technical", "train", "MIXED", 16)
    checks = {"1_no_loop_isolation": check_isolation(w), "2_parameter_counts": check_params(),
              "3_shuffle": check_shuffle(), "4_cut": check_cut(w), "5_gradient": check_grad(w),
              "6_determinism": check_determinism()}
    out = {"stage": STAGE, "checks": checks, "all_pass": all(c["pass"] for c in checks.values()),
           "learner_records": 0, "cpu_s": round(time.process_time() - t0, 1)}
    d = HERE / "results" / STAGE
    d.mkdir(parents=True, exist_ok=True)
    (d / "CHECKS.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    import runtime
    runtime.write_closure()


if __name__ == "__main__":
    main()
