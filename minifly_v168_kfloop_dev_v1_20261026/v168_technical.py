"""V168 C1 technical (design section 4): checks of the learner code on V168's technical worlds (small fixtures).
Learner records: 1 per line per pass of a learner computation; training fixtures count lines x epochs. Run through
launch.py:
    python launch.py run --stage technical -- /tmp/venv_t/bin/python v168_technical.py
1 TBPTT-1 identity; 2 exact-RTRL identity; 3 unbiasedness by enumeration of the sign sequences; 4 SHUF; 5 diagonal
traces (e-prop's query update == TBPTT-1's); 6 determinism; 7 learns; alignment at initialisation (non-gating).
The budget config (batch, lr) for 6-7 is the first qualifying one ignoring the records cap (testbed-003). Writes results/technical/CHECKS.json.
"""
import itertools
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import v168_kf as KF  # noqa: E402
import v168_lib as L  # noqa: E402

torch.set_num_threads(1)
STAGE = os.environ.get("MINIFLY_STAGE", "unlaunched")
C1 = L.SET["c1"]
REC = {"n": 0}
RQ = ("Wx", "Wh", "bh", "Wr", "Wq", "bq")          # recurrent and query groups (the readout is exact in every rule)
RA = ("Wx", "Wh", "bh", "Wr")
F64 = torch.float64


def count(n):
    REC["n"] += int(n)


def b64(w, idx):
    return L.L166.batch(w, np.asarray(idx), "LOOP", F64)


def net64():
    return L.L166.Net("LOOP").double()


def autograd(net, x, K, V, ws, tg, rule):
    net.zero_grad()
    L.ref_grad(rule)(net, x, K, V, ws, tg)
    return {k: p.grad.detach().clone() for k, p in net.named_parameters()}


def flat(g, keys):
    return torch.cat([g[k].flatten() for k in keys])


def rel(a, b, keys=KF.NAMES):
    return float((flat(a, keys) - flat(b, keys)).norm() / flat(b, keys).norm())


def cos(a, b, keys=RQ):
    fa, fb = flat(a, keys), flat(b, keys)
    return float(fa @ fb / (fa.norm() * fb.norm()))


def sub(a, b):
    return {k: a[k] - b[k] for k in a}


def check_identities(w):
    idx = np.arange(C1["exact_lines"])
    net = net64()
    x, K, V, ws, tg = b64(w, idx)
    tb1_auto, bptt_auto = autograd(net, x, K, V, ws, tg, "TBPTT-1"), autograd(net, x, K, V, ws, tg, "BPTT")
    tb1, _ = KF.run(net, x, K, V, ws, tg, "TB1")
    count(len(idx))
    ex, _ = KF.run(net, x, K, V, ws, tg, "EXACT")
    count(len(idx))
    r1, r2 = rel(tb1, tb1_auto), rel(ex, bptt_auto)
    per1 = {k: rel(tb1, tb1_auto, (k,)) for k in KF.NAMES}
    per2 = {k: rel(ex, bptt_auto, (k,)) for k in KF.NAMES}
    carried_share = float(flat(sub(bptt_auto, tb1_auto), RQ).norm() / flat(bptt_auto, RQ).norm())
    return ({"rel_diff": r1, "per_parameter": per1, "pass": r1 <= C1["exact_tol"]},
            {"rel_diff": r2, "per_parameter": per2, "bptt_minus_tb1_share_of_norm": carried_share,
             "pass": r2 <= C1["exact_tol"]})


def check_unbiased(w):
    i = int(np.flatnonzero(w["wstep"][:, 0] <= 1)[0])
    x, K, V, ws, tg = b64(w, [i])
    net = net64()
    ts = C1["enum_prefix"] - 1
    ex = KF.run(net, x, K, V, ws, tg, "EXACT", t_stop=ts)
    out = {"line": i, "first_write_step": int(w["wstep"][i, 0]), "prefix_steps": ts + 1}
    ok = True
    for g, gi in (("recurrent", 0), ("query", 1)):
        acc, last = 0, None
        for combo in itertools.product([-1.0, 1.0], repeat=ts):
            S = torch.ones(L.T, 2, 1, dtype=F64)
            S[1:ts + 1, gi, 0] = torch.tensor(combo, dtype=F64)
            tr = KF.run(net, x, K, V, ws, tg, "KF", signs=S, t_stop=ts)
            count(1)
            A, u = (tr["Aa"], tr["ua"]) if gi == 0 else (tr["Aq"], tr["uq"])
            last = A[:, :, :, None] * u[:, None, None, :]
            acc = acc + last
        J = ex["Ja"] if gi == 0 else ex["Jq"]
        mean = acc / 2 ** ts
        r = float((mean - J).norm() / J.norm())
        out[g] = {"sequences": 2 ** ts, "rel_diff_of_mean": r, "rel_diff_single_draw": float((last - J).norm() / J.norm()),
                  "J_norm": float(J.norm())}
        ok &= r <= C1["exact_tol"] and float(J.norm()) > 0
    return {**out, "pass": bool(ok)}


def check_shuf(w):
    rng = np.random.default_rng(L.SET["kf"]["shuf_seed"])
    der = all(not np.any(KF.derangement(rng, n) == np.arange(n)) for n in (2, 4) for _ in range(1000))
    idx = np.arange(C1["shuf_lines"])
    net = net64()
    x, K, V, ws, tg = b64(w, idx)
    perm = torch.from_numpy(KF.derangement(np.random.default_rng(L.SET["kf"]["shuf_seed"]), len(idx)))
    gen = lambda: torch.Generator().manual_seed(L.SET["kf"]["sign_seed"])  # noqa: E731
    kf, kf_tb1 = KF.run(net, x, K, V, ws, tg, "KF", gen=gen())
    sh, sh_tb1 = KF.run(net, x, K, V, ws, tg, "KF", perm=perm, gen=gen())
    idp, _ = KF.run(net, x, K, V, ws, tg, "KF", perm=torch.arange(len(idx)), gen=gen())
    count(3 * len(idx))
    tb1_eq = all(torch.equal(kf_tb1[k], sh_tb1[k]) for k in KF.NAMES)
    carried_differs = any(not torch.equal(kf[k], sh[k]) for k in RQ)
    ident = all(torch.equal(kf[k], idp[k]) for k in KF.NAMES)
    return {"derangements_2000_draws": bool(der), "perm": perm.tolist(), "tb1_part_identical": tb1_eq,
            "carried_part_differs": carried_differs, "identity_perm_reproduces_kf": ident,
            "pass": bool(der and tb1_eq and carried_differs and ident)}


def check_eprop_and_alignment(w):
    idx = np.arange(C1["eprop_lines"])
    net = net64()
    x, K, V, ws, tg = b64(w, idx)
    tb1, bptt = autograd(net, x, K, V, ws, tg, "TBPTT-1"), autograd(net, x, K, V, ws, tg, "BPTT")
    ep, _ = KF.run(net, x, K, V, ws, tg, "EPROP")
    count(len(idx))
    rq = rel(ep, tb1, ("Wq", "bq"))
    dB, dE = flat(sub(bptt, tb1), RA), flat(sub(ep, tb1), RA)
    eprop = {"query_update_rel_diff_vs_tbptt1": rq, "pass": rq <= C1["eprop_tol"],
             "non_gating": {"cos_eprop_tbptt1": cos(ep, tb1), "cos_eprop_bptt": cos(ep, bptt),
                            "cos_tbptt1_bptt": cos(tb1, bptt),
                            "recurrent_group_projection_share_of_bptt_carried": float(dE @ dB / (dB @ dB)),
                            "recurrent_group_norm_ratio_eprop_extra_to_bptt_carried": float(dE.norm() / dB.norm())}}
    ia = np.arange(C1["align_lines"])
    x, K, V, ws, tg = b64(w, ia)
    tb1, bptt = autograd(net, x, K, V, ws, tg, "TBPTT-1"), autograd(net, x, K, V, ws, tg, "BPTT")
    gen = torch.Generator().manual_seed(L.SET["kf"]["sign_seed"])
    draws = [KF.run(net, x, K, V, ws, tg, "KF", gen=gen)[0] for _ in range(C1["align_draws"])]
    count(len(ia) * C1["align_draws"])
    mean = {k: sum(d[k] for d in draws) / len(draws) for k in KF.NAMES}
    align = {"cos_single_draw_bptt": [cos(d, bptt) for d in draws], "cos_mean_of_draws_bptt": cos(mean, bptt),
             "cos_tbptt1_bptt": cos(tb1, bptt),
             "cos_carried_single_draw_vs_bptt_carried": [cos(sub(d, tb1), sub(bptt, tb1)) for d in draws],
             "cos_carried_mean_vs_bptt_carried": cos(sub(mean, tb1), sub(bptt, tb1)), "lines": len(ia)}
    return eprop, align


def check_determinism(cfg):
    tr = L.world("v168_technical", "train", C1["det_lines"])
    ca = L.world("v168_technical", "calib", 32)
    c = dict(cfg, n_train=C1["det_lines"], max_epochs=1)
    a, ia = L.train("KF", c, tr, ca, KF.learner_grad("KF"))
    b, ib = L.train("KF", c, tr, ca, KF.learner_grad("KF"))
    count(ia["line_presentations"] + ib["line_presentations"])
    same = all(torch.equal(a.state_dict()[k], b.state_dict()[k]) for k in a.state_dict())
    return {"identical_parameters": bool(same), "calib_bits": [ia["calib_bits"], ib["calib_bits"]], "pass": bool(same)}


def check_learns(cfg):
    tr = L.world("v168_technical", "train", C1["learn_lines"])
    ca = L.world("v168_technical", "calib", 32)
    before = L.L166.evaluate(L.L166.Net("LOOP"), tr, "LOOP")["all_bits"]
    net, info = L.train("KF", dict(cfg, n_train=C1["learn_lines"], max_epochs=1), tr, ca, KF.learner_grad("KF"))
    count(info["line_presentations"])
    after = L.L166.evaluate(net, tr, "LOOP")["all_bits"]
    return {"train_bits_before": before, "train_bits_after": after, "updates": -(-C1["learn_lines"] // cfg["batch"]),
            "pass": bool(after < before)}


def main():
    t0 = time.process_time()
    tb3 = json.loads((HERE / "results" / "testbed" / "TESTBED_3.json").read_text())
    sel = tb3["checks"]["T0-c_budget_grid"]["first_qualifying_ignoring_cap"]     # E1: the config C2 would need
    cfg = L.SET["configs"][sel]
    w = L.world("v168_technical", "train", 64)
    c1, c2 = check_identities(w)
    checks = {"1_tbptt1_identity": c1, "2_exact_rtrl_identity": c2, "3_unbiased_enumeration": check_unbiased(w),
              "4_shuf": check_shuf(w)}
    ep, align = check_eprop_and_alignment(w)
    checks["5_diagonal_traces_eprop"] = ep
    checks["6_determinism"] = check_determinism(cfg)
    checks["7_learns"] = check_learns(cfg)
    assert REC["n"] <= C1["max_records"], REC["n"]
    out = {"stage": STAGE, "selected_config": sel, "config": cfg, "checks": checks,
           "all_pass": all(c["pass"] for c in checks.values()), "alignment_at_init_non_gating": align,
           "learner_records": REC["n"], "cpu_s": round(time.process_time() - t0, 1)}
    d = HERE / "results" / STAGE
    d.mkdir(parents=True, exist_ok=True)
    (d / "CHECKS.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    import runtime
    runtime.write_closure()


if __name__ == "__main__":
    main()
