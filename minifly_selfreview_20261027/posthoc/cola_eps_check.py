"""Post-hoc check of V164 (non-gating; no delivery; no new arm or configuration).
Question (Codex synthesis, memo 02 section 9): are COLA's EMA statistics of the same order as the fixed eps = 1e-8 in
alpha = S / (Q + eps) and mu = S_AB / (S_A2 + eps)? If so, V164's near-zero alpha and small mu are partly numerical.
Part 1 runs V164's own COLA code, unchanged, for 5 batches of 16 lines from V164's initialisation and schedule.
Part 2 re-executes V164's committed C2 COLA training (same code, settings and seeds) and reads the final statistics;
it must reproduce the alpha and mu quantiles in V164's results/dev/RESULTS.json before its other numbers are read.
"Without eps" values are computed from the same EMAs (the EMAs themselves were accumulated with eps in alpha).
Run: /tmp/venv_t/bin/python cola_eps_check.py
"""
import json
import sys
from pathlib import Path

import torch

V164 = Path(__file__).resolve().parents[2] / "minifly_v164_cola_dev_v1_20261026"
sys.path.insert(0, str(V164))
import v164_lib as L  # noqa: E402
import v164_cola as CO  # noqa: E402

C, S = L.SET["cola"], L.SET["rnn"]
GAIN = {n: r["gain"]["g"] for n, r in json.loads((V164 / "results/testbed/TESTBED.json").read_text())["results"].items()}
DEV = json.loads((V164 / "results/dev/RESULTS.json").read_text())["results"]
med = lambda v: float(v.median())  # noqa: E731


def report(name, co, tr, sc, net):
    """Statistics with eps and without eps; the cap share uses per-step phi' from one batch run with frozen statistics."""
    a_no = (co.S / co.Q.clamp(min=1e-30)).clamp(-C["alpha_clip"], C["alpha_clip"])
    mu_no = co.SAB / co.SA2.clamp(min=1e-30)
    co.freeze = True
    idx, mask, pos = L.V.line_batches(tr, list(range(16)))
    ctr, valid = L.targets(tr, sc, pos)
    co(net, idx, valid, ctr, record=True)
    inline = (idx[:, :, 0] != L.V.PAD)
    ph = torch.stack([t[1] for t in co.trace], 1)[inline]                      # (in-line steps, units)
    cap_eps = float((co.mu * ph).abs().ge(C["mu_cap"]).float().mean())
    cap_no = float((mu_no * ph).abs().ge(C["mu_cap"]).float().mean())
    print("  median Q %.1e, median S_A2 %.1e (eps %.0e)" % (med(co.Q), med(co.SA2), C["eps"]))
    print("  median |alpha|: with eps %.4f, without eps %.4f" % (med(co.alpha.abs()), med(a_no.abs())))
    print("  median mu:      with eps %.3f, without eps %.3f" % (med(co.mu), med(mu_no)))
    print("  share of in-line unit-steps with |mu phi'| at the cap %.2f: with eps %.3f, without eps %.3f"
          % (C["mu_cap"], cap_eps, cap_no))


def part1(name, tr, sc, iv):
    net = L.make_rnn(sc.K, GAIN[name], iv)
    opt = torch.optim.AdamW(net.parameters(), lr=S["lr"], weight_decay=S["wd"])
    order = torch.randperm(len(tr["n"]), generator=torch.Generator().manual_seed(S["order_seed"])).tolist()
    co = CO.COLA(S["H"])
    for b in range(5):
        idx, mask, pos = L.V.line_batches(tr, order[16 * b:16 * b + 16])
        ctr, valid = L.targets(tr, sc, pos)
        opt.zero_grad()
        co(net, idx, valid, ctr, record=True)
        torch.nn.utils.clip_grad_norm_(net.parameters(), S["clip"])
        opt.step()
        e2 = float(torch.stack([t[0] ** 2 for t in co.trace]).mean())
        print("  batch %d  mean e^2 %.1e  median Q %.1e  median |S| %.1e  median S_A2 %.1e  median |S_AB| %.1e"
              % (b, e2, med(co.Q), med(co.S.abs()), med(co.SA2), med(co.SAB.abs())))
    report(name, co, tr, sc, net)


def part2(name, tr, ca, sc, iv, me, pa):
    co = CO.COLA(S["H"])
    net, info = L.train(tr, ca, sc, GAIN[name], "cola", iv, me, pa, grad_fn=co)
    got, ref = co.summary(), DEV[name]["stats"]
    same = all(abs(a - b) < 1e-6 for k in ("alpha_q05_50_95", "mu_q05_50_95") for a, b in zip(got[k], ref[k]))
    print("  reproduces V164's final alpha and mu quantiles: %s (epochs run %d)" % (same, info["epochs_run"]))
    report(name, co, tr, sc, net)


if __name__ == "__main__":
    tasks = {x[0]: x for x in L.settings()}
    for name in ("S-COPY", "TEXT"):
        _, tr, ca, _, sc, iv, me, pa = tasks[name]
        print("%s, part 1: first 5 batches from initialisation" % name)
        part1(name, tr, sc, iv)
    for name in ("S-COPY", "TEXT"):
        _, tr, ca, _, sc, iv, me, pa = tasks[name]
        print("%s, part 2: end of V164's C2 COLA training, re-executed" % name)
        part2(name, tr, ca, sc, iv, me, pa)
