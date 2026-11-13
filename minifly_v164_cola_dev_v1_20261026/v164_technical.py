"""V164 C1 technical (design section 3). Run through launch.py:
python launch.py run --stage technical -- /tmp/venv_t/bin/python v164_technical.py
Checks: with mu fixed at 0 COLA's gradient equals TBPTT-1's (float64, <= 1e-10), on S-COPY and on text batches; the teaching
signal equals Eq. 23 with non-zero mu; alpha and mu equal Eqs. 25-30 on a hand-computed two-unit case; determinism.
Writes results/technical/CHECKS.json.
"""
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v164_lib as L  # noqa: E402
import v164_cola as CO  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

STAGE = os.environ.get("MINIFLY_STAGE", "unlaunched")


def check(out, name, ok, detail=None):
    out.append({"check": name, "pass": bool(ok), "detail": detail})
    print(("PASS " if ok else "FAIL ") + name, "" if detail is None else detail, flush=True)


def grads_equal(split, scorer, in_var, lines):
    net = L.make_rnn(scorer.K, 3.0, in_var).double()
    idx, mask, pos = L.V.line_batches(split, lines)
    ctr, valid = L.targets(split, scorer, pos)
    net.zero_grad()
    F.cross_entropy(net.out(net.run(idx, truncate=True))[valid], ctr[valid]).backward()
    ref = {k: v.grad.clone() for k, v in net.named_parameters()}
    net.zero_grad()
    CO.COLA(net.Wrr.shape[0], torch.float64, freeze_stats=True)(net, idx, valid, ctr)
    return max(float((ref[k] - v.grad).abs().max()) for k, v in net.named_parameters())


def main():
    t0 = time.process_time()
    out, records = [], 0
    tasks = {x[0]: x for x in L.settings()}
    _, tr, ca, te, sc, iv, _, _ = tasks["S-COPY"]
    A, cls = L.text_splits()
    torch.set_default_dtype(torch.float64)     # V163's run() creates its zero state in the default dtype
    e1 = grads_equal(tr, sc, iv, list(range(16)))
    e2 = grads_equal(A["train"], cls, L.V.SET["lane_L"]["init_var_inv"], list(range(16)))
    check(out, "mu = 0: COLA gradient equals TBPTT-1 autograd (float64, <= 1e-10), S-COPY and text", max(e1, e2) <= 1e-10,
          {"S-COPY": e1, "text": e2})
    # Eq. 23 with non-zero mu
    net = L.make_rnn(sc.K, 3.0, iv).double()
    co = CO.COLA(net.Wrr.shape[0], torch.float64)
    co.mu = torch.linspace(-1.5, 1.5, net.Wrr.shape[0], dtype=torch.float64)
    co.freeze = True
    idx, mask, pos = L.V.line_batches(tr, list(range(4)))
    ctr, valid = L.targets(tr, sc, pos)
    co(net, idx, valid, ctr, record=True)
    errs = []
    for e, ph, d, mu in co.trace:
        den = torch.clamp(1 - torch.clamp(mu * ph, -0.99, 0.99), min=1e-3)
        errs.append(float((d - e / den).abs().max()))
    check(out, "teaching signal equals Eq. 23 with the cap (non-zero mu)", max(errs) <= 1e-12, {"max_abs": max(errs)})
    torch.set_default_dtype(torch.float32)
    # hand-computed two-unit case for Eqs. 25-30
    co = CO.COLA(2, torch.float64)
    d_prev, d = torch.tensor([[1.0, 2.0]], dtype=torch.float64), torch.tensor([[0.5, -1.0]], dtype=torch.float64)
    m = torch.tensor([True])
    co.update_alpha(d, d_prev, m)
    S, Q = 0.005 * np.array([0.5, -2.0]), 0.005 * np.array([1.0, 4.0])
    alpha = np.clip(S / (Q + 1e-8), -0.99, 0.99)
    e_prev, e = torch.tensor([[0.2, -0.4]], dtype=torch.float64), torch.tensor([[0.1, 0.3]], dtype=torch.float64)
    ph_prev, ph = torch.tensor([[0.9, 0.5]], dtype=torch.float64), torch.tensor([[0.8, 0.6]], dtype=torch.float64)
    co.update_mu(e, e_prev, ph, ph_prev, m)
    Ah = alpha * np.array([0.8, 0.6]) * np.array([0.2, -0.4]) - np.array([0.9, 0.5]) * np.array([0.1, 0.3])
    Bh = alpha * np.array([0.2, -0.4]) - np.array([0.1, 0.3])
    mu = (0.02 * Ah * Bh) / (0.02 * Ah ** 2 + 1e-8)
    ok = np.allclose(co.alpha.numpy(), alpha, atol=1e-12) and np.allclose(co.mu.numpy(), mu, atol=1e-12)
    check(out, "alpha and mu equal Eqs. 25-30 on a hand-computed two-unit case", ok,
          {"alpha": co.alpha.tolist(), "alpha_hand": alpha.tolist(), "mu": co.mu.tolist(), "mu_hand": mu.tolist()})
    # determinism: one epoch of COLA on 16 S-COPY lines, twice
    sub = {"idx": tr["idx"][:16 * 120], "y": tr["y"][:16 * 120], "n": tr["n"][:16]}
    g = 3.1475108458573424
    nets = [L.train(sub, ca, sc, g, "cola", iv, 1, 1, grad_fn=CO.COLA(256))[0] for _ in range(2)]
    records += 2 * 16
    check(out, "COLA training is deterministic (one epoch, 16 lines)",
          all(torch.equal(a, b) for a, b in zip(nets[0].state_dict().values(), nets[1].state_dict().values())))
    d = HERE / "results" / STAGE
    d.mkdir(parents=True, exist_ok=True)
    (d / "CHECKS.json").write_text(json.dumps({"checks": out, "learner_records": records,
                                              "cpu_s": round(time.process_time() - t0, 1)}, indent=1))
    print(sum(o["pass"] for o in out), "/", len(out), "pass; records", records)
    import runtime
    runtime.write_closure()
    sys.exit(0 if all(o["pass"] for o in out) else 1)


if __name__ == "__main__":
    main()
