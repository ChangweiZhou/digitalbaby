"""Eligibility-trace (e-prop style) version of the sign rule for the GRU of Experiment 3.

Local in space: every backward weight is the sign of the forward weight (as in all sign-rule arms).
Local in time: no backpropagation through the state. Instead each recurrent synapse keeps an
eligibility trace eps_t = z_t * eps_{t-1} + (direct partial of h_t w.r.t. the synapse at step t),
where z_t is the GRU update gate of the postsynaptic unit (its own one-step self-dependence).
The learning signal L_t for a hidden unit is the sign-feedback error projected to h_t within the
step (exactly the signal the no-credit-through-time arm uses). The weight update is sum_t L_t * eps_t.
With the trace decay forced to zero this reduces exactly to the within-step rule (arm W), which is
the qualification check. The readout (l2, l3) and the byte embedding use the within-step rule.

Arms (same protocol as exp3_state.life, GRU only):
  E   warm start from the developed birth, trace rule
  ER  random birth, trace rule
"""
import os
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[k] = "1"
import json
import sys
import time
from itertools import islice
from pathlib import Path

import torch as T
from torch.nn.functional import cross_entropy

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp3_state as X

T.set_num_threads(1)
OUT = X.OUT
SEEDS = X.SEEDS


def forward_with_internals(m, x, decay_traces=True):
    """GRU forward in local mode with the previous state detached; returns logits, per-step internals, hs (retain_grad)."""
    u = m.window(x)
    B, L, _ = u.shape
    h = T.zeros(B, X.H)
    hs, internals = [], []
    for t in range(L):
        ut = u[:, t]
        hp = h.detach()
        pz = X.lin(ut, m.lz, True) + X.lin(hp, m.mz, True)
        pr = X.lin(ut, m.lr, True) + X.lin(hp, m.mr, True)
        mnh = X.lin(hp, m.mn, True)
        z, r = T.sigmoid(pz), T.sigmoid(pr)
        n = T.tanh(X.lin(ut, m.ln, True) + r * mnh)
        h = (1 - z) * n + z * hp
        hs.append(h)
        internals.append(dict(u=ut.detach(), hp=hp, z=z.detach(), r=r.detach(), n=n.detach(), mnh=mnh.detach()))
    hs = T.stack(hs, 1)
    hs.retain_grad()
    z2 = T.tanh(X.lin(hs, m.l2, True))
    return X.lin(z2, m.l3, True), internals, hs


@T.no_grad()
def trace_grads(m, internals, Lh, decay=True):
    """Accumulate sum_t L_t * eps_t for the six gate matrices and their biases."""
    B = Lh.shape[0]
    Hd, In = X.H, 8 * X.W
    eps = {k: T.zeros(B, Hd, In if k[0] == "l" else Hd) for k in ("lz", "lr", "ln", "mz", "mr", "mn")}
    epsb = {k: T.zeros(B, Hd) for k in eps}
    G = {k: T.zeros(Hd, In if k[0] == "l" else Hd) for k in eps}
    Gb = {k: T.zeros(Hd) for k in eps}
    for t, s in enumerate(internals):
        u, hp, z, r, n, mnh = s["u"], s["hp"], s["z"], s["r"], s["n"], s["mnh"]
        # per-unit direct partials of h_t w.r.t. each gate pre-activation
        dn = (1 - z) * (1 - n * n)                 # dh/dpre_n
        dz = (hp - n) * z * (1 - z)                # dh/dpre_z
        dr = dn * mnh * r * (1 - r)                # dh/dpre_r
        dec = z.unsqueeze(2) if decay else T.zeros_like(z).unsqueeze(2)
        for key, dpre in (("z", dz), ("r", dr), ("n", dn)):
            for side, inp in (("l", u), ("m", hp)):
                k = side + key
                d = dpre * r if k == "mn" else dpre     # mn's pre-activation (weights and bias) is gated by r
                eps[k] = dec * eps[k] + d.unsqueeze(2) * inp.unsqueeze(1)
                epsb[k] = dec.squeeze(2) * epsb[k] + d
        L = Lh[:, t]                                # B x H learning signal at step t
        for k in eps:
            G[k] += T.einsum("bj,bji->ji", L, eps[k])
            Gb[k] += (L * epsb[k]).sum(0)
    return G, Gb


def eprop_update(m, opt, rows, decay=True):
    data = X.batch_tensor(rows)
    x, y = data[:, :-1], data[:, 1:]
    logits, internals, hs = forward_with_internals(m, x)
    mask = T.zeros_like(y, dtype=T.bool)
    for i, r in enumerate(rows):
        mask[i, len(r) - 3:len(r)] = True
    loss = cross_entropy(logits[mask], y[mask])
    opt.zero_grad()
    loss.backward()                                 # within-step grads for everything + hs.grad = learning signal
    G, Gb = trace_grads(m, internals, hs.grad, decay)
    for k in G:
        layer = getattr(m, k)
        layer.weight.grad = G[k]
        layer.bias.grad = Gb[k]
    opt.step()
    return float(loss.detach())


def life(arm, seed):
    OUT.mkdir(exist_ok=True)
    path = OUT / f"life_gru_{arm}_{seed}.json"
    mod = 73
    taught, monitor, sealed = X.split(mod, .25, seed)
    T.manual_seed(seed)
    m = X.StateNet("gru")
    if arm == "E":
        m.load_state_dict(T.load(OUT / f"dev_gru_{seed}.pt"))
    opt = T.optim.AdamW(m.parameters(), lr=.001, weight_decay=1., betas=(.9, .98))
    items_s = X.panel(mod, sealed, seed, per_k=300)
    items_t = X.panel(mod, taught, seed, ks=(0, 4, 8, 12), per_k=300)
    before = X.score(m, items_s)
    st = X.stream(mod, taught, seed + mod)
    t0 = time.process_time()
    curve = []
    for stop in range(X.BLOCK, X.LIFE_UPDATES + 1, X.BLOCK):
        losses = [eprop_update(m, opt, rows) for rows in islice(st, X.BLOCK)]
        if stop % 2048 == 0:
            row = dict(step=stop, loss=sum(losses) / len(losses), taught=X.score(m, items_t), sealed=X.score(m, items_s), cpu=time.process_time() - t0)
            curve.append(row)
            print("gru", arm, seed, stop, {k: v[0] for k, v in row["sealed"].items()}, f"cpu {row['cpu']:.0f}s", flush=True)
            path.write_text(json.dumps(dict(arch="gru", arm=arm, seed=seed, frac=.25, sealed_before=before,
                                            shortcuts=X.shortcuts(mod, taught, items_s), curve=curve), indent=1))


def qualify():
    out = {}
    T.manual_seed(5)
    m = X.StateNet("gru")
    m.load_state_dict(T.load(OUT / f"dev_gru_{SEEDS[0]}.pt"))
    taught, _, _ = X.split(73, .25, SEEDS[0])
    rows = next(X.stream(73, taught, 11))
    data = X.batch_tensor(rows)
    x, y = data[:, :-1], data[:, 1:]
    mask = T.zeros_like(y, dtype=T.bool)
    for i, r in enumerate(rows):
        mask[i, len(r) - 3:len(r)] = True

    def grads(local, detach):
        m.zero_grad()
        logits = m(x, local=local, detach=detach)
        cross_entropy(logits[mask], y[mask]).backward()
        return {n: p.grad.detach().clone() for n, p in m.named_parameters() if p.grad is not None}

    within = grads(True, True)          # arm W
    through = grads(True, False)        # arm F: sign rule with credit through time
    bptt = grads(False, False)          # true gradient
    # trace grads with decay 0 must equal the within-step grads
    m.zero_grad()
    logits, internals, hs = forward_with_internals(m, x)
    cross_entropy(logits[mask], y[mask]).backward()
    G0, Gb0 = trace_grads(m, internals, hs.grad, decay=False)
    G1, Gb1 = trace_grads(m, internals, hs.grad, decay=True)
    cos = lambda a, b: float(T.nn.functional.cosine_similarity(a.flatten(), b.flatten(), 0))
    for k in G0:
        assert T.allclose(G0[k], within[f"{k}.weight"], atol=1e-5, rtol=1e-3), (k, cos(G0[k], within[f"{k}.weight"]))
        assert T.allclose(Gb0[k], within[f"{k}.bias"], atol=1e-5, rtol=1e-3), k
    out["decay0_equals_within_step"] = True
    # with traces, alignment to the credit-through-time sign gradient and to the true gradient should rise
    out["cosine_to_sign_through_time"] = {k: dict(within=round(cos(within[f"{k}.weight"], through[f"{k}.weight"]), 3),
                                                   trace=round(cos(G1[k], through[f"{k}.weight"]), 3)) for k in G1}
    out["cosine_to_true_bptt"] = {k: dict(within=round(cos(within[f"{k}.weight"], bptt[f"{k}.weight"]), 3),
                                          trace=round(cos(G1[k], bptt[f"{k}.weight"]), 3)) for k in G1}
    # smoke: a few updates, finite
    T.manual_seed(6)
    mm = X.StateNet("gru")
    opt = T.optim.AdamW(mm.parameters(), lr=.001, weight_decay=1.)
    st = X.stream(73, taught, 12)
    for r in islice(st, 3):
        l = eprop_update(mm, opt, r)
    assert l == l
    out["smoke"] = l
    (OUT / "QUALIFY_EPROP.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "qualify":
        qualify()
    else:
        life(sys.argv[1], int(sys.argv[2]))
