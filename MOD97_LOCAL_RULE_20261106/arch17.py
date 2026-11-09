"""E17: the window removed. Two learners that take one byte at a time, so that all working memory is recurrent state.

    cell1   the E11 cell exactly as it is (input-driven gate r, h = r h + (1 - r) c, 512 units), with the window's 80-d input
            replaced by the current byte's 8-d embedding: the minimal change that isolates the window.
    bank    a bank of diagonal units with FIXED retention rates (five groups: 0.5, 0.8, 0.95, 0.99, 0.999), an input-driven
            write gate w = sigmoid(Ww p) so that letters need not be written, c = tanh(Wc p), h = R h + w c; no learned
            recurrence at all: a trainable temporal basis.

Both keep the exact per-synapse eligibility trace of E11 for the same reason: no gate reads the state. The backward scan is
    Lambda_s = L_s + r_{s+1} Lambda_{s+1};  dL/dWc = sum_s (Lambda_s * w_s (1 - c_s^2))^T p_s;
    cell1:  dL/dWr = sum_s (Lambda_s * (h_{s-1} - c_s) r_s (1 - r_s))^T p_s;    bank:  dL/dWw = sum_s (Lambda_s * c_s w_s (1 - w_s))^T p_s.
Everything downstream of the state (readout z2 = tanh(W2 z1 + Wm h), logits) and upstream (z1, p) learns by the sign rule as before.
Used by exp16_integrated.py through E16_ARCH=cell1|bank (default window, unchanged); births by exp16's `dev` on these architectures.
"""
import json
import sys
import time
from pathlib import Path
from random import Random

import torch as T
from torch import nn
from torch.nn.functional import cross_entropy

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp11_cell as E
import exp45 as X
from exp3_state import lin

H, D, C, BOS = E.H, E.D, E.C, E.BOS
RATES = (0.5, 0.8, 0.95, 0.99, 0.999)


class Net17(nn.Module):
    def __init__(self, arch):
        super().__init__()
        assert arch in ("cell1", "bank")
        self.arch = arch
        self.book = nn.Embedding(258, 8, padding_idx=BOS)
        self.l1 = nn.Linear(8, H)            # the current byte only (the window was 8 * W)
        self.lp = nn.Linear(H, D)
        self.lc = nn.Linear(D, C)
        if arch == "cell1":
            self.lr = nn.Linear(D, C)
        else:
            self.lw = nn.Linear(D, C)
            self.register_buffer("R", T.tensor([RATES[i * len(RATES) // C] for i in range(C)]))
        self.lm = nn.Linear(C, H)
        self.l2 = nn.Linear(H, H)
        self.l3 = nn.Linear(H, 257)


def new_net(seed, arch):
    T.manual_seed(seed)
    return Net17(arch)


def forward(m, x, local, through_time, keep=False, silence=False):
    u = m.book(x)                                   # (B, L, 8)
    z1 = T.tanh(lin(u, m.l1, local))
    p = T.tanh(lin(z1, m.lp, local))
    c = T.tanh(lin(p, m.lc, local))
    if m.arch == "cell1":
        r = T.sigmoid(lin(p, m.lr, local)); w = 1 - r
    else:
        w = T.sigmoid(lin(p, m.lw, local)); r = m.R.expand_as(c)
    B, L, _ = c.shape
    h = T.zeros(B, C)
    hs, prev = [], []
    for t in range(L):
        hp = h if through_time else h.detach()
        prev.append(hp)
        h = r[:, t] * hp + w[:, t] * c[:, t]
        if keep:
            h.retain_grad()
        hs.append(h)
    Hs = T.stack(hs, 1)
    if silence:
        Hs = T.zeros_like(Hs)
    z2 = T.tanh(lin(z1, m.l2, local) + lin(Hs, m.lm, local))
    return lin(z2, m.l3, local), dict(p=p, c=c, r=r, w=w, hs=hs, prev=prev)


@T.no_grad()
def trace_grads_scan(m, it):
    p, c, r, w, hs, prev = it["p"], it["c"], it["r"], it["w"], it["hs"], it["prev"]
    Ls = T.stack([h.grad if h.grad is not None else T.zeros_like(h) for h in hs], 1)
    Lam = T.zeros_like(Ls)
    acc = T.zeros_like(Ls[:, 0])
    for t in range(Ls.shape[1] - 1, -1, -1):
        acc = Ls[:, t] + (r[:, t + 1] * acc if t + 1 < Ls.shape[1] else 0)
        Lam[:, t] = acc
    hp = T.stack(prev, 1)
    gc = w * (1 - c * c)
    m.lc.weight.grad = T.einsum("btc,btd->cd", Lam * gc, p); m.lc.bias.grad = (Lam * gc).sum((0, 1))
    if m.arch == "cell1":
        gr = (hp - c) * r * (1 - r)
        m.lr.weight.grad = T.einsum("btc,btd->cd", Lam * gr, p); m.lr.bias.grad = (Lam * gr).sum((0, 1))
    else:
        gw = c * w * (1 - w)
        m.lw.weight.grad = T.einsum("btc,btd->cd", Lam * gw, p); m.lw.bias.grad = (Lam * gw).sum((0, 1))


trace_grads = trace_grads_scan


def cell_params(arch):
    return ("lc.weight", "lc.bias") + (("lr.weight", "lr.bias") if arch == "cell1" else ("lw.weight", "lw.bias"))


def install(arch):
    """Point exp11's module-level names at this architecture, so exp11.update / respond / score and exp16's life run on it."""
    E.forward = forward
    E.trace_grads_scan = trace_grads_scan
    E.trace_grads = trace_grads_scan
    E.new_net = lambda seed: new_net(seed, arch)
    E.CELL_PARAMS = cell_params(arch)


def qualify(arch):
    """Trace against backprop through time on the state's synapses, on a mixed batch; pace per update; parameter count."""
    install(arch)
    out = {"arch": arch, "rates": (RATES if arch == "bank" else None)}
    rng = Random(1); seed = X.SEEDS[0]
    m = new_net(seed, arch)
    items = [E.render(t, a, b, rng.randint(0, 12), rng) for t in E.DEV for a, b in E.split(t, .5, seed)[0][:16]]
    x, y, mask = X.batch_tensor(items)
    logits, _ = forward(m, x, False, through_time=True)
    loss = cross_entropy(logits[mask], y[mask]); m.zero_grad(); loss.backward()
    ref = {n: m.get_parameter(n).grad.clone() for n in cell_params(arch)}
    logits2, it = forward(m, x, False, through_time=False, keep=True)
    loss2 = cross_entropy(logits2[mask], y[mask]); m.zero_grad(); loss2.backward()
    within = {n: m.get_parameter(n).grad.clone() for n in cell_params(arch)}
    trace_grads_scan(m, it)
    out["trace_vs_bptt_max_rel_err"] = {n: float((m.get_parameter(n).grad - ref[n]).abs().max() / ref[n].abs().max()) for n in cell_params(arch)}
    out["within_step_vs_bptt_max_rel_err"] = {n: float((within[n] - ref[n]).abs().max() / ref[n].abs().max()) for n in cell_params(arch)}
    out["loss_same_forward"] = [float(loss.detach()), float(loss2.detach())]
    out["params"] = sum(p.numel() for p in m.parameters())
    opt = T.optim.AdamW(m.parameters(), lr=.001)
    bt = E.batches([(t, E.split(t, .5, seed)[0]) for t in E.DEV], 5, 12)
    sm = {}
    for arm in ("D", "T"):
        t0 = time.process_time()
        for _ in range(10):
            E.update(m, opt, next(bt), arm)
        sm[arm] = (time.process_time() - t0) / 10 * 1000
    out["ms_per_update"] = sm
    (E.OUT / f"QUALIFY_E17_{arch}.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    qualify(sys.argv[1])
