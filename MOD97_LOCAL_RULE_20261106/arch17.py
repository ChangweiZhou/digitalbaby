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


ARCHS = ("cell1", "bank", "cell1d2", "bank2")
# E17 Phase 1d (9 Nov): the projection feeding each bank's write (lp, lp2) otherwise learns from within-step credit only (hp detached), while
# the write synapses (lc, lw) get exact credit through time from the scan. TRACEUP extends the scan one layer upstream: with Lambda_s in hand,
# dL/dp_s = (Lambda_s * gc_s) Wc + (Lambda_s * gg_s) Wg, delivered through sign(W) (FB_SIGN) or W, then dL/dWp = sum_s (dL/dp_s * (1 - p_s^2))^T x_s.
# No new approximation: given Lambda, this is backprop through time's own credit for lp. Set by exp16 (E16_TRACEUP, E16_FEEDBACK).
TRACEUP = False
FB_SIGN = True
# E18 (10 Nov): LOCAL_HEADS gives the first bank its own next-byte head (l2a, lma, l3a) and cuts every gradient path between the two banks:
# the second bank and the main readout read the first bank's state and first layer detached. Each bank then learns from its own
# prediction error only; no error is relayed between banks. HEAD1_OUT makes forward return the first bank's head (scoring only).
LOCAL_HEADS = False
HEAD1_OUT = False
# 10 Nov audit: with FB_SIGN the projection trace (_proj) relayed raw sign(W) for the content and gate matrices, while every other backward
# relay uses sign(W) * mean|W| (SignLinear). The two matrices have different mean magnitudes, so the raw form weights the two terms
# differently and changes the direction of the projection's gradient. PROJ_SCALED uses the same map as everywhere else. Default False
# reproduces the Phase 3 and Phase 4 receipts; exp16 sets it from E16_PROJFB=scaled (tag _tups).
PROJ_SCALED = False
# 10 Nov audit: LOCAL_DETACH False keeps the first bank's own head but leaves every gradient path between the banks open (E16_HEADS=aux):
# the one-factor control for "no relay between banks" (same heads, same objective, only the detach toggled).
LOCAL_DETACH = True
# E18c (11 Nov 2026, after the internal audit): LOCAL_DETACH cuts three reads (Hs -> lp2, Hs -> lm, z1 -> l2). Only the first two cross
# between banks. Z1_DETACH None follows LOCAL_DETACH (every earlier receipt); E16_HEADS=relay sets LOCAL_DETACH False and Z1_DETACH True:
# the two inter-bank paths open, the first layer's direct path into the main head still cut.
Z1_DETACH = None
# E23 (11 Nov 2026, timescale of the upper bank's target): FAR_HEAD adds a third head that reads the second bank's state alone
# (lmf, l3f; created after every other layer, so all other initial weights are unchanged) and is trained on the byte FAR_DELTA steps
# ahead (set and applied by exp16_integrated.update16). With LOCAL_DETACH its error reaches only the second bank. Default off.
FAR_HEAD = False
# Phase 0b variants (8 Nov 23:45 EST, after P17.0 was refuted for cell1 and bank):
#   cell1d2  cell1 with a deeper readout: two hidden layers between (z1, h) and the logits, so two digits held in the state can interact before the output
#   bank2    two stacked fixed-rate banks: the second bank's write is computed from the FIRST bank's state (p2 = tanh(Wp2 h1)), so writes can depend on
#            accumulated context; each bank's own trace stays exact given its input (credit from the readout through the second bank's memory to the
#            first bank's synapses is truncated, the e-prop approximation)


class Net17(nn.Module):
    def __init__(self, arch):
        super().__init__()
        assert arch in ARCHS
        self.arch = arch
        self.book = nn.Embedding(258, 8, padding_idx=BOS)
        self.l1 = nn.Linear(8, H)            # the current byte only (the window was 8 * W)
        self.lp = nn.Linear(H, D)
        self.lc = nn.Linear(D, C)
        if arch in ("cell1", "cell1d2"):
            self.lr = nn.Linear(D, C)
        else:
            self.lw = nn.Linear(D, C)
            self.register_buffer("R", T.tensor([RATES[i * len(RATES) // C] for i in range(C)]))
        if arch == "bank2":
            self.lp2 = nn.Linear(C, D); self.lc2 = nn.Linear(D, C); self.lw2 = nn.Linear(D, C); self.lm2 = nn.Linear(C, H)
        self.lm = nn.Linear(C, H)
        self.l2 = nn.Linear(H, H)
        if arch == "cell1d2":
            self.l2b = nn.Linear(H, H)
        self.l3 = nn.Linear(H, 257)
        if arch == "bank2" and LOCAL_HEADS:     # created last, so every other layer's initial weights are those of the plain bank2 net of the same seed
            self.l2a = nn.Linear(H, H); self.lma = nn.Linear(C, H); self.l3a = nn.Linear(H, 257)
        if arch == "bank2" and LOCAL_HEADS and FAR_HEAD:     # created after l2a/lma/l3a: the E18 local-heads net's initial weights are unchanged
            self.lmf = nn.Linear(C, H); self.l3f = nn.Linear(H, 257)


def new_net(seed, arch):
    T.manual_seed(seed)
    return Net17(arch)


def forward(m, x, local, through_time, keep=False, silence=False):
    u = m.book(x)                                   # (B, L, 8)
    z1 = T.tanh(lin(u, m.l1, local))
    p = T.tanh(lin(z1, m.lp, local))
    c = T.tanh(lin(p, m.lc, local))
    if m.arch in ("cell1", "cell1d2"):
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
        if keep and h.requires_grad:     # a bank whose whole input pathway is frozen (book, l1, lp, lc, lw) has no gradient path; its scan is then unused
            h.retain_grad()
        hs.append(h)
    Hs = T.stack(hs, 1)
    extra = {}
    lh = hasattr(m, "l3a")
    if m.arch == "bank2":   # second bank written from the first bank's state
        p2 = T.tanh(lin(Hs.detach() if (lh and LOCAL_DETACH) else Hs, m.lp2, local))
        c2 = T.tanh(lin(p2, m.lc2, local)); w2 = T.sigmoid(lin(p2, m.lw2, local)); r2 = m.R.expand_as(c2)
        h2 = T.zeros(B, C); hs2, prev2 = [], []
        for t in range(L):
            hp2 = h2 if through_time else h2.detach()
            prev2.append(hp2)
            h2 = r2[:, t] * hp2 + w2[:, t] * c2[:, t]
            if keep and h2.requires_grad:
                h2.retain_grad()
            hs2.append(h2)
        Hs2 = T.stack(hs2, 1)
        extra = dict(p2=p2, c2=c2, r2=r2, w2=w2, hs2=hs2, prev2=prev2)
    if silence:
        Hs = T.zeros_like(Hs)
        if m.arch == "bank2":
            Hs2 = T.zeros_like(Hs2)
    if lh:
        extra["logits1"] = lin(T.tanh(lin(z1, m.l2a, local) + lin(Hs, m.lma, local)), m.l3a, local)
        zd = LOCAL_DETACH if Z1_DETACH is None else Z1_DETACH
        pre = lin(z1.detach() if zd else z1, m.l2, local) + lin(Hs.detach() if LOCAL_DETACH else Hs, m.lm, local)
    else:
        pre = lin(z1, m.l2, local) + lin(Hs, m.lm, local)
    if m.arch == "bank2":
        pre = pre + lin(Hs2, m.lm2, local)
    z2 = T.tanh(pre)
    if m.arch == "cell1d2":
        z2 = T.tanh(lin(z2, m.l2b, local))
    if hasattr(m, "l3f"):        # E23: the far head reads the second bank's state only
        extra["logits_far"] = lin(T.tanh(lin(Hs2, m.lmf, local)), m.l3f, local)
    out = extra["logits1"] if (lh and HEAD1_OUT) else lin(z2, m.l3, local)
    return out, dict(p=p, c=c, r=r, w=w, hs=hs, prev=prev, z1=z1, **extra)


@T.no_grad()
def _scan(p, c, r, w, hs, prev, lc, lgate, gate_is_r):
    Ls = T.stack([h.grad if h.grad is not None else T.zeros_like(h) for h in hs], 1)
    Lam = T.zeros_like(Ls)
    acc = T.zeros_like(Ls[:, 0])
    for t in range(Ls.shape[1] - 1, -1, -1):
        acc = Ls[:, t] + (r[:, t + 1] * acc if t + 1 < Ls.shape[1] else 0)
        Lam[:, t] = acc
    hp = T.stack(prev, 1)
    gc = w * (1 - c * c)
    lc.weight.grad = T.einsum("btc,btd->cd", Lam * gc, p); lc.bias.grad = (Lam * gc).sum((0, 1))
    gg = (hp - c) * r * (1 - r) if gate_is_r else c * w * (1 - w)
    lgate.weight.grad = T.einsum("btc,btd->cd", Lam * gg, p); lgate.bias.grad = (Lam * gg).sum((0, 1))
    return Lam, gc, gg


@T.no_grad()
def _proj(Lam, gc, gg, lc, lgate, lp, x, p):
    """TRACEUP: credit through time for the projection p = tanh(Wp x) that feeds the write, replacing its within-step gradient."""
    sc = (lambda W: W.sign() * W.abs().mean()) if PROJ_SCALED else (lambda W: W.sign())   # 10 Nov audit: raw signs mix the two terms differently from SignLinear
    Wc = sc(lc.weight) if FB_SIGN else lc.weight
    Wg = sc(lgate.weight) if FB_SIGN else lgate.weight
    dpre = ((Lam * gc) @ Wc + (Lam * gg) @ Wg) * (1 - p * p)          # (B, T, D)
    lp.weight.grad = T.einsum("btd,bth->dh", dpre, x); lp.bias.grad = dpre.sum((0, 1))


@T.no_grad()
def trace_grads_scan(m, it):
    cellish = m.arch in ("cell1", "cell1d2")
    Lam, gc, gg = _scan(it["p"], it["c"], it["r"], it["w"], it["hs"], it["prev"], m.lc, m.lr if cellish else m.lw, cellish)
    if TRACEUP and m.lp.weight.requires_grad:
        _proj(Lam, gc, gg, m.lc, m.lr if cellish else m.lw, m.lp, it["z1"], it["p"])
    if m.arch == "bank2":
        Lam2, gc2, gg2 = _scan(it["p2"], it["c2"], it["r2"], it["w2"], it["hs2"], it["prev2"], m.lc2, m.lw2, False)
        if TRACEUP:
            _proj(Lam2, gc2, gg2, m.lc2, m.lw2, m.lp2, T.stack(it["hs"], 1), it["p2"])


trace_grads = trace_grads_scan


def cell_params(arch):
    base = ("lc.weight", "lc.bias") + (("lr.weight", "lr.bias") if arch in ("cell1", "cell1d2") else ("lw.weight", "lw.bias"))
    return base + (("lc2.weight", "lc2.bias", "lw2.weight", "lw2.bias") if arch == "bank2" else ())


def install(arch):
    """Point exp11's module-level names at this architecture, so exp11.update / respond / score and exp16's life run on it."""
    E.forward = forward
    E.trace_grads_scan = trace_grads_scan
    E.trace_grads = trace_grads_scan
    E.new_net = lambda seed: new_net(seed, arch)
    E.CELL_PARAMS = cell_params(arch)


def qualify(arch, up=False):
    """Trace against backprop through time on the state's synapses, on a mixed batch; pace per update; parameter count.
    up=True (Phase 1d): TRACEUP with exact feedback, the projections lp/lp2 included in the comparison; with the first bank frozen
    (lp.requires_grad False) the second bank's path is exact given its input, so lp2's trace must match BPTT to rounding."""
    global TRACEUP, FB_SIGN
    install(arch)
    TRACEUP, FB_SIGN = up, not up
    out = {"arch": arch, "rates": (RATES if arch == "bank" else None), "traceup": up}
    rng = Random(1); seed = X.SEEDS[0]
    m = new_net(seed, arch)
    names = cell_params(arch) + ((("lp.weight", "lp.bias") + (("lp2.weight", "lp2.bias") if arch == "bank2" else ())) if up else ())
    if up and arch == "bank2":   # the Phase 1b/1c/1d setting: first bank frozen
        for n_ in ("l1", "lp", "lc", "lw"):
            getattr(m, n_).weight.requires_grad_(False); getattr(m, n_).bias.requires_grad_(False)
        names = tuple(n for n in names if not n.startswith(("lp.", "lc.", "lw.")))
    items = [E.render(t, a, b, rng.randint(0, 12), rng) for t in E.DEV for a, b in E.split(t, .5, seed)[0][:16]]
    x, y, mask = X.batch_tensor(items)
    logits, _ = forward(m, x, False, through_time=True)
    loss = cross_entropy(logits[mask], y[mask]); m.zero_grad(); loss.backward()
    ref = {n: m.get_parameter(n).grad.clone() for n in names}
    logits2, it = forward(m, x, False, through_time=False, keep=True)
    loss2 = cross_entropy(logits2[mask], y[mask]); m.zero_grad(); loss2.backward()
    within = {n: m.get_parameter(n).grad.clone() for n in names}
    trace_grads_scan(m, it)
    out["trace_vs_bptt_max_rel_err"] = {n: float((m.get_parameter(n).grad - ref[n]).abs().max() / ref[n].abs().max()) for n in names}
    out["within_step_vs_bptt_max_rel_err"] = {n: float((within[n] - ref[n]).abs().max() / ref[n].abs().max()) for n in names}
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
    (E.OUT / f"QUALIFY_E17_{arch}{'_up' if up else ''}.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    qualify(sys.argv[1], up=(len(sys.argv) > 2 and sys.argv[2] == "up"))
