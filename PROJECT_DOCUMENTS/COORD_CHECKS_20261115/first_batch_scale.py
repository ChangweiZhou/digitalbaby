"""Coordinator check for the E34 brief (15 Nov 2026). Read-only. First batch of seed 73129, untrained network: size of the weights and of
the parameter gradients against AdamW's eps, without and with division by the sum of the weights; bit-equality of the unit-weight loss with
the mean cross-entropy.  python3 -I first_batch_scale.py"""
import sys
from pathlib import Path
H = Path(__file__).resolve().parents[2] / "MOD97_LOCAL_RULE_20261106"
sys.path.insert(0, str(H))
import exp33_targets as X33
T = X33.T; arch26 = X33.arch26; E = X33.E; C = X33.C
from torch.nn.functional import cross_entropy, log_softmax
arch26.install()
items = X33.first_batches(73129, 1)[0]
x, y, amask, real, fine = X33.masks(items)
print("vocab", arch26.forward(E.new_net(73129), x, True, True)[0].shape, "real", int(real.sum()), "ans", int(amask.sum()))
for NORM, tau in [(n, t) for n in (False, True) for t in (0.5, 1.0, 2.0, float("inf"))]:
    m = E.new_net(73129); opt = C.make_opt(m)
    lg = arch26.forward(m, x, True, True)[0]
    with T.no_grad():
        lp = log_softmax(lg[real], -1); Hh = -(lp.exp() * lp).sum(-1); w = T.exp(-Hh / tau)
    ce = cross_entropy(lg[real], y[real], reduction="none")
    loss = (w * ce).sum() / (w.sum() if NORM else real.sum())
    ref = cross_entropy(lg[real], y[real])
    loss.backward()
    g = T.cat([p.grad.abs().flatten() for p in m.parameters() if p.grad is not None])
    print("normalised" if NORM else "unnormalised", tau, "H0 %.3f w0 %.2e loss %.3e ref %.6f biteq %s |g| median %.2e 10pct %.2e eps %s" % (float(Hh.mean()), float(w.mean()), float(loss), float(ref), bool(T.equal(loss, ref)), float(g.median()), float(g.quantile(.1)), opt.param_groups[0]["eps"]))
