"""Coordinator's independent check of the E19 worker's forward-only rule (10 Nov 2026).
My own forward pass, written from the brief's four equations with plain tensor operations (not arch19.forward), differentiated by
autograd in float64, against the worker's rule_grads with exact weights; then a finite-difference probe on a few entries of A."""
import sys, torch as T
from torch.nn.functional import cross_entropy
import exp19_kbank as K, arch19 as A19

def my_loss(m, x, y, mask):
    B, L = x.shape
    h1 = T.zeros(B, A19.C, dtype=T.float64); h2 = T.zeros(B, A19.C, dtype=T.float64); out = []
    onehot = T.nn.functional.one_hot(x, 258).double()                 # the table as a linear map of the one-hot byte
    for t in range(L):
        h1 = h1 * m.R1 + onehot[:, t] @ m.A.t()
        h2 = h2 * m.R2 + T.tanh(h1) @ m.W2.weight.t() + m.W2.bias
        z = T.tanh(T.tanh(h1) @ m.Wm1.weight.t() + T.tanh(h2) @ m.Wm2.weight.t() + onehot[:, t] @ m.Etab.t())
        out.append(z @ m.l3.weight.t() + m.l3.bias)
    lg = T.stack(out, 1)
    return cross_entropy(lg[mask], y[mask])

worst = 0.0
for seed in (11, 22, 33):
    m = A19.new_net(seed).double()
    with T.no_grad():                                                   # move away from the small-weight regime of a fresh init
        for p in m.parameters(): p.mul_(3.0)
    x, y, mask = K.dev_batch(73129, seed % 5)
    m.zero_grad(); L = my_loss(m, x, y, mask); L.backward()
    ref = {n: p.grad.clone() for n, p in m.named_parameters()}
    K.rule_grads(m, x, y, mask, sign=False, deep=True)
    errs = {n: K.rel(p.grad, ref[n]) for n, p in m.named_parameters()}
    worst = max(worst, max(errs.values()))
    # the control arm must differ from the true gradient on A and only on A
    K.rule_grads(m, x, y, mask, sign=False, deep=False)
    nodeep = {n: K.rel(p.grad, ref[n]) for n, p in m.named_parameters()}
    # finite differences on three entries of A that the batch touches
    fd = []
    bytes_present = T.unique(x)[:3]
    for b in bytes_present:
        j = int(T.randint(0, A19.C, (1,)))
        with T.no_grad():
            m.A[j, b] += 1e-6; lp = float(my_loss(m, x, y, mask)); m.A[j, b] -= 2e-6; lm = float(my_loss(m, x, y, mask)); m.A[j, b] += 1e-6
        fd.append(abs((lp - lm) / 2e-6 - float(ref["A"][j, b])) / (abs(float(ref["A"][j, b])) + 1e-12))
    print(seed, "max rel err rule vs my autograd:", f"{max(errs.values()):.2e}", "| no-deep arm, A:", f"{nodeep['A']:.3f}", "others max:", f"{max(v for n, v in nodeep.items() if n != 'A'):.1e}", "| finite-diff rel:", [f"{v:.1e}" for v in fd])
print("WORST", f"{worst:.2e}")
