"""E49: a limited interface. E42's F10 (256 cells, 10 fixed local connections) with two more fixed masks, as runnable code; coordinator's
specification file, 16 Nov 2026 (the job the audit of E47 names as next). Each cell reads K_IN of the 512 values of the byte's code, the
same ones for its three gates; each of the 512 readout units reads K_OUT of the 256 cells. K_IN = 512 and K_OUT = 256 is E42's F10.
    python3 -I e49_interface_spec.py        self-checks, no training"""
import sys
from pathlib import Path
from random import Random

HERE = Path(__file__).resolve()
for _cand in (HERE.parent, HERE.parents[2] / "MOD97_LOCAL_RULE_20261106"):
    if (_cand / "exp26_candidate.py").exists():
        sys.path.insert(0, str(_cand)); break
sys.path.insert(0, str(HERE.parent))
import exp26_candidate as C26        # noqa: E402
import arch26                        # noqa: E402
import torch as T                    # noqa: E402
from torch.nn.functional import linear  # noqa: E402
from exp3_state import lin           # noqa: E402
import e42_cells_spec as E42         # noqa: E402  (beside this file in $K; the runner's copy is e42_cells.py: adjust the import there)

U, D = 256, 512


def draws(seed, k_in, k_out):
    """in_src[i]: the K_IN input values cell i reads; out_src[j]: the K_OUT cells readout unit j reads. random.Random only."""
    ri = Random(f"e49|input|{seed}|{k_in}"); ro = Random(f"e49|readout|{seed}|{k_out}")
    in_src = [list(range(D)) if k_in == D else sorted(ri.sample(range(D), k_in)) for _ in range(U)]
    out_src = [list(range(U)) if k_out == U else sorted(ro.sample(range(U), k_out)) for _ in range(D)]
    return in_src, out_src


class Interface:
    """The two masks, and the start: the kept weights of the input map and of lm are scaled by sqrt(full / kept), as E42 scaled the kept
    recurrent weights; the others are set to zero. Biases are untouched."""

    def __init__(self, m, seed, k_in, k_out):
        self.k_in, self.k_out = k_in, k_out
        in_src, out_src = draws(seed, k_in, k_out)
        self.mask_in = T.zeros(U, D, dtype=T.bool); self.mask_out = T.zeros(D, U, dtype=T.bool)
        for i, s in enumerate(in_src): self.mask_in[i, s] = True
        for j, s in enumerate(out_src): self.mask_out[j, s] = True
        self.full = k_in == D and k_out == U
        with T.no_grad():
            if k_in < D:
                m.rnn.weight_ih_l0.mul_(self.mask_in.repeat(3, 1).float() * (D / k_in) ** 0.5)
            if k_out < U:
                m.lm.weight.mul_(self.mask_out.float() * (U / k_out) ** 0.5)

    def counts(self, m, n_rec):
        fixed = 8 * D + m.l2.weight.numel() + m.l3.weight.numel()             # the byte's code, l2, l3: products, no biases
        iface = 3 * U * self.k_in + D * self.k_out
        return dict(fixed=fixed, interface=iface, cells=n_rec, multiply_adds=fixed + iface + n_rec,
                    weights=sum(q.numel() for q in m.parameters()) - m.rnn.weight_ih_l0.numel() - m.lm.weight.numel() - m.rnn.weight_hh_l0.numel() + iface + n_rec)


def masked(x, W, b, maskf, local):
    """A masked linear map under either rule: exact, or E42's masked sign rule with the mean magnitude inside the mask as its scale."""
    if not local:
        return linear(x, W * maskf, b)
    return E42.MaskedSign.apply(x, W, b, maskf, (W.detach().abs() * maskf).sum() / maskf.sum())


def forward(m, layer, iface, x, local, silence=False):
    """E42's forward for a layer of period 1 (F10), with the input map and lm masked. With both masks full it calls exactly what
    E42.forward calls."""
    assert layer.K == 1, "E49 runs F10: every cell at period 1"
    u = m.book(x)
    z1 = T.tanh(lin(u, m.l1, local))
    B, L = x.shape
    Wi, bi = m.rnn.weight_ih_l0, m.rnn.bias_ih_l0
    gi = lin(z1, arch26.Mat(Wi, bi), local) if iface.k_in == D else masked(z1, Wi, bi, iface.mask_in.repeat(3, 1).float(), local)
    Wh, bh = m.rnn.weight_hh_l0, m.rnn.bias_hh_l0
    scale = layer.sign_scale(m) if (local and not layer.full) else None
    h = T.zeros(B, U); hs = []
    for t in range(L):
        gh = E42.rec(layer, m, h, Wh, bh, local, scale)
        i_r, i_z, i_n = gi[:, t].chunk(3, 1); h_r, h_z, h_n = gh.chunk(3, 1)
        r = T.sigmoid(i_r + h_r); z = T.sigmoid(i_z + h_z); n = T.tanh(i_n + r * h_n)
        h = (1 - z) * n + z * h
        hs.append(h)
    Hs = T.stack(hs, 1)
    if silence:
        Hs = T.zeros_like(Hs)
    out = lin(Hs, m.lm, local) if iface.k_out == U else masked(Hs, m.lm.weight, m.lm.bias, iface.mask_out.float(), local)
    z2 = T.tanh(lin(z1, m.l2, local) + out)
    return lin(z2, m.l3, local), hs


if __name__ == "__main__":
    import hashlib, json
    T.set_num_threads(1); arch26.install(); E, X = C26.E, C26.X
    seed = 73129
    tp = [(t, E.split(t, .5, seed)[0]) for t in E.DEV]
    items = next(E.batches(tp, seed, E.KMAX["M"])); x, y, mask = X.batch_tensor(items)

    def run(m, fwd):
        for q in m.parameters(): q.grad = None
        lg = fwd(); loss = E.cross_entropy(lg[mask], y[mask]); loss.backward(); return lg.detach(), float(loss)

    # 1. with both masks full the network is E42's F10: logits and gradients under both rules
    for local in (False, True):
        m1 = arch26.new_net(seed); l1 = E42.Layer(m1, seed, N=10); lgA, _ = run(m1, lambda: E42.forward(m1, l1, x, local)[0])
        m2 = arch26.new_net(seed); l2 = E42.Layer(m2, seed, N=10); f2 = Interface(m2, seed, D, U); lgB, _ = run(m2, lambda: forward(m2, l2, f2, x, local)[0])
        print(f"1. full masks, local={local}: logits equal E42's F10 {bool(T.equal(lgA, lgB))}; gradients equal {all(T.equal(a.grad, b.grad) for a, b in zip(m1.parameters(), m2.parameters()))}")
    # 2. the settings: counts, coverage, the start, gradients zero outside the masks
    dg = lambda v: hashlib.sha256(json.dumps(v).encode()).hexdigest()[:16]
    for k_in, k_out in ((64, 32), (16, 8), (16, U), (D, 8)):
        m = arch26.new_net(seed); lay = E42.Layer(m, seed, N=10); f = Interface(m, seed, k_in, k_out)
        c = f.counts(m, int(lay.mask3().sum()))
        readers_in = f.mask_in.sum(0); readers_out = f.mask_out.sum(0)
        res = {}
        for local in (False, True):
            lg, loss = run(m, lambda: forward(m, lay, f, x, local)[0])
            gi_out = m.rnn.weight_ih_l0.grad[~f.mask_in.repeat(3, 1)]; go_out = m.lm.weight.grad[~f.mask_out]
            res[local] = (loss, bool((gi_out == 0).all()) and bool((go_out == 0).all()), bool(T.isfinite(lg).all()))
        gB = None
        print(f"2. K_IN {k_in}, K_OUT {k_out}: multiply-adds a position {c['multiply_adds']:,} (interface {c['interface']:,}, cells {c['cells']:,}, the rest {c['fixed']:,}); weights {c['weights']:,};"
              f" every input value is read by {int(readers_in.min())} to {int(readers_in.max())} cells, every cell by {int(readers_out.min())} to {int(readers_out.max())} readout units;"
              f" first-batch loss B {res[False][0]:.3f} S {res[True][0]:.3f}; gradients zero outside the masks {res[False][1] and res[True][1]}; finite {res[False][2] and res[True][2]};"
              f" digests of the draws {dg(draws(seed, k_in, k_out)[0])} {dg(draws(seed, k_in, k_out)[1])}")
