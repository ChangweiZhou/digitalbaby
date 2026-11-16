"""E48: a new family learned only in connections gated by its own code. The added terms and the forward as runnable code; coordinator's
specification file, 16 Nov 2026 (requirements: the reviewer's memo, revision 19, section 27, where the job is called E47; that number was
already taken by the locality job). Everything of the lived network is frozen. For a gated item, the input map and the recurrent matrix
each gain one term of rank r, a product of two matrices that starts at zero; the four new code rows of the byte table may also change.
    python3 -I e48_gated_spec.py        self-checks on E26's sweep life of seed 73129 (no training)"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
for _cand in (HERE.parent, HERE.parents[2] / "MOD97_LOCAL_RULE_20261106"):
    if (_cand / "exp26_candidate.py").exists():
        sys.path.insert(0, str(_cand)); MDIR = _cand; break
import exp26_candidate as C26        # noqa: E402
import arch26                        # noqa: E402
import torch as T                    # noqa: E402
from exp3_state import lin           # noqa: E402

CODES = (101, 102, 103, 104)         # E40's family: the bytes e, f, g, h


class Term:
    """One family's two added terms. down: (r, in), drawn; up: (out, r), zero at the start. No bias."""

    def __init__(self, seed, r):
        g = T.Generator().manual_seed(seed * 1000 + r)
        self.r = r
        self.down_i = T.nn.Parameter(T.empty(r, 512).uniform_(-512 ** -0.5, 512 ** -0.5, generator=g))   # on the input map: 768 x 512
        self.up_i = T.nn.Parameter(T.zeros(768, r))
        self.down_h = T.nn.Parameter(T.empty(r, 256).uniform_(-1 / 16, 1 / 16, generator=g))            # on the recurrent matrix: 768 x 256
        self.up_h = T.nn.Parameter(T.zeros(768, r))
        self.zr, self.z768 = T.zeros(r), T.zeros(768)

    def parameters(self):
        return [self.down_i, self.up_i, self.down_h, self.up_h]

    def count(self):
        return sum(q.numel() for q in self.parameters()) + 8 * len(CODES)


def added(x, down, up, zr, z768, local):
    """up (down x): two linear maps without bias, each crossed back by the rule of the run (exact, or the sweep's signs)."""
    return lin(lin(x, arch26.Mat(down, zr), local), arch26.Mat(up, z768), local)


def forward(m, term, x, local, gate, silence=False):
    """arch26.forward; with gate=True the two terms are added. gate=False is arch26.forward itself, value for value."""
    u = m.book(x)
    z1 = T.tanh(lin(u, m.l1, local))
    B, L = x.shape
    gi = lin(z1, arch26.Mat(m.rnn.weight_ih_l0, m.rnn.bias_ih_l0), local)
    if gate:
        gi = gi + added(z1, term.down_i, term.up_i, term.zr, term.z768, local)
    hh = arch26.Mat(m.rnn.weight_hh_l0, m.rnn.bias_hh_l0)
    h = T.zeros(B, m.units); hs = []
    for t in range(L):
        gh = lin(h, hh, local)
        if gate:
            gh = gh + added(h, term.down_h, term.up_h, term.zr, term.z768, local)
        i_r, i_z, i_n = gi[:, t].chunk(3, 1); h_r, h_z, h_n = gh.chunk(3, 1)
        r = T.sigmoid(i_r + h_r); z = T.sigmoid(i_z + h_z); n = T.tanh(i_n + r * h_n)
        h = (1 - z) * n + z * h
        hs.append(h)
    Hs = T.stack(hs, 1)
    if silence:
        Hs = T.zeros_like(Hs)
    z2 = T.tanh(lin(z1, m.l2, local) + lin(Hs, m.lm, local))
    return lin(z2, m.l3, local), hs


def freeze(m):
    """Every parameter of the lived network is frozen but the byte table, whose gradient is kept for the four code rows only."""
    for n, q in m.named_parameters():
        q.requires_grad_(n == "book.weight")


def keep_code_rows(m):
    """Call after backward: zero the byte table's gradient everywhere but the four code rows."""
    g = m.book.weight.grad
    keep = T.zeros(g.shape[0], 1); keep[list(CODES)] = 1.0
    g.mul_(keep)


def make_opt(m, term, lr=0.001, wd=0.25):
    """The terms decay as the network's weights do; the byte table does not, as everywhere else."""
    return T.optim.AdamW([{"params": term.parameters(), "weight_decay": wd}, {"params": [m.book.weight], "weight_decay": 0.0}], lr=lr, betas=(.9, .98))


if __name__ == "__main__":
    import e40_new_family as E40
    T.set_num_threads(1); arch26.install(); E, X = C26.E, C26.X
    seed = 73129
    items = next(E40.batches(seed)); x, y, mask = X.batch_tensor(items)
    for rule, local in (("S", True), ("B", False)):
        m = arch26.new_net(seed); m.load_state_dict(T.load(MDIR / "exp45" / "e26" / f"e26_life_{rule}_{seed}.pt"))
        before = {n: q.detach().clone() for n, q in m.named_parameters()}
        for r in (4, 16, 64):
            term = Term(seed, r); freeze(m); opt = make_opt(m, term)
            with T.no_grad():
                off = forward(m, term, x, local, False)[0]; ref = arch26.forward(m, x, local, True)[0]; on0 = forward(m, term, x, local, True)[0]
            lg = forward(m, term, x, local, True)[0]; loss = E.cross_entropy(lg[mask], y[mask]); opt.zero_grad(); loss.backward(); keep_code_rows(m)
            rows = (m.book.weight.grad.abs().sum(1) > 0).nonzero().flatten().tolist()
            g0 = {k: float(getattr(term, k).grad.abs().max()) for k in ("up_i", "down_i", "up_h", "down_h")}
            opt.step()
            lg1 = forward(m, term, x, local, True)[0]; loss1 = E.cross_entropy(lg1[mask], y[mask]); opt.zero_grad(); loss1.backward(); keep_code_rows(m)
            g1 = {k: float(getattr(term, k).grad.abs().max()) for k in ("down_i", "down_h")}
            changed = [n for n, q in m.named_parameters() if not T.equal(q.detach(), before[n])]
            rows_changed = (m.book.weight.detach() != before["book.weight"]).any(1).nonzero().flatten().tolist()
            with T.no_grad(): off1 = forward(m, term, x, local, False)[0]
            print(f"{rule} rank {r}: numbers that may change {term.count():,}; gate off equals arch26.forward {bool(T.equal(off, ref))}; gate on at the start equals it {bool(T.equal(on0, ref))};"
                  f" byte-table rows with gradient {rows}; first-step gradients up {g0['up_i']:.1e}/{g0['up_h']:.1e}, down {g0['down_i']:.1e}/{g0['down_h']:.1e}; after one step down {g1['down_i']:.1e}/{g1['down_h']:.1e};"
                  f" tensors changed {changed}, rows {rows_changed}; loss {float(loss):.3f} -> {float(loss1):.3f}; gate-off logits unchanged by the step on rows other than the codes: see check 2")
            m.load_state_dict(before)
    # the item-for-item check on an earlier family: gate off after a step, on items that contain no new code
    m = arch26.new_net(seed); m.load_state_dict(T.load(MDIR / "exp45" / "e26" / f"e26_life_S_{seed}.pt"))
    old = next(E.batches([(t, E.split(t, .5, seed)[0]) for t in E.DEV], seed, 12)); xo = X.batch_tensor(old)[0]
    with T.no_grad(): a0 = arch26.forward(m, xo, True, True)[0]
    term = Term(seed, 16); freeze(m); opt = make_opt(m, term)
    for _ in range(3):
        lg = forward(m, term, x, True, True)[0]; opt.zero_grad(); E.cross_entropy(lg[mask], y[mask]).backward(); keep_code_rows(m); opt.step()
    with T.no_grad(): a1 = forward(m, term, xo, True, False)[0]
    print("check 2: after three gated updates, an earlier family's logits with the gate off are bit-equal to the lived network's:", bool(T.equal(a0, a1)))
