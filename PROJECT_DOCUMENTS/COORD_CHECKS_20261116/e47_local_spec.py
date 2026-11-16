"""E47: locality at the input and the output. The layer as runnable code; coordinator's specification file, 16 Nov 2026 (requirements: the
reviewer's memo, revision 18, section 26.3). Cells stand on a SIDE x SIDE grid with joined edges. Only the 256 cells of one 16 x 16 patch
(the interface) read the byte and are read by the readout. Every cell reads N cells of its neighbourhood and its own previous value, by
index: no matrix over all pairs of cells is built. With SIDE = 16 every cell is at the interface and the layer is E42's F10.
    python3 -I e47_local_spec.py        self-checks, no training"""
import sys
from pathlib import Path
from random import Random

HERE = Path(__file__).resolve()
for _cand in (HERE.parent, HERE.parents[2] / "MOD97_LOCAL_RULE_20261106"):
    if (_cand / "exp26_candidate.py").exists():
        sys.path.insert(0, str(_cand)); break
sys.path.insert(0, str(HERE.parent))
import exp26_candidate as C26        # noqa: E402  first: sets the harness environment
import arch26                        # noqa: E402
import torch as T                    # noqa: E402
from exp3_state import lin           # noqa: E402

PATCH, RADIUS = 16, 3                # the interface is a 16 x 16 patch; neighbourhood radius 3 (48 cells)


def neighbourhood(i, side):
    r, c = divmod(i, side)
    return sorted({((r + dr) % side) * side + (c + dc) % side for dr in range(-RADIUS, RADIUS + 1) for dc in range(-RADIUS, RADIUS + 1)} - {i})


def interface(side):
    return [r * side + c for r in range(PATCH) for c in range(PATCH)]        # rows 0..15, columns 0..15, in this order


def distance(i, side):
    """Steps of the grid (larger of row and column distance, edges joined) from cell i to the nearest interface cell."""
    r, c = divmod(i, side)
    dr = 0 if r < PATCH else min(r - (PATCH - 1), side - r)
    dc = 0 if c < PATCH else min(c - (PATCH - 1), side - c)
    return max(dr, dc)


def draws(seed, side, N, periods):
    """connections (list of sorted lists), periods p, phases f. E42's strings, so SIDE = 16 gives E42's draws."""
    C = side * side
    rc = Random(f"e42|connections|{seed}")
    A = [sorted(rc.sample(neighbourhood(i, side), N)) for i in range(C)]
    far = [i for i in range(C) if distance(i, side) > 0]
    p = [1] * C
    if periods != "one" and far:
        by_place = sorted(far, key=lambda i: (distance(i, side), i))          # nearest the interface first
        per = [2 ** (1 + (4 * k) // len(far)) for k in range(len(far))]         # four equal groups by distance: periods 2, 4, 8, 16
        order = by_place
        if periods == "scatter":                                             # the same periods, placed at random among the far cells
            order = list(far); Random(f"e47|periods|{seed}|{side}").shuffle(order)
        for i, q in zip(order, per):
            p[i] = q
    rp = Random(f"e42|phases|{seed}")
    f = [rp.randrange(p[i]) for i in range(C)]
    return A, p, f


class IndexSign(T.autograd.Function):
    """out[b, g, i] = sum_k W[g, i, k] * h[b, idx[i, k]] + bias[g, i]. Back: the error crosses sign(W) * scale; weight gradient exact."""

    @staticmethod
    def forward(ctx, h, W, bias, idx, scale):
        ctx.save_for_backward(h, W, idx, scale)
        return T.einsum("bik,gik->bgi", h[:, idx], W) + bias

    @staticmethod
    def backward(ctx, g):
        h, W, idx, scale = ctx.saved_tensors
        B, C = h.shape
        back = T.einsum("bgi,gik->bik", g, W.sign() * scale)                 # (B, C, N+1): what each connection sends to its source
        gx = T.zeros(B, C, dtype=h.dtype).index_add_(1, idx.reshape(-1), back.reshape(B, -1))
        gw = T.einsum("bgi,bik->gik", g, h[:, idx])
        return gx, gw, g.sum(0), None, None


class Local:
    def __init__(self, m, seed, side=16, N=10, periods="one", structure=None):
        """m: arch26.new_net(seed). Its byte table, l1, input map (for the 256 interface cells), lm, l2 and l3 are used as they are.
        structure=(A, p, f, Wc, bc, a) loads a given structure and weights (used to compare with E42)."""
        self.side, self.C, self.N, self.seed, self.periods = side, side * side, N, seed, periods
        self.ifx = T.tensor(interface(side)); far = [i for i in range(self.C) if distance(i, side) > 0]; self.farx = T.tensor(far, dtype=T.long)
        if structure is None:
            A, p, f = draws(seed, side, N, periods)
            g = T.Generator().manual_seed(seed * 1000 + side)
            bound = (N + 1) ** -0.5                                          # fan-in N + 1; E42's start has this range (1/16 * sqrt(256 / 11))
            Wc = T.empty(3, self.C, N + 1).uniform_(-bound, bound, generator=g)
            bc = T.empty(3, self.C).uniform_(-1 / 16, 1 / 16, generator=g)
            a = None
        else:
            A, p, f, Wc, bc, a = structure
        self.idx = T.tensor([[i] + A[i] for i in range(self.C)])             # column 0 is the cell itself
        self.p, self.f = T.tensor(p), T.tensor(f); self.K = max(p)
        self.kmask = (T.arange(self.K)[:, None] < self.p[None, :]).float()
        self.Wc, self.bc = T.nn.Parameter(Wc.clone()), T.nn.Parameter(bc.clone())
        gf = T.Generator().manual_seed(seed * 1000 + side + 1)
        self.bfar = T.nn.Parameter(T.empty(3, len(far)).uniform_(-1 / 16, 1 / 16, generator=gf))   # far cells read no byte: a bias only
        self.a = T.nn.Parameter((self.kmask / self.p[None, :].float()) if a is None else a.clone()) if self.K > 1 else None

    def parameters(self):
        return [q for q in (self.Wc, self.bc, self.bfar, self.a) if q is not None and q.numel()]

    def coef(self):
        if self.K == 1:
            return T.ones(1, self.C)
        fixed = self.kmask / self.p[None, :].float()
        return T.where((self.p == 1)[None, :], fixed, self.a * self.kmask)

    def counts(self, m):
        """weights, and multiply-adds for one item at one position (forward), with every cell computed at every step."""
        fixed = 8 * m.l1.out_features + m.l2.weight.numel() + m.l3.weight.numel()
        iface = m.rnn.weight_ih_l0.numel() + m.lm.weight.numel()
        cells = 3 * self.C * (self.N + 1)
        nw = sum(q.numel() for n, q in m.named_parameters() if n != "rnn.weight_hh_l0" and n != "rnn.bias_hh_l0") + self.Wc.numel() + self.bc.numel() + self.bfar.numel() \
            + (int((self.kmask * (self.p > 1)[None, :]).sum()) if self.a is not None else 0)
        per_step = sum(3 * (self.N + 1) * int((self.p == q).sum()) / q for q in sorted(set(self.p.tolist())))
        return dict(weights=nw, multiply_adds=fixed + iface + cells, multiply_adds_with_periods=fixed + iface + per_step)


def forward(m, L, x, local, silence=False, far_off=False, far_mean=None):
    """Returns logits and the list of states. Two diagnostics: far_off holds the cells away from the interface at zero after every position;
    far_mean, a tensor (positions, far cells), holds them at the given values (their means over the scored items at each position)."""
    u = m.book(x)
    z1 = T.tanh(lin(u, m.l1, local))
    B, Lx = x.shape
    gi_if = lin(z1, arch26.Mat(m.rnn.weight_ih_l0, m.rnn.bias_ih_l0), local).reshape(B, Lx, 3, PATCH * PATCH)   # only the interface reads the byte
    scale = L.Wc.detach().abs().mean() if local else None
    c = L.coef()                                                             # (K, C)
    h = T.zeros(B, L.C); hs, G_hist, H_hist = [], [], []
    for t in range(Lx):
        G = T.zeros(B, 3, L.C)
        G = G.index_copy(2, L.ifx, gi_if[:, t])
        if len(L.farx):
            G = G.index_copy(2, L.farx, L.bfar[None].expand(B, -1, -1))
        H = IndexSign.apply(h, L.Wc, L.bc, L.idx, scale) if local else T.einsum("bik,gik->bgi", h[:, L.idx], L.Wc) + L.bc
        G_hist.append(G); H_hist.append(H)
        Gs = sum(c[k] * G_hist[t - k] for k in range(min(L.K, t + 1)))
        Hsum = sum(c[k] * H_hist[t - k] for k in range(min(L.K, t + 1)))
        r = T.sigmoid(Gs[:, 0] + Hsum[:, 0]); z = T.sigmoid(Gs[:, 1] + Hsum[:, 1]); n = T.tanh(Gs[:, 2] + r * Hsum[:, 2])
        hn = (1 - z) * n + z * h
        upd = (t >= L.f) & ((t - L.f) % L.p == 0)
        h = T.where(upd[None, :], hn, h)
        if far_off and len(L.farx):
            h = h.index_fill(1, L.farx, 0.0)
        if far_mean is not None and len(L.farx):
            h = h.index_copy(1, L.farx, far_mean[t][None].expand(B, -1))
        hs.append(h)
    Hs = T.stack(hs, 1)[:, :, L.ifx]                                         # only the interface is read
    if silence:
        Hs = T.zeros_like(Hs)
    z2 = T.tanh(lin(z1, m.l2, local) + lin(Hs, m.lm, local))
    return lin(z2, m.l3, local), hs


def from_e42(m42, lay42):
    """An E42 layer (masked matrix) as a Local structure with the same connections and weights, SIDE = 16."""
    U = 256; W = m42.rnn.weight_hh_l0.detach(); b = m42.rnn.bias_hh_l0.detach()
    A = [[j for j in range(U) if lay42.mask[i, j] and j != i] for i in range(U)]
    idx = T.tensor([[i] + A[i] for i in range(U)])
    Wc = T.stack([W[g * U:(g + 1) * U].gather(1, idx) for g in range(3)])
    return (A, lay42.p_list, lay42.f_list, Wc, b.reshape(3, U), lay42.a.detach() if lay42.a is not None else None), idx


if __name__ == "__main__":
    import time
    import e42_cells_spec as E42
    T.set_num_threads(1); arch26.install(); E, X = C26.E, C26.X
    seed = 73129
    tp = [(t, E.split(t, .5, seed)[0]) for t in E.DEV]
    items = next(E.batches(tp, seed, E.KMAX["M"])); x, y, mask = X.batch_tensor(items)

    def loss_grads(fwd, params):
        for q in params: q.grad = None
        lg = fwd(); E.cross_entropy(lg[mask], y[mask]).backward(); return lg.detach()

    # 1. at 256 cells the layer is E42's: same logits and gradients under both rules, for F10 and for P8
    for name, kw in (("F10", dict(N=10)), ("P8", dict(N=10, period_set="8", variant="step"))):
        m42 = arch26.new_net(seed); lay42 = E42.Layer(m42, seed, **kw)
        mL = arch26.new_net(seed); mL.load_state_dict(m42.state_dict())
        struct, idx = from_e42(m42, lay42); L = Local(mL, seed, 16, 10, structure=struct)
        for local in (False, True):
            lg42 = loss_grads(lambda: E42.forward(m42, lay42, x, local)[0], list(m42.parameters()) + lay42.added_parameters())
            lgL = loss_grads(lambda: forward(mL, L, x, local)[0], list(mL.parameters()) + L.parameters())
            gW42 = T.stack([m42.rnn.weight_hh_l0.grad[g * 256:(g + 1) * 256].gather(1, idx) for g in range(3)])
            rel = lambda a_, b_: float((a_ - b_).abs().max() / b_.abs().max())
            shared = max(rel(dict(mL.named_parameters())[n].grad, q.grad) for n, q in m42.named_parameters() if n not in ("rnn.weight_hh_l0", "rnn.bias_hh_l0"))
            shared = max(shared, rel(L.bc.grad, m42.rnn.bias_hh_l0.grad.reshape(3, 256)))
            extra = rel(L.a.grad, lay42.a.grad) if L.a is not None else 0.0
            print(f"1. {name} local={local}: logits differ by {float((lgL - lg42).abs().max()):.1e}; gradients (largest relative): recurrent {rel(L.Wc.grad, gW42):.1e}, the others {shared:.1e}, a {extra:.1e}")
    # 2. the draws at SIDE = 16 are E42's
    A16, _, _ = draws(seed, 16, 10, "one"); A42 = E42.draws(seed, 10, "1")[2]
    print("2. connections at 16 x 16 equal E42's:", A16 == A42)
    # 3. sizes: weights, multiply-adds, periods by place, how far the interface is
    for side in (16, 32, 64):
        for periods in (("one",) if side == 16 else ("one", "place", "scatter")):
            m = arch26.new_net(seed); L = Local(m, seed, side, 10, periods)
            cnt = L.counts(m); A, p, f = draws(seed, side, 10, periods)
            gap = sum(abs(p[i].bit_length() - p[j].bit_length()) for i in range(L.C) for j in A[i]) / (L.C * 10)
            byp = {q: p.count(q) for q in sorted(set(p))}
            print(f"3. {L.C} cells, periods {periods}: weights {cnt['weights']:,}; multiply-adds {cnt['multiply_adds']:,} (with periods {cnt['multiply_adds_with_periods']:,.0f}); cells by period {byp};"
                  f" mean difference in log2 period across a connection {gap:.2f}; farthest cell {max(distance(i, side) for i in range(L.C))} grid steps = {-(-max(distance(i, side) for i in range(L.C)) // RADIUS)} connections")
    # 4. locality and time at 1,024 cells with periods by place
    m = arch26.new_net(seed); L = Local(m, seed, 32, 10, "place")
    with T.no_grad():
        hs = T.stack(forward(m, L, x, False)[1], 1)
        x2 = x.clone(); t0 = 9; x2[:, t0] = (x2[:, t0] + 1) % 256
        hs2 = T.stack(forward(m, L, x2, False)[1], 1)
        d = (hs != hs2).any(0)                                               # (L, C): cells whose state differs
        first = [int(d[:, i].float().argmax()) if bool(d[:, i].any()) else -1 for i in range(L.C)]
        ok_time = all(fi == -1 or fi >= t0 + -(-distance(i, 32) // RADIUS) for i, fi in enumerate(first))
        upd = T.stack([(t >= L.f) & ((t - L.f) % L.p == 0) for t in range(x.shape[1])]); changed = (hs[:, 1:] != hs[:, :-1]).any(0)
        lg_off = forward(m, L, x, False, far_off=True)[0]; hs_off = T.stack(forward(m, L, x, False, far_off=True)[1], 1)
    print("4. a changed byte reaches a cell no sooner than its distance allows:", ok_time, "| changes only at update positions:", bool((~changed | upd[1:]).all()),
          "| with far cells off they are zero:", bool((hs_off[:, :, L.farx] == 0).all()), "| logits finite:", bool(T.isfinite(lg_off).all()))
    with T.no_grad():
        mean = hs[:, :, L.farx].mean(0)                                      # (positions, far cells): the batch's mean state at each position
        hs_mean = T.stack(forward(m, L, x, False, far_mean=mean)[1], 1)
    print("   with far cells held at their mean they equal it:", bool(T.equal(hs_mean[:, :, L.farx], mean[None].expand(x.shape[0], -1, -1))))
    import hashlib, json
    dg = lambda v: hashlib.sha256(json.dumps(v).encode()).hexdigest()[:16]
    WANT = {16: ("fde78220c57da3d4", "0cc7c65a16f482a9", "0cc7c65a16f482a9", "ba1a044f61b89809"), 32: ("f3361f3823d86f9d", "8394011ba0130f32", "1f2074a6ca43ab00", "624270ab87410e73"),
            64: ("a70f0c440580f1bb", "10ef2ba7a17c347c", "772686e61a77fb83", "80295a954d239c28")}
    for side, want in WANT.items():
        Ap, pp, fp = draws(seed, side, 10, "place"); _, ps_, _ = draws(seed, side, 10, "scatter")
        print(f"   draws at {side} a side equal the audit's digests (connections, periods place, periods scatter, phases):", (dg(Ap), dg(pp), dg(ps_), dg(fp)) == want)
    for local in (False, True):
        loss_grads(lambda: forward(m, L, x, local)[0], list(m.parameters()) + L.parameters())
        print(f"   local={local}: every parameter has a finite gradient: {all(bool(T.isfinite(q.grad).all()) for q in list(m.parameters()) + L.parameters() if q.grad is not None)};"
              f" share of far cells' recurrent weights with a non-zero gradient {float((L.Wc.grad[:, L.farx] != 0).float().mean()):.2f}")
    # 5. cost per forward and backward (this host; indicative only)
    for side, periods in ((16, "one"), (32, "one"), (32, "place"), (64, "one"), (64, "place")):
        m = arch26.new_net(seed); L = Local(m, seed, side, 10, periods); t = time.process_time()
        for _ in range(2): loss_grads(lambda: forward(m, L, x, True)[0], list(m.parameters()) + L.parameters())
        print(f"5. {L.C} cells, periods {periods}: {(time.process_time() - t) / 2:.3f} CPU-seconds per forward and backward (sweep)")
