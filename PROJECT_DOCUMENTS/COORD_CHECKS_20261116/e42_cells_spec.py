"""E42: the cell layer, as runnable code. Coordinator's specification file, 16 Nov 2026 (the design is the reviewer's memo, revision 10,
section 22). It defines the layer's forward, the masked sign rule and the replacement of connections; a runner imports it and adds the
birth loop, scoring and receipts.            python3 -I e42_cells_spec.py          runs the self-checks (no training)

Conventions: U = 256 cells; Wh = m.rnn.weight_hh_l0 (768 x 256: rows r, z, n for each cell; column j is the source cell);
mask[i, j] is True if cell i reads cell j (its N connections and itself). Position t = 0 is the start symbol; h(-1) = 0."""
import sys
from pathlib import Path
from random import Random

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2] / "MOD97_LOCAL_RULE_20261106"))
import exp26_candidate as C26        # noqa: E402  first: sets the harness environment (E16_BOOKDECAY=off and the rest)
import arch26                        # noqa: E402
import torch as T                    # noqa: E402
from torch.nn.functional import linear  # noqa: E402
from exp3_state import lin           # noqa: E402

U, SIDE, RADIUS = 256, 16, 3
PERIOD_SETS = {"1": {1: 256}, "4": {1: 128, 2: 64, 4: 64}, "8": {1: 128, 2: 64, 4: 32, 8: 32}, "16": {1: 128, 2: 64, 4: 32, 8: 16, 16: 16}}
VARIANTS = ("last", "mean", "step", "source")
EVENT, MIN_AGE = 256, 512            # replacement every 256 updates; a connection can be given up once it is 512 updates old


# ---------------------------------------------------------------- geometry and draws (random.Random only; torch's generator is untouched)
def neighbourhood(i, wide=False):
    if wide:
        return [j for j in range(U) if j != i]
    r, c = divmod(i, SIDE)
    return sorted({((r + dr) % SIDE) * SIDE + (c + dc) % SIDE for dr in range(-RADIUS, RADIUS + 1) for dc in range(-RADIUS, RADIUS + 1)} - {i})


def draws(seed, N, period_set, wide=False):
    """periods p (U,), phases f (U,), connections A (list of sorted lists). N = 255 means every cell to every other."""
    order = list(range(U)); Random(f"e42|periods|{seed}").shuffle(order)
    p, k = [0] * U, 0
    for per, n in PERIOD_SETS[period_set].items():
        for i in order[k:k + n]:
            p[i] = per
        k += n
    rp = Random(f"e42|phases|{seed}"); f = [rp.randrange(p[i]) for i in range(U)]
    rc = Random(f"e42|connections|{seed}")
    A = [[j for j in range(U) if j != i] if N == U - 1 else sorted(rc.sample(sorted(neighbourhood(i, wide)), N)) for i in range(U)]
    return p, f, A


# ---------------------------------------------------------------- the masked sign rule
class MaskedSign(T.autograd.Function):
    """linear(x, weight * maskf, bias). Back: the error crosses sign(weight) inside the mask times `scale`; the weight gradient is the
    usual outer product, zero outside the mask."""

    @staticmethod
    def forward(ctx, x, weight, bias, maskf, scale):
        ctx.save_for_backward(x, weight, maskf, scale)
        return linear(x, weight * maskf, bias)

    @staticmethod
    def backward(ctx, g):
        x, weight, maskf, scale = ctx.saved_tensors
        gx = g @ ((weight * maskf).sign() * scale)
        gw = (g.reshape(-1, g.shape[-1]).T @ x.reshape(-1, x.shape[-1])) * maskf
        gb = g.reshape(-1, g.shape[-1]).sum(0)
        return gx, gw, gb, None, None


class Layer:
    """The structure of one run's cell layer and its added parameters. `full` is True for configuration G."""

    def __init__(self, m, seed, N=10, period_set="1", variant="step", wide=False):
        assert variant in VARIANTS
        self.N, self.variant, self.seed = N, variant, seed
        self.p_list, self.f_list, A = draws(seed, N, period_set, wide)
        self.full = N == U - 1
        self.wide = wide
        self.mask = T.eye(U, dtype=T.bool)
        for i, a in enumerate(A):
            self.mask[i, a] = True
        self.made = T.zeros(U, U, dtype=T.long)                     # the update at which each connection was made (0 = at the start)
        self.p, self.f = T.tensor(self.p_list), T.tensor(self.f_list)
        self.K = max(self.p_list)
        self.kmask = (T.arange(self.K)[:, None] < self.p[None, :]).float()        # (K, U): step k belongs to cell i's period
        W = m.rnn.weight_hh_l0
        with T.no_grad():                                           # start: outside the mask zero; inside scaled to keep the input variance
            if not self.full:
                W.mul_(self.mask3().float() * (U / (N + 1)) ** 0.5)
        self.a = None; self.Wx = None
        if variant in ("step", "source") and self.K > 1:
            self.a = T.nn.Parameter(self.kmask / self.p[None, :].float())        # a[k, i], 1 / p_i inside the period
        if variant == "source" and self.K > 1:
            with T.no_grad():
                W.div_(self.p.repeat(3)[:, None].float())                       # Wh[0] = masked Wh / p_i
                self.Wx = T.nn.Parameter(W.detach()[None].repeat(self.K - 1, 1, 1) * self.kmask[1:].repeat(1, 3)[:, :, None])   # Wh[1..K-1]

    def mask3(self):
        return self.mask.repeat(3, 1)                               # (768, U)

    def added_parameters(self):
        return [q for q in (self.a, self.Wx) if q is not None]

    def coef(self):
        """c[k, i]: the weight cell i gives to step t - k of its period."""
        if self.variant == "last" or self.K == 1:
            c = T.zeros(self.K, U); c[0] = 1.0; return c
        if self.variant == "mean":
            return self.kmask / self.p[None, :].float()
        fixed = self.kmask / self.p[None, :].float()
        return T.where((self.p == 1)[None, :], fixed, self.a * self.kmask)     # a cell of period 1 has no learned weight: the same cell in every variant

    def sign_scale(self, m):
        """One scale for the recurrent map: the mean magnitude of the weights inside the mask (all steps stacked in `source`)."""
        m3 = self.mask3().float(); W = m.rnn.weight_hh_l0.detach()
        tot, cnt = (W.abs() * m3).sum(), m3.sum()
        if self.Wx is not None:
            km = self.kmask[1:].repeat(1, 3)[:, :, None] * m3[None]
            tot, cnt = tot + (self.Wx.detach().abs() * km).sum(), cnt + km.sum()
        return tot / cnt

    def strength(self, m):
        """(U, U): the sum of the magnitudes of a connection's weights over the three gates and, in `source`, over the steps."""
        s = m.rnn.weight_hh_l0.detach().abs().reshape(3, U, U).sum(0)
        if self.Wx is not None:
            s = s + self.Wx.detach().abs().reshape(self.K - 1, 3, U, U).sum((0, 1))
        return s


def rec(layer, m, h, W, bias, local, scale, kstep=None):
    """One crossing of the recurrent map: (W * mask) h + bias."""
    if layer.full:
        return lin(h, arch26.Mat(W, bias), local)                  # E26's own call, unchanged
    m3 = layer.mask3().float()
    if kstep is not None:
        m3 = m3 * layer.kmask[kstep].repeat(3)[:, None]
    return MaskedSign.apply(h, W, bias, m3, scale) if local else linear(h, W * m3, bias)


def forward(m, layer, x, local, silence=False):
    """arch26.forward with the cell layer in place of the GRU step. Returns logits and the list of states after each position."""
    u = m.book(x)
    z1 = T.tanh(lin(u, m.l1, local))
    B, L = x.shape
    gi = lin(z1, arch26.Mat(m.rnn.weight_ih_l0, m.rnn.bias_ih_l0), local)     # every cell reads the whole current byte
    Wh, bh = m.rnn.weight_hh_l0, m.rnn.bias_hh_l0
    scale = layer.sign_scale(m) if (local and not layer.full) else None
    c = layer.coef().repeat(1, 3)                                             # (K, 768)
    zero_b = T.zeros_like(bh)
    h = T.zeros(B, U); hs, gh_hist, hprev_hist = [], [], []
    for t in range(L):
        hprev_hist.append(h)                                                  # h(t-1)
        if layer.variant != "source" or layer.K == 1:
            gh_hist.append(rec(layer, m, h, Wh, bh, local, scale))            # gh(t) = (Wh * mask) h(t-1) + bh
        G = sum(c[k] * gi[:, t - k] for k in range(min(layer.K, t + 1)))      # steps before the start symbol contribute nothing
        if layer.variant == "source" and layer.K > 1:
            H = 0
            for k in range(min(layer.K, t + 1)):                              # a separate strength for every connection at every step
                Wk = Wh if k == 0 else layer.Wx[k - 1]
                H = H + rec(layer, m, hprev_hist[t - k], Wk, zero_b, local, scale, kstep=k) + c[k] * bh
        else:
            H = sum(c[k] * gh_hist[t - k] for k in range(min(layer.K, t + 1)))
        i_r, i_z, i_n = G.chunk(3, 1); h_r, h_z, h_n = H.chunk(3, 1)
        r = T.sigmoid(i_r + h_r); z = T.sigmoid(i_z + h_z); n = T.tanh(i_n + r * h_n)
        hn = (1 - z) * n + z * h
        upd = (t >= layer.f) & ((t - layer.f) % layer.p == 0)                 # (U,): the cells updated at this position
        h = T.where(upd[None, :], hn, h)
        hs.append(h)
    Hs = T.stack(hs, 1)
    if silence:
        Hs = T.zeros_like(Hs)
    z2 = T.tanh(lin(z1, m.l2, local) + lin(Hs, m.lm, local))
    return lin(z2, m.l3, local), hs


def replace(layer, m, opt, update, theta, rr):
    """One replacement event. Returns the number of connections replaced. rr: Random(f"e42|replace|{seed}"), created once per run."""
    s = layer.strength(m); eye = T.eye(U, dtype=T.bool); n_rep = 0
    params = [m.rnn.weight_hh_l0] + ([layer.Wx] if layer.Wx is not None else [])
    for i in range(U):
        conn = [j for j in range(U) if layer.mask[i, j] and j != i]
        mean = sum(float(s[i, j]) for j in conn) / len(conn)
        drop = [j for j in conn if update - int(layer.made[i, j]) >= MIN_AGE and float(s[i, j]) < theta * mean]
        for j in drop:                                                        # in increasing order of source
            cand = sorted(q for q in neighbourhood(i, layer.wide) if not layer.mask[i, q] and q not in drop)
            new = rr.choice(cand)
            layer.mask[i, j] = False; layer.mask[i, new] = True; layer.made[i, new] = update; layer.made[i, j] = 0
            rows = [i, U + i, 2 * U + i]
            with T.no_grad():
                for prm in params:
                    for col in (j, new):                                      # both ends: weights zero, optimizer's running averages zero
                        prm[..., rows, col] = 0.0
                        st = opt.state.get(prm)
                        if st:
                            st["exp_avg"][..., rows, col] = 0.0; st["exp_avg_sq"][..., rows, col] = 0.0
            n_rep += 1
    assert not (layer.mask & ~eye).sum(1).ne(layer.N).any() and bool(layer.mask.diagonal().all())
    return n_rep


# ---------------------------------------------------------------- self-checks (no training)
if __name__ == "__main__":
    import time
    T.set_num_threads(1)
    E, X = C26.E, C26.X
    arch26.install()
    seed = 73129
    tp = [(t, E.split(t, .5, seed)[0]) for t in E.DEV]
    items = next(E.batches(tp, seed, E.KMAX["M"])); x, y, mask = X.batch_tensor(items)

    def net_and_layer(**kw):
        m = arch26.new_net(seed); return m, Layer(m, seed, **kw)

    def grads(m, layer, local):
        for q in list(m.parameters()) + layer.added_parameters():
            q.grad = None
        lg = forward(m, layer, x, local)[0]; E.cross_entropy(lg[mask], y[mask]).backward()
        return lg.detach(), {n: q.grad.clone() for n, q in m.named_parameters() if q.grad is not None}

    # 1. G is E26's network and forward
    m, lay = net_and_layer(N=255); m0 = arch26.new_net(seed)
    print("1. G: parameters bit-equal to arch26.new_net:", all(T.equal(a, b) for a, b in zip(m.state_dict().values(), m0.state_dict().values())))
    for local in (False, True):
        lg, g = grads(m, lay, local)
        for q in m0.parameters(): q.grad = None
        lg0 = arch26.forward(m0, x, local, True)[0]; E.cross_entropy(lg0[mask], y[mask]).backward()
        print(f"   local={local}: logits equal {bool(T.equal(lg, lg0.detach()))}; gradients equal {all(T.equal(g[n], q.grad) for n, q in m0.named_parameters())}")
    # 2. counts
    for N in (5, 10, 20):
        m, lay = net_and_layer(N=N)
        inside = int(lay.mask3().sum()); total = sum(q.numel() for q in m.parameters()) - m.rnn.weight_hh_l0.numel() + inside
        print(f"2. N={N}: recurrent weights inside the mask {inside}; network {total}; neighbourhood size {len(neighbourhood(0))}; scale at start {(U / (N + 1)) ** 0.5:.3f}")
    for ps in ("4", "8", "16"):
        print(f"   periods {ps}: learned weights a_i (cells of period above 1) {sum(q for q in draws(seed, 10, ps)[0] if q > 1)}")
    # 3. mean, step, source agree at the start
    ref = None
    for v in ("mean", "step", "source", "last"):
        m, lay = net_and_layer(N=10, period_set="8", variant=v)
        with T.no_grad(): lg = forward(m, lay, x, False)[0]
        if ref is None: ref = lg
        print(f"3. P8 variant {v}: largest difference from `mean` {float((lg - ref).abs().max()):.2e}")
    # 4. time: a later byte changes no earlier state; a cell changes only at its own update positions; zero before its phase
    m, lay = net_and_layer(N=10, period_set="8", variant="step")
    with T.no_grad():
        hs = T.stack(forward(m, lay, x, False)[1], 1)
        x2 = x.clone(); t0 = 9; x2[:, t0] = (x2[:, t0] + 1) % 256
        hs2 = T.stack(forward(m, lay, x2, False)[1], 1)
    upd = T.stack([(t >= lay.f) & ((t - lay.f) % lay.p == 0) for t in range(x.shape[1])])        # (L, U)
    changed = (hs[:, 1:] != hs[:, :-1]).any(0)                                                   # (L-1, U)
    print("4. no change before the changed byte:", bool(T.equal(hs[:, :t0], hs2[:, :t0])), "| changes only at update positions:", bool((~changed | upd[1:]).all()),
          "| zero before the phase:", bool(all((hs[:, :int(lay.f[i]), i] == 0).all() for i in range(U))))
    # 5. gradients: zero outside the mask under both rules; the two rules differ
    for name, kw in (("F10", dict(N=10)), ("P8", dict(N=10, period_set="8", variant="step")), ("P8-source", dict(N=10, period_set="8", variant="source"))):
        m, lay = net_and_layer(**kw)
        gB = grads(m, lay, False)[1]; gS = grads(m, lay, True)[1]; out = ~lay.mask3()
        print(f"5. {name}: recurrent gradient zero outside the mask B {bool((gB['rnn.weight_hh_l0'][out] == 0).all())} S {bool((gS['rnn.weight_hh_l0'][out] == 0).all())};"
              f" rules differ on l1 {not T.allclose(gB['l1.weight'], gS['l1.weight'])}; cosine of the recurrent gradients {float(T.nn.functional.cosine_similarity(gB['rnn.weight_hh_l0'].flatten(), gS['rnn.weight_hh_l0'].flatten(), 0)):.3f}")
    # 6. replacement: the limit holds, added connections change no logit, replaced connections lie in the neighbourhood
    m, lay = net_and_layer(N=10); opt = C26.make_opt(m); rr = Random(f"e42|replace|{seed}")
    before = lay.mask.clone()
    with T.no_grad(): m.rnn.weight_hh_l0[:, ::7] *= 0.01                     # make some connections weak
    n = replace(lay, m, opt, 512, 0.25, rr)
    new = lay.mask & ~before
    with T.no_grad():
        lg1 = forward(m, lay, x, False)[0]
        w = m.rnn.weight_hh_l0.clone(); w[new.repeat(3, 1)] = 0; same_w = bool(T.equal(w, m.rnn.weight_hh_l0))
    print(f"6. replaced {n} connections; new connections' weights zero: {same_w}; all inside the neighbourhood: {all(j in neighbourhood(i) for i, j in new.nonzero().tolist())};"
          f" weights outside the mask zero: {bool((m.rnn.weight_hh_l0[~lay.mask3()] == 0).all())}")
    # 7. cost per update (this host; indicative)
    for name, kw in (("G", dict(N=255)), ("A10", dict(N=10)), ("P8", dict(N=10, period_set="8", variant="step")), ("P8-source", dict(N=10, period_set="8", variant="source"))):
        m, lay = net_and_layer(**kw); t = time.process_time()
        for _ in range(3): grads(m, lay, True)
        print(f"7. {name}: {(time.process_time() - t) / 3:.3f} CPU-seconds per forward and backward (sweep)")
