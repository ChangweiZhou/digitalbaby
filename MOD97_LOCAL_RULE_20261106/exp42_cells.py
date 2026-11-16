"""E42 runner (brief OPUS_TASK_E42_CELLS_WITH_LIMITED_LOCAL_CONNECTIONS_AND_PERIODS_20261116.md, revision 2): the present network with its
recurrent layer made of cells with a limited number of local connections and their own periods, born on E26's task under the sweep and under
backpropagation.

The layer is the specification file e42_cells.py (a byte-identical copy of PROJECT_DOCUMENTS/COORD_CHECKS_20261116/e42_cells_spec.py),
imported and never rewritten: its geometry and draws (Layer), its masked sign rule (MaskedSign), its coefficients (Layer.coef), its sign scale
(Layer.sign_scale) and its replacement of connections (replace). This runner adds a faster forward (check 9 ties it to the file's), the birth
loop, the scoring and the receipts.

    python3 exp42_cells.py run <config> <S|B> <seed> [--theta T] [--dose N] [--out DIR] [--answers]
    python3 exp42_cells.py check <1|2|3|5|6|7|8|9|10|12> [seed] [--qual DIR]
    configs: G F10 A10 P8 N5 N20 W10 P4 P16 P8-last P8-source

The faster forward (`forward`) computes what e42_cells.forward computes, with three changes of arrangement only:
  * the sums over a period's steps of the input map, G(t) = sum_k c[k] gi(t - k), are formed for all positions at once before the loop
    (gi is known for every position), from a copy of gi padded with zeros before the start symbol;
  * the recurrent sums H(t) = sum_k c[k] [crossing k at t] are kept in a rolling accumulator: at position t the crossing made from h(t - 1)
    is added, with its coefficient c[k], to the sum of position t + k, for every k at once; in `source` the K crossings W_k h(t - 1) are one
    matrix product with the K masked matrices stacked. A crossing made at position s >= 0 always belongs to steps at or after the start
    symbol, so "a step before the start symbol contributes nothing" holds by construction;
  * the masked recurrent matrix (W * mask) and, under the sweep, its sign times the scale are formed once per forward, not at every position;
    the weight gradient reaches W through the product with the mask, so it is zero outside the mask, as in MaskedSign.
For G (every cell reads every other, all periods 1) the forward is arch26.forward itself.
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from random import Random

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exp26_candidate as C26        # noqa: E402  first harness import (environment gate: E16_BOOKDECAY=off and the rest)

import torch as T                    # noqa: E402
from torch.nn.functional import linear, pad  # noqa: E402

import e42_cells as L42              # noqa: E402  the specification file, imported, not rewritten
import arch26                        # noqa: E402

R_, X16, E, X = C26.R, C26.X16, C26.E, C26.X
T.set_num_threads(1)

U = L42.U
SEEDS = (73129, 93683, 116089)
DOSE = 24576
ROW = 256
RULES = {"S": True, "B": False}
OUT = X.OUT / "e42"
QUAL = X.OUT / "e42_qual"
SPEC = HERE.parent / "PROJECT_DOCUMENTS" / "COORD_CHECKS_20261116" / "e42_cells_spec.py"
SOURCES = ("exp42_cells.py", "e42_cells.py") + C26.SOURCES
LEN_TASK = ("add", 89, b"p")

#            N    period set  variant   wide   changing
CONFIGS = {
    "G":         (255, "1",  "step",   False, False),
    "F10":       (10,  "1",  "step",   False, False),
    "A10":       (10,  "1",  "step",   False, True),
    "P8":        (10,  "8",  "step",   False, True),
    "N5":        (5,   "1",  "step",   False, True),
    "N20":       (20,  "1",  "step",   False, True),
    "W10":       (10,  "1",  "step",   True,  True),
    "P4":        (10,  "4",  "step",   False, True),
    "P16":       (10,  "16", "step",   False, True),
    "P8-last":   (10,  "8",  "last",   False, True),
    "P8-source": (10,  "8",  "source", False, True),
}
STAGE1 = ("G", "F10", "A10", "P8")
STAGE2 = ("N5", "N20", "W10", "P4", "P16", "P8-last", "P8-source")
WEIGHT_COUNT = {"G": 1124113, "N5": 932113, "F10": 935953, "A10": 935953, "W10": 935953, "P8-last": 935953, "N20": 943633,
                "P4": 936337, "P8": 936465, "P16": 936593, "P8-source": 949137}
THETA = None          # set once by check 10; the runner takes it from --theta and records it


# ---------------------------------------------------------------- the faster forward
class SignLin(T.autograd.Function):
    """linear(x, Wm, b) whose error crosses the constant matrix S (the sign of the masked weights times the scale) on the way back. Wm is
    the masked matrix (W * mask, formed once per forward); its gradient is the usual outer product and reaches W through the product with
    the mask, so it is zero outside the mask."""

    @staticmethod
    def forward(ctx, x, Wm, b, S):
        ctx.save_for_backward(x, S)
        return linear(x, Wm, b)

    @staticmethod
    def backward(ctx, g):
        x, S = ctx.saved_tensors
        return g @ S, g.T @ x, g.sum(0), None


def forward(m, layer, x, local, silence=False):
    """Same values as e42_cells.forward (check 9). Returns logits and the list of states after each position."""
    if layer.full:
        lg, d = arch26.forward(m, x, local, True, silence=silence)
        return lg, d["h"]
    u = m.book(x)
    z1 = T.tanh(L42.lin(u, m.l1, local))
    B, L = x.shape
    gi = L42.lin(z1, arch26.Mat(m.rnn.weight_ih_l0, m.rnn.bias_ih_l0), local)          # (B, L, 768): every cell reads the whole byte
    Wh, bh = m.rnn.weight_hh_l0, m.rnn.bias_hh_l0
    src = layer.variant == "source" and layer.K > 1
    Kc = 1 if (layer.K == 1 or layer.variant == "last") else layer.K                  # `last`: c[k] = 0 for k > 0, so those steps are skipped
    upd = T.stack([(t >= layer.f) & ((t - layer.f) % layer.p == 0) for t in range(L)])   # (L, U)
    if Kc > 1:
        c = layer.coef().repeat(1, 3)                                                  # (K, 768)
        # the step a position s is for, in each cell's period: the cell's next update at or after s is at s + kidx[s]
        kidx = ((layer.f[None, :] - T.arange(L)[:, None]) % layer.p[None, :]).repeat(1, 3)            # (L, 768)
        wtab = c.gather(0, kidx)                                                       # (L, 768): c[kidx[s], i]
        keep = T.cat([T.zeros(1, 3 * U), (~upd[:-1]).float().repeat(1, 3)], 0)        # (L, 768): 0 just after a cell's update
    m3 = layer.mask3().float()
    if src:
        Wall = T.cat([Wh[None], layer.Wx], 0) * (m3[None] * layer.kmask.repeat(1, 3)[:, :, None])     # (K, 768, U), masked
        # the crossing made at position s for a cell uses the matrix of its step kidx[s]; kidx is periodic in s with period K
        Wc = Wall[kidx[:layer.K], T.arange(3 * U)[None, :]]                           # (K, 768, U): row i of matrix q is Wall[kidx[q, i], i]
    else:
        Wc = (Wh * m3)[None]
    if local:
        Sc = Wc.detach().sign() * layer.sign_scale(m)
    # input side, all positions at once: G(t) = sum_k c[k] gi(t - k), steps before the start symbol contribute nothing
    if Kc == 1:
        G = gi
    else:
        gp = pad(gi, (0, 0, Kc - 1, 0))
        G = sum(c[k] * gp[:, Kc - 1 - k:Kc - 1 - k + L] for k in range(Kc))
    h = T.zeros(B, U); hs = []
    acc = None
    for t in range(L):
        q = t % Wc.shape[0]
        y = SignLin.apply(h, Wc[q], bh, Sc[q]) if local else linear(h, Wc[q], bh)    # (B, 768): the crossing made from h(t-1)
        if Kc == 1:
            H = y
        else:
            # running sum over the cell's current period, restarted after each of its updates; exact at the cell's update positions,
            # which are the only positions where its value is kept
            acc = wtab[t] * y if acc is None else keep[t] * acc + wtab[t] * y
            H = acc
        i_r, i_z, i_n = G[:, t].chunk(3, 1); h_r, h_z, h_n = H.chunk(3, 1)
        r = T.sigmoid(i_r + h_r); z = T.sigmoid(i_z + h_z); n = T.tanh(i_n + r * h_n)
        hn = (1 - z) * n + z * h
        h = T.where(upd[t][None, :], hn, h)
        hs.append(h)
    Hs = T.stack(hs, 1)
    if silence:
        Hs = T.zeros_like(Hs)
    z2 = T.tanh(L42.lin(z1, m.l2, local) + L42.lin(Hs, m.lm, local))
    return L42.lin(z2, m.l3, local), hs


# ---------------------------------------------------------------- network, layer, optimizer
def build(config, seed):
    N, ps, variant, wide, _ = CONFIGS[config]
    m = arch26.new_net(seed)
    layer = L42.Layer(m, seed, N=N, period_set=ps, variant=variant, wide=wide)
    return m, layer


def make_opt(m, layer):
    opt = C26.make_opt(m)                         # asserts E26's two groups by name
    added = layer.added_parameters()
    if added:
        g = [g for g in opt.param_groups if g["weight_decay"] == C26.WD]
        assert len(g) == 1
        g[0]["params"].extend(added)              # the layer's added parameters join the decayed group
    return opt


def opt_record(m, layer, opt):
    names = {id(p): n for n, p in m.named_parameters()}
    if layer.a is not None:
        names[id(layer.a)] = "a"
    if layer.Wx is not None:
        names[id(layer.Wx)] = "Wx"
    return dict(name="AdamW", lr=C26.LR, weight_decay=C26.WD, betas=list(C26.BETAS),
                groups=[dict(weight_decay=g["weight_decay"], lr=g["lr"], betas=list(g["betas"]), params=[names[id(p)] for p in g["params"]])
                        for g in opt.param_groups])


def weight_count(m, layer):
    n = sum(q.numel() for q in m.parameters()) - m.rnn.weight_hh_l0.numel() + int(layer.mask3().sum())
    n += sum(p for p in layer.p_list if p > 1) if layer.a is not None else 0
    if layer.Wx is not None:
        n += sum((p - 1) * 3 * (layer.N + 1) for p in layer.p_list)
    return n


def install(layer):
    """arch26.install(), then exp11_cell.forward replaced by a function of the harness's signature that runs this run's layer."""
    arch26.install()

    def fwd(m, x, local, through_time, keep=False, silence=False):
        lg, hs = forward(m, layer, x, local, silence=silence)
        return lg, dict(h=hs)
    E.forward = fwd


def step(m, layer, opt, items, local):
    """exp26_candidate.step's text with this file's forward in place of arch26.forward."""
    x, y, mask = X.batch_tensor(items)
    logits = forward(m, layer, x, local)[0]
    loss = E.cross_entropy(logits[mask], y[mask])
    opt.zero_grad()
    loss.backward()
    opt.step()
    return float(loss.detach())


# ---------------------------------------------------------------- scoring (E41's three accuracies, copied from exp41_scale.py at 88b8b7b3)
def taught_held(f, seed):
    return {t: E.split(t, f, seed) for t in E.DEV}


def reduction_sample(t, held, f, seed):
    op = t[0]
    pairs = [(a, b) for a, b in held if (a + b >= 89 if op == "add" else a - b < 0)]
    return Random(f"e41|{seed}|{round(f * 100)}|{t[0]}").sample(sorted(pairs), 1000)


def panels(seed, f=.5):
    th = taught_held(f, seed)
    A = {E.NAME(*t): E.panel(t, th[t][1], seed, ks=(12,), n=300)[12] for t in E.DEV}
    TA = {E.NAME(*t): E.panel(t, th[t][0], seed + 1, ks=(12,), n=300)[12] for t in E.DEV}
    Rs = {}
    for t in E.DEV[:2]:
        s = reduction_sample(t, th[t][1], f, seed)
        Rs[E.NAME(*t)] = [E.render(t, a, b, 0, Random(0)) for a, b in s]
    LP = E.panel(LEN_TASK, E.split(LEN_TASK, .5, seed)[1], seed, ks=(12, 40), n=300)      # E30's length test
    return A, Rs, TA, LP


def score_all(m, A, Rs, TA, LP):
    a = {n: E.score(m, it)["exact"] for n, it in A.items()}
    r = {n: E.score(m, it)["exact"] for n, it in Rs.items()}
    ta = {n: E.score(m, it)["exact"] for n, it in TA.items()}
    ln = {str(k): E.score(m, it)["exact"] for k, it in LP.items()}
    return dict(A_counts=a, R_counts=r, taught_counts=ta, length_counts=ln,
                A=100 * (a["add89"] + a["sub89"]) / 600, R=100 * (r["add89"] + r["sub89"]) / 2000,
                taught=100 * (ta["add89"] + ta["sub89"]) / 600, len12=100 * ln["12"] / 300, len40=100 * ln["40"] / 300)


# ---------------------------------------------------------------- structure, digests, checks inside a run
def tsha(t):
    t = t.detach().contiguous().cpu()
    return hashlib.sha256(str(t.dtype).encode() + str(tuple(t.shape)).encode() + t.numpy().tobytes()).hexdigest()


def structure(layer):
    conn = [[j for j in range(U) if layer.mask[i, j] and j != i] for i in range(U)]
    return dict(periods=layer.p_list, phases=layer.f_list, connections=conn,
                made=[[int(layer.made[i, j]) for j in cs] for i, cs in enumerate(conn)])


def check_limit(m, layer):
    """Check 4, at every scored row: exactly N connections, all inside the neighbourhood, none to itself beyond its own reading; every weight
    outside the mask exactly zero."""
    eye = T.eye(U, dtype=T.bool)
    nb_ok = all(all(j in set(L42.neighbourhood(i, layer.wide)) for j in range(U) if layer.mask[i, j] and j != i) for i in range(U)) \
        if not layer.full else True
    cnt_ok = bool((layer.mask & ~eye).sum(1).eq(layer.N).all()) and bool(layer.mask.diagonal().all())
    out = ~layer.mask3()
    w_ok = bool((m.rnn.weight_hh_l0[out] == 0).all())
    if layer.Wx is not None:
        w_ok = w_ok and bool((layer.Wx[:, out] == 0).all())
    return dict(count=cnt_ok, neighbourhood=nb_ok, outside_zero=w_ok)


def check_opt_outside(layer, m, opt):
    out = ~layer.mask3(); ok = True
    for prm in [m.rnn.weight_hh_l0] + ([layer.Wx] if layer.Wx is not None else []):
        st = opt.state.get(prm)
        if st:
            ok = ok and bool((st["exp_avg"][..., out] == 0).all()) and bool((st["exp_avg_sq"][..., out] == 0).all())
    return ok


def rec_grads(m, layer, x, y, mask, local):
    """Gradients of the recurrent matrix (and of Wx) on one batch; the optimizer is not touched."""
    prm = [m.rnn.weight_hh_l0] + ([layer.Wx] if layer.Wx is not None else [])
    for q in list(m.parameters()) + layer.added_parameters():
        q.grad = None
    lg = forward(m, layer, x, local)[0]
    E.cross_entropy(lg[mask], y[mask]).backward()
    g = [q.grad.detach().clone() for q in prm]
    for q in list(m.parameters()) + layer.added_parameters():
        q.grad = None
    return g


def cosines(m, layer, items):
    x, y, mask = X.batch_tensor(items)
    gB, gS = rec_grads(m, layer, x, y, mask, False), rec_grads(m, layer, x, y, mask, True)
    cs = lambda a, b: float(T.nn.functional.cosine_similarity(a.flatten(), b.flatten(), 0))
    out = dict(recurrent_matrix=cs(gB[0], gS[0]))
    if len(gB) > 1:
        out["recurrent_matrix_and_Wx"] = cs(T.cat([g.flatten() for g in gB]), T.cat([g.flatten() for g in gS]))
    return out


# ---------------------------------------------------------------- one run
def receipt_path(out, config, rule, seed, dose):
    tag = "" if dose == DOSE else f"_dry{dose}"
    return Path(out) / f"e42_{config}_{rule}{tag}_{seed}.json"


def run(config, rule, seed, out, cmd, theta=None, dose=None, answers=False):
    N, ps, variant, wide, changing = CONFIGS[config]
    local = RULES[rule]
    dose = dose or DOSE
    assert dose % ROW == 0
    if changing:
        assert theta is not None, "a configuration with changing connections needs --theta (set once by check 10)"
    rp = receipt_path(out, config, rule, seed, dose)
    paths = dict(receipt=rp, weights=rp.with_suffix(".pt"), structure=rp.with_name(rp.stem + "_structure.json"),
                 small=rp.with_name(rp.stem + "_recurrent.pt"), answers=rp.with_name(rp.stem + "_answers.json"))
    R_.refuse_existing(*paths.values(), str(paths["weights"]) + ".tmp")
    Path(out).mkdir(parents=True, exist_ok=True)
    m, layer = build(config, seed)
    install(layer)
    opt = make_opt(m, layer)
    wc = weight_count(m, layer)
    assert wc == WEIGHT_COUNT[config], (config, wc)
    A, Rs, TA, LP = panels(seed)
    tp = [(t, E.split(t, .5, seed)[0]) for t in E.DEV]
    bt = E.batches(tp, seed, E.KMAX["M"]); dg = C26.Digest()
    rr = Random(f"e42|replace|{seed}")
    struct0 = structure(layer)
    rec = dict(experiment="E42", kind="birth", config=config, rule=rule, local=local, seed=seed, dose=dose, row_every=ROW,
               N=N, period_set=ps, variant=variant, wide_neighbourhood=wide, changing=changing, theta=theta if changing else None,
               batch=X.BATCH, kmax=E.KMAX["M"], weight_count=wc, params_module=sum(q.numel() for q in m.parameters()),
               added_parameters={k: (None if v is None else list(v.shape)) for k, v in (("a", layer.a), ("Wx", layer.Wx))},
               optimizer=opt_record(m, layer, opt),
               update="exp26_candidate.step's text with exp42_cells.forward(m, layer, x, local) in place of arch26.forward",
               data="exp11_cell.batches([(t, split(t, .5, seed)[0]) for t in DEV], seed, 12)",
               forward="exp42_cells.forward (the faster forward; check 9 ties it to e42_cells.forward)",
               structure_start_sha256=dict(mask=tsha(layer.mask), made=tsha(layer.made)),
               provenance=provenance(cmd), complete=False, curve=[])
    curve = rec["curve"]; cpu_up = cpu_sc = cpu_rep = 0.0; w0 = time.time(); losses = []; n_rep_since = 0; first = None

    def score_row(u):
        nonlocal cpu_sc, n_rep_since
        c = time.process_time()
        row = dict(step=u, **score_all(m, A, Rs, TA, LP))
        cpu_sc += time.process_time() - c
        lim = check_limit(m, layer)
        assert all(lim.values()), f"check 4 failed at row {u}: {lim}"
        row.update(replaced_since_previous_row=n_rep_since, limit=lim, cpu_updates=cpu_up, cpu_scoring=cpu_sc, cpu_replacement=cpu_rep,
                   weight_items=wc * X.BATCH * u, data_digest=dg.value(),
                   loss=(sum(losses) / len(losses)) if losses else None, finite_losses=all(v == v and abs(v) != float("inf") for v in losses),
                   wall=time.time() - w0)
        n_rep_since = 0
        curve.append(row)
        print(f"e42 {config} {rule} {seed} {u}: A {row['A']:.2f} R {row['R']:.2f} taught {row['taught']:.2f} len12 {row['len12']:.1f} "
              f"len40 {row['len40']:.1f} replaced {row['replaced_since_previous_row']} cpu {cpu_up:.0f}+{cpu_sc:.0f}s", flush=True)

    score_row(0)
    for u in range(1, dose + 1):
        items = next(bt)
        if first is None:
            first = items
            rec["cosine_first_batch_start"] = cosines(m, layer, first)
        dg.add(items)
        c = time.process_time(); losses.append(step(m, layer, opt, items, local)); cpu_up += time.process_time() - c
        if u % ROW == 0:
            if changing and u % L42.EVENT == 0:
                c = time.process_time()
                n_rep_since += L42.replace(layer, m, opt, u, theta, rr)
                assert check_opt_outside(layer, m, opt), f"optimizer averages not zero outside the mask after the event at {u}"
                cpu_rep += time.process_time() - c
            score_row(u); losses = []
            if u % 4096 == 0 or u == dose:
                rp.write_text(json.dumps(rec, indent=1))
    rec["cosine_first_batch_end"] = cosines(m, layer, first)
    sd = m.state_dict()
    full = dict(sd)
    full.update(mask=layer.mask.clone(), periods=layer.p.clone(), phases=layer.f.clone(), made=layer.made.clone())
    if layer.a is not None:
        full["a"] = layer.a.detach().clone()
    if layer.Wx is not None:
        full["Wx"] = layer.Wx.detach().clone()
    tmp = Path(str(paths["weights"]) + ".tmp"); T.save(full, tmp); os.replace(tmp, paths["weights"])
    small = {k: v for k, v in full.items() if k in ("rnn.weight_hh_l0", "a", "Wx", "mask", "made", "periods", "phases")}
    T.save(small, paths["small"])
    st = structure(layer)
    paths["structure"].write_text(json.dumps(dict(config=config, rule=rule, seed=seed, update=dose, start=struct0, end=st)))
    if answers:
        paths["answers"].write_text(json.dumps({str(k): E.respond(m, [p for p, _ in it]) for k, it in LP.items()}))
    rec.update(complete=True, weights=paths["weights"].name, weights_sha256=X16.sha_file(paths["weights"]),
               tensors_sha256=R_.tensors_sha(sd), small_file=paths["small"].name, structure_file=paths["structure"].name,
               structure_end_sha256=dict(mask=tsha(layer.mask), made=tsha(layer.made)),
               added_parameters_sha256={k: tsha(v) for k, v in (("a", layer.a), ("Wx", layer.Wx)) if v is not None},
               optimizer_steps=R_.opt_steps(opt), optimizer_read_back=opt_record(m, layer, opt), data_digest=dg.value(),
               cpu_updates=cpu_up, cpu_scoring=cpu_sc, cpu_replacement=cpu_rep, wall=time.time() - w0,
               processes_beside=os.environ.get("E42_BESIDE", "not recorded"))
    if answers:
        rec["answers_file"] = paths["answers"].name
    rp.write_text(json.dumps(rec, indent=1))
    print("e42 complete", rp.name, rec["tensors_sha256"][:16], flush=True)
    return rec


# ---------------------------------------------------------------- provenance
def provenance(cmd):
    p = C26.provenance(cmd)
    p["sources_sha256"] = {s: X16.sha_file(HERE / s) for s in SOURCES}
    p["runner_sha256"] = X16.sha_file(HERE / "exp42_cells.py")
    p["spec_sha256"] = X16.sha_file(SPEC) if SPEC.exists() else None
    return p


# ---------------------------------------------------------------- stage 0 checks
DRY2048 = {"S": "79618ab29390e6ecfdacd49a955afff7a70f282466dd92fd14e6ef3ba1ddd19f",
           "B": "adf933d7f2ab1a8f28f9bb1633f8a40da1ce1b3bad69b482e9d5d49263f23a0b"}
E26_DIGEST_4096 = {73129: "bbd71ab7ca27b72e", 93683: "65b1c7e06b17459a", 116089: "1df203fb8f299dc0"}
DIGEST_2048 = "4f622d94e0156993"
CHECK12_COUNTS = {   # E41 check 7(b): (A add, sub), (R add, sub), (taught add, sub); E30: length test at 12 and 40 letters (add)
    ("S", 73129): ((295, 293), (962, 955), (298, 298), (295, 288)), ("S", 93683): ((296, 293), (988, 979), (297, 298), (296, 291)),
    ("S", 116089): ((300, 299), (993, 996), (300, 296), (300, 299)), ("B", 73129): ((295, 296), (976, 987), (295, 297), (295, 293)),
    ("B", 93683): ((298, 294), (970, 968), (298, 298), (298, 296)), ("B", 116089): ((284, 279), (955, 895), (285, 288), (284, 277)),
}


def check1(seed, qual, cmd):
    """G is E26: the thirteen tensors bit-equal at the start; after 2,048 updates E26's dry-birth tensors_sha256 under both rules (seed 73129)."""
    out = dict(start={})
    for s in SEEDS:
        m, layer = build("G", s); m0 = arch26.new_net(s)
        a, b = m.state_dict(), m0.state_dict()
        out["start"][str(s)] = list(a) == list(b) and all(T.equal(a[k], b[k]) for k in a)
    out["runs"] = {}
    for rule in "SB":
        rec = run("G", rule, 73129, qual, cmd, dose=2048)
        out["runs"][rule] = dict(tensors_sha256=rec["tensors_sha256"], expected=DRY2048[rule], equal=rec["tensors_sha256"] == DRY2048[rule],
                                 digest_2048=rec["data_digest"]["sha256"], digest_equal=rec["data_digest"]["sha256"].startswith(DIGEST_2048))
    out["pass"] = all(out["start"].values()) and all(v["equal"] and v["digest_equal"] for v in out["runs"].values())
    return out


def check2(seed, qual, cmd):
    """The batch digest at row 4,096 is E26's. The stream depends on the seed only, not on the configuration or the rule; every full run's
    receipt carries its own digest at row 4,096, compared again by the tables."""
    tp = [(t, E.split(t, .5, seed)[0]) for t in E.DEV]
    bt = E.batches(tp, seed, E.KMAX["M"]); dg = C26.Digest()
    for _ in range(4096):
        dg.add(next(bt))
    d = dg.value()["sha256"]
    return dict(digest_4096=d, expected_prefix=E26_DIGEST_4096[seed], **{"pass": d.startswith(E26_DIGEST_4096[seed])})


def check3(seed, qual, cmd):
    """The specification file: the copy is byte-identical; its self-checks, run on the specification's own place, print what the
    coordinator's output has for checks 1 to 6, 8 and 9 (check 7 is timing)."""
    import subprocess
    copy = HERE / "e42_cells.py"
    same = copy.read_bytes() == SPEC.read_bytes()
    res = subprocess.run([sys.executable, "-I", str(SPEC)], capture_output=True, text=True)
    mine = res.stdout.splitlines()
    ref = (SPEC.parent / "OUTPUT_E42_SPEC_MAC.txt").read_text().splitlines()
    keep = lambda ls: [s for s in ls if not s.startswith("7.")]
    diff = [(a, b) for a, b in zip(keep(mine), keep(ref)) if a != b]
    runner_src = (HERE / "exp42_cells.py").read_text()
    only_imported = "import e42_cells as L42" in runner_src and "def replace(" not in runner_src and "class Layer" not in runner_src
    return dict(copy_byte_identical=same, spec_sha256=X16.sha_file(SPEC), copy_sha256=X16.sha_file(copy), returncode=res.returncode,
                stdout=mine, stderr=res.stderr[-2000:], lines_compared=len(keep(ref)), lines_equal=len(keep(mine)) == len(keep(ref)) and not diff,
                differences=diff, spec_only_imported_by_runner=only_imported,
                **{"pass": same and res.returncode == 0 and not diff and len(keep(mine)) == len(keep(ref)) and only_imported})


def first_batch(seed):
    tp = [(t, E.split(t, .5, seed)[0]) for t in E.DEV]
    return X.batch_tensor(next(E.batches(tp, seed, E.KMAX["M"])))


def all_grads(m, layer, fwd, x, y, mask, local):
    ps = [(n, q) for n, q in m.named_parameters()] + [(n, q) for n, q in (("a", layer.a), ("Wx", layer.Wx)) if q is not None]
    for _, q in ps:
        q.grad = None
    lg = fwd(m, layer, x, local)[0]
    E.cross_entropy(lg[mask], y[mask]).backward()
    g = {n: q.grad.detach().clone() for n, q in ps if q.grad is not None}
    for _, q in ps:
        q.grad = None
    return lg.detach(), g


def rel_diff(ga, gb):
    return {n: float((ga[n] - gb[n]).abs().max()) / max(float(gb[n].abs().max()), 1e-30) for n in gb}


def time_props(fwd, m, layer, x):
    with T.no_grad():
        hs = T.stack(fwd(m, layer, x, False)[1], 1)
        x2 = x.clone(); t0 = 9; x2[:, t0] = (x2[:, t0] + 1) % 256
        hs2 = T.stack(fwd(m, layer, x2, False)[1], 1)
    upd = T.stack([(t >= layer.f) & ((t - layer.f) % layer.p == 0) for t in range(x.shape[1])])
    changed = (hs[:, 1:] != hs[:, :-1]).any(0)
    return dict(no_change_before_changed_byte=bool(T.equal(hs[:, :t0], hs2[:, :t0])), changes_only_at_update_positions=bool((~changed | upd[1:]).all()),
                zero_before_phase=bool(all((hs[:, :int(layer.f[i]), i] == 0).all() for i in range(U))))


def perturb(layer, m, gen):
    """Moves the added parameters and the recurrent weights inside the mask away from their starting values, so that the equality of the two
    forwards is also tested where the weights a differ from 1/p and Wx from Wh (an addition to check 9, reported beside it)."""
    with T.no_grad():
        m3 = layer.mask3().float()
        m.rnn.weight_hh_l0.add_(0.05 * T.randn(m.rnn.weight_hh_l0.shape, generator=gen) * m3)
        if layer.a is not None:
            layer.a.mul_(1 + 0.5 * T.rand(layer.a.shape, generator=gen))
        if layer.Wx is not None:
            km = layer.kmask[1:].repeat(1, 3)[:, :, None] * m3[None]
            layer.Wx.add_(0.05 * T.randn(layer.Wx.shape, generator=gen) * km)


def check9(seed, qual, cmd):
    """The faster forward against the specification's, for every configuration, on the first batch of seed 73129: logits to 1e-6, every
    parameter gradient (added ones included) to 1e-5 of the tensor's largest entry, under both rules; check 5's time properties for the four
    variants with periods 1 to 16; then CPU-seconds per update and per scored row on an idle host, and a forecast for the worker's fifteen runs."""
    seed = 73129
    x, y, mask = first_batch(seed)
    out = dict(equality={}, perturbed={}, time={}, cost={}); ok = True
    for cfg in CONFIGS:
        for label, pert in (("start", False), ("perturbed", True)):
            m, layer = build(cfg, seed)
            if pert:
                perturb(layer, m, T.Generator().manual_seed(7))
            r = {}
            for rule, local in RULES.items():
                l_spec, g_spec = all_grads(m, layer, L42.forward, x, y, mask, local)
                l_fast, g_fast = all_grads(m, layer, forward, x, y, mask, local)
                rd = rel_diff(g_fast, g_spec)
                r[rule] = dict(logits_max_abs_diff=float((l_fast - l_spec).abs().max()), grads_max_rel=max(rd.values()),
                               grads_rel=rd, same_tensors=sorted(g_fast) == sorted(g_spec))
                r[rule]["pass"] = r[rule]["logits_max_abs_diff"] <= 1e-6 and r[rule]["grads_max_rel"] <= 1e-5 and r[rule]["same_tensors"]
            (out["equality"] if label == "start" else out["perturbed"])[cfg] = r
            if label == "start":
                ok = ok and all(v["pass"] for v in r.values())
    for v in L42.VARIANTS:
        m = arch26.new_net(seed); layer2 = L42.Layer(m, seed, N=10, period_set="16", variant=v)
        out["time"][f"P16 {v}"] = time_props(forward, m, layer2, x)
        ok = ok and all(out["time"][f"P16 {v}"].values())
    out["equality_pass"] = ok
    # cost, serial, on this host
    tp = [(t, E.split(t, .5, seed)[0]) for t in E.DEV]
    n_up = 48
    for cfg in CONFIGS:
        for rule in (("S", "B") if cfg in STAGE1 else ("S",)):
            m, layer = build(cfg, seed); install(layer); opt = make_opt(m, layer)
            bt = E.batches(tp, seed, E.KMAX["M"])
            step(m, layer, opt, next(bt), RULES[rule])         # warm-up, not timed
            c = time.process_time()
            for _ in range(n_up):
                step(m, layer, opt, next(bt), RULES[rule])
            per_u = (time.process_time() - c) / n_up
            A, Rs, TA, LP = panels(seed)
            c = time.process_time(); score_all(m, A, Rs, TA, LP); per_row = time.process_time() - c
            rows = DOSE // ROW + 1
            out["cost"][f"{cfg} {rule}"] = dict(cpu_per_update=per_u, cpu_per_scored_row=per_row,
                                                forecast_cpu_h=(DOSE * per_u + rows * per_row) / 3600)
            print(f"cost {cfg} {rule}: {per_u:.4f} s/update, {per_row:.2f} s/row, {out['cost'][f'{cfg} {rule}']['forecast_cpu_h']:.2f} CPU-h", flush=True)
    tot = sum(v["forecast_cpu_h"] for v in out["cost"].values())
    out["forecast_15_runs_cpu_h"] = tot
    out["forecast_limit_cpu_h"] = 30
    out["forecast_within_limit"] = tot <= 30
    out["pass"] = ok
    return out


def ref_forward(m, layer, x, local):
    """A second form, written cell by cell, without masks or buffers: for every position and every cell updated there, its gates from its
    own connections (index lists) over the steps of its period. Backpropagation only (check 6)."""
    assert not local
    u = m.book(x)
    z1 = T.tanh(linear(u, m.l1.weight, m.l1.bias))
    B, L = x.shape
    gi = linear(z1, m.rnn.weight_ih_l0, m.rnn.bias_ih_l0)
    Wh, bh = m.rnn.weight_hh_l0, m.rnn.bias_hh_l0
    reads = [[j for j in range(U) if bool(layer.mask[i, j])] for i in range(U)]       # its connections and itself
    hist = [T.zeros(B, U)]                                                             # hist[s + 1] = h(s); hist[0] = h(-1) = 0
    for t in range(L):
        hprev = hist[-1]; cells = []
        for i in range(U):
            p, f = layer.p_list[i], layer.f_list[i]
            if not (t >= f and (t - f) % p == 0):
                cells.append(hprev[:, i]); continue
            rows = [i, U + i, 2 * U + i]; cols = reads[i]
            steps = [k for k in range(p) if t - k >= 0]
            if layer.variant == "last":
                coef = {k: (1.0 if k == 0 else 0.0) for k in steps}
            elif p == 1:
                coef = {0: 1.0}
            elif layer.variant == "mean":
                coef = {k: 1.0 / p for k in steps}
            else:
                coef = {k: layer.a[k, i] for k in steps}
            Gs = sum(coef[k] * gi[:, t - k, rows] for k in steps)
            Hs = 0
            for k in steps:
                if layer.variant == "source":
                    Wk = Wh if k == 0 else layer.Wx[k - 1]
                    Hs = Hs + coef[k] * (hist[t - k][:, cols] @ Wk[rows][:, cols].T + bh[rows])
                else:
                    Hs = Hs + coef[k] * (hist[t - k][:, cols] @ Wh[rows][:, cols].T + bh[rows])
            r = T.sigmoid(Gs[:, 0] + Hs[:, 0]); z = T.sigmoid(Gs[:, 1] + Hs[:, 1]); n = T.tanh(Gs[:, 2] + r * Hs[:, 2])
            cells.append((1 - z) * n + z * hprev[:, i])
        hist.append(T.stack(cells, 1))
    Hs_ = T.stack(hist[1:], 1)
    z2 = T.tanh(linear(z1, m.l2.weight, m.l2.bias) + linear(Hs_, m.lm.weight, m.lm.bias))
    return linear(z2, m.l3.weight, m.l3.bias), hist[1:]


def check6(seed, qual, cmd):
    """Gradients against a second form, for P8, P8-source and F10 (seed 73129, first 16 items of the first batch: the reference is slow).
    Under backpropagation equal to 1e-5 of each tensor's largest entry; under the sweep they differ; outside the mask zero under both."""
    seed = 73129
    x, y, mask = first_batch(seed)
    x, y, mask = x[:16], y[:16], mask[:16]
    out = {}; ok = True
    for cfg in ("F10", "P8", "P8-source"):
        m, layer = build(cfg, seed)
        perturb(layer, m, T.Generator().manual_seed(11))           # so that a differs from 1/p and Wx from Wh
        _, g_ref = all_grads(m, layer, ref_forward, x, y, mask, False)
        _, g_B = all_grads(m, layer, forward, x, y, mask, False)
        _, g_S = all_grads(m, layer, forward, x, y, mask, True)
        _, g_specB = all_grads(m, layer, L42.forward, x, y, mask, False)
        rdB = rel_diff(g_B, g_ref); rdS = rel_diff(g_S, g_ref); rdSpec = rel_diff(g_specB, g_ref)
        outm = ~layer.mask3()
        zero = dict(B=bool((g_B["rnn.weight_hh_l0"][outm] == 0).all()), S=bool((g_S["rnn.weight_hh_l0"][outm] == 0).all()))
        if layer.Wx is not None:
            zero["B_Wx"] = bool((g_B["Wx"][:, outm] == 0).all()); zero["S_Wx"] = bool((g_S["Wx"][:, outm] == 0).all())
        r = dict(B_max_rel=max(rdB.values()), B_rel=rdB, spec_B_max_rel=max(rdSpec.values()), S_tensors_differing=[n for n, v in rdS.items() if v > 1e-5],
                 outside_mask_zero=zero, tensors=sorted(g_ref))
        r["pass"] = r["B_max_rel"] <= 1e-5 and r["spec_B_max_rel"] <= 1e-5 and bool(r["S_tensors_differing"]) and all(zero.values())
        ok = ok and r["pass"]; out[cfg] = r
        print(cfg, "check 6", r["pass"], r["B_max_rel"], flush=True)
    out["note"] = ("the reference reads a cell's own connections by index lists and loops over cells; it has no mask, no accumulator and no "
                   "padding; the weights were first moved off their starting values (perturb, generator 11), and the first 16 items are used")
    out["pass"] = ok
    return out


def check7(seed, qual, cmd, theta):
    """Replacement: adding the new connections changes no logit; the same command run twice gives the same tensors_sha256 and the same digests
    of the mask, of `made` and of the added parameters, with changing connections (P8, 1,024 updates, the chosen theta)."""
    seed = 73129
    x, y, mask = first_batch(seed)
    m, layer = build("A10", seed); opt = make_opt(m, layer); rr = Random(f"e42|replace|{seed}")
    tp = [(t, E.split(t, .5, seed)[0]) for t in E.DEV]; bt = E.batches(tp, seed, E.KMAX["M"])
    for _ in range(512):
        step(m, layer, opt, next(bt), True)
    before = layer.mask.clone()
    n = L42.replace(layer, m, opt, 512, theta, rr)
    new = layer.mask & ~before
    with T.no_grad():
        lg_after = forward(m, layer, x, False)[0]
        keep = layer.mask.clone(); layer.mask = layer.mask & ~new
        lg_without_new = forward(m, layer, x, False)[0]
        layer.mask = keep
    out = dict(event=dict(update=512, replaced=n, new_connections=int(new.sum()),
                          logits_equal_with_and_without_new=bool(T.equal(lg_after, lg_without_new)),
                          optimizer_zero_outside=check_opt_outside(layer, m, opt)))
    reps = []
    for i in range(2):
        rec = run("P8", "S", seed, Path(qual) / f"rep{i}", cmd, theta=theta, dose=1024)
        reps.append(dict(tensors_sha256=rec["tensors_sha256"], mask=rec["structure_end_sha256"]["mask"], made=rec["structure_end_sha256"]["made"],
                         added=rec["added_parameters_sha256"], replaced=[r["replaced_since_previous_row"] for r in rec["curve"]]))
    out["repeats"] = reps
    out["repeat_equal"] = reps[0] == reps[1]
    out["pass"] = out["event"]["logits_equal_with_and_without_new"] and out["event"]["optimizer_zero_outside"] and out["repeat_equal"] \
        and sum(reps[0]["replaced"]) > 0
    return out


def check10(seed, qual, cmd):
    """theta: three dry runs of A10 under the sweep, 4,096 updates, theta 0.1, 0.25, 0.5; the one whose mean share of connections replaced per
    event over the events at 768 to 4,096 is nearest 5%. No accuracy is used."""
    seed = 73129
    out = dict(runs={})
    for th in (0.1, 0.25, 0.5):
        rec = run("A10", "S", seed, Path(qual) / f"theta{th}", cmd, theta=th, dose=4096)
        ev = [(r["step"], r["replaced_since_previous_row"]) for r in rec["curve"] if 768 <= r["step"] <= 4096]
        shares = [n / (U * 10) for _, n in ev]
        out["runs"][str(th)] = dict(events=ev, mean_share=sum(shares) / len(shares))
    ms = {th: v["mean_share"] for th, v in out["runs"].items()}
    out["chosen_theta"] = float(min(ms, key=lambda th: abs(ms[th] - 0.05)))
    out["stop"] = all(v < 0.01 for v in ms.values()) or all(v > 0.30 for v in ms.values())
    out["pass"] = not out["stop"]
    return out


def check8(seed, qual, cmd, theta):
    """The reviewer's reader: dry runs of 2,048 updates of F10, P8 and P8-source (seed 73129, under the sweep) with their weights files and
    their answers on the length-test panel."""
    seed = 73129; out = {}
    for cfg in ("F10", "P8", "P8-source"):
        rec = run(cfg, "S", seed, qual, cmd, theta=theta, dose=2048, answers=True)
        out[cfg] = dict(receipt=receipt_path(qual, cfg, "S", seed, 2048).name, weights=rec["weights"], weights_sha256=rec["weights_sha256"],
                        answers=rec["answers_file"], len12=rec["curve"][-1]["len12"], len40=rec["curve"][-1]["len40"])
    out["pass"] = True
    return out


def check12(seed, qual, cmd):
    """The scorer: with G's layer and the six E26 births, E41's check 7(b) counts for A, R and taught, and E30's accuracies at 12 and 40."""
    out = dict(counts={}); worst = 0
    for rule in "SB":
        for fs in SEEDS:
            m, layer = build("G", fs)
            f_ = X.OUT / "e26" / f"e26_dev_{rule}_{fs}.pt"
            m.load_state_dict(T.load(f_)); install(layer)
            A, Rs, TA, LP = panels(fs)
            sc = score_all(m, A, Rs, TA, LP)
            got = ((sc["A_counts"]["add89"], sc["A_counts"]["sub89"]), (sc["R_counts"]["add89"], sc["R_counts"]["sub89"]),
                   (sc["taught_counts"]["add89"], sc["taught_counts"]["sub89"]), (sc["length_counts"]["12"], sc["length_counts"]["40"]))
            exp = CHECK12_COUNTS[(rule, fs)]
            d = max(abs(g - e) for gg, ee in zip(got, exp) for g, e in zip(gg, ee)); worst = max(worst, d)
            out["counts"][f_.name] = dict(file_sha256=X16.sha_file(f_), got=got, expected=exp, max_abs_diff=d, A=sc["A"], R=sc["R"],
                                          taught=sc["taught"], len12=sc["len12"], len40=sc["len40"])
            print(f_.name, got, exp, flush=True)
    out["max_abs_diff"] = worst
    out["verdict"] = "equal" if worst == 0 else "differs by one or two (reported)" if worst <= 2 else "FAIL"
    out["pass"] = worst <= 2
    return out


CHECKS = {"1": check1, "2": check2, "3": check3, "6": check6, "9": check9, "10": check10, "12": check12}
CHECKS_THETA = {"7": check7, "8": check8}


# ---------------------------------------------------------------- main
def main(argv):
    cmd = "python3 " + " ".join(argv)
    a = argv[1:]
    opt = lambda name, default: (a[a.index(name) + 1] if name in a else default)
    if a[0] == "run":
        config, rule, seed = a[1], a[2], int(a[3])
        assert config in CONFIGS and rule in RULES and seed in SEEDS
        th = opt("--theta", None); d = opt("--dose", None)
        run(config, rule, seed, Path(opt("--out", str(OUT))), cmd, theta=float(th) if th else None, dose=int(d) if d else None,
            answers="--answers" in a)
    elif a[0] == "check":
        n = a[1]; seed = int(a[2]) if len(a) > 2 and not a[2].startswith("--") else 73129
        qual = Path(opt("--qual", str(QUAL))); qual.mkdir(parents=True, exist_ok=True)
        p = qual / f"check{n}_{seed}.json"
        R_.refuse_existing(p)
        t0 = time.process_time(); w0 = time.time()
        if n in CHECKS_THETA:
            res = CHECKS_THETA[n](seed, qual / f"check{n}_{seed}_runs", cmd, float(opt("--theta", None)))
        else:
            res = CHECKS[n](seed, qual / f"check{n}_{seed}_runs", cmd)
        res.update(check=int(n), seed=seed, cpu=time.process_time() - t0, wall=time.time() - w0, provenance=provenance(cmd))
        p.write_text(json.dumps(res, indent=1))
        print(f"e42 check {n} seed {seed}:", "PASS" if res["pass"] else "FAIL", flush=True)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv)
