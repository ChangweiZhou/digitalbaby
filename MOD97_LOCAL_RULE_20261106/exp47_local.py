"""E47 runner (brief OPUS_TASK_E47_LOCALITY_AT_THE_INPUT_AND_THE_OUTPUT_20261116.md, revision 2): the present network whose recurrent layer is
a grid of cells, of which only one 16 x 16 patch (the interface) reads the byte and is read by the readout; every cell reads 10 cells of its
neighbourhood and itself, by index. Born on E26's task under the sweep and under backpropagation.

The layer is the specification file e47_local.py (a byte-identical copy of PROJECT_DOCUMENTS/COORD_CHECKS_20261116/e47_local_spec.py),
imported and never rewritten: its draws, its Local (structure and starting values), coef, counts, from_e42 and its IndexSign. E42's runner
exp42_cells.py (byte-identical from claude/gallant-dijkstra-i9zgsr) is imported for E42's data, optimizer, panels and scores. This file adds a
faster forward (check 3 ties it to e47_local.forward), the two diagnostics in scoring, the birth loop and the receipts.

    python3 exp47_local.py run <side> <arm> <S|B> <seed> [--dose N] [--out DIR]
    python3 exp47_local.py check <1|2|3|4|5|6|7> [--qual DIR]
    sides 16 32 64; arms one place scatter (16: one only, built from E42's F10 layer of the seed)

The faster forward (`forward`) computes what e47_local.forward computes, arranged differently:
  * at each position only the cells updated there are computed, in one group per period; a cell's state is otherwise carried unchanged
    (the specification computes every cell at every position and keeps the result only at update positions);
  * a cell of period p updated at t has gates from sum_{k < min(p, t+1)} c[k] (W h(t-1-k)[sources] + b): the step weights are applied to
    the sources' states first (11 multiply-adds a step) and the weights after (33), which is the same sum. Under the sweep a custom function
    (SweepCross) returns: to the weights W and biases b the exact gradient; to the step weights c the exact gradient g . (W h_k + b); to
    the sources' states c[k] times the error crossed by sign(W) * scale, the scale being the specification's (mean |W| over all cells).
    That is IndexSign followed by the specification's sum over steps;
  * the input side of a far cell is its bias times the sum of its step weights over the steps that exist; an interface cell has period 1.
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
import exp26_candidate as C26        # noqa: E402  first harness import (environment gate)

import torch as T                    # noqa: E402

import arch26                        # noqa: E402
import e42_cells as L42              # noqa: E402  E42's specification file (for the 16-a-side start and check 2)
import e47_local as L47              # noqa: E402  the specification file, imported, not rewritten
import exp42_cells as R42            # noqa: E402  E42's runner: data, optimizer, panels, scores, check 12's counts

R_, X16, E, X = C26.R, C26.X16, C26.E, C26.X
T.set_num_threads(1)

SEEDS = (73129, 93683, 116089)
DOSE = 24576
ROW = 256
RULES = {"S": True, "B": False}
SIDES = (16, 32, 64)
ARMS = ("one", "place", "scatter")
N = 10
OUT = X.OUT / "e47"
QUAL = X.OUT / "e47_qual"
KDIR = HERE.parent / "PROJECT_DOCUMENTS" / "COORD_CHECKS_20261116"
SPEC = KDIR / "e47_local_spec.py"
SOURCES = ("exp47_local.py", "e47_local.py", "exp42_cells.py", "e42_cells.py") + C26.SOURCES
WEIGHT_COUNT = {(16, "one"): 935953, (32, "one"): 965905, (32, "place"): 971665, (32, "scatter"): 971665,
                (64, "one"): 1085713, (64, "place"): 1114513, (64, "scatter"): 1114513}
E42_SOURCE = "origin/claude/gallant-dijkstra-i9zgsr at 0de93296 (exp42_cells.py, e42_cells.py: the blobs of 534df8f0)"


# ---------------------------------------------------------------- the faster forward
class SweepCross(T.autograd.Function):
    """out[b, g, i] = sum_p c[p, i] (sum_k W[g, i, k] hk[p, b, i, k] + bias[g, i]). Back: weights, biases and c exact; the sources' states
    receive c[p, i] * (g . S), S = sign(W) * scale."""

    @staticmethod
    def forward(ctx, hk, W, bias, c, S):
        hbar = T.einsum("pbik,pi->bik", hk, c)
        csum = c.sum(0)
        ctx.save_for_backward(hk, W, bias, c, S, hbar)
        return T.einsum("bik,gik->bgi", hbar, W) + (bias * csum[None])[None]

    @staticmethod
    def backward(ctx, g):
        hk, W, bias, c, S, hbar = ctx.saved_tensors
        csum = c.sum(0)
        gW = T.einsum("bgi,bik->gik", g, hbar)
        gb = g.sum(0) * csum[None]
        v = T.einsum("bgi,gik->bik", g, W)                                    # the exact error into the summed states
        gc = T.einsum("pbik,bik->pi", hk, v) + (g * bias[None]).sum((0, 1))[None]
        vs = T.einsum("bgi,gik->bik", g, S)                                   # the sweep's error into the sources
        ghk = c[:, None, :, None] * vs[None]
        return ghk, gW, gb, gc, None


def plan(L, Lx):
    """For each position t < Lx: groups (P, cells, ifpos, farpos) of the cells updated there, one group per period; P = steps that exist.
    The list is cached on the layer and grown as longer inputs come."""
    out = L.__dict__.setdefault("_e47_plan", [])
    if len(out) >= Lx:
        return out
    assert bool((L.p[L.ifx] == 1).all()), "the faster forward assumes period 1 at the interface"
    isif = T.zeros(L.C, dtype=T.bool); isif[L.ifx] = True
    ifpos_of = T.full((L.C,), -1, dtype=T.long); ifpos_of[L.ifx] = T.arange(len(L.ifx))
    farpos_of = T.full((L.C,), -1, dtype=T.long)
    if len(L.farx):
        farpos_of[L.farx] = T.arange(len(L.farx))
    for t in range(len(out), Lx):
        upd = (t >= L.f) & ((t - L.f) % L.p == 0)
        groups = []
        for q in sorted(set(L.p.tolist())):
            sel = upd & (L.p == q)
            if not bool(sel.any()):
                continue
            ci = T.nonzero(sel & isif).flatten(); cf = T.nonzero(sel & ~isif).flatten()
            cells = T.cat([ci, cf])
            groups.append(dict(P=min(q, t + 1), cells=cells, ifpos=ifpos_of[ci], farpos=farpos_of[cf], n_if=len(ci), idx=L.idx[cells]))
        allc = T.cat([g_["cells"] for g_ in groups])
        out.append((groups, allc))
    return out


def forward(m, L, x, local, silence=False, far_off=False, far_mean=None, poke=None, state=None):
    """Same values as e47_local.forward (check 3). Returns logits and the list of states after each position.
    poke=(t, values (B, far cells)): after position t the far cells' states are set by hand to these values (check 4 only).
    state: the dict returned in L._e47_state by a previous call on the preceding positions of the same sequences (greedy decoding without
    recomputing the prefix; scoring only). x then holds the following positions, and t counts from the start of the sequence."""
    u = m.book(x)
    z1 = T.tanh(L47.lin(u, m.l1, local))
    B, Lx = x.shape
    gi_if = L47.lin(z1, arch26.Mat(m.rnn.weight_ih_l0, m.rnn.bias_ih_l0), local).reshape(B, Lx, 3, L47.PATCH * L47.PATCH)
    c = L.coef()                                                             # (K, C), with the graph to a
    if local:
        Sfull = L.Wc.detach().sign() * L.Wc.detach().abs().mean()
    t0 = 0 if state is None else state["t"]
    hist = [T.zeros(B, L.C)] if state is None else list(state["hist"])      # hist[s + 1] = h(s); hist[0] = h(-1) = 0
    h = hist[-1]
    hs = []
    pl = plan(L, t0 + Lx)
    for t in range(t0, t0 + Lx):
        groups, allc = pl[t]
        tl = t - t0                                                          # the position within x
        hn_parts = []
        for g_ in groups:
            P, cells, idx = g_["P"], g_["cells"], g_["idx"]
            cg = c[:P, cells]                                                # (P, n)
            Wg, bg = L.Wc[:, cells], L.bc[:, cells]
            hk = T.stack([hist[t - k][:, idx] for k in range(P)])           # (P, B, n, N + 1): h(t-1-k) at the sources
            if local:
                H = SweepCross.apply(hk, Wg, bg, cg, Sfull[:, cells])
            else:
                hbar = T.einsum("pbik,pi->bik", hk, cg)
                H = T.einsum("bik,gik->bgi", hbar, Wg) + (bg * cg.sum(0)[None])[None]
            parts = []
            if g_["n_if"]:
                parts.append(gi_if[:, tl][:, :, g_["ifpos"]])                # period 1: c[0] = 1
            if len(g_["farpos"]):
                nf = g_["n_if"]
                parts.append((L.bfar[:, g_["farpos"]] * cg[:, nf:].sum(0)[None])[None].expand(B, -1, -1))
            Gs = parts[0] if len(parts) == 1 else T.cat(parts, 2)
            r = T.sigmoid(Gs[:, 0] + H[:, 0]); z = T.sigmoid(Gs[:, 1] + H[:, 1]); n = T.tanh(Gs[:, 2] + r * H[:, 2])
            hn_parts.append((1 - z) * n + z * h[:, cells])
        hn = hn_parts[0] if len(hn_parts) == 1 else T.cat(hn_parts, 1)
        h = h.index_copy(1, allc, hn)
        if far_off and len(L.farx):
            h = h.index_fill(1, L.farx, 0.0)
        if far_mean is not None and len(L.farx):
            h = h.index_copy(1, L.farx, far_mean[t][None].expand(B, -1))
        if poke is not None and poke[0] == t:
            h = h.index_copy(1, L.farx, poke[1])
        hist.append(h); hs.append(h)
    if not T.is_grad_enabled():
        L._e47_state = dict(t=t0 + Lx, hist=hist)                           # the whole history: hist[t - k] stays an absolute index
    Hs = T.stack(hs, 1)[:, :, L.ifx]
    if silence:
        Hs = T.zeros_like(Hs)
    z2 = T.tanh(L47.lin(z1, m.l2, local) + L47.lin(Hs, m.lm, local))
    return L47.lin(z2, m.l3, local), hs


# ---------------------------------------------------------------- network, layer, optimizer
def build(side, arm, seed):
    m = arch26.new_net(seed)
    if side == 16:
        assert arm == "one"
        lay42 = L42.Layer(m, seed, N=N, period_set="1", variant="step", wide=False)      # E42's F10 (exp42_cells.CONFIGS["F10"])
        struct, _ = L47.from_e42(m, lay42)
        L = L47.Local(m, seed, 16, N, structure=struct)
    else:
        L = L47.Local(m, seed, side, N, arm)
    L.periods_arm = arm
    return m, L


def make_opt(m, L):
    """E26's optimizer (C26.make_opt, its two groups asserted by name); the module's rnn.weight_hh_l0 and rnn.bias_hh_l0 are taken out (they
    take no part), and Wc, bc, bfar, a join the decayed group."""
    opt = C26.make_opt(m)
    g = [g_ for g_ in opt.param_groups if g_["weight_decay"] == C26.WD]
    assert len(g) == 1
    drop = {id(m.rnn.weight_hh_l0), id(m.rnn.bias_hh_l0)}
    g[0]["params"] = [q for q in g[0]["params"] if id(q) not in drop]
    g[0]["params"].extend(L.parameters())
    return opt


def names(m, L):
    d = {id(p): n for n, p in m.named_parameters()}
    for n, q in (("Wc", L.Wc), ("bc", L.bc), ("bfar", L.bfar), ("a", L.a)):
        if q is not None:
            d[id(q)] = n
    return d


def opt_record(m, L, opt):
    nm = names(m, L)
    return dict(name="AdamW", lr=C26.LR, weight_decay=C26.WD, betas=list(C26.BETAS),
                groups=[dict(weight_decay=g["weight_decay"], lr=g["lr"], betas=list(g["betas"]), params=[nm[id(p)] for p in g["params"]])
                        for g in opt.param_groups])


E_RESPOND = E.respond                    # exp11_cell.respond as imported (whole sequence again for every answer byte)


def install(L, incremental=True):
    """arch26.install(), then exp11_cell.forward replaced by a function of the harness's signature that runs this layer, and
    exp11_cell.respond (used by exp11_cell.score, hence by E42's score_all) by this file's greedy decoding, incremental unless told not."""
    arch26.install()

    def fwd(m, x, local, through_time, keep=False, silence=False):
        lg, hs = forward(m, L, x, local, silence=silence)
        return lg, dict(h=hs)
    E.forward = fwd

    def respond(m, prompts, store=None, cap=4, silence=False):
        assert store is None and not silence
        return generate(m, L, prompts, cap)[0]
    E.respond = respond if incremental else E_RESPOND


def step(m, L, opt, items, local):
    """exp26_candidate.step's text with this file's forward in place of arch26.forward."""
    x, y, mask = X.batch_tensor(items)
    logits = forward(m, L, x, local)[0]
    loss = E.cross_entropy(logits[mask], y[mask])
    opt.zero_grad()
    loss.backward()
    opt.step()
    return float(loss.detach())


def weights(m, L):
    """The network's weights as saved: the module's tensors except rnn.weight_hh_l0 and rnn.bias_hh_l0, and the layer's."""
    sd = {k: v.detach().clone() for k, v in m.state_dict().items() if k not in ("rnn.weight_hh_l0", "rnn.bias_hh_l0")}
    sd.update(Wc=L.Wc.detach().clone(), bc=L.bc.detach().clone(), bfar=L.bfar.detach().clone())
    if L.a is not None:
        sd["a"] = L.a.detach().clone()
    return sd


def save_weights(m, L, path):
    full = weights(m, L)
    full.update(sources=L.idx.clone(), periods=L.p.clone(), phases=L.f.clone(), interface=L.ifx.clone(), far=L.farx.clone())
    tmp = Path(str(path) + ".tmp"); T.save(full, tmp); os.replace(tmp, path)
    return R_.tensors_sha(weights(m, L))


# ---------------------------------------------------------------- scoring with the two diagnostics
@T.no_grad()
def generate(m, L, prompts, cap=4, far_off=False, far_mean=None, full=False, incremental=True):
    """exp11_cell.respond's greedy decoding (same grouping, chunks of 256, stop at EOS) with the diagnostics passed to this file's forward.
    incremental=True: the prompt is run once and each answer byte adds one position to the carried state, instead of running the whole
    sequence again for every byte as exp11_cell.respond does (the same values by causality; check 3 compares the answers given).
    full=True runs all cap steps (for the means) and also returns the final sequences."""
    out = [None] * len(prompts); seqs = [None] * len(prompts)
    groups = {}
    for i, p in enumerate(prompts):
        groups.setdefault(len(p), []).append(i)
    for _, idx in groups.items():
        for j in range(0, len(idx), 256):
            ids = idx[j:j + 256]
            p = T.tensor([[E.BOS, *prompts[i]] for i in ids])
            gen = [[] for _ in ids]; done = [False] * len(ids)
            st = None
            for _ in range(cap):
                if not incremental:
                    lg = forward(m, L, p, False, far_off=far_off, far_mean=far_mean)[0][:, -1]
                elif st is None:
                    lg = forward(m, L, p, False, far_off=far_off, far_mean=far_mean)[0][:, -1]; st = L._e47_state
                else:
                    lg = forward(m, L, p[:, -1:], False, far_off=far_off, far_mean=far_mean, state=st)[0][:, -1]; st = L._e47_state
                yy = lg.argmax(1)
                for q in range(len(ids)):
                    if not done[q]:
                        gen[q].append(int(yy[q]))
                        if int(yy[q]) == E.EOS:
                            done[q] = True
                p = T.cat([p, yy[:, None]], 1)
                if all(done) and not full:
                    break
            for q, i in enumerate(ids):
                out[i] = gen[q]; seqs[i] = p[q]
    return out, seqs


@T.no_grad()
def far_means(m, L, items):
    """Each far cell's mean state after each position over the scored items, from the plain forward on the items' prompts followed by the
    network's own plain answers (greedy, all 4 steps), so that every position the held run can reach has a mean. One length per scored set."""
    prompts = [p for p, _ in items]
    assert len({len(p) for p in prompts}) == 1, "a scored set has items of one length"
    _, seqs = generate(m, L, prompts, full=True)
    S = T.stack(seqs)
    hs = []
    for j in range(0, len(S), 256):
        hs.append(T.stack(forward(m, L, S[j:j + 256], False)[1], 1)[:, :, L.farx])
    return T.cat(hs).mean(0)                                                # (positions, far cells)


def score_diag(m, L, items, mode):
    prompts = [p for p, _ in items]
    if mode == "off":
        outs, _ = generate(m, L, prompts, far_off=True)
    else:
        outs, _ = generate(m, L, prompts, far_mean=far_means(m, L, items))
    return sum(o == t for o, (_, t) in zip(outs, items)), outs


def score_row_all(m, L, A, Rs, TA, LP, answers=False):
    sc = R42.score_all(m, A, Rs, TA, LP)                                    # E42's scores, plain (E.score through the installed forward)
    diag = {}
    ans = {}
    for mode in ("off", "mean"):
        a = {n: score_diag(m, L, it, mode)[0] for n, it in A.items()}
        r = {n: score_diag(m, L, it, mode)[0] for n, it in Rs.items()}
        ln = {}
        for k, it in LP.items():
            ln[str(k)], outs = score_diag(m, L, it, mode)
            if answers:
                ans.setdefault(mode, {})[str(k)] = outs
        diag[mode] = dict(A_counts=a, R_counts=r, length_counts=ln, A=100 * (a["add89"] + a["sub89"]) / 600,
                          R=100 * (r["add89"] + r["sub89"]) / 2000, len12=100 * ln["12"] / 300, len40=100 * ln["40"] / 300)
    if answers:
        ans["plain"] = {str(k): E.respond(m, [p for p, _ in it]) for k, it in LP.items()}
    if not len(L.farx):
        diag = None
    return sc, diag, ans


# ---------------------------------------------------------------- measures inside a run
def magnitudes(L):
    w = L.Wc.detach().abs()
    return dict(interface=float(w[:, L.ifx].mean()), far=float(w[:, L.farx].mean()) if len(L.farx) else None, scale=float(w.mean()))


def wc_grad(m, L, x, y, mask, local):
    ps = list(m.parameters()) + L.parameters()
    for q in ps:
        q.grad = None
    lg = forward(m, L, x, local)[0]
    E.cross_entropy(lg[mask], y[mask]).backward()
    g = L.Wc.grad.detach().clone()
    for q in ps:
        q.grad = None
    return g


def cosines(m, L, items):
    x, y, mask = X.batch_tensor(items)
    gB, gS = wc_grad(m, L, x, y, mask, False), wc_grad(m, L, x, y, mask, True)
    cs = lambda a, b: float(T.nn.functional.cosine_similarity(a.flatten(), b.flatten(), 0))
    out = dict(all=cs(gB, gS), interface=cs(gB[:, L.ifx], gS[:, L.ifx]))
    if len(L.farx):
        out.update(far=cs(gB[:, L.farx], gS[:, L.farx]),
                   far_nonzero_share_B=float((gB[:, L.farx] != 0).float().mean()), far_nonzero_share_S=float((gS[:, L.farx] != 0).float().mean()))
    return out


def tsha(t):
    t = t.detach().contiguous().cpu()
    return hashlib.sha256(str(t.dtype).encode() + str(tuple(t.shape)).encode() + t.numpy().tobytes()).hexdigest()


# ---------------------------------------------------------------- one run
def receipt_path(out, side, arm, rule, seed, dose):
    tag = "" if dose == DOSE else f"_dry{dose}"
    return Path(out) / f"e47_{side}_{arm}_{rule}{tag}_{seed}.json"


def run(side, arm, rule, seed, out, cmd, dose=None, verbose=True):
    local = RULES[rule]
    dose = dose or DOSE
    assert dose % ROW == 0
    rp = receipt_path(out, side, arm, rule, seed, dose)
    paths = dict(receipt=rp, start=rp.with_name(rp.stem + "_start.pt"), weights=rp.with_suffix(".pt"), answers=rp.with_name(rp.stem + "_answers.json"))
    R_.refuse_existing(*paths.values(), str(paths["weights"]) + ".tmp", str(paths["start"]) + ".tmp")
    Path(out).mkdir(parents=True, exist_ok=True)
    m, L = build(side, arm, seed)
    install(L)
    opt = make_opt(m, L)
    cnt = L.counts(m)
    assert cnt["weights"] == WEIGHT_COUNT[(side, arm)], (side, arm, cnt)
    A, Rs, TA, LP = R42.panels(seed)
    tp = [(t, E.split(t, .5, seed)[0]) for t in E.DEV]
    bt = E.batches(tp, seed, E.KMAX["M"]); dg = C26.Digest()
    start_sha = save_weights(m, L, paths["start"])
    rec = dict(experiment="E47", kind="birth", side=side, cells=L.C, arm=arm, rule=rule, local=local, seed=seed, dose=dose, row_every=ROW, N=N,
               interface_cells=len(L.ifx), far_cells=len(L.farx), cells_by_period={str(q): int((L.p == q).sum()) for q in sorted(set(L.p.tolist()))},
               start="E42's F10 layer of the seed (from_e42, Local(..., structure=...))" if side == 16 else
                     "Local(m, seed, side, 10, arm): recurrent weights and biases from T.Generator().manual_seed(seed * 1000 + side)",
               batch=X.BATCH, kmax=E.KMAX["M"], weight_count=cnt["weights"], multiply_adds=cnt["multiply_adds"],
               multiply_adds_with_periods=cnt["multiply_adds_with_periods"], optimizer=opt_record(m, L, opt),
               update="exp26_candidate.step's text with exp47_local.forward(m, L, x, local) in place of arch26.forward",
               data="exp11_cell.batches([(t, split(t, .5, seed)[0]) for t in DEV], seed, 12)",
               forward="exp47_local.forward (only the updated cells; check 3 ties it to e47_local.forward)",
               scores="exp42_cells.score_all (plain); with the far cells at zero and at their mean: A, R, length test at 12 and 40",
               structure_sha256=dict(sources=tsha(L.idx), periods=tsha(L.p), phases=tsha(L.f)),
               start_weights=paths["start"].name, start_tensors_sha256=start_sha, start_file_sha256=X16.sha_file(paths["start"]),
               provenance=provenance(cmd), complete=False, curve=[])
    curve = rec["curve"]; cpu_up = cpu_sc = 0.0; w0 = time.time(); losses = []; first = None

    def score_row(u, last=False):
        nonlocal cpu_sc
        c = time.process_time()
        sc, diag, ans = score_row_all(m, L, A, Rs, TA, LP, answers=last)
        dt = time.process_time() - c; cpu_sc += dt
        row = dict(step=u, **sc, far_off=diag["off"] if diag else None, far_mean=diag["mean"] if diag else None, magnitudes=magnitudes(L),
                   cpu_updates=cpu_up, cpu_scoring=cpu_sc, cpu_this_row=dt, weight_items=cnt["weights"] * X.BATCH * u,
                   multiply_add_items=cnt["multiply_adds_with_periods"] * u, data_digest=dg.value(),
                   loss=(sum(losses) / len(losses)) if losses else None, finite_losses=all(v == v and abs(v) != float("inf") for v in losses),
                   wall=time.time() - w0)
        curve.append(row)
        if verbose:
            d = (f" | off A {diag['off']['A']:.1f} R {diag['off']['R']:.1f} 40 {diag['off']['len40']:.1f}"
                 f" | mean A {diag['mean']['A']:.1f} R {diag['mean']['R']:.1f} 40 {diag['mean']['len40']:.1f}") if diag else ""
            print(f"e47 {side} {arm} {rule} {seed} {u}: A {row['A']:.2f} R {row['R']:.2f} taught {row['taught']:.2f} len12 {row['len12']:.1f} "
                  f"len40 {row['len40']:.1f}{d} cpu {cpu_up:.0f}+{cpu_sc:.0f}s", flush=True)
        return ans

    score_row(0)
    for u in range(1, dose + 1):
        items = next(bt)
        if first is None:
            first = items
            rec["cosine_first_batch_start"] = cosines(m, L, first)
        dg.add(items)
        c = time.process_time(); losses.append(step(m, L, opt, items, local)); cpu_up += time.process_time() - c
        if u % ROW == 0:
            ans = score_row(u, last=(u == dose)); losses = []
            if u % 4096 == 0 or u == dose:
                rp.write_text(json.dumps(rec, indent=1))
    rec["cosine_first_batch_end"] = cosines(m, L, first)
    sha = save_weights(m, L, paths["weights"])
    paths["answers"].write_text(json.dumps(dict(update=dose, panel="length test (add89, code p), 300 items at 12 and 40 letters",
                                                prompts={str(k): [p for p, _ in it] for k, it in LP.items()},
                                                targets={str(k): [t for _, t in it] for k, it in LP.items()}, answers=ans)))
    rec.update(complete=True, weights=paths["weights"].name, weights_sha256=X16.sha_file(paths["weights"]), tensors_sha256=sha,
               answers_file=paths["answers"].name, answers_sha256=X16.sha_file(paths["answers"]),
               module_recurrent_untouched=dict(weight_hh_grad_none=m.rnn.weight_hh_l0.grad is None, bias_hh_grad_none=m.rnn.bias_hh_l0.grad is None),
               optimizer_steps=R_.opt_steps(opt), optimizer_read_back=opt_record(m, L, opt), data_digest=dg.value(),
               cpu_updates=cpu_up, cpu_scoring=cpu_sc, cpu_per_update=cpu_up / dose, wall=time.time() - w0,
               processes_beside=os.environ.get("E47_BESIDE", "not recorded"))
    rp.write_text(json.dumps(rec, indent=1))
    print("e47 complete", rp.name, sha[:16], flush=True)
    return rec


# ---------------------------------------------------------------- provenance
def provenance(cmd):
    p = C26.provenance(cmd)
    p["sources_sha256"] = {s: X16.sha_file(HERE / s) for s in SOURCES}
    p["runner_sha256"] = X16.sha_file(HERE / "exp47_local.py")
    p["spec_sha256"] = X16.sha_file(SPEC) if SPEC.exists() else None
    p["e42_runner_source"] = E42_SOURCE
    p["torch"] = T.__version__; p["python"] = sys.version.split()[0]
    return p


# ---------------------------------------------------------------- stage 0 checks
def first_batch(seed):
    return R42.first_batch(seed)


def all_grads(m, L, fwd, x, y, mask, local, **kw):
    ps = [(n, q) for n, q in m.named_parameters()] + [(n, q) for n, q in (("Wc", L.Wc), ("bc", L.bc), ("bfar", L.bfar), ("a", L.a))
                                                       if q is not None and q.numel()]
    for _, q in ps:
        q.grad = None
    lg = fwd(m, L, x, local, **kw)[0]
    E.cross_entropy(lg[mask], y[mask]).backward()
    g = {n: q.grad.detach().clone() for n, q in ps if q.grad is not None}
    for _, q in ps:
        q.grad = None
    return lg.detach(), g


def rel_diff(ga, gb):
    return {n: float((ga[n] - gb[n]).abs().max()) / max(float(gb[n].abs().max()), 1e-30) for n in gb}


def check1(qual, cmd):
    """The specification file: the copy is byte-identical; its self-checks, run on the specification's own place, print what the
    coordinator's output has for checks 1 to 4 (check 5 is timing)."""
    import subprocess
    copy = HERE / "e47_local.py"
    same = copy.read_bytes() == SPEC.read_bytes()
    env = {k: v for k, v in os.environ.items() if not k.startswith(("E11_", "E16_", "EXP45_"))}
    res = subprocess.run([sys.executable, "-I", str(SPEC)], capture_output=True, text=True, env=env, cwd=str(KDIR))
    mine = res.stdout.splitlines()
    ref = (KDIR / "OUTPUT_E47_SPEC_MAC.txt").read_text().splitlines()
    import re
    keep = lambda ls: [s for s in ls if not s.startswith("5.")]
    diff = [(a, b) for a, b in zip(keep(mine), keep(ref)) if a != b]
    # Lines of self-check 1 print rounding residues, which differ between hosts; they are compared by their text with the numbers taken
    # out, and the numbers against the tolerances (logits 1e-6, gradients 1e-5). Every other line must be identical.
    strip = lambda s: re.sub(r"[0-9.]+e[-+][0-9]+", "#", s)
    nums_ok = lambda s: all((float(v) <= 1e-6) if i == 0 else (float(v) <= 1e-5) for i, v in enumerate(re.findall(r"[0-9.]+e[-+][0-9]+", s)))
    unexplained = [(a, b) for a, b in diff if not (a.startswith("1.") and b.startswith("1.") and strip(a) == strip(b) and nums_ok(a))]
    src = (HERE / "exp47_local.py").read_text()
    only_imported = "import e47_local as L47" in src and ("class " + "Local") not in src and ("def " + "draws(") not in src
    return dict(copy_byte_identical=same, spec_sha256=X16.sha_file(SPEC), copy_sha256=X16.sha_file(copy), returncode=res.returncode,
                stdout=mine, stderr=res.stderr[-2000:], lines_compared=len(keep(ref)), lines_identical=len(keep(mine)) == len(keep(ref)) and not diff,
                differences=diff, differences_only_rounding_residues_within_tolerance=not unexplained, unexplained_differences=unexplained,
                spec_only_imported_by_runner=only_imported,
                **{"pass": same and res.returncode == 0 and not unexplained and len(keep(mine)) == len(keep(ref)) and only_imported})


def check2(qual, cmd):
    """At 16 a side the runner's layer is E42's F10: first batch of seed 73129, logits to 1e-6 and every gradient mapped by index to 1e-5 of
    the tensor's largest entry under each rule; losses of the first 16 updates of both runners; a dry run of 2,048 updates under the sweep
    beside E42's e42_F10_S_dry2048_73129."""
    seed = 73129
    x, y, mask = first_batch(seed)
    out = dict(first_batch={}, losses16={}); ok = True
    for rule, local in RULES.items():
        m42, lay42 = R42.build("F10", seed)
        m, L = build(16, "one", seed)
        sd42 = m42.state_dict(); sd = m.state_dict()
        same_start = all(T.equal(sd[k], sd42[k]) for k in sd if k not in ("rnn.weight_hh_l0", "rnn.bias_hh_l0"))
        _, idx = L47.from_e42(m42, lay42)
        same_start = same_start and bool(T.equal(L.idx, idx))
        lg42, g42 = R42.all_grads(m42, lay42, R42.forward, x, y, mask, local)
        lg, g = all_grads(m, L, forward, x, y, mask, local)
        gW42 = T.stack([g42["rnn.weight_hh_l0"][q * 256:(q + 1) * 256].gather(1, idx) for q in range(3)])
        rd = {n: v for n, v in rel_diff({n: g[n] for n in g42 if n in g and n not in ("rnn.weight_hh_l0", "rnn.bias_hh_l0")},
                                         {n: g42[n] for n in g42 if n in g and n not in ("rnn.weight_hh_l0", "rnn.bias_hh_l0")}).items()}
        rd["Wc vs rnn.weight_hh_l0 (by index)"] = rel_diff({"w": g["Wc"]}, {"w": gW42})["w"]
        rd["bc vs rnn.bias_hh_l0"] = rel_diff({"b": g["bc"]}, {"b": g42["rnn.bias_hh_l0"].reshape(3, 256)})["b"]
        outside = g42["rnn.weight_hh_l0"].clone()
        for q in range(3):
            outside[q * 256:(q + 1) * 256].scatter_(1, idx, 0.0)
        r = dict(same_start=same_start, logits_max_abs_diff=float((lg - lg42).abs().max()), grads_rel=rd, grads_max_rel=max(rd.values()),
                 e42_grad_outside_connections_zero=bool((outside == 0).all()))
        r["pass"] = same_start and r["logits_max_abs_diff"] <= 1e-6 and r["grads_max_rel"] <= 1e-5
        ok = ok and r["pass"]; out["first_batch"][rule] = r
        # the first 16 updates of both runners
        R42.install(lay42); opt42 = R42.make_opt(m42, lay42)
        install(L); opt = make_opt(m, L)
        tp = [(t, E.split(t, .5, seed)[0]) for t in E.DEV]
        b1 = E.batches(tp, seed, E.KMAX["M"]); b2 = E.batches(tp, seed, E.KMAX["M"])
        l42, l47 = [], []
        for _ in range(16):
            it1 = next(b1); it2 = next(b2)
            assert it1 == it2
            R42.install(lay42); l42.append(R42.step(m42, lay42, opt42, it1, local))
            install(L); l47.append(step(m, L, opt, it2, local))
        d = [abs(a - b) for a, b in zip(l47, l42)]
        out["losses16"][rule] = dict(e42=l42, e47=l47, abs_diff=d, first_within_1e6=d[0] <= 1e-6, sixteenth_within_1e3=d[15] <= 1e-3)
        print(f"check 2 {rule}: logits {r['logits_max_abs_diff']:.1e} grads {r['grads_max_rel']:.1e} loss diff first {d[0]:.1e} 16th {d[15]:.1e}", flush=True)
    out["first_batch_pass"] = ok
    out["losses_flag"] = all(v["first_within_1e6"] and v["sixteenth_within_1e3"] for v in out["losses16"].values())
    rec = run(16, "one", "S", seed, Path(qual) / "dry", cmd, dose=2048)
    out["dry2048_S"] = dict(receipt=receipt_path(Path(qual) / "dry", 16, "one", "S", seed, 2048).name, tensors_sha256=rec["tensors_sha256"],
                            curve=[{k: r_[k] for k in ("step", "A", "R", "taught", "len12", "len40", "loss")} for r_ in rec["curve"]])
    ref = HERE / "exp45" / "e47_qual" / "e42_reference" / "e42_F10_S_dry2048_73129.json"
    if ref.exists():
        r42 = json.loads(ref.read_text())
        out["dry2048_S"]["e42_reference"] = dict(file=str(ref.relative_to(HERE)), source=E42_SOURCE, sha256=X16.sha_file(ref),
                                                 curve=[{k: r_[k] for k in ("step", "A", "R", "taught", "len12", "len40", "loss")} for r_ in r42["curve"]])
    out["pass"] = ok and out["losses_flag"]
    return out


def perturb(m, L, gen):
    with T.no_grad():
        L.Wc.add_(0.05 * T.randn(L.Wc.shape, generator=gen))
        L.bc.add_(0.05 * T.randn(L.bc.shape, generator=gen))
        if L.bfar.numel():
            L.bfar.add_(0.05 * T.randn(L.bfar.shape, generator=gen))
        if L.a is not None:
            L.a.mul_(1 + 0.5 * T.rand(L.a.shape, generator=gen))


def time_props(m, L, x):
    """The time properties of the specification's self-check 4, on this file's forward."""
    with T.no_grad():
        hs = T.stack(forward(m, L, x, False)[1], 1)
        x2 = x.clone(); t0 = 9; x2[:, t0] = (x2[:, t0] + 1) % 256
        hs2 = T.stack(forward(m, L, x2, False)[1], 1)
        d = (hs != hs2).any(0)
        first = [int(d[:, i].float().argmax()) if bool(d[:, i].any()) else -1 for i in range(L.C)]
        ok_time = all(fi == -1 or fi >= t0 + -(-L47.distance(i, L.side) // L47.RADIUS) for i, fi in enumerate(first))
        upd = T.stack([(t >= L.f) & ((t - L.f) % L.p == 0) for t in range(x.shape[1])]); changed = (hs[:, 1:] != hs[:, :-1]).any(0)
        hs_off = T.stack(forward(m, L, x, False, far_off=True)[1], 1)
        mean = hs[:, :, L.farx].mean(0)
        hs_mean = T.stack(forward(m, L, x, False, far_mean=mean)[1], 1)
    return dict(changed_byte_reaches_cell_no_sooner_than_distance=ok_time, changes_only_at_update_positions=bool((~changed | upd[1:]).all()),
                far_off_zero=bool((hs_off[:, :, L.farx] == 0).all()),
                far_mean_equal=bool(T.equal(hs_mean[:, :, L.farx], mean[None].expand(x.shape[0], -1, -1))),
                cells_reached=sum(1 for fi in first if fi != -1))


def check3(qual, cmd):
    """The faster forward against the specification's on the first batch, at every size and arm: logits to 1e-6, every parameter gradient to
    1e-5 of the tensor's largest entry, under both rules, at the start and with the weights moved off their start (Wc, bc, bfar, a); also with
    the far cells at zero and at a mean; and the time properties of self-check 4."""
    seed = 73129
    x, y, mask = first_batch(seed)
    out = dict(equality={}, time={}); ok = True
    cases = [(16, "one")] + [(s, a) for s in (32, 64) for a in ARMS]
    for side, arm in cases:
        for label in ("start", "perturbed"):
            m, L = build(side, arm, seed)
            if label == "perturbed":
                perturb(m, L, T.Generator().manual_seed(7))
            diag_cases = [("plain", {})]
            if len(L.farx) and label == "perturbed":
                with T.no_grad():
                    mean = T.stack(L47.forward(m, L, x, False)[1], 1)[:, :, L.farx].mean(0)
                diag_cases += [("far_off", dict(far_off=True)), ("far_mean", dict(far_mean=mean))]
            for dname, kw in diag_cases:
                for rule, local in RULES.items():
                    l_spec, g_spec = all_grads(m, L, L47.forward, x, y, mask, local, **kw)
                    l_fast, g_fast = all_grads(m, L, forward, x, y, mask, local, **kw)
                    rd = rel_diff(g_fast, g_spec)
                    r = dict(logits_max_abs_diff=float((l_fast - l_spec).abs().max()), grads_max_rel=max(rd.values()), grads_rel=rd,
                             same_tensors=sorted(g_fast) == sorted(g_spec))
                    r["pass"] = r["logits_max_abs_diff"] <= 1e-6 and r["grads_max_rel"] <= 1e-5 and r["same_tensors"]
                    ok = ok and r["pass"]
                    out["equality"][f"{side} {arm} {label} {dname} {rule}"] = r
                    print(f"check 3 {side} {arm} {label} {dname} {rule}: logits {r['logits_max_abs_diff']:.1e} grads {r['grads_max_rel']:.1e} {r['pass']}", flush=True)
        if side > 16:
            m, L = build(side, arm, seed)
            tp_ = time_props(m, L, x)
            out["time"][f"{side} {arm}"] = tp_
            ok = ok and all(v for k, v in tp_.items() if k != "cells_reached")
    out["decoding"] = decoding_equal()
    ok = ok and out["decoding"]["pass"]
    out["pass"] = ok
    return out


def decoding_equal():
    """Scoring arrangement: the answers given by the incremental greedy decoding equal those of exp11_cell.respond (the whole sequence
    again for every answer byte) through the same forward, and those of this file's decoding without the carried state, plain and under
    both diagnostics. On E26's birth S 73129 loaded by index (as in check 6) and on 1,024-cell networks moved off their start."""
    seed = 73129
    A, Rs, TA, LP = R42.panels(seed)
    sets = dict(A_add=A["add89"], len12=LP[12], len40=LP[40])
    out = {}; ok = True
    nets = []
    m = arch26.new_net(seed); m.load_state_dict(T.load(X.OUT / "e26" / f"e26_dev_S_{seed}.pt"))
    W = m.rnn.weight_hh_l0.detach(); b = m.rnn.bias_hh_l0.detach()
    Al = [[j for j in range(256) if j != i] for i in range(256)]; idx = T.tensor([[i] + Al[i] for i in range(256)])
    Wc = T.stack([W[g * 256:(g + 1) * 256].gather(1, idx) for g in range(3)])
    nets.append(("e26_dev_S_73129 by index", m, L47.Local(m, seed, 16, 255, structure=(Al, [1] * 256, [0] * 256, Wc, b.reshape(3, 256), None))))
    for arm in ("one", "place"):
        m2, L2 = build(32, arm, seed); perturb(m2, L2, T.Generator().manual_seed(3)); nets.append((f"32 {arm} perturbed", m2, L2))
    for name, m_, L_ in nets:
        for sname, items in sets.items():
            prompts = [p for p, _ in items]
            install(L_, incremental=False); ref = E.respond(m_, prompts)
            inc = generate(m_, L_, prompts)[0]
            r = dict(plain_equal=inc == ref, exact_incremental=sum(o == t for o, (_, t) in zip(inc, items)),
                     exact_respond=sum(o == t for o, (_, t) in zip(ref, items)))
            if len(L_.farx):
                fm = far_means(m_, L_, items)
                r["far_off_equal"] = generate(m_, L_, prompts, far_off=True)[0] == generate(m_, L_, prompts, far_off=True, incremental=False)[0]
                r["far_mean_equal"] = generate(m_, L_, prompts, far_mean=fm)[0] == generate(m_, L_, prompts, far_mean=fm, incremental=False)[0]
            out[f"{name} {sname}"] = r
            ok = ok and all(v for k, v in r.items() if k.endswith("_equal"))
            print("decoding", name, sname, r, flush=True)
            install(L_)
    out["pass"] = ok
    return out


def check4(qual, cmd):
    """Locality, on the runner's forward, at 1,024 cells under one and place: a changed byte changes no far cell at that position and reaches
    a far cell no sooner than its distance allows; the logits at a position do not change when a far cell's state at that position is changed
    by hand (they change only through later positions); with far_off the far cells are zero."""
    seed = 73129
    x, y, mask = first_batch(seed)
    out = {}; ok = True
    for arm in ("one", "place"):
        m, L = build(32, arm, seed)
        perturb(m, L, T.Generator().manual_seed(5))                         # moved off the start so that no weight is accidentally symmetric
        with T.no_grad():
            lg, hs = forward(m, L, x, False); hs = T.stack(hs, 1)
            t0 = 9; x2 = x.clone(); x2[:, t0] = (x2[:, t0] + 1) % 256
            hs2 = T.stack(forward(m, L, x2, False)[1], 1)
            far_same_at_t0 = bool(T.equal(hs[:, :t0 + 1, L.farx], hs2[:, :t0 + 1, L.farx]))
            iface_changed_at_t0 = bool((hs[:, t0, L.ifx] != hs2[:, t0, L.ifx]).any())
            d = (hs != hs2).any(0)
            first = [int(d[:, i].float().argmax()) if bool(d[:, i].any()) else -1 for i in range(L.C)]
            ok_time = all(fi == -1 or fi >= t0 + -(-L47.distance(i, 32) // L47.RADIUS) for i, fi in enumerate(first))
            far_reached = sum(1 for i in L.farx.tolist() if first[i] != -1)
            pokes = {}
            for tp in (5, 9, 14):
                vals = T.rand(x.shape[0], len(L.farx), generator=T.Generator().manual_seed(tp)) * 2 - 1
                lgp, hsp = forward(m, L, x, False, poke=(tp, vals))
                pokes[str(tp)] = dict(logits_up_to_t_equal=bool(T.equal(lgp[:, :tp + 1], lg[:, :tp + 1])),
                                      logits_after_t_changed=bool((lgp[:, tp + 1:] != lg[:, tp + 1:]).any()),
                                      far_state_set=bool(T.equal(hsp[tp][:, L.farx], vals)))
            hs_off = T.stack(forward(m, L, x, False, far_off=True)[1], 1)
            off_zero = bool((hs_off[:, :, L.farx] == 0).all())
        r = dict(far_unchanged_at_changed_position=far_same_at_t0, interface_changed_at_changed_position=iface_changed_at_t0,
                 no_sooner_than_distance=ok_time, far_cells_reached_by_end=far_reached, far_cells=len(L.farx), pokes=pokes, far_off_zero=off_zero)
        r["pass"] = far_same_at_t0 and iface_changed_at_t0 and ok_time and off_zero and all(
            v["logits_up_to_t_equal"] and v["far_state_set"] for v in pokes.values())
        ok = ok and r["pass"]; out[arm] = r
    out["pass"] = ok
    return out


def check5(qual, cmd):
    """The batch digest at row 4,096 is E26's for the seed."""
    seed = 73129
    tp = [(t, E.split(t, .5, seed)[0]) for t in E.DEV]
    bt = E.batches(tp, seed, E.KMAX["M"]); dg = C26.Digest()
    for _ in range(4096):
        dg.add(next(bt))
    d = dg.value()["sha256"]
    return dict(digest_4096=d, expected_prefix=R42.E26_DIGEST_4096[seed], **{"pass": d.startswith(R42.E26_DIGEST_4096[seed])})


def check6(qual, cmd):
    """E42's check 12 on E26's six births, each loaded by index with structure=: every cell's sources itself and the 255 others."""
    out = dict(counts={}); worst = 0
    for rule in "SB":
        for fs in SEEDS:
            m = arch26.new_net(fs)
            f_ = X.OUT / "e26" / f"e26_dev_{rule}_{fs}.pt"
            m.load_state_dict(T.load(f_))
            W = m.rnn.weight_hh_l0.detach(); b = m.rnn.bias_hh_l0.detach()
            A = [[j for j in range(256) if j != i] for i in range(256)]
            idx = T.tensor([[i] + A[i] for i in range(256)])
            Wc = T.stack([W[g * 256:(g + 1) * 256].gather(1, idx) for g in range(3)])
            L = L47.Local(m, fs, 16, 255, structure=(A, [1] * 256, [0] * 256, Wc, b.reshape(3, 256), None))
            install(L)
            A_, Rs, TA, LP = R42.panels(fs)
            sc = R42.score_all(m, A_, Rs, TA, LP)
            got = ((sc["A_counts"]["add89"], sc["A_counts"]["sub89"]), (sc["R_counts"]["add89"], sc["R_counts"]["sub89"]),
                   (sc["taught_counts"]["add89"], sc["taught_counts"]["sub89"]), (sc["length_counts"]["12"], sc["length_counts"]["40"]))
            exp = R42.CHECK12_COUNTS[(rule, fs)]
            d = max(abs(g - e) for gg, ee in zip(got, exp) for g, e in zip(gg, ee)); worst = max(worst, d)
            out["counts"][f_.name] = dict(file_sha256=X16.sha_file(f_), got=got, expected=exp, max_abs_diff=d, A=sc["A"], R=sc["R"],
                                          taught=sc["taught"], len12=sc["len12"], len40=sc["len40"])
            print(f_.name, got, exp, flush=True)
    out["max_abs_diff"] = worst
    out["verdict"] = "equal" if worst == 0 else "differs by one or two (reported)" if worst <= 2 else "FAIL"
    out["pass"] = worst <= 2
    return out


def check7(qual, cmd):
    """CPU-seconds per update and per scored row (plain and the two diagnostics) at each size and arm, serial, on this host; a forecast for
    the trial and for the roster as planned."""
    seed = 73129
    tp = [(t, E.split(t, .5, seed)[0]) for t in E.DEV]
    out = dict(cost={}, beside=os.environ.get("E47_BESIDE", "not recorded")); n_up = 32
    A, Rs, TA, LP = R42.panels(seed)
    rows = DOSE // ROW + 1
    for side, arm in [(16, "one")] + [(s, a) for s in (32, 64) for a in ARMS]:
        for rule in "SB":
            m, L = build(side, arm, seed); install(L); opt = make_opt(m, L)
            bt = E.batches(tp, seed, E.KMAX["M"])
            step(m, L, opt, next(bt), RULES[rule])
            c = time.process_time()
            for _ in range(n_up):
                step(m, L, opt, next(bt), RULES[rule])
            per_u = (time.process_time() - c) / n_up
            # the specification's forward and backward, for comparison
            x, y, mask = first_batch(seed)
            c = time.process_time(); all_grads(m, L, L47.forward, x, y, mask, RULES[rule]); spec_fb = time.process_time() - c
            c = time.process_time(); all_grads(m, L, forward, x, y, mask, RULES[rule]); fast_fb = time.process_time() - c
            if rule == "S":
                c = time.process_time(); score_row_all(m, L, A, Rs, TA, LP); per_row = time.process_time() - c
            else:
                per_row = out["cost"][f"{side} {arm} S"]["cpu_per_scored_row"]
            out["cost"][f"{side} {arm} {rule}"] = dict(cpu_per_update=per_u, cpu_per_scored_row=per_row, spec_forward_backward=spec_fb,
                                                       fast_forward_backward=fast_fb, forecast_cpu_h=(DOSE * per_u + rows * per_row) / 3600,
                                                       multiply_adds_with_periods=L.counts(m)["multiply_adds_with_periods"])
            print(f"cost {side} {arm} {rule}: {per_u:.3f} s/update (spec f+b {spec_fb:.3f}, fast f+b {fast_fb:.3f}), {per_row:.1f} s/row, "
                  f"{out['cost'][f'{side} {arm} {rule}']['forecast_cpu_h']:.1f} CPU-h", flush=True)
    fc = lambda k: out["cost"][k]["forecast_cpu_h"]
    out["forecast_trial_cpu_h"] = {k: fc(k) for k in ("32 one S", "32 one B", "32 place S", "32 place B")}
    out["forecast_trial_total_cpu_h"] = sum(out["forecast_trial_cpu_h"].values())
    out["forecast_roster_cpu_h"] = {k: fc(k) for k in ("32 scatter S", "32 scatter B", "64 one B", "64 one S")}
    out["forecast_roster_total_cpu_h"] = sum(out["forecast_roster_cpu_h"].values())
    out["note"] = "measured serially with no other job on the host, one thread; a run beside two others is slower (reported from the trial)"
    out["pass"] = True
    return out


CHECKS = {"1": check1, "2": check2, "3": check3, "4": check4, "5": check5, "6": check6, "7": check7}


# ---------------------------------------------------------------- main
def main(argv):
    cmd = "python3 " + " ".join(argv)
    a = argv[1:]
    opt = lambda name, default: (a[a.index(name) + 1] if name in a else default)
    if a[0] == "run":
        side, arm, rule, seed = int(a[1]), a[2], a[3], int(a[4])
        assert side in SIDES and arm in ARMS and rule in RULES and seed in SEEDS and (side > 16 or arm == "one")
        d = opt("--dose", None)
        run(side, arm, rule, seed, Path(opt("--out", str(OUT))), cmd, dose=int(d) if d else None)
    elif a[0] == "check":
        n = a[1]
        qual = Path(opt("--qual", str(QUAL))); qual.mkdir(parents=True, exist_ok=True)
        p = qual / f"check{n}.json"
        R_.refuse_existing(p)
        t0 = time.process_time(); w0 = time.time()
        res = CHECKS[n](qual / f"check{n}_runs", cmd)
        res.update(check=int(n), cpu=time.process_time() - t0, wall=time.time() - w0, provenance=provenance(cmd))
        p.write_text(json.dumps(res, indent=1))
        print(f"e47 check {n}:", "PASS" if res["pass"] else "FAIL", flush=True)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv)
