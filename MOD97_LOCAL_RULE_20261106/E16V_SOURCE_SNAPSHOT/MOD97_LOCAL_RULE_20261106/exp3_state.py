"""Experiment 3 (rung 3): working state for multi-step computation.

Task: modular addition with a run of k distractor bytes between '+' and the second operand:
    <code> a a + D1 ... Dk b b = c c EOS        distractors are uppercase letters, k in 0..12 during development,
                                               0..20 at test (14-20 is extrapolation)
The first operand must be held across the run. A fixed byte window cannot see it once k exceeds the window.

Architectures (all share the byte embedding, a hidden readout l2 and the 257-way head l3):
    win   windowed MLP, no state  (the harness network, W=10)
    rnn   Elman state:  h_t = tanh(L(u_t) + M(h_{t-1}))
    gru   gated state:  z,r sigmoid gates, candidate tanh, h_t = (1-z) n + z h_{t-1}
Every linear map goes through one function that uses the true weight forward and, in LOCAL mode, the
sign of the weight (scaled to its mean magnitude) backward: sign-symmetric feedback for any architecture.
In local mode the previous state is detached, so no credit flows through time (a within-step local rule).

Development: backprop with full BPTT on mod 97 (code r) and mod 89 (code q), AdamW lr .001 decay 0.3.
Life: held-out mod 73 (code s), 25% of pairs taught, arms
    W   warm start from the developed birth, within-step sign rule
    R   random birth, within-step sign rule
    B   random birth, backprop with BPTT (reference)
"""
import os
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[k] = "1"
import json
import sys
import time
from collections import Counter, defaultdict
from itertools import islice
from pathlib import Path
from random import Random

import torch as T
from torch import nn
from torch.nn.functional import cross_entropy, linear, pad

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mod97_local as M

T.set_num_threads(1)
EOS, BOS, PAD = 256, 257, 257
SEEDS = M.SEEDS
BATCH, BLOCK = 64, 512
OUT = Path(__file__).resolve().parent / "exp3"
W, H = 10, 128
DEV_UPDATES, LIFE_UPDATES = 24576, 12288
KMAX_DEV, KS_TEST = 12, (0, 2, 4, 6, 8, 10, 12, 14, 16, 20)
CODE = {97: b"r", 89: b"q", 73: b"s"}
DISTRACT = list(b"ABCDEFGHIJKLMNOPQRSTUVWXYZ")
ARCHS = ("win", "rnn", "gru")


# ---------------------------------------------------------------- task
def render(mod, a, b, k, rng):
    d = [rng.choice(DISTRACT) for _ in range(k)]
    prompt = list(CODE[mod]) + list(f"{a:02d}".encode()) + [ord("+")] + d + list(f"{b:02d}".encode()) + [ord("=")]
    return prompt + list(f"{(a + b) % mod:02d}".encode()) + [EOS]


def split(mod, frac, seed):
    groups = [(a, b) for a in range(mod) for b in range(a, mod)]
    rng = Random(seed * 1000 + round(frac * 100) + mod)
    rng.shuffle(groups)
    kk = round(frac * len(groups))
    taught, rest = groups[:kk], groups[kk:]
    monitor, sealed = rest[0::2], rest[1::2]
    expand = lambda gs: [p for a, b in gs for p in (((a, b), (b, a)) if a != b else ((a, b),))]
    return expand(sorted(taught)), expand(sorted(monitor)), expand(sorted(sealed))


def stream(mod, taught, seed):
    """Each presentation draws k uniformly in 0..KMAX_DEV and fresh distractors."""
    order, draw = Random(seed + 1), Random(seed + 2)
    chunk = []
    while True:
        for a, b in order.sample(taught, len(taught)):
            chunk.append(render(mod, a, b, draw.randint(0, KMAX_DEV), draw))
            if len(chunk) == BATCH:
                yield chunk
                chunk = []


def panel(mod, pairs, seed, ks=KS_TEST, per_k=None):
    """Fixed evaluation prompts: each pair at each k with seeded distractors (subsampled to per_k pairs per k)."""
    rng = Random(seed + 3)
    sub = pairs if per_k is None or len(pairs) <= per_k else rng.sample(pairs, per_k)
    return [(k, (a, b), render(mod, a, b, k, rng)) for k in ks for a, b in sub]


# ---------------------------------------------------------------- sign-feedback linear map
class SignLinear(T.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        ctx.save_for_backward(x, weight)
        return linear(x, weight, bias)

    @staticmethod
    def backward(ctx, g):
        x, weight = ctx.saved_tensors
        S = weight.sign() * weight.abs().mean()
        gx = g @ S
        gw = g.reshape(-1, g.shape[-1]).T @ x.reshape(-1, x.shape[-1])
        gb = g.reshape(-1, g.shape[-1]).sum(0)
        return gx, gw, gb


def lin(x, layer, local):
    return SignLinear.apply(x, layer.weight, layer.bias) if local else linear(x, layer.weight, layer.bias)


class StateNet(nn.Module):
    def __init__(self, arch):
        super().__init__()
        self.arch = arch
        self.book = nn.Embedding(258, 8, padding_idx=PAD)
        self.l1 = nn.Linear(8 * W, H)
        if arch == "rnn":
            self.m = nn.Linear(H, H)
        if arch == "gru":
            self.lz, self.lr, self.ln = nn.Linear(8 * W, H), nn.Linear(8 * W, H), nn.Linear(8 * W, H)
            self.mz, self.mr, self.mn = nn.Linear(H, H), nn.Linear(H, H), nn.Linear(H, H)
        self.l2 = nn.Linear(H, H)
        self.l3 = nn.Linear(H, 257)

    def window(self, x):
        return self.book(pad(x, (W - 1, 0), value=PAD).unfold(1, W, 1)).flatten(2)

    def forward(self, x, local=False, detach=None):
        detach = local if detach is None else detach
        u = self.window(x)                      # B x L x 8W
        B, L, _ = u.shape
        h = T.zeros(B, H)
        hs = []
        for t in range(L):
            ut = u[:, t]
            hp = h.detach() if detach else h
            if self.arch == "win":
                h = T.tanh(lin(ut, self.l1, local))
            elif self.arch == "rnn":
                h = T.tanh(lin(ut, self.l1, local) + lin(hp, self.m, local))
            else:
                z = T.sigmoid(lin(ut, self.lz, local) + lin(hp, self.mz, local))
                r = T.sigmoid(lin(ut, self.lr, local) + lin(hp, self.mr, local))
                n = T.tanh(lin(ut, self.ln, local) + r * lin(hp, self.mn, local))
                h = (1 - z) * n + z * hp
            hs.append(h)
        hs = T.stack(hs, 1)
        z2 = T.tanh(lin(hs, self.l2, local))
        return lin(z2, self.l3, local)


# ---------------------------------------------------------------- training and scoring
def batch_tensor(rows):
    L = max(len(r) for r in rows) + 1
    data = T.full((len(rows), L), PAD, dtype=T.long)
    for i, r in enumerate(rows):
        data[i, 0] = BOS
        data[i, 1:len(r) + 1] = T.tensor(r)
    return data


ANSWER_ONLY = True   # train on the two answer bytes and EOS only; distractor letters are unpredictable noise (see EXP3 report)


def update(m, opt, rows, local, detach=None):
    data = batch_tensor(rows)
    x, y = data[:, :-1], data[:, 1:]
    logits = m(x, local=local, detach=detach)
    if ANSWER_ONLY:
        mask = T.zeros_like(y, dtype=T.bool)
        for i, r in enumerate(rows):
            mask[i, len(r) - 3:len(r)] = True          # targets: answer byte, answer byte, EOS
    else:
        mask = y != PAD
    loss = cross_entropy(logits[mask], y[mask])
    opt.zero_grad()
    loss.backward()
    opt.step()
    return float(loss.detach())


@T.no_grad()
def respond(m, prompts, cap=4):
    """Greedy generation; prompts grouped by length so each group batches cleanly."""
    out = [None] * len(prompts)
    groups = defaultdict(list)
    for i, p in enumerate(prompts):
        groups[len(p)].append(i)
    for L, idx in groups.items():
        for j in range(0, len(idx), 256):
            ids = idx[j:j + 256]
            p = T.tensor([[BOS, *prompts[i]] for i in ids])
            gen = [[] for _ in ids]
            done = [False] * len(ids)
            for _ in range(cap):
                y = m(p)[:, -1].argmax(1)
                for q in range(len(ids)):
                    if not done[q]:
                        gen[q].append(int(y[q]))
                        if int(y[q]) == EOS:
                            done[q] = True
                p = T.cat([p, y[:, None]], 1)
                if all(done):
                    break
            for q, i in enumerate(ids):
                out[i] = gen[q]
    return out


def score(m, items):
    """items: list of (k, pair, row). Returns exact counts by k."""
    prompts = [r[:-3] for _, _, r in items]
    gold = [r[-3:] for _, _, r in items]
    outs = respond(m, prompts)
    by_k = defaultdict(lambda: [0, 0])
    for (k, _, _), o, g in zip(items, outs, gold):
        by_k[k][0] += o == g
        by_k[k][1] += 1
    return {str(k): v for k, v in sorted(by_k.items())}


def shortcuts(mod, taught, sealed_items):
    """Floors on the sealed panel for strategies that ignore the held operand: most frequent answer,
    answer from the second operand alone (b + most common a), and the no-wrap sum."""
    ans = lambda a, b: (a + b) % mod
    freq = Counter(ans(a, b) for a, b in taught).most_common(1)[0][0]
    by_b = defaultdict(Counter)
    for a, b in taught:
        by_b[b][ans(a, b)] += 1
    return dict(n=len(sealed_items), frequency=sum(ans(*p) == freq for _, p, _ in sealed_items),
                second_operand_only=sum(by_b[p[1]].most_common(1)[0][0] == ans(*p) if by_b[p[1]] else 0 for _, p, _ in sealed_items),
                no_wrap=sum(p[0] + p[1] == ans(*p) for _, p, _ in sealed_items),
                exact_lookup=sum(p in set(taught) for _, p, _ in sealed_items))


# ---------------------------------------------------------------- phases
def develop(arch, seed):
    OUT.mkdir(exist_ok=True)
    path = OUT / f"dev_{arch}_{seed}.pt"
    if path.exists():
        return
    T.manual_seed(seed)
    m = StateNet(arch)
    opt = T.optim.AdamW(m.parameters(), lr=.001, weight_decay=.3, betas=(.9, .98))
    streams = [stream(mod, split(mod, .5, seed)[0], seed + mod) for mod in (97, 89)]
    panels = {mod: panel(mod, split(mod, .5, seed)[2], seed, per_k=300) for mod in (97, 89)}
    t0 = time.process_time()
    curve = []
    for stop in range(BLOCK, DEV_UPDATES + 1, BLOCK):
        losses = []
        for i in range(BLOCK):
            losses.append(update(m, opt, next(streams[i % 2]), local=False))
        if stop % 4096 == 0 or stop == DEV_UPDATES:
            row = dict(step=stop, loss=sum(losses) / len(losses), cpu=time.process_time() - t0,
                       sealed={str(mod): score(m, panels[mod]) for mod in panels})
            curve.append(row)
            print("dev", arch, seed, stop, {mod: {k: v[0] for k, v in row["sealed"][str(mod)].items()} for mod in (97, 89)}, flush=True)
    T.save(m.state_dict(), path)
    (OUT / f"dev_{arch}_{seed}.json").write_text(json.dumps(dict(arch=arch, seed=seed, params=sum(p.numel() for p in m.parameters()), curve=curve), indent=1))


def life(arch, arm, seed, frac=.25):
    OUT.mkdir(exist_ok=True)
    path = OUT / f"life_{arch}_{arm}_{seed}.json"
    mod = 73
    taught, monitor, sealed = split(mod, frac, seed)
    T.manual_seed(seed)
    m = StateNet(arch)
    # W warm, sign rule, no credit through time | R random, same | B random, backprop+BPTT
    # T warm, TRUE gradient but no credit through time | F warm, sign rule WITH credit through time | G warm, backprop+BPTT
    if arm in ("W", "T", "F", "G"):
        m.load_state_dict(T.load(OUT / f"dev_{arch}_{seed}.pt"))
    local = arm in ("W", "R", "F")
    detach = arm in ("W", "R", "T")
    opt = T.optim.AdamW(m.parameters(), lr=.001, weight_decay=(.3 if arm in ("B", "T", "G") else 1.), betas=(.9, .98))
    items_s = panel(mod, sealed, seed, per_k=300)
    items_t = panel(mod, taught, seed, ks=(0, 4, 8, 12), per_k=300)
    before = score(m, items_s)
    st = stream(mod, taught, seed + mod)
    t0 = time.process_time()
    curve = []
    for stop in range(BLOCK, LIFE_UPDATES + 1, BLOCK):
        losses = [update(m, opt, rows, local=local, detach=detach) for rows in islice(st, BLOCK)]
        if stop % 2048 == 0:
            row = dict(step=stop, loss=sum(losses) / len(losses), taught=score(m, items_t), sealed=score(m, items_s), cpu=time.process_time() - t0)
            curve.append(row)
            print(arch, arm, seed, stop, {k: v[0] for k, v in row["sealed"].items()}, f"cpu {row['cpu']:.0f}s", flush=True)
            path.write_text(json.dumps(dict(arch=arch, arm=arm, seed=seed, frac=frac, params=sum(p.numel() for p in m.parameters()),
                                            sealed_before=before, shortcuts=shortcuts(mod, taught, items_s), curve=curve), indent=1))


# ---------------------------------------------------------------- qualification
def qualify():
    out = {}
    # 1. the sign-feedback autograd path reproduces the manual local rule on the window network
    T.manual_seed(1)
    mm = M.Net(W=10)
    sn = StateNet("win")
    sn.book.weight.data.copy_(mm.book.weight.data); sn.l1.weight.data.copy_(mm.l1.weight.data); sn.l1.bias.data.copy_(mm.l1.bias.data)
    sn.l2.weight.data.copy_(mm.l2.weight.data); sn.l2.bias.data.copy_(mm.l2.bias.data); sn.l3.weight.data.copy_(mm.l3.weight.data); sn.l3.bias.data.copy_(mm.l3.bias.data)
    train, _, _ = M.panels(SEEDS[0])
    rows = list(islice(M.stream(train, 1), 1))[0]
    g, _ = M.local_grads(mm, dict(mode="sign", drop=0., rng=T.Generator().manual_seed(8)), rows, False)
    data = M.rows_tensor(rows)
    logits = sn(data[:, :-1], local=True)
    loss = cross_entropy(logits.flatten(0, 1), data[:, 1:].flatten())
    loss.backward()
    P = dict(sn.named_parameters())
    agree = {n: bool(T.allclose(P[n].grad, g[n], atol=1e-5, rtol=1e-3)) for n in ("l3.weight", "l3.bias", "l2.weight", "l2.bias", "l1.weight", "l1.bias", "book.weight")}
    assert all(agree.values()), agree
    out["sign_autograd_equals_manual_rule"] = agree
    # 2. task: splits, rendering, oracle, distractor vocabulary disjoint from digits/codes
    for mod in (97, 89, 73):
        t, mo, se = split(mod, .25, SEEDS[0])
        assert len(t) + len(mo) + len(se) == mod * mod and not (set(t) & set(se))
        rng = Random(0)
        for a, b in t[:200]:
            for k in (0, 5, 20):
                r = render(mod, a, b, k, rng)
                assert len(r) == 7 + k + 3 and int(bytes(r[-3:-1])) == (a + b) % mod and r[-1] == EOS
                assert not (set(r[4:4 + k]) & set(b"0123456789+=pqrs"))
    out["task_checks"] = True
    # 3. the windowed net provably cannot see the first operand for k > W - 5: check by perturbation
    T.manual_seed(2); sn = StateNet("win")
    rng = Random(1)
    for k in (0, 5, 6, 12):
        r1 = render(97, 12, 34, k, rng); r2 = list(r1); r2[1:3] = list(b"56")   # change the first operand only
        with T.no_grad():
            d = float((sn(T.tensor([[BOS, *r1[:-3]]]))[0, -1] - sn(T.tensor([[BOS, *r2[:-3]]]))[0, -1]).abs().max())
        out[f"win_logit_change_when_first_operand_changes_k{k}"] = d
    assert out["win_logit_change_when_first_operand_changes_k12"] == 0.0
    # 4. smoke: each architecture trains finitely in both modes; local mode leaves no gradient through time
    for arch in ARCHS:
        for local in (False, True):
            T.manual_seed(3); m = StateNet(arch)
            opt = T.optim.AdamW(m.parameters(), lr=.001)
            st = stream(97, split(97, .25, 1)[0], 1)
            for rows in islice(st, 4):
                l = update(m, opt, rows, local)
            assert l == l
            out[f"smoke_{arch}_{'local' if local else 'bptt'}"] = l
        out[f"params_{arch}"] = sum(p.numel() for p in StateNet(arch).parameters())
    # 5. shortcut floors on the mod-73 sealed panel
    t, _, se = split(73, .25, SEEDS[0])
    out["shortcuts_mod73_25pct"] = shortcuts(73, t, panel(73, se, SEEDS[0], per_k=300))
    OUT.mkdir(exist_ok=True)
    (OUT / "QUALIFY.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "qualify":
        qualify()
    elif cmd == "develop":
        develop(sys.argv[2], int(sys.argv[3]))
    elif cmd == "life":
        life(sys.argv[2], sys.argv[3], int(sys.argv[4]))
