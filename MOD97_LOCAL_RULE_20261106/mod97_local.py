"""Mod-97 addition from raw bytes: backprop versus V181-style direct-feedback local rule.

Task, split, rendering, model shape, loss and scoring follow Codex's
minifly_three_remaining_experiments_20261106 mod-97 run (fixtures.py, models.py,
official_mod97.py at commit 501ca0c). The KC association residual is omitted
(it contributed 0.9-1.6% of held sums in that run). The only difference between
arms is how the weight update is computed:

  bp       autograd backpropagation
  dfa      direct feedback alignment as in V181's learner.py: the output error is
           projected to each hidden layer by a fixed random matrix, and every weight
           update is local (presynaptic activity x projected error). The embedding
           receives its own fixed random projection of the output error.
  dfa_fix  same as dfa, embedding frozen at its random birth (fixed input code).

All arms use Adam lr 0.003, batch 64, no weight decay, same data order per seed.
"""
import os
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[k] = "1"
import json
import math
import sys
import time
from itertools import combinations_with_replacement, islice
from pathlib import Path
from random import Random

import torch as T
from torch import nn
from torch.nn.functional import cross_entropy, pad

T.set_num_threads(1)
EOS = 256
BOS = 257
SPLIT_SEED = 57467          # Codex fixtures.SEED
SEEDS = (73129, 93683, 116089)
BATCH = 64
BLOCK = 512
OUT = Path(__file__).resolve().parent / "runs"


# ---------------------------------------------------------------- task (verbatim logic from fixtures.py)
def carry(pair):
    a, b = pair
    return int(a % 10 + b % 10 >= 10)


def mod_split(seed=SPLIT_SEED):
    rng = Random(seed)
    diag = set(rng.sample(range(97), 48))
    extra = set(rng.sample(range(97), 48))
    out = [[], [], []]
    for residue in range(97):
        groups = [[p for p in combinations_with_replacement(range(97), 2)
                   if sum(p) % 97 == residue and p[0] != p[1] and carry(p) == c] for c in range(2)]
        d = (residue * 49) % 97
        n = 24 + (residue in extra) - (residue in diag)
        first = round(n * len(groups[0]) / 48)
        for g, k in zip(groups, (first, n - first)):
            rng.shuffle(g)
            out[0].extend(g[:k])
            rest = g[k:]
            side = 1 if len(out[1]) <= len(out[2]) else 2
            out[side].extend(rest[::2])
            out[3 - side].extend(rest[1::2])
        side = 1 if len(out[1]) <= len(out[2]) else 2
        out[0 if residue in diag else side].append((d, d))
    return out


def expand(pairs):
    return [p for a, b in pairs for p in (((a, b), (b, a)) if a != b else ((a, b),))]


def arithmetic(pair, mod=97):
    a, b = pair
    prompt = f"{a:02d}+{b:02d}".encode() + b"="
    y = (a + b) % mod
    return list(prompt) + list(f"{y:02d}".encode()) + [EOS]


def panels(seed):
    return tuple(map(expand, mod_split(seed)))


def stream(train, seed):
    targets = [arithmetic(p) for p in train]
    order = Random(seed + 1)
    chunk = []
    while True:
        for row in order.sample(targets, len(targets)):
            chunk.append(row)
            if len(chunk) == BATCH:
                yield chunk
                chunk = []


# ---------------------------------------------------------------- model (shape from models.Modular, shared path only)
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.book = nn.Embedding(258, 8, padding_idx=BOS)
        self.l1 = nn.Linear(64, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, 257)

    def window(self, x):
        return self.book(pad(x, (7, 0), value=BOS).unfold(1, 8, 1)).flatten(2)

    def forward(self, x, keep=False, drop=0., rng=None):
        u = self.window(x)
        a1 = self.l1(u); z1 = T.tanh(a1)
        a2 = self.l2(z1); z2 = T.tanh(a2)
        mask = None
        if drop > 0:   # inverted dropout on the second hidden layer, as V181's T layer
            mask = (T.rand(z2.shape, generator=rng) >= drop).float() / (1 - drop)
            z2 = z2 * mask
        logits = self.l3(z2)
        if keep:
            return logits, (u, z1, z2, mask)
        return logits


V181_OPT = dict(lr=.001, weight_decay=1e-4)   # V181 learner.py: lr .001, coupled L2 1e-4, dropout .1 on 2nd hidden
DROP = {"bp_v181": .1, "dfa_v181": .1}


def make(seed, arm="bp"):
    T.manual_seed(seed)
    m = Net()
    g = T.Generator().manual_seed(seed + 99)
    F = {  # fixed random feedback, scale as V181 (1/sqrt(257))
        "F2": T.randn(257, 128, generator=g) / math.sqrt(257),
        "F1": T.randn(257, 128, generator=g) / math.sqrt(257),
        "F0": T.randn(257, 64, generator=g) / math.sqrt(257),
    }
    F["drop"] = DROP.get(arm, 0.)
    F["rng"] = T.Generator().manual_seed(seed + 7)
    F["mode"] = "sign" if arm.startswith("sign") else "dfa"
    if arm.startswith("bpw") or arm.startswith("signw"):   # decoupled weight decay (AdamW), lr .001, as in the grokking papers
        wd = WD[arm[-2:]]
        opt = T.optim.AdamW(m.parameters(), lr=.001, weight_decay=wd, betas=(.9, .98))
    elif arm.endswith("_v181"):
        opt = T.optim.Adam(m.parameters(), **V181_OPT)
    elif arm == "dfa_outwd":      # weight decay on the output layer only
        out = [m.l3.weight, m.l3.bias]
        rest = [p for n, p in m.named_parameters() if not n.startswith("l3.")]
        opt = T.optim.Adam([dict(params=rest, weight_decay=0.), dict(params=out, weight_decay=1e-4)], lr=.003)
    elif arm == "dfa_lowlr":      # lower learning rate, no decay
        opt = T.optim.Adam(m.parameters(), lr=.001)
    else:
        opt = T.optim.Adam(m.parameters(), lr=.003)
    return m, F, opt


def rows_tensor(rows):
    return T.tensor([[BOS, *r] for r in rows])


def update_bp(m, F, opt, rows):
    data = rows_tensor(rows)
    logits = m(data[:, :-1], drop=F["drop"], rng=F["rng"])
    loss = cross_entropy(logits.flatten(0, 1), data[:, 1:].flatten())
    opt.zero_grad()
    loss.backward()
    opt.step()
    return float(loss.detach())


@T.no_grad()
def local_grads(m, F, rows, freeze_emb):
    """Direct feedback alignment. Returns dict of grads keyed by parameter name."""
    data = rows_tensor(rows)
    x = data[:, :-1]
    y = data[:, 1:].flatten()
    logits, (u, z1, z2, mask) = m(x, keep=True, drop=F["drop"], rng=F["rng"])
    N = y.numel()
    L = logits.flatten(0, 1)
    e = L.softmax(1)
    e[T.arange(N), y] -= 1
    e /= N                                   # d loss / d logits (mean CE)
    loss = float(-(L.log_softmax(1)[T.arange(N), y]).mean())
    u = u.flatten(0, 1); z1 = z1.flatten(0, 1); z2 = z2.flatten(0, 1)
    g = {}
    g["l3.weight"] = e.T @ z2
    g["l3.bias"] = e.sum(0)
    sign = F.get("mode") == "sign"

    def S(W):   # sign-symmetric feedback: sign of the forward weight, scaled to its mean magnitude
        return W.sign() * W.abs().mean()

    if mask is None:
        tp2 = 1 - z2 * z2
        d2 = (e @ (S(m.l3.weight) if sign else F["F2"])) * tp2
    else:
        mask = mask.flatten(0, 1)
        pre = z2 / mask.clamp_min(1e-6) * (mask > 0)   # undo the scaling where kept
        d2 = (e @ (S(m.l3.weight) if sign else F["F2"])) * (1 - pre * pre) * mask
    g["l2.weight"] = d2.T @ z1
    g["l2.bias"] = d2.sum(0)
    d1 = ((d2 @ S(m.l2.weight)) if sign else (e @ F["F1"])) * (1 - z1 * z1)
    g["l1.weight"] = d1.T @ u
    g["l1.bias"] = d1.sum(0)
    if not freeze_emb:
        d0 = ((d1 @ S(m.l1.weight)) if sign else (e @ F["F0"])).reshape(N, 8, 8)  # per slot, per embedding dim
        tok = pad(x, (7, 0), value=BOS).unfold(1, 8, 1).flatten(0, 1)   # N x 8 token ids
        ge = T.zeros(258, 8)
        ge.index_add_(0, tok.flatten(), d0.reshape(-1, 8))
        ge[BOS] = 0
        g["book.weight"] = ge
    return g, loss


def update_dfa(m, F, opt, rows, freeze_emb=False):
    g, loss = local_grads(m, F, rows, freeze_emb)
    opt.zero_grad()
    for name, p in m.named_parameters():
        p.grad = g[name] if name in g else None   # frozen embedding: no grad, Adam skips it
    opt.step()
    return loss


# ---------------------------------------------------------------- scoring (as official_mod97.score)
@T.no_grad()
def respond(m, prompts, cap=5, batch=256):
    outs = []
    for i in range(0, len(prompts), batch):
        p = T.tensor([[BOS, *x] for x in prompts[i:i + batch]])
        done = T.zeros(len(p), dtype=T.bool)
        gen = [[] for _ in range(len(p))]
        for _ in range(cap):
            y = m(p)[:, -1].argmax(1)
            for j in range(len(p)):
                if not done[j]:
                    gen[j].append(int(y[j]))
                    if int(y[j]) == EOS:
                        done[j] = True
            p = T.cat([p, y[:, None]], 1)
            if bool(done.all()):
                break
        outs.extend(gen)
    return outs


def score(m, pairs):
    rows = [arithmetic(p) for p in pairs]
    prompts = [r[:6] for r in rows]
    gold = [r[6:] for r in rows]
    outs = respond(m, prompts)
    exact = {p: o == g for p, o, g in zip(pairs, outs, gold)}
    return dict(
        n=len(pairs),
        exact=sum(exact.values()),
        first=sum(o[:1] == g[:1] for o, g in zip(outs, gold)),
        eos=sum(EOS in o for o in outs),
        both_orders=sum(exact[a, b] and exact[b, a] for a, b in pairs if a < b),
        both_order_n=sum(a < b for a, b in pairs),
        wrap=[sum(exact[a, b] for a, b in pairs if a + b >= 97), sum(a + b >= 97 for a, b in pairs)],
        decimal_carry=[sum(exact[a, b] for a, b in pairs if a % 10 + b % 10 >= 10),
                       sum(a % 10 + b % 10 >= 10 for a, b in pairs)],
    )


# ---------------------------------------------------------------- qualification
def qualify():
    train, monitor, sealed = panels(SEEDS[0])
    assert len(train) == 4704 and len(monitor) + len(sealed) == 4705, (len(train), len(monitor), len(sealed))
    allp = set(train) | set(monitor) | set(sealed)
    assert len(allp) == 97 * 97 and not (set(train) & set(monitor)) and not (set(train) & set(sealed))
    assert all(int(bytes(arithmetic(p)[6:8])) == sum(p) % 97 for p in allp)
    # swapped form of a held pair is also held
    assert all((b, a) in sealed for a, b in sealed)
    # output-layer and layer-2 local grads reduce to autograd when F2 = W3
    m, F, opt = make(1)
    rows = list(islice(stream(train, 1), 1))[0]
    F["F2"] = m.l3.weight.detach().clone()
    g, _ = local_grads(m, F, rows, False)
    data = rows_tensor(rows)
    loss = cross_entropy(m(data[:, :-1]).flatten(0, 1), data[:, 1:].flatten())
    loss.backward()
    for name in ("l3.weight", "l3.bias", "l2.weight", "l2.bias"):
        p = dict(m.named_parameters())[name]
        assert T.allclose(p.grad, g[name], atol=1e-6, rtol=1e-4), name
    # true backprop d1 is NOT of the DFA form, so l1 is expected to differ; record the cosine
    cos = T.nn.functional.cosine_similarity(dict(m.named_parameters())["l1.weight"].grad.flatten(), g["l1.weight"].flatten(), 0)
    # short training smoke: 32 updates each arm, finite
    res = {}
    for arm in ("bp", "dfa", "dfa_fix"):
        m, F, opt = make(1)
        for rows in islice(stream(train, 1), 32):
            l = (update_bp if arm == "bp" else lambda *a: update_dfa(*a, freeze_emb=(arm == "dfa_fix")))(m, F, opt, rows)
            assert math.isfinite(l)
        res[arm] = dict(loss=l, taught=score(m, train[:256])["exact"])
    out = dict(split=[len(train), len(monitor), len(sealed)], l1_cosine_bp_vs_dfa_with_F2_eq_W3=float(cos), smoke=res)
    OUT.mkdir(exist_ok=True)
    (OUT / "QUALIFY.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


# ---------------------------------------------------------------- main run
WD = {"01": .1, "03": .3, "10": 1.}
ARMS = ("bp", "dfa", "dfa_fix", "bp_v181", "dfa_v181", "dfa_outwd", "dfa_lowlr", "sign",
        "bpw01", "bpw03", "bpw10", "signw10")


def subsample(train, frac, seed):
    """Keep a fraction of ALL 4,753 unordered pairs as taught, drawn from the taught groups; both orders stay together."""
    groups = sorted({(min(a, b), max(a, b)) for a, b in train})
    k = round(frac * 4753)
    assert k <= len(groups), (k, len(groups))
    rng = Random(seed + 500 + round(frac * 1000))
    keep = rng.sample(groups, k)
    return expand(sorted(keep))


def run(arm, seed, dose, frac=.5):
    OUT.mkdir(exist_ok=True)
    path = OUT / (f"{arm}_{seed}.json" if frac == .5 else f"{arm}_{seed}_f{round(frac * 100)}.json")
    assert arm in ARMS, arm
    train, monitor, sealed = panels(seed)
    if frac != .5:
        train = subsample(train, frac, seed)
    m, F, opt = make(seed, arm)
    upd = update_bp if arm.startswith("bp") else (lambda *a: update_dfa(*a, freeze_emb=(arm == "dfa_fix")))
    batches = stream(train, seed)
    curve = []
    t0 = time.process_time()
    for stop in range(BLOCK, dose + 1, BLOCK):
        losses = [upd(m, F, opt, rows) for rows in islice(batches, BLOCK)]
        row = dict(step=stop, loss=sum(losses) / len(losses), taught=score(m, train), monitor=score(m, monitor),
                   sealed=score(m, sealed), cpu=time.process_time() - t0)
        curve.append(row)
        print(arm, seed, stop, f"loss {row['loss']:.3f}", f"taught {row['taught']['exact']}/{row['taught']['n']}",
              f"monitor {row['monitor']['exact']}/{row['monitor']['n']}", f"sealed {row['sealed']['exact']}/{row['sealed']['n']}",
              f"cpu {row['cpu']:.0f}s", flush=True)
        path.write_text(json.dumps(dict(arm=arm, seed=seed, dose=dose, frac=frac, taught_n=len(train),
                                        params=sum(p.numel() for p in m.parameters()), curve=curve), indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "qualify":
        qualify()
    else:
        run(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]) if len(sys.argv) > 4 else .5)
