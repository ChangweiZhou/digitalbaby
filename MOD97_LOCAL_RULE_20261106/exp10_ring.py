"""E10: the ring with innate operations (predictions: E45_PREDICTIONS.md, E10 section).

Parsing by position is supplied. Learned: digits->bump, operator byte->op choice, bump->digits.
Innate on bumps over 97 positions: add (circular convolution), sub (circular correlation),
max and min (from cumulative distributions). Flat ablation: learned MLP replaces the innate ops.
"""
import os
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[k] = "1"
import json
import sys
import time
from pathlib import Path
from random import Random

import torch as T
from torch import nn
from torch.nn.functional import cross_entropy

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp3_state import lin
import exp7_decon as E7

T.set_num_threads(1)
OUT = Path(__file__).resolve().parent / "exp45"
SEEDS = (73129, 93683, 116089)
MOD = 97
OPS = ["+", "-", ">", "<"]
OPB = {o: ord(o) for o in OPS}
PAIRS = E7.PAIRS; HELD_PAIRS = E7.HELD_PAIRS; TAUGHT_PAIRS = E7.TAUGHT_PAIRS
CONF_TAUGHT = [("+", "+"), ("-", "-")]; CONF_HELD = [("+", "-"), ("-", "+")]
BATCH, BLOCK = 64, 512
DEV_UPDATES = int(os.environ.get("E10_DEV_DOSE", "8192")); LIFE_UPDATES = 8192
TAG = "" if DEV_UPDATES == 8192 else f"_d{DEV_UPDATES}"
IDX = T.arange(MOD)
ADD_IDX = (IDX[:, None] - IDX[None, :]) % MOD      # k - i  -> index of b for add
SUB_IDX = (IDX[:, None] + IDX[None, :]) % MOD      # k + i  -> ... (sub: a - b = k  <=> b = a - k)


# ---------------------------------------------------------------- innate operations on bumps (B x 97)
def op_add(a, b):      # P(k) = sum_i a(i) b((k-i) mod 97)
    return T.einsum("bi,bki->bk", a, b[:, ADD_IDX])


def op_sub(a, b):      # P(k) = sum_i a(i) b((i-k) mod 97)
    return T.einsum("bi,bki->bk", a, b[:, (IDX[None, :] - IDX[:, None]) % MOD])


def op_max(a, b):
    Fa, Fb = a.cumsum(1), b.cumsum(1)
    F = Fa * Fb
    return T.cat([F[:, :1], F[:, 1:] - F[:, :-1]], 1)


def op_min(a, b):
    Sa = a.flip(1).cumsum(1).flip(1); Sb = b.flip(1).cumsum(1).flip(1)
    S = Sa * Sb
    return T.cat([S[:, :-1] - S[:, 1:], S[:, -1:]], 1)


INNATE = [op_add, op_sub, op_max, op_min]


class RingNet(nn.Module):
    def __init__(self, flat=False, H=128):
        super().__init__()
        self.flat = flat
        self.digit = nn.Embedding(10, 16)
        self.enc1 = nn.Linear(32, H); self.enc2 = nn.Linear(H, MOD)           # two digits -> bump
        self.opsel = nn.Embedding(256, 4)                                      # operator byte -> op logits
        self.dec = nn.Linear(MOD, 20)                                          # bump -> tens, units
        if flat:
            self.mix1 = nn.Linear(2 * MOD + 4, H); self.mix2 = nn.Linear(H, MOD)

    def bump(self, tens, units, local):
        x = T.cat([self.digit(tens), self.digit(units)], 1)
        return lin(T.tanh(lin(x, self.enc1, local)), self.enc2, local).softmax(1)

    def combine(self, a, b, opbyte, local):
        w = self.opsel(opbyte).softmax(1)                                      # B x 4
        if self.flat:
            z = T.tanh(lin(T.cat([a, b, w], 1), self.mix1, local))
            return lin(z, self.mix2, local).softmax(1)
        outs = T.stack([f(a, b) for f in INNATE], 1)                           # B x 4 x 97
        return (w[:, :, None] * outs).sum(1)

    def forward(self, a, b, o1, c=None, o2=None, local=False):
        """a, b, c: (tens, units) tensors; o1, o2: operator bytes. Returns digit logits of the result bump(s)."""
        x = self.combine(self.bump(*a, local), self.bump(*b, local), o1, local)
        inter = lin(x, self.dec, local)
        if c is None:
            return inter, None
        y = self.combine(x, self.bump(*c, local), o2, local)
        return inter, lin(y, self.dec, local)


def digits(v):
    v = T.tensor(v); return v // 10, v % 10


def batch_single(items):
    a, b, o, y = zip(*items)
    return digits(list(a)), digits(list(b)), T.tensor([OPB[x] for x in o]), T.tensor(y)


def batch_pair(items):
    a, b, c, o1, o2, i, f = zip(*items)
    return digits(list(a)), digits(list(b)), digits(list(c)), T.tensor([OPB[x] for x in o1]), T.tensor([OPB[x] for x in o2]), T.tensor(i), T.tensor(f)


def loss_digits(logits, y):
    return cross_entropy(logits[:, :10], y // 10) + cross_entropy(logits[:, 10:], y % 10)


def update(m, opt, items, local, pair):
    if pair:
        a, b, c, o1, o2, i, f = batch_pair(items)
        li, lf = m(a, b, o1, c, o2, local=local)
        loss = loss_digits(lf, f) + loss_digits(li, i)
    else:
        a, b, o, y = batch_single(items)
        li, _ = m(a, b, o, local=local)
        loss = loss_digits(li, y)
    opt.zero_grad(); loss.backward(); opt.step()
    return float(loss.detach())


@T.no_grad()
def score_pairs(m, items):
    a, b, c, o1, o2, i, f = batch_pair(items)
    li, lf = m(a, b, o1, c, o2)
    pf = li[:, :10].argmax(1) * 10 + li[:, 10:].argmax(1); ff = lf[:, :10].argmax(1) * 10 + lf[:, 10:].argmax(1)
    return dict(n=len(items), final=int((ff == f).sum()), intermediate=int((pf == i).sum()))


@T.no_grad()
def score_single(m, items):
    a, b, o, y = batch_single(items)
    li, _ = m(a, b, o)
    p = li[:, :10].argmax(1) * 10 + li[:, 10:].argmax(1)
    return dict(n=len(items), exact=int((p == y).sum()))


def single_items(seed, frac=.5):
    out, held = [], {}
    for o in OPS:
        t, rest = E7.single_split(o, frac, seed)
        out += [(a, b, o, E7.apply(o, a, b)) for a, b in t]
        held[o] = [(a, b, o, E7.apply(o, a, b)) for a, b in Random(seed + 2).sample(rest, 600)]
    return out, held


def pair_items(pair, trs):
    o1, o2 = pair
    return [(a, b, c, o1, o2, E7.apply(o1, a, b), E7.apply(o2, E7.apply(o1, a, b), c)) for a, b, c in trs]


def panels(seed):
    p = {f"{o1}{o2}": pair_items((o1, o2), E7.triples((o1, o2), seed + 1, 600)) for o1, o2 in HELD_PAIRS}
    within = []
    for pr in (("+", "+"), ("-", "-"), (">", ">"), ("<", "<")):
        within += pair_items(pr, E7.triples(pr, seed + 1, 150, exclude=frozenset(E7.triples(pr, seed, 2000))))
    p["within"] = within
    return p


def develop(seed, flat):
    tag = "flat" if flat else "ring"
    path = OUT / f"e10_dev_{tag}{TAG}_{seed}.pt"
    if path.exists():
        return
    T.manual_seed(seed); m = RingNet(flat)
    opt = T.optim.AdamW(m.parameters(), lr=.003, weight_decay=.01)
    items, held = single_items(seed)
    order = Random(seed + 1); t0 = time.process_time(); curve = []
    for stop in range(BLOCK, DEV_UPDATES + 1, BLOCK):
        losses = [update(m, opt, order.sample(items, BATCH), False, False) for _ in range(BLOCK)]
        if stop % 4096 == 0 or stop == DEV_UPDATES:
            row = dict(step=stop, loss=sum(losses) / len(losses), cpu=time.process_time() - t0, single={o: score_single(m, it) for o, it in held.items()})
            curve.append(row); print("e10 dev", tag, seed, stop, {o: v["exact"] for o, v in row["single"].items()}, flush=True)
    T.save(m.state_dict(), path)
    (OUT / f"e10_dev_{tag}{TAG}_{seed}.json").write_text(json.dumps(dict(seed=seed, flat=flat, params=sum(p.numel() for p in m.parameters()), curve=curve), indent=1))


def life(arm, seed):
    """arms: ring_zero, ring_sign, ring_bp, ring_conf_sign, ring_conf_bp, flat_bp, flat_conf_bp"""
    path = OUT / f"e10_life_{arm}{TAG}_{seed}.json"
    flat = arm.startswith("flat")
    T.manual_seed(seed); m = RingNet(flat)
    m.load_state_dict(T.load(OUT / f"e10_dev_{'flat' if flat else 'ring'}{TAG}_{seed}.pt"))
    conf = "conf" in arm
    local = arm.endswith("sign")
    taught_pairs = CONF_TAUGHT if conf else TAUGHT_PAIRS
    teach = [it for p in taught_pairs for it in pair_items(p, E7.triples(p, seed, 2000))]
    pan = panels(seed)
    if conf:
        pan = {f"{o1}{o2}": pair_items((o1, o2), E7.triples((o1, o2), seed + 1, 600)) for o1, o2 in CONF_HELD}
        pan["within"] = [it for p in CONF_TAUGHT for it in pair_items(p, E7.triples(p, seed + 1, 300, exclude=frozenset(E7.triples(p, seed, 2000))))]
    _, held_single = single_items(seed)
    curve = [dict(step=0, compose={k: score_pairs(m, it) for k, it in pan.items()}, single={o: score_single(m, it) for o, it in held_single.items()})]
    print("e10", arm, seed, 0, {k: v["final"] for k, v in curve[-1]["compose"].items()}, flush=True)
    if arm != "ring_zero":
        opt = T.optim.AdamW(m.parameters(), lr=.001, weight_decay=(1. if local else .3), betas=(.9, .98))
        order = Random(seed + 3); t0 = time.process_time()
        for stop in range(BLOCK, LIFE_UPDATES + 1, BLOCK):
            losses = [update(m, opt, order.sample(teach, BATCH), local, True) for _ in range(BLOCK)]
            if stop % 2048 == 0:
                curve.append(dict(step=stop, loss=sum(losses) / len(losses), cpu=time.process_time() - t0, compose={k: score_pairs(m, it) for k, it in pan.items()}, single={o: score_single(m, it) for o, it in held_single.items()}))
                print("e10", arm, seed, stop, {k: v["final"] for k, v in curve[-1]["compose"].items()}, flush=True)
    path.write_text(json.dumps(dict(arm=arm, seed=seed, flat=flat, conf=conf, curve=curve), indent=1))


def qualify():
    out = {}
    # innate ops on one-hot bumps reproduce integer arithmetic exactly
    rng = Random(0); ok = {o: 0 for o in OPS}; n = 300
    for _ in range(n):
        a, b = rng.randrange(MOD), rng.randrange(MOD)
        ba, bb = T.zeros(1, MOD), T.zeros(1, MOD); ba[0, a] = 1; bb[0, b] = 1
        for o, f in zip(OPS, INNATE):
            ok[o] += int(f(ba, bb).argmax(1).item() == E7.apply(o, a, b))
    out["innate_exact_on_onehot"] = {o: f"{v}/{n}" for o, v in ok.items()}; assert all(v == n for v in ok.values())
    # distributions stay normalized
    ba, bb = T.rand(4, MOD).softmax(1), T.rand(4, MOD).softmax(1)
    out["mass_after_ops"] = [round(float(f(ba, bb).sum(1).mean()), 6) for f in INNATE]; assert all(abs(x - 1) < 1e-5 for x in out["mass_after_ops"])
    # composition by wiring: one-hot bumps through two ops
    for _ in range(100):
        a, b, c = (rng.randrange(MOD) for _ in range(3)); o1, o2 = rng.choice(OPS), rng.choice(OPS)
        bumps = [T.zeros(1, MOD) for _ in range(3)]
        for t, v in zip(bumps, (a, b, c)): t[0, v] = 1
        x = INNATE[OPS.index(o1)](bumps[0], bumps[1]); y = INNATE[OPS.index(o2)](x, bumps[2])
        assert y.argmax(1).item() == E7.apply(o2, E7.apply(o1, a, b), c)
    out["composition_by_wiring_ok"] = True
    out["params"] = dict(ring=sum(p.numel() for p in RingNet(False).parameters()), flat=sum(p.numel() for p in RingNet(True).parameters()))
    # smoke both modes
    m = RingNet(False); opt = T.optim.AdamW(m.parameters(), lr=.001)
    items, _ = single_items(1); l = update(m, opt, items[:64], True, False); assert l == l
    teach = pair_items(("+", "-"), E7.triples(("+", "-"), 1, 64)); l = update(m, opt, teach, False, True); assert l == l
    out["smoke"] = l
    (OUT / "QUALIFY_E10.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    if sys.argv[1] == "qualify":
        qualify()
    elif sys.argv[1] == "dev":
        develop(int(sys.argv[2]), sys.argv[3] == "flat")
    else:
        life(sys.argv[1], int(sys.argv[2]))
