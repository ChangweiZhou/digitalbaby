"""E7: de-confounded composition with four operators (predictions: E45_PREDICTIONS.md, E7 section).

Ops mod 97: add r '+', sub t '-', max x '>', min n '<'. Two-step: c a o1 b o2 c = ; 12 taught ordered pairs, 4 held cross pairs.
Window 18, hidden 256 (EXP45_WINDOW=18 set here before importing exp45).
"""
import os
os.environ["EXP45_WINDOW"] = "18"
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[k] = "1"
import json
import sys
import time
from collections import Counter
from pathlib import Path
from random import Random

import torch as T

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp45 as X

T.set_num_threads(1)
assert X.W == 18
EOS, BOS = X.EOS, X.BOS
OUT = X.OUT
SEEDS = X.SEEDS
MOD = 97
OPS = {"+": ("add", b"r"), "-": ("sub", b"t"), ">": ("max", b"x"), "<": ("min", b"n")}
PAIRS = [(a, b) for a in OPS for b in OPS]
HELD_PAIRS = [("+", "-"), ("-", "+"), (">", "<"), ("<", ">")]
TAUGHT_PAIRS = [p for p in PAIRS if p not in HELD_PAIRS]
ARMS = ("SCR", "SCRc", "SCRcsh", "SCRbp", "SCRcbp")
DOSE = int(os.environ.get("E7_DOSE", "8192"))
DTAG = "" if DOSE == 8192 else f"_d{DOSE}"


def apply(op, a, b):
    return {"+": (a + b) % MOD, "-": (a - b) % MOD, ">": max(a, b), "<": min(a, b)}[op]


def render_single(op, a, b):
    code = OPS[op][1]
    return list(code + f"{a:02d}".encode() + op.encode() + f"{b:02d}".encode() + b"="), list(f"{apply(op, a, b):02d}".encode()) + [EOS]


def single_split(op, frac, seed):
    groups = [(a, b) for a in range(MOD) for b in range(MOD)]          # ordered pairs: max/min are symmetric, add is, sub is not
    rng = Random(seed * 1000 + round(frac * 100) + ord(op))
    rng.shuffle(groups)
    k = round(frac * len(groups))
    return groups[:k], groups[k:]


def render_pair(o1, o2, a, b, c, canonical):
    prompt = list(b"c" + f"{a:02d}".encode() + o1.encode() + f"{b:02d}".encode() + o2.encode() + f"{c:02d}".encode() + b"=")
    i = apply(o1, a, b); f = apply(o2, i, c)
    if canonical:
        target = list(OPS[o1][1] + f"{a:02d}".encode() + o1.encode() + f"{b:02d}".encode() + b"=") + list(f"{i:02d}".encode()) + [ord(";")] + \
                 list(OPS[o2][1] + f"{i:02d}".encode() + o2.encode() + f"{c:02d}".encode() + b"=") + list(f"{f:02d}".encode()) + [EOS]
    else:
        target = list(f"{i:02d}".encode()) + [ord(";")] + list(f"{f:02d}".encode()) + [EOS]
    return prompt, target


def triples(pair, seed, n, exclude=frozenset()):
    rng = Random(seed * 31 + ord(pair[0]) * 7 + ord(pair[1]))
    out, seen = [], set(exclude)
    while len(out) < n:
        tr = (rng.randrange(MOD), rng.randrange(MOD), rng.randrange(MOD))
        if tr not in seen:
            seen.add(tr); out.append(tr)
    return out


def score(m, items, canonical):
    outs = X.respond(m, [p for p, _ in items], None, cap=22 if canonical else 7)
    res = dict(n=len(items), exact=sum(o == t for o, (_, t) in zip(outs, items)), final=sum(o[-3:] == t[-3:] for o, (_, t) in zip(outs, items)))
    if canonical:
        res["intermediate"] = sum(o[7:9] == t[7:9] for o, (_, t) in zip(outs, items))
        res["step1_prompt"] = sum(o[:7] == t[:7] for o, (_, t) in zip(outs, items))
        res["step2_prompt"] = sum(o[10:17] == t[10:17] for o, (_, t) in zip(outs, items))
    else:
        res["intermediate"] = sum(o[:3] == t[:3] for o, (_, t) in zip(outs, items))
    return res


def develop(seed):
    path = OUT / f"e7_dev{X.WTAG[4:]}_{seed}.pt"
    if path.exists():
        return
    m = X.new_net(seed)
    opt = T.optim.AdamW(m.parameters(), lr=.001, weight_decay=.3, betas=(.9, .98))
    base = [render_single(op, a, b) for op in OPS for a, b in single_split(op, .5, seed)[0]]
    rng, order = Random(seed + 21), Random(seed + 1)
    t0 = time.process_time(); curve = []
    for stop in range(X.BLOCK, X.DEV_UPDATES + 1, X.BLOCK):
        losses = [X.update(m, opt, X.with_left_noise(order.sample(base, X.BATCH), rng), False) for _ in range(X.BLOCK)]
        if stop % 4096 == 0 or stop == X.DEV_UPDATES:
            held = {op: [render_single(op, a, b) for a, b in Random(seed + 2).sample(single_split(op, .5, seed)[1], 600)] for op in OPS}
            row = dict(step=stop, loss=sum(losses) / len(losses), cpu=time.process_time() - t0, sealed={op: X.score(m, it) for op, it in held.items()})
            curve.append(row)
            print("e7 dev", seed, stop, {op: v["exact"] for op, v in row["sealed"].items()}, flush=True)
    T.save(m.state_dict(), path)
    (OUT / f"e7_dev{X.WTAG[4:]}_{seed}.json").write_text(json.dumps(dict(seed=seed, curve=curve), indent=1))


def life(arm, seed):
    path = OUT / f"e7_life_{arm}{X.WTAG[4:]}{DTAG}_{seed}.json"
    m = X.new_net(seed)
    m.load_state_dict(T.load(OUT / f"e7_dev{X.WTAG[4:]}_{seed}.pt"))
    canonical = arm.startswith("SCRc")
    local = arm not in ("SCRbp", "SCRcbp")
    opt = T.optim.AdamW(m.parameters(), lr=.001, weight_decay=(1. if local else .3), betas=(.9, .98))
    taught = {p: triples(p, seed, 2000) for p in TAUGHT_PAIRS}
    teach = [render_pair(*p, *tr, canonical) for p in TAUGHT_PAIRS for tr in taught[p]]
    panels = {f"{o1}{o2}": [render_pair(o1, o2, *tr, canonical) for tr in triples((o1, o2), seed + 1, 600)] for o1, o2 in HELD_PAIRS}
    within = []
    for p in (("+", "+"), ("-", "-"), (">", ">"), ("<", "<")):
        within += [render_pair(*p, *tr, canonical) for tr in triples(p, seed + 1, 150, exclude=frozenset(taught[p]))]
    panels["within"] = within
    single = {op: [render_single(op, a, b) for a, b in Random(seed + 2).sample(single_split(op, .5, seed)[1], 600)] for op in OPS}
    if arm == "SCRcsh":
        rng = Random(seed + 5); finals = [tt[-3:-1] for _, tt in teach]; rng.shuffle(finals)
        teach = [(p, tt[:-3] + f + [EOS]) for (p, tt), f in zip(teach, finals)]
    shortcuts = {}
    for name, items in panels.items():
        finals = Counter(tuple(tt[-3:-1]) for _, tt in teach).most_common(1)[0][0]
        shortcuts[name] = dict(n=len(items), frequency=sum(tuple(tt[-3:-1]) == finals for _, tt in items))
    t0 = time.process_time()
    curve = [dict(step=0, compose={k: score(m, it, canonical) for k, it in panels.items()}, single={op: X.score(m, it) for op, it in single.items()})]
    bt = X.batches(teach, seed + 3)
    for stop in range(X.BLOCK, DOSE + 1, X.BLOCK):
        losses = [X.update(m, opt, next(bt), local) for _ in range(X.BLOCK)]
        if stop % 2048 == 0:
            curve.append(dict(step=stop, loss=sum(losses) / len(losses), cpu=time.process_time() - t0,
                              compose={k: score(m, it, canonical) for k, it in panels.items()}, single={op: X.score(m, it) for op, it in single.items()}))
            print("e7", arm, seed, stop, {k: v["final"] for k, v in curve[-1]["compose"].items()}, f"cpu {curve[-1]['cpu']:.0f}s", flush=True)
    path.write_text(json.dumps(dict(arm=arm, seed=seed, canonical=canonical, shortcuts=shortcuts, curve=curve), indent=1))


def qualify():
    out = {}
    for op in OPS:
        for a, b in Random(0).sample([(a, b) for a in range(MOD) for b in range(MOD)], 50):
            p, t = render_single(op, a, b); assert len(p) == 7 and int(bytes(t[:2])) == apply(op, a, b)
    for pair in PAIRS:
        for tr in triples(pair, 1, 20):
            p, t = render_pair(*pair, *tr, True); i = apply(pair[0], tr[0], tr[1]); f = apply(pair[1], i, tr[2])
            assert int(bytes(t[7:9])) == i and int(bytes(t[17:19])) == f and bytes(t[10:11]) == OPS[pair[1]][1]
            p2, t2 = render_pair(*pair, *tr, False); assert int(bytes(t2[:2])) == i and int(bytes(t2[3:5])) == f
    # every operator appears in both positions among taught pairs; held pairs disjoint
    firsts = {a for a, _ in TAUGHT_PAIRS}; seconds = {b for _, b in TAUGHT_PAIRS}
    assert firsts == set(OPS) and seconds == set(OPS) and not (set(HELD_PAIRS) & set(TAUGHT_PAIRS)) and len(TAUGHT_PAIRS) == 12
    out["taught_pairs"] = ["".join(p) for p in TAUGHT_PAIRS]; out["held_pairs"] = ["".join(p) for p in HELD_PAIRS]
    p, t = render_pair("+", "-", 24, 81, 7, True)
    x = T.tensor([[BOS, *p, *t[:14]]]); win = T.nn.functional.pad(x, (X.W - 1, 0), value=BOS).unfold(1, X.W, 1)[0, -1].tolist()
    out["window_when_writing_third_operand"] = bytes(b for b in win if b < 256).decode(); assert "07" in out["window_when_writing_third_operand"]
    (OUT / "QUALIFY_E7.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    if sys.argv[1] == "qualify":
        qualify()
    elif sys.argv[1] == "dev":
        develop(int(sys.argv[2]))
    else:
        life(sys.argv[1], int(sys.argv[2]))
