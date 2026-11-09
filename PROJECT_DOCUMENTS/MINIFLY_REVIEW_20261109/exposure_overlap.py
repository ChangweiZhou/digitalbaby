"""How much of the "held" material in the E16 stream was taught in another form? Pure Python, no harness import.

Re-derives exp11_cell.split and exp16_integrated.triples (as at codex/minifly-consolidated-20261023 c8cfbbbc) and reports,
per seed: (1) the share of each mod-97 operation's sealed panel pairs (a, b) that are also the first step of a TAUGHT
composition with that operator, where the intermediate a op b is supervised; (2) for the held compositions, the share
whose first step, second step, or both occur inside a taught composition.
Run with:  python3 -I exposure_overlap.py [SEED ...]
"""
import sys
from random import Random

SYM = {"add": "+", "sub": "-", "max": ">", "min": "<"}
OPS = {v: k for k, v in SYM.items()}
PAIRS = [(a, b) for a in SYM.values() for b in SYM.values()]
HELD = [("+", "-"), ("-", "+"), (">", "<"), ("<", ">")]
TAUGHT = [p for p in PAIRS if p not in HELD]
F1 = [("add", 97, b"r"), ("sub", 97, b"t"), ("max", 97, b"x"), ("min", 97, b"n")]


def apply(op, m, a, b):
    return {"add": (a + b) % m, "sub": (a - b) % m, "max": max(a, b), "min": min(a, b)}[op]


def split(task, frac, seed):
    op, m, code = task
    pairs = [(a, b) for a in range(m) for b in range(m)]
    Random(seed * 1000 + round(frac * 100) + m * 7 + code[0]).shuffle(pairs)
    k = round(frac * len(pairs))
    return pairs[:k], pairs[k:]


def triples(pair, seed, n, exclude=frozenset(), split_="taught"):
    rng = Random(seed * 31 + ord(pair[0]) * 7 + ord(pair[1])) if split_ == "taught" else Random(f"{split_}|{seed}|{pair[0]}{pair[1]}")
    out, seen = [], set(exclude)
    while len(out) < n:
        tr = (rng.randrange(97), rng.randrange(97), rng.randrange(97))
        if tr not in seen:
            seen.add(tr); out.append(tr)
    return out


for seed in [int(s) for s in sys.argv[1:]] or [73129, 200003, 600013]:
    taught = {p: triples(p, seed, 2000) for p in TAUGHT}
    step1, step2 = {}, {}
    for (o1, o2), trs in taught.items():
        for a, b, c in trs:
            step1.setdefault(o1, set()).add((a, b))
            step2.setdefault(o2, set()).add((apply(OPS[o1], 97, a, b), c))
    f1_taught = {SYM[t[0]]: set(split(t, .25, seed)[0]) for t in F1}
    shares = []
    for t in F1:
        panel = Random(seed + 3).sample(split(t, .25, seed)[1], 400)
        shares.append(f"{t[0]}97 {100 * sum(p in step1[SYM[t[0]]] for p in panel) / 400:.0f}%")
    all_taught = frozenset().union(*[set(v) for v in taught.values()])
    s1 = s2 = both = s1f = n = 0
    for o1, o2 in HELD:
        for a, b, c in triples((o1, o2), seed, 600, exclude=all_taught, split_="held"):
            h1 = (a, b) in step1[o1]
            h2 = (apply(OPS[o1], 97, a, b), c) in step2[o2]
            s1 += h1; s2 += h2; both += h1 and h2; s1f += h1 or (a, b) in f1_taught[o1]; n += 1
    print(f"seed {seed}: sealed operation pairs taught as a composition's first step: {', '.join(shares)}")
    print(f"   held compositions: first step inside a taught composition {100 * s1 / n:.0f}%, second step {100 * s2 / n:.0f}%, "
          f"both {100 * both / n:.0f}%; first step there or in the mod-97 family {100 * s1f / n:.0f}%")
