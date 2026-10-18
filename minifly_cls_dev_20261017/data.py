"""Task data: the single-digit split of the lineage comparison / addition diagnosis (same draws), plus order contrasts.

No learner imports this module's answers except through the taught records it is given.
"""
import random

WORLDS = (202610143901, 202610143902, 202610143903)


def split(world):
    # Identical to addition_core_lineage_comparison_dev_v1_20261015/validate_design.py split().
    rng = random.Random(world)
    diagonal = [(d, d) for d in range(10)]
    original = [(a, b) for a in range(10) for b in range(a + 1, 10)]
    for _ in range(1, 10001):
        off = original.copy()
        rng.shuffle(off)
        held, new, old = off[:10], off[10:20], diagonal + off[20:]
        answers = {a + b for a, b in old}
        degrees = [sum(d in pair for pair in old) for d in range(10)]
        if answers == set(range(19)) and min(degrees) >= 3:
            return dict(old=old, new=new, held=held)
    raise ValueError("split failed")


def ordered(orbits):
    return sorted({p for a, b in orbits for p in ((a, b), (b, a))})


def record(pair, op="+"):
    a, b = pair
    return f"{a}{op}{b}=".encode(), str(a + b if op == "+" else a * b).encode()


def addition(world):
    sp = split(world)
    taught = ordered(sp["old"]) + ordered(sp["new"])
    held = ordered(sp["held"])
    return dict(taught_pairs=taught, held_pairs=held,
                taught=[record(p) for p in taught], held=[record(p) for p in held],
                mul_taught=[record(p, "*") for p in taught], mul_held=[record(p, "*") for p in held])


def order_contrasts():
    """G2: prompts that contain the same bytes in a different order (or differ only by brackets / an early byte),
    each with its own correct answer. Memorising all of them requires a key that keeps order."""
    recs = []
    for a, b, c in ((2, 3, 4), (5, 3, 2), (1, 4, 3), (3, 2, 5), (4, 1, 2), (2, 5, 3)):
        for expr in (f"({a}+{b})*{c}", f"{a}+{b}*{c}", f"{a}*({b}+{c})", f"{a}*{b}+{c}"):
            recs.append((f"{expr}=".encode(), str(eval(expr)).encode()))
    for a, b in ((9, 2), (7, 3), (8, 5), (6, 1)):
        recs.append((f"{a}-{b}=".encode(), str(a - b).encode()))
        recs.append((f"{b}-{a}=".encode(), str(b - a).encode()))
    return recs
