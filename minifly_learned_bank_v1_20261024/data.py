"""Worlds for LEARNED-BANK V1: single-digit prompts `a+b=`, orbit splits and tasks (evaluator side only)."""
import hashlib

import numpy as np

ORBITS = [(a, b) for a in range(10) for b in range(a, 10)]          # 55 unordered pairs
HELD_ORBITS = 15
MIN_TRAIN_ORBITS_PER_DIGIT = 4


def seed_of(name):
    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little")


def answer(task, a, b):
    if task == "MOD":
        return ord(str((a + b) % 10))
    if task == "COPY":
        return ord(str(a))
    raise ValueError(task)


def prompt(a, b):
    return f"{a}+{b}=".encode()


def ordered(orbits):
    out = []
    for a, b in orbits:
        out.append((a, b))
        if a != b:
            out.append((b, a))
    return out


def make_world(name):
    """Deterministic split of the 55 orbits: 15 held out (both orientations never taught), 40 for training."""
    rng = np.random.default_rng(seed_of("split:" + name))
    for _ in range(10000):
        idx = rng.permutation(len(ORBITS))
        held = sorted(ORBITS[i] for i in idx[:HELD_ORBITS])
        train = sorted(ORBITS[i] for i in idx[HELD_ORBITS:])
        counts = np.zeros(10, int)
        for a, b in train:
            counts[a] += 1
            counts[b] += 1 if a != b else 0
        if counts.min() >= MIN_TRAIN_ORBITS_PER_DIGIT:
            return {"name": name, "held": held, "train": train,
                    "held_pairs": ordered(held), "train_pairs": ordered(train),
                    "model_seed": seed_of("model:" + name), "order_seed": seed_of("order:" + name)}
    raise RuntimeError("no admissible split")


def derangement(task, pairs, name):
    """Fixed reassignment of training answers among training pairs so that no pair keeps its true answer."""
    rng = np.random.default_rng(seed_of("shuffle:" + task + ":" + name))
    truth = [answer(task, a, b) for a, b in pairs]
    for _ in range(100000):
        perm = rng.permutation(len(pairs))
        taught = [truth[i] for i in perm]
        if all(t != u for t, u in zip(taught, truth)):
            return dict(zip(pairs, taught))
    raise RuntimeError("no derangement")


def schedule(world, passes):
    """Interleaved schedule: every pass presents all training pairs once, reshuffled per pass."""
    rng = np.random.default_rng(world["order_seed"])
    pairs = world["train_pairs"]
    out = []
    for _ in range(passes):
        out.extend(pairs[i] for i in rng.permutation(len(pairs)))
    return out
