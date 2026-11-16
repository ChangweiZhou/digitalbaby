"""E43: the unrelated task, as runnable code. Coordinator's specification file, 16 Nov 2026. A task on bytes no family uses: two symbols are
shown, then 0 to 12 fillers, then a query byte; the answer is the two symbols and the end symbol. random.Random only.
    python3 -I e43_unrelated_task_spec.py      self-checks on E26's birth of seed 73129 (no training)"""
import hashlib
import sys
from pathlib import Path
from random import Random

SYM = list(range(128, 192))          # 64 symbols
FILL = list(range(192, 224))         # 32 fillers
CODE, ASK = 224, 225                 # the task's code byte and its query byte
EOS = 256                            # the harness's end symbol (mod97_local.py line 34)
KMAX, TAUGHT = 12, 2048


def split(seed):
    pairs = [(a, b) for a in SYM for b in SYM]                       # 4,096 ordered pairs
    Random(f"e43|{seed}|split").shuffle(pairs)
    return pairs[:TAUGHT], pairs[TAUGHT:]


def render(a, b, k, rng):
    return [CODE, a, b] + [rng.choice(FILL) for _ in range(k)] + [ASK], [a, b, EOS]


def stream(seed):
    taught, _ = split(seed); rng = Random(f"e43|{seed}|stream")
    while True:
        order = list(range(len(taught))); rng.shuffle(order)
        for i in order:
            yield render(*taught[i], rng.randint(0, KMAX), rng)


def batches(seed, batch=64):
    st = stream(seed)
    while True:
        yield [next(st) for _ in range(batch)]


def panel(seed, which, n=300, k=12):
    taught, held = split(seed); rng = Random(f"e43|{seed}|panel|{which}")
    return [render(a, b, k, rng) for a, b in rng.sample(taught if which == "taught" else held, n)]


def digest(seed, updates):
    h = hashlib.sha256(); bt = batches(seed)
    for _ in range(updates):
        for p, y in next(bt):
            h.update(bytes([len(p)]) + bytes(p) + bytes(v % 256 for v in y))
    return h.hexdigest()


if __name__ == "__main__":
    HERE = Path(__file__).resolve()
    for cand in (HERE.parent, HERE.parents[2] / "MOD97_LOCAL_RULE_20261106"):
        if (cand / "exp26_candidate.py").exists():
            sys.path.insert(0, str(cand)); M = cand; break
    import exp26_candidate as C26
    import arch26
    import torch as T
    T.set_num_threads(1); arch26.install(); E, X = C26.E, C26.X
    for seed in (73129, 93683, 116089):
        t, h = split(seed)
        print(f"seed {seed}: taught {len(t)}, held {len(h)}, disjoint {set(t).isdisjoint(h)}; digest of the first 64 updates {digest(seed, 64)[:16]}; held panel digest "
              f"{hashlib.sha256(repr(panel(seed, 'held')).encode()).hexdigest()[:16]}")
    used = set(b"0123456789+-><=;ABCDEFGHIJKLMNOPQRSTUVWXYZpqvwrtxnsuyzcefgh")
    mine = set(SYM) | set(FILL) | {CODE, ASK}
    print("no byte shared with any family's prompts or answers (the end symbol aside):", used.isdisjoint(mine), "| every byte below 256:", max(mine) < 256)
    seed = 73129
    items = next(batches(seed)); x, y, mask = X.batch_tensor(items)
    print("batch tensor:", tuple(x.shape), "answer positions:", int(mask.sum()), "(64 items x 3)")
    m = arch26.new_net(seed); birth = M / "exp45" / "e26" / f"e26_dev_S_{seed}.pt"; m0 = arch26.new_net(seed)
    m.load_state_dict(T.load(birth))
    rows = sorted(mine)
    print(f"embedding rows of the task's bytes: largest difference between the birth and the initial network {float((m.book.weight[rows] - m0.book.weight[rows]).abs().max()):.1e}")
    print("score of the born network on the held panel before any update:", E.score(m, panel(seed, "held")))
    lg = arch26.forward(m, x, True, True)[0]; loss = E.cross_entropy(lg[mask], y[mask]); loss.backward()
    print(f"first-batch loss {float(loss):.3f}; finite gradient: {all(bool(T.isfinite(p.grad).all()) for p in m.parameters() if p.grad is not None)}")
