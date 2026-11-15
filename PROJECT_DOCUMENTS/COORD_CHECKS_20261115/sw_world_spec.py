"""Small-world generator: the specification of job L0, as runnable pure Python (no torch, no numpy). Coordinator's prototype, 15 Nov 2026.
Everything a life's data depends on comes from the world seed through random.Random objects seeded with strings, so that a second
implementation can reproduce it byte for byte.   python3 -I sw_world_spec.py [world_seed] [visits]"""
import hashlib
import sys
from random import Random

BOS, EOS = 256, 257                       # as exp11_cell: BOS never a target, EOS ends the answer
MODULI = list(range(8, 38))               # 30 moduli; operands and answers are two decimal digits
LINEAR = [(c1, c2) for c1 in (1, 2, -1, -2) for c2 in (1, 2, -1, -2)]          # (c1*a + c2*b) mod p: 16 operations
OPS = [("lin", c) for c in LINEAR] + [("max", None), ("min", None)]            # 18 operations
SYMBOLS = b"+-*/%&|^~!?:;<>@$#"           # one byte per operation, the same under every modulus (18 bytes)
LETTERS = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ"   # distractors
CODE_BYTES = b"abcdefghijklmnopqrstuvwxyz"
KMAX = 4                                  # 0..4 distractor letters
GROUP, BATCH, TAUGHT_PER_BATCH = 2, 64, 48          # tasks per group; items per update; taught items per update (the other 16 are asked)
VISIT_UPDATES = 256                       # updates per visit
P_NEW = 0.2                               # chance that a visit after the first brings a new group
TAUGHT_FRACTION = 0.5


def apply(op, p, a, b):
    kind, c = op
    if kind == "lin":
        return (c[0] * a + c[1] * b) % p
    return max(a, b) if kind == "max" else min(a, b)


def needs_wrap(op, p, a, b):              # for lin operations: the plain integer value falls outside 0..p-1
    kind, c = op
    return kind == "lin" and not (0 <= c[0] * a + c[1] * b < p)


class World:
    def __init__(self, seed):
        self.seed = seed
        r = Random(f"world|{seed}|tasks")
        tasks = [(oi, p) for oi in range(len(OPS)) for p in MODULI]             # 540 tasks
        r.shuffle(tasks)
        codes = [bytes([x, y]) for x in CODE_BYTES for y in CODE_BYTES]         # 676 two-byte codes
        r.shuffle(codes)
        self.tasks = [dict(id=i, op=OPS[oi], symbol=SYMBOLS[oi], p=p, code=codes[i]) for i, (oi, p) in enumerate(tasks)]
        for t in self.tasks:                                                    # each task's pairs are split once
            pairs = [(a, b) for a in range(t["p"]) for b in range(t["p"])]
            Random(f"world|{seed}|split|{t['id']}").shuffle(pairs)
            k = round(TAUGHT_FRACTION * len(pairs))
            t["taught"], t["asked"] = pairs[:k], pairs[k:]
        self.groups = [self.tasks[i:i + GROUP] for i in range(0, len(self.tasks), GROUP)]   # 270 groups, met in this order

    def schedule(self, visits):
        """The group of each visit. Visit 0 is group 0. Later: with chance P_NEW the next unmet group; otherwise a return, whose absence
        class is drawn uniformly from short (last seen 1-4 visits ago), medium (5-32) and long (33 or more), and the group is drawn
        uniformly among met groups in that class; an empty class falls back to the nearest non-empty one in the order short, medium, long."""
        r = Random(f"world|{self.seed}|schedule")
        out, last, met = [], {}, 0
        for v in range(visits):
            new = v == 0 or (r.random() < P_NEW and met < len(self.groups))
            if new:
                g = met; met += 1
            else:
                cls = r.randrange(3)
                bins = [[], [], []]
                for h, lv in last.items():
                    d = v - lv
                    bins[0 if d <= 4 else 1 if d <= 32 else 2].append(h)
                order = [cls] + [c for c in (0, 1, 2) if c != cls]
                pick = next(b for b in (bins[c] for c in order) if b)
                g = sorted(pick)[r.randrange(len(pick))]
            out.append(dict(visit=v, group=g, new=new, absence=(None if g not in last else v - last[g])))
            last[g] = v
        return out

    def render(self, t, a, b, r):
        d = bytes(r.choice(LETTERS) for _ in range(r.randint(0, KMAX)))
        prompt = t["code"] + f"{a:02d}".encode() + bytes([t["symbol"]]) + d + f"{b:02d}".encode() + b"="
        return list(prompt), list(f"{apply(t['op'], t['p'], a, b):02d}".encode()) + [EOS]

    def visit_batches(self, v, group):
        """VISIT_UPDATES batches for visit v: each has TAUGHT_PER_BATCH taught items then BATCH - TAUGHT_PER_BATCH asked items; every item's
        task is drawn uniformly from the group, its pair uniformly from that task's taught (or asked) pairs. Yields (taught, asked), each a
        list of (task_id, a, b, prompt, answer)."""
        r = Random(f"world|{self.seed}|visit|{v}")
        ts = self.groups[group]
        for _ in range(VISIT_UPDATES):
            taught, asked = [], []
            for i in range(BATCH):
                t = ts[r.randrange(len(ts))]
                pool = t["taught"] if i < TAUGHT_PER_BATCH else t["asked"]
                a, b = pool[r.randrange(len(pool))]
                (taught if i < TAUGHT_PER_BATCH else asked).append((t["id"], a, b, *self.render(t, a, b, r)))
            yield taught, asked


def digest(world, visits):
    h = hashlib.sha256()
    for s in world.schedule(visits):
        for taught, asked in world.visit_batches(s["visit"], s["group"]):
            for tid, a, b, p, y in taught + asked:
                h.update(bytes([tid // 256, tid % 256, a, b, len(p)]) + bytes(p) + bytes(v % 256 for v in y))
    return h.hexdigest()


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    visits = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
    w = World(seed)
    sch = w.schedule(visits)
    new = sum(s["new"] for s in sch); ab = [s["absence"] for s in sch if s["absence"] is not None]
    cls = [sum(1 for d in ab if lo <= d <= hi) for lo, hi in ((1, 4), (5, 32), (33, 10 ** 9))]
    assert all(set(t["taught"]).isdisjoint(t["asked"]) for t in w.tasks)
    assert len({t["code"] for t in w.tasks}) == len(w.tasks) and len(SYMBOLS) == len(OPS) == len(set(SYMBOLS))
    print(f"world {seed}: {len(w.tasks)} tasks in {len(w.groups)} groups; {visits} visits: {new} new groups ({new * GROUP} tasks), returns by absence short/medium/long {cls}, longest absence {max(ab)}")
    print("pairs per task min/max:", min(len(t['taught']) + len(t['asked']) for t in w.tasks), max(len(t['taught']) + len(t['asked']) for t in w.tasks),
          "| prompt length max:", 2 + 2 + 1 + KMAX + 2 + 1, "| first tasks:", [(t['code'].decode(), chr(t['symbol']), t['op'], t['p']) for t in w.tasks[:3]])
    print("digest of the first 16 visits:", digest(w, 16))
