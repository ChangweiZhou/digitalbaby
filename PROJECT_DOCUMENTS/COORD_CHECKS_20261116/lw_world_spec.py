"""The long-life world in the full-scale format: the specification, as runnable pure Python (no torch in the generator). Coordinator's file,
16 Nov 2026 (the proposal is the reviewer's memo, revision 16, section 25.4). It replaces the task set of sw_world_spec.py and keeps its
schedule rule, item rule and digest rule. Everything follows from the world's seed through random.Random objects seeded with strings.
    python3 -I lw_world_spec.py [world_seed] [visits]         prints a world"""
import hashlib
import sys
from random import Random

EOS, BOS = 256, 257                        # mod97_local.py lines 34-35
USED_MODULI = (73, 83, 89, 97)             # F3, E40's family, the birth family, F1
LOWEST = 41
MODULI = [m for m in range(LOWEST, 101) if m not in USED_MODULI]    # 56 families, one modulus each
OPS = ("add", "sub", "max", "min")         # exp11_cell.DEV's four, with exp11_cell.SYM's symbols
SYM = {"add": ord("+"), "sub": ord("-"), "max": ord(">"), "min": ord("<")}
CODE_BYTES = list(range(128, 256))         # a task's code is two of these: bytes no family and no birth has used
LETTERS = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ"    # exp11_cell.DISTRACT
KMAX = 12                                  # 0..12 letters, as at full scale
BATCH, TAUGHT_PER_BATCH = 64, 48
VISIT_UPDATES = 256
P_NEW = 0.05                               # a life of 1,024 visits then meets about 51 new families of the 52 scheduled
TAUGHT_FRACTION = 0.10
PROBE_FAMILIES = 4                         # the last four families are never scheduled: the world's probe tasks


def apply(op, p, a, b):
    return {"add": (a + b) % p, "sub": (a - b) % p, "max": max(a, b), "min": min(a, b)}[op]


def needs_wrap(op, p, a, b):               # a reduction pair
    return (op == "add" and a + b >= p) or (op == "sub" and a - b < 0)


class World:
    def __init__(self, seed, visit_updates=VISIT_UPDATES, p_new=P_NEW, taught_fraction=TAUGHT_FRACTION, lowest=LOWEST):
        self.seed, self.visit_updates, self.p_new = seed, visit_updates, p_new
        r = Random(f"lw|{seed}|families")
        mods = [m for m in MODULI if m >= lowest]; r.shuffle(mods)             # family g has modulus mods[g]; lowest = 41 gives all 56
        codes = r.sample(range(len(CODE_BYTES) ** 2), 4 * len(mods))           # distinct two-byte codes, after the moduli, same generator
        self.tasks = []
        for g, p in enumerate(mods):
            for j, op in enumerate(OPS):                                       # task id = 4 g + j
                c = codes[4 * g + j]
                self.tasks.append(dict(id=4 * g + j, family=g, op=op, symbol=SYM[op], p=p,
                                       code=bytes([CODE_BYTES[c // len(CODE_BYTES)], CODE_BYTES[c % len(CODE_BYTES)]])))
        for t in self.tasks:                                                   # each task's pairs are split once
            pairs = [(a, b) for a in range(t["p"]) for b in range(t["p"])]
            Random(f"lw|{seed}|split|{t['id']}").shuffle(pairs)
            k = round(taught_fraction * len(pairs))
            t["taught"], t["asked"] = pairs[:k], pairs[k:]
            t["taught_set"] = set(t["taught"])
        self.groups = [self.tasks[4 * g:4 * g + 4] for g in range(len(mods))]  # a group is a family

    def schedule(self, visits, in_turn=False):
        """As sw_world_spec.World.schedule, with families as groups. in_turn=True (calibration only): visit v is family v."""
        if in_turn:
            return [dict(visit=v, group=v, new=True, absence=None) for v in range(visits)]
        r = Random(f"lw|{self.seed}|schedule")
        out, last, met = [], {}, 0
        for v in range(visits):
            new = v == 0 or (r.random() < self.p_new and met < len(self.groups) - PROBE_FAMILIES)
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

    def mirror(self, t, a, b):
        return int(t["op"] != "sub" and a != b and (b, a) in t["taught_set"])

    def visit_batches(self, v, group):
        """visit_updates batches: 48 taught then 16 asked items, each from a task of the family drawn at random; yields (taught, asked),
        each a list of (task_id, a, b, prompt, answer)."""
        r = Random(f"lw|{self.seed}|visit|{v}")
        ts = self.groups[group]
        for _ in range(self.visit_updates):
            taught, asked = [], []
            for i in range(BATCH):
                t = ts[r.randrange(len(ts))]
                pool = t["taught"] if i < TAUGHT_PER_BATCH else t["asked"]
                a, b = pool[r.randrange(len(pool))]
                (taught if i < TAUGHT_PER_BATCH else asked).append((t["id"], a, b, *self.render(t, a, b, r)))
            yield taught, asked


def digest(world, visits, in_turn=False):
    h = hashlib.sha256()
    for s in world.schedule(visits, in_turn):
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
    first_new = [s["visit"] for s in sch if s["new"]][:6]
    assert all(set(t["taught"]).isdisjoint(t["asked"]) for t in w.tasks) and len({t["code"] for t in w.tasks}) == len(w.tasks)
    print(f"world {seed}: {len(w.groups)} families, {len(w.tasks)} tasks; {visits} visits: {new} new families, returns short/medium/long {cls}, longest absence {max(ab)}; first new visits {first_new}")
    print(f"  first families' moduli {[g[0]['p'] for g in w.groups[:8]]}; taught pairs per task min/max {min(len(t['taught']) for t in w.tasks)}/{max(len(t['taught']) for t in w.tasks)};"
          f" reduction share of asked add pairs, family 0: {sum(needs_wrap('add', w.groups[0][0]['p'], a, b) for a, b in w.groups[0][0]['asked']) / len(w.groups[0][0]['asked']):.3f}")
    print(f"  digest of the first 4 visits: {digest(w, 4)}")
    print(f"  digest of 8 visits in turn at 64 updates: {digest(World(seed, visit_updates=64), 8, in_turn=True)}")
