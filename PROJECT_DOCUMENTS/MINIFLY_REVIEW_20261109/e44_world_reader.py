"""A second implementation of E44's long-life world, written from the text of the brief (revision 1, on the default branch at
436d9677) and, where that text says "as in the small world", from the rules the E39 brief states in full.

Pure Python, standard library only.  No line of the coordinator's specification file (lw_world_spec.py) is imported, run or
copied; the layout differs on purpose (tasks are tuples, the schedule is a list of families, the stream is one generator).

Where the brief's words leave a choice, the choice is an argument, so that the other reading can be computed beside it:
    half_up     the number of taught pairs is round(0.10 p^2).  Python's round sends a half to the even number (202 at p = 45);
                half_up=True rounds a half upwards (203).  Six moduli end in 5.

    python3 -I e44_world_reader.py [world ...]          prints what the coordinator's file prints for a world
"""
import hashlib
import sys
from random import Random

EOS, BOS = 256, 257
USED_MODULI = (73, 83, 89, 97)
OPERATIONS = ("add", "sub", "max", "min")
SYMBOL = {"add": ord("+"), "sub": ord("-"), "max": ord(">"), "min": ord("<")}
LETTERS = [ord(c) for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
N_FAMILIES, PROBE_FAMILIES = 56, 4
ITEMS, TAUGHT_ITEMS = 64, 48


def answer(op, p, a, b):
    if op == "add":
        return (a + b) % p
    if op == "sub":
        return (a - b) % p
    return max(a, b) if op == "max" else min(a, b)


def needs_wrap(op, p, a, b):
    return (op == "add" and a + b >= p) or (op == "sub" and a - b < 0)


class World:
    def __init__(self, seed, visit_updates=256, p_new=0.05, taught_share=0.10, half_up=False):
        self.seed, self.visit_updates, self.p_new = seed, visit_updates, p_new
        g = Random(f"lw|{seed}|families")
        self.moduli = [m for m in range(41, 101) if m not in USED_MODULI]
        assert len(self.moduli) == N_FAMILIES
        g.shuffle(self.moduli)                                              # family f has the f-th
        values = g.sample(range(128 * 128), 4 * N_FAMILIES)                 # after the moduli, the same generator
        self.task = []                                                      # task 4 f + j: (family, operation, modulus, code)
        for f, p in enumerate(self.moduli):
            for j, op in enumerate(OPERATIONS):
                c = values[4 * f + j]
                self.task.append((f, op, p, (128 + c // 128, 128 + c % 128)))
        self.taught, self.asked = [], []
        for i, (f, op, p, code) in enumerate(self.task):
            pairs = [(a, b) for a in range(p) for b in range(p)]
            Random(f"lw|{seed}|split|{i}").shuffle(pairs)
            n = (p * p + 5) // 10 if half_up else round(taught_share * p * p)
            if half_up:
                assert taught_share == 0.10
            self.taught.append(pairs[:n]); self.asked.append(pairs[n:])
        self._taught_set = [set(t) for t in self.taught]

    # ------------------------------------------------------------------------------------------------ schedule
    def schedule(self, visits, in_turn=False):
        """The family of each visit.  in_turn: visit v is family v (the calibration)."""
        if in_turn:
            return list(range(visits))
        r = Random(f"lw|{self.seed}|schedule")
        out, last_seen, met = [0], {0: 0}, 1
        for v in range(1, visits):
            x = r.random()                                                  # one draw at every visit after the first
            if x < self.p_new and met < N_FAMILIES - PROBE_FAMILIES:
                f = met; met += 1
            else:
                c = r.randrange(3)                                          # the absence class is drawn first
                classes = ([], [], [])
                for h, lv in last_seen.items():
                    d = v - lv
                    classes[0 if d <= 4 else (1 if d <= 32 else 2)].append(h)
                chosen = classes[c] if classes[c] else next(k for k in classes if k)     # else the first non-empty of short, medium, long
                f = sorted(chosen)[r.randrange(len(chosen))]
            out.append(f); last_seen[f] = v
        return out

    def absences(self, visits):
        """(visit, family, new, visits since the family's last visit or None) for each visit."""
        last, rows = {}, []
        for v, f in enumerate(self.schedule(visits)):
            rows.append((v, f, f not in last, None if f not in last else v - last[f]))
            last[f] = v
        return rows

    # ------------------------------------------------------------------------------------------------ items
    def item(self, i, a, b, letters):
        f, op, p, code = self.task[i]
        prompt = list(code) + [ord(ch) for ch in f"{a:02d}"] + [SYMBOL[op]] + list(letters) + [ord(ch) for ch in f"{b:02d}"] + [ord("=")]
        return prompt, [ord(ch) for ch in f"{answer(op, p, a, b):02d}"] + [EOS]

    def visit(self, v, family, r=None):
        """One visit: visit_updates updates, each (taught, asked) with 48 and 16 entries (task id, a, b, prompt, answer).
        r: another generator in place of the visit's own (the runner's `branch` command draws by the same rule from its own)."""
        r = r or Random(f"lw|{self.seed}|visit|{v}")
        ids = [4 * family + j for j in range(4)]
        for _ in range(self.visit_updates):
            taught, asked = [], []
            for slot in range(ITEMS):
                i = ids[r.randrange(4)]
                pool = self.taught[i] if slot < TAUGHT_ITEMS else self.asked[i]
                a, b = pool[r.randrange(len(pool))]
                k = r.randint(0, 12)                                        # the number of letters, then the letters by choice
                letters = [r.choice(LETTERS) for _ in range(k)]
                (taught if slot < TAUGHT_ITEMS else asked).append((i, a, b, *self.item(i, a, b, letters)))
            yield taught, asked

    def stream(self, visits, in_turn=False):
        for v, f in enumerate(self.schedule(visits, in_turn)):
            for taught, asked in self.visit(v, f):
                yield v, f, taught, asked

    def digest(self, visits, in_turn=False):
        return self.stream_digest(self.stream(visits, in_turn))

    @staticmethod
    def stream_digest(stream):
        h = hashlib.sha256()
        for v, f, taught, asked in stream:
            for i, a, b, prompt, target in taught + asked:
                h.update(bytes([i // 256, i % 256, a, b, len(prompt)]) + bytes(prompt) + bytes(x % 256 for x in target))
        return h.hexdigest()

    def branch(self, n0, families):
        """The stream of `sw_run.py branch` from a state after n0 visits over the named families: its k-th visit draws, by the
        visit's item rule, from Random(f"world|{seed}|branch|{n0}|{k}") (the runner's string at 436d9677, which says "world")."""
        for k, f in enumerate(families):
            for taught, asked in self.visit(n0 + k, f, Random(f"world|{self.seed}|branch|{n0}|{k}")):
                yield n0 + k, f, taught, asked

    def probe_list(self, families, per_task=100):
        """A list for the question "what is kept": for add and for sub of each family, the first per_task asked pairs, in the
        split's order, that need the reduction; no letters.  Entries (task id, a, b, prompt, answer)."""
        out = []
        for f in families:
            for j in (0, 1):
                i = 4 * f + j
                red = [(a, b) for a, b in self.asked[i] if self.wrap(i, a, b)][:per_task]
                out += [(i, a, b, *self.item(i, a, b, [])) for a, b in red]
        return out

    # ------------------------------------------------------------------------------------------------ marks on an item
    def mirror(self, i, a, b):
        return int(self.task[i][1] != "sub" and a != b and (b, a) in self._taught_set[i])

    def wrap(self, i, a, b):
        f, op, p, code = self.task[i]
        return needs_wrap(op, p, a, b)


def describe(seed, visits=1024):
    """The lines the coordinator's file prints for a world, from this implementation."""
    w = World(seed)
    rows = w.absences(visits)
    new = sum(1 for r in rows if r[2]); ab = [r[3] for r in rows if r[3] is not None]
    cls = [sum(1 for d in ab if lo <= d <= hi) for lo, hi in ((1, 4), (5, 32), (33, 10 ** 9))]
    first_new = [r[0] for r in rows if r[2]][:6]
    assert all(set(w.taught[i]).isdisjoint(w.asked[i]) for i in range(len(w.task))) and len({t[3] for t in w.task}) == len(w.task)
    red = sum(needs_wrap("add", w.moduli[0], a, b) for a, b in w.asked[0]) / len(w.asked[0])
    return [f"world {seed}: {N_FAMILIES} families, {len(w.task)} tasks; {visits} visits: {new} new families, returns short/medium/long {cls}, "
            f"longest absence {max(ab)}; first new visits {first_new}",
            f"  first families' moduli {w.moduli[:8]}; taught pairs per task min/max {min(map(len, w.taught))}/{max(map(len, w.taught))};"
            f" reduction share of asked add pairs, family 0: {red:.3f}",
            f"  digest of the first 4 visits: {w.digest(4)}",
            f"  digest of 8 visits in turn at 64 updates: {World(seed, visit_updates=64).digest(8, in_turn=True)}"]


if __name__ == "__main__":
    for s in [int(v) for v in sys.argv[1:]] or [1, 2, 3]:
        print("\n".join(describe(s)))
