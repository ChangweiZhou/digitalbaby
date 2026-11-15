"""Second implementation of the E39 small world: generator and scorer (15 November 2026).

Written by the reviewer from the TEXT of OPUS_TASK_E39_SMALL_WORLD_PROTOCOL_L0_L1_20261115.md (revision 1, 18751457),
before reading the coordinator's sw_world_spec.py.  Standard library and numpy only; nothing trains.

Mode `text` uses, wherever the brief's text is silent, the reviewer's own conventions (each marked TEXT-SILENT below).
It cannot be expected to give the coordinator's digests, and the list of TEXT-SILENT points is what the brief must
state for a second implementation to do so.  Mode `spec` (added after the comparison) follows the rules of the
coordinator's file, re-implemented here independently, and is the one whose digests are compared with check 1.

Run from the repository root:
  python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e39_world.py text stats      # schedule figures, text-only reading
  python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e39_world.py text tasks      # the task space: coincidences, wraps
  python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e39_world.py text alloc      # taught items per task over a life
  python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e39_world.py text blind      # the blind predictor's score, no network
  python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e39_world.py text scorer     # the scorer on a hand-made event file
  python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e39_world.py spec-file stats # the coordinator's rules with the file's end symbol (257)
  python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e39_world.py spec stats      # the same rules with the harness's end symbol (256)
  python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e39_world.py spec alloc      # taught items per task under the coordinator's schedule
"""
import hashlib, math, sys
from random import Random
import numpy as np

EOS, BOS = 256, 257
MODULI = list(range(8, 38))                                   # the brief: 30 moduli, 8 to 37
COEF = (1, 2, -1, -2)
OPS = [("lin", c1, c2) for c1 in COEF for c2 in COEF] + [("max",), ("min",)]      # 18 operations
GROUP, VISIT_UPDATES, TAUGHT, ASKED, P_NEW, KMAX = 2, 256, 48, 16, 0.2, 4
LETTERS = list(b"ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def answer(op, p, a, b):
    if op[0] == "lin":
        return (op[1] * a + op[2] * b) % p
    return max(a, b) if op[0] == "max" else min(a, b)


def needs_wrap(op, p, a, b):
    """TEXT-SILENT: the brief names `needs_wrap` and does not define it.  Here: the raw value leaves 0..p-1."""
    if op[0] != "lin":
        return False
    raw = op[1] * a + op[2] * b
    return raw < 0 or raw >= p


class World:
    """One world, from its seed.  Every draw comes from a random.Random seeded with a string, as the brief says."""

    def __init__(self, seed, moduli=MODULI, p_new=P_NEW, visit_updates=VISIT_UPDATES):
        self.seed, self.p_new, self.visit_updates = seed, p_new, visit_updates
        rng = Random(f"world|{seed}|tasks")
        tasks = [(op, p) for op in OPS for p in moduli]                     # TEXT-SILENT: the order of the 540 tasks
        # TEXT-SILENT: which bytes are the 18 operation symbols.  Here: punctuation, never a digit, a letter or '='.
        self.symbol = dict(zip(OPS, b"+-*/<>!?#$%&@^~:;|"))
        # TEXT-SILENT: the alphabet of the two-byte codes and how they are drawn.  Here: two lower-case letters, no repeats.
        codes = rng.sample([(x, y) for x in range(97, 123) for y in range(97, 123)], len(tasks))
        self.code = dict(zip(tasks, codes))
        rng.shuffle(tasks)                                                  # TEXT-SILENT: how tasks are put into groups of 2
        self.groups = [tasks[i:i + GROUP] for i in range(0, len(tasks), GROUP)]
        self.tasks = tasks
        self.taught, self.asked = {}, {}
        for t in tasks:
            op, p = t
            pairs = [(a, b) for a in range(p) for b in range(p)]
            Random(f"world|{seed}|split|{self.code[t]}").shuffle(pairs)      # TEXT-SILENT: the generator and order of the split
            half = len(pairs) // 2                                          # TEXT-SILENT: which half gets the odd pair (p odd)
            self.taught[t], self.asked[t] = pairs[:half], pairs[half:]
            assert not set(self.taught[t]) & set(self.asked[t])

    # ------------------------------------------------------------ the schedule
    def schedule(self, visits):
        """Group index of every visit.  Visit 0 is group 0; then a new group with chance p_new, else a return."""
        rng = Random(f"world|{self.seed}|schedule")
        sched, last, nxt = [0], {0: 0}, 1
        for v in range(1, visits):
            if nxt < len(self.groups) and rng.random() < self.p_new:
                g = nxt; nxt += 1
            else:
                # TEXT-SILENT: how "about equally common" is made.  Here: a class drawn uniformly among the classes
                # that have a group, then a group uniformly within it.  Absence = visits since the group was last visited.
                cls = {0: [], 1: [], 2: []}
                for h, lv in last.items():
                    d = v - lv
                    cls[0 if d <= 4 else 1 if d <= 32 else 2].append(h)
                c = rng.choice([k for k in (0, 1, 2) if cls[k]])
                g = rng.choice(cls[c])
            sched.append(g); last[g] = v
        return sched

    # ------------------------------------------------------------ the stream
    def item(self, t, pair, rng, with_answer=True):
        op, p = t; a, b = pair
        k = rng.randint(0, KMAX)                                            # TEXT-SILENT: how k and the letters are drawn
        prompt = list(self.code[t]) + list(f"{a:02d}".encode()) + [self.symbol[op]] + [rng.choice(LETTERS) for _ in range(k)] + list(f"{b:02d}".encode()) + [ord("=")]
        return prompt, list(f"{answer(op, p, a, b):02d}".encode()) + [EOS]

    def batches(self, visits):
        """Yields (visit, update, taught_items, asked_items); an item is (task, pair, prompt, target)."""
        rng = Random(f"world|{self.seed}|items")
        u = 0
        for v, g in enumerate(self.schedule(visits)):
            grp = self.groups[g]
            for _ in range(self.visit_updates):
                taught, asked = [], []
                for n, src, out in ((TAUGHT, self.taught, taught), (ASKED, self.asked, asked)):   # TEXT-SILENT: order of draws
                    for _ in range(n):
                        t = rng.choice(grp); pair = rng.choice(src[t])
                        out.append((t, pair) + self.item(t, pair, rng))
                yield v, u, taught, asked
                u += 1

    def digest(self, visits):
        """TEXT-SILENT: what is hashed.  Here: SHA-256 over every item's prompt and target bytes (two bytes per value)."""
        h = hashlib.sha256()
        for v, u, taught, asked in self.batches(visits):
            for kind, items in ((b"T", taught), (b"A", asked)):
                for t, pair, prompt, target in items:
                    h.update(kind); h.update(np.array(prompt + target, dtype="<u2").tobytes())
        return h.hexdigest()


# ---------------------------------------------------------------- the coordinator's rules, re-implemented
class SpecWorld(World):
    """The rules of PROJECT_DOCUMENTS/COORD_CHECKS_20261115/sw_world_spec.py (18751457), written again here from a reading of
    that file, not by importing or running it.  Each rule below is one the brief's text does not state.

      tasks     (operation index, modulus), operation outer and modulus inner, shuffled by Random("world|S|tasks");
                a task's id is its place after the shuffle
      codes     the 676 two-lower-case-letter codes in order (first letter outer), shuffled by the SAME generator after the
                tasks; task i takes code i
      symbols   the bytes of "+-*/%&|^~!?:;<>@$#" for the operations in order: the 16 linear ones with c1 outer and c2 inner
                over (1, 2, -1, -2), then max, then min
      split     pairs in order (a outer, b inner) shuffled by Random("world|S|split|<id>"); the first round(p*p/2) are taught
      groups    tasks 2g and 2g+1 form group g; groups are met in this order
      schedule  one generator Random("world|S|schedule").  At every visit after the first one random() is drawn; the visit is
                new if it is below 0.2 and an unmet group remains.  Otherwise randrange(3) draws the absence class (1-4, 5-32,
                33 or more visits since the group's last visit); if that class is empty the first non-empty of short, medium,
                long is used; the group is sorted(class)[randrange(len(class))]
      items     a fresh generator Random("world|S|visit|<v>") for each visit.  Per update, 64 items in order, the first 48
                taught: task = group[randrange(2)], pair = pool[randrange(len(pool))], k = randint(0, 4), then k letters by
                choice
      digest    SHA-256 over every item in order (taught then asked): the bytes [id // 256, id % 256, a, b, len(prompt)],
                the prompt, and each answer value mod 256
    The file sets BOS, EOS = 256, 257; the harness has EOS = 256 and BOS = 257.  `eos` is a parameter here so that both the
    file's digests and the corrected ones can be computed.
    """

    def __init__(self, seed, eos=EOS, moduli=MODULI, p_new=P_NEW, visit_updates=VISIT_UPDATES, reserve=0):
        self.seed, self.p_new, self.visit_updates, self.eos, self.reserve = seed, p_new, visit_updates, eos, reserve
        r = Random(f"world|{seed}|tasks")
        idx = [(oi, p) for oi in range(len(OPS)) for p in moduli]
        r.shuffle(idx)
        codes = [(x, y) for x in range(97, 123) for y in range(97, 123)]
        r.shuffle(codes)
        self.tasks = [(OPS[oi], p) for oi, p in idx]
        self.tid = {t: i for i, t in enumerate(self.tasks)}
        self.code = {t: codes[i] for i, t in enumerate(self.tasks)}
        self.symbol = dict(zip(OPS, b"+-*/%&|^~!?:;<>@$#"))
        self.taught, self.asked = {}, {}
        for i, t in enumerate(self.tasks):
            pairs = [(a, b) for a in range(t[1]) for b in range(t[1])]
            Random(f"world|{seed}|split|{i}").shuffle(pairs)
            k = round(0.5 * len(pairs))
            self.taught[t], self.asked[t] = pairs[:k], pairs[k:]
        self.groups = [self.tasks[i:i + GROUP] for i in range(0, len(self.tasks), GROUP)]

    def schedule(self, visits):
        r = Random(f"world|{self.seed}|schedule")
        sched, last, met = [], {}, 0
        for v in range(visits):
            new = v == 0
            if v > 0:
                new = r.random() < self.p_new and met < len(self.groups) - self.reserve
            if new:
                g = met; met += 1
            else:
                c = r.randrange(3)
                bins = ([], [], [])
                for h, lv in last.items():
                    d = v - lv
                    bins[0 if d <= 4 else 1 if d <= 32 else 2].append(h)
                pick = bins[c] or next(b for b in bins if b)
                g = sorted(pick)[r.randrange(len(pick))]
            sched.append(g); last[g] = v
        return sched

    def batches(self, visits):
        u = 0
        for v, g in enumerate(self.schedule(visits)):
            r = Random(f"world|{self.seed}|visit|{v}")
            grp = self.groups[g]
            for _ in range(self.visit_updates):
                taught, asked = [], []
                for i in range(TAUGHT + ASKED):
                    t = grp[r.randrange(len(grp))]
                    pool = self.taught[t] if i < TAUGHT else self.asked[t]
                    a, b = pool[r.randrange(len(pool))]
                    k = r.randint(0, KMAX)
                    letters = [r.choice(LETTERS) for _ in range(k)]
                    prompt = list(self.code[t]) + list(f"{a:02d}".encode()) + [self.symbol[t[0]]] + letters + list(f"{b:02d}".encode()) + [ord("=")]
                    target = list(f"{answer(t[0], t[1], a, b):02d}".encode()) + [self.eos]
                    (taught if i < TAUGHT else asked).append((t, (a, b), prompt, target))
                yield v, u, taught, asked
                u += 1

    def digest(self, visits):
        h = hashlib.sha256()
        for v, u, taught, asked in self.batches(visits):
            for t, (a, b), prompt, target in taught + asked:
                i = self.tid[t]
                h.update(bytes([i // 256, i % 256, a, b, len(prompt)]) + bytes(prompt) + bytes(x % 256 for x in target))
        return h.hexdigest()


# ---------------------------------------------------------------- the scorer, from the brief's table
def score(ev, visits, visit_updates=VISIT_UPDATES, window=32, level=0.8):
    """ev: dict of equal-length arrays in stream order: update, visit, task, correct, blind, wrap, learned.
    Returns the brief's quantities.  Readings made where the table is not explicit:
      * a mean "of its last 32 asked items" exists only once the task has 32 asked items (a shorter window is not used);
      * the n-th task met is ordered by its first asked item in the stream;
      * a return visit is a visit to a group met before; absence = visits since the task's previous visit;
      * relearning within a visit uses a window that starts at the visit's first asked item of the task.
    """
    upd, vis, task, cor = (np.asarray(ev[k]) for k in ("update", "visit", "task", "correct"))
    blind, wrap, learned = (np.asarray(ev[k]) for k in ("blind", "wrap", "learned"))
    out = {}
    out["score_by_age"] = np.array([cor[vis == v].mean() if (vis == v).any() else np.nan for v in range(visits)])
    out["blind_by_age"] = np.array([blind[vis == v].mean() if (vis == v).any() else np.nan for v in range(visits)])
    order = []
    for t in task:
        if t not in order:
            order.append(int(t))
    cost, kept, relearn = [], [], []
    for n, t in enumerate(order, 1):
        m = task == t
        c, lrn, v_t, w_t = cor[m].astype(float), learned[m], vis[m], wrap[m]
        run = np.convolve(c, np.ones(window) / window, mode="valid") if len(c) >= window else np.array([])
        hit = np.nonzero(run >= level - 1e-12)[0]
        if len(hit):
            cost.append((n, t, int(lrn[hit[0] + window - 1]), None))
        else:
            cost.append((n, t, None, float(c[-window:].mean()) if len(c) else float("nan")))
        visits_t = sorted(set(int(x) for x in v_t))
        for prev, v in zip(visits_t, visits_t[1:]):
            mv = v_t == v
            cv, lv, wv = c[mv], lrn[mv], w_t[mv]
            first = cv[:window]
            rec = dict(task=t, visit=v, absence=v - prev, kept=float(first.mean()),
                       kept_wrap=float(first[wv[:window] == 1].mean()) if (wv[:window] == 1).any() else None,
                       kept_nowrap=float(first[wv[:window] == 0].mean()) if (wv[:window] == 0).any() else None)
            kept.append(rec)
            runv = np.convolve(cv, np.ones(window) / window, mode="valid") if len(cv) >= window else np.array([])
            hv = np.nonzero(runv >= level - 1e-12)[0]
            relearn.append(dict(task=t, visit=v, items=int(lv[hv[0] + window - 1] - lv[0]) if len(hv) else None))
    out["cost"], out["kept"], out["relearn"] = cost, kept, relearn
    q = visits - max(1, visits // 4)
    out["late_level"] = float(np.nanmean(out["score_by_age"][q:]))
    half = np.arange(visits // 2, visits)
    out["score_trend"] = float(np.polyfit(np.log(half + 1), out["score_by_age"][half], 1)[0]) if len(half) > 1 else float("nan")
    reached = [(n, c) for n, t, c, lvl in cost if c is not None and n >= 2 and c > 0]
    out["cost_trend"] = float(np.polyfit(np.log([n for n, c in reached]), np.log([c for n, c in reached]), 1)[0]) if len(reached) > 1 else float("nan")
    return out


def blind_events(world, visits):
    """The event-file fields that need no network: update, visit, task, blind-correct, needs-wrap, taught items so far."""
    tid = {t: i for i, t in enumerate(world.tasks)}
    last, n_taught, ev = {}, {}, dict(update=[], visit=[], task=[], correct=[], blind=[], wrap=[], learned=[])
    for v, u, taught, asked in world.batches(visits):
        for t, pair, prompt, target in asked:                              # asked items are answered before the update
            ev["update"].append(u); ev["visit"].append(v); ev["task"].append(tid[t])
            ev["blind"].append(int(last.get(t) == target)); ev["correct"].append(0)
            ev["wrap"].append(int(needs_wrap(t[0], t[1], *pair))); ev["learned"].append(n_taught.get(t, 0))
        for t, pair, prompt, target in taught:
            last[t] = target; n_taught[t] = n_taught.get(t, 0) + 1
    return {k: np.array(x) for k, x in ev.items()}


# ---------------------------------------------------------------- command line
def stats(mk):
    print("schedule, 1,024 visits: new groups met; returns in the three absence classes (1-4, 5-32, 33+); longest absence")
    for s in (1, 2, 3):
        w = mk(s); sc = w.schedule(1024); last, cls, longest, met = {}, [0, 0, 0], 0, set()
        for v, g in enumerate(sc):
            if g in met:
                d = v - last[g]; cls[0 if d <= 4 else 1 if d <= 32 else 2] += 1; longest = max(longest, d)
            met.add(g); last[g] = v
        print(f"   world {s}: {len(met)} groups met of {len(w.groups)}; returns {cls[0]}/{cls[1]}/{cls[2]}; longest absence {longest}")
    for s in (1, 2, 3):
        print(f"   world {s}: digest of the first 16 visits {mk(s).digest(16)}")


def tasks_report():
    print("task space: operations that coincide on every pair, per modulus; answer ranges; wrap shares")
    same = {}
    for p in MODULI:
        tabs = {op: tuple(answer(op, p, a, b) for a in range(p) for b in range(p)) for op in OPS}
        for i, o1 in enumerate(OPS):
            for o2 in OPS[i + 1:]:
                if tabs[o1] == tabs[o2]:
                    same.setdefault(p, []).append((o1, o2))
    print("   coinciding pairs of operations:", same if same else "none at any modulus 8..37")
    for op in OPS:
        if op[0] != "lin":
            continue
        shares = [np.mean([needs_wrap(op, p, a, b) for a in range(p) for b in range(p)]) for p in MODULI]
        vals = [len({answer(op, p, a, b) for a in range(p) for b in range(p)}) / p for p in MODULI]
        inj = [len({answer(op, p, a, 0) for a in range(p)}) / p for p in MODULI]
        print(f"   ({op[1]:+d})a + ({op[2]:+d})b: pairs needing a wrap {min(shares):.2f}-{max(shares):.2f}; share of the p answers that occur {min(vals):.2f}-{max(vals):.2f}; distinct answers as a varies / p {min(inj):.2f}-{max(inj):.2f}")
    print("   largest operand and answer:", max(MODULI) - 1, "(two digits)")


def alloc(mk):
    print("taught items a task receives (24 per update on average, 256 updates per visit = 6,144 per visit)")
    for V in (16, 64, 256, 1024):
        rows = []
        for s in range(1, 9):
            sc = mk(s).schedule(V); n = {}
            for g in sc:
                n[g] = n.get(g, 0) + 1
            first = {}
            for v, g in enumerate(sc):
                first.setdefault(g, v)
            rows.append((len(n), np.array(list(n.values())), np.array([first[g] for g in n])))
        G = np.mean([r[0] for r in rows]); cnt = np.concatenate([r[1] for r in rows]); fst = np.concatenate([r[2] for r in rows])
        late = cnt[fst >= V // 2]
        print(f"   {V:5d} visits (8 worlds): groups met {G:.1f}; visits per group mean {cnt.mean():.1f}, median {np.median(cnt):.0f}; share of groups with 1 visit {np.mean(cnt == 1):.2f}, 2-3 {np.mean((cnt >= 2) & (cnt <= 3)):.2f}, 4+ {np.mean(cnt >= 4):.2f}, 8+ {np.mean(cnt >= 8):.2f}; groups first met in the second half: median visits {np.median(late) if len(late) else float('nan'):.0f}")
        print(f"         taught items per task: median {int(np.median(cnt)) * 6144:,}; mean {cnt.mean() * 6144:,.0f}; updates per group: median {int(np.median(cnt)) * 256:,}, mean {cnt.mean() * 256:,.0f}")


if __name__ == "__main__":
    mode, what = sys.argv[1], sys.argv[2]
    kw = {}
    if mode == "spec-file":
        kw = dict(eos=257)                                   # the file as it stands: BOS, EOS = 256, 257
    mk = (lambda s: World(s)) if mode == "text" else (lambda s: SpecWorld(s, **kw))
    if what == "stats":
        stats(mk)
    elif what == "tasks":
        tasks_report()
    elif what == "alloc":
        alloc(mk)
    elif what == "blind":
        for s in (1, 2):
            ev = blind_events(mk(s), 16)
            print(f"world {s}, 16 visits: blind predictor right on {100 * ev['blind'].mean():.1f}% of {len(ev['blind']):,} asked items; needs-wrap share {ev['wrap'].mean():.2f}")
    elif what == "scorer":
        # a hand-made event file: two tasks, one visit each of 2 "visits" of 8 updates, 4 asked items per task per update
        ev = dict(update=[], visit=[], task=[], correct=[], blind=[], wrap=[], learned=[])
        for u in range(32):
            for t in (0, 1):
                for j in range(4):
                    ev["update"].append(u); ev["visit"].append(u // 16); ev["task"].append(t)
                    ev["correct"].append(int(u >= (6 if t == 0 else 40))); ev["blind"].append(0); ev["wrap"].append(j % 2); ev["learned"].append(u * 24)
        r = score({k: np.array(v) for k, v in ev.items()}, visits=2, visit_updates=16, window=32, level=0.8)
        # task 0 is right from update 6: its 32-item window (8 updates) first holds 26 of 32 (0.8125) when it ends inside update 12 (items of updates 5..12 -> 7 right updates = 28/32 at end of update 12; first >= 0.8 is the 2nd item of update 12)
        print("hand-made file: cost", r["cost"], "| score by age", r["score_by_age"], "| late level", r["late_level"])
