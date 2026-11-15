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
  python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e39_world.py spec events <events.npz> <world>   # a runner's event file against the generator
  python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e39_world.py spec score <events.npz> [visits]   # the scorer on a runner's event file
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


# ---------------------------------------------------------------- the scorer
def symmetric(op):
    return op[0] != "lin" or op[1] == op[2]


def score(ev, visits, visit_updates=VISIT_UPDATES, window=32, level=0.8):
    """The brief's quantities, with the definitions the audit asks for where the table was not explicit or not sound.

    ev: equal-length arrays in stream order, one record per asked item:
        update, visit, task, correct, blind, wrap, seen (the task's stream taught items presented before this update).
    Only the first `visits` visits are read, so a shorter life is scored from the file of a longer one.

      score_by_age, blind_by_age   share right in each visit
      first_visit                  per task, in order of meeting (first visit, then task id): n, task, the mean of its last
                                   `window` asked items in its FIRST visit (every task has one full first visit, so nothing is
                                   censored), and its taught items before the running mean first reaches `level` in that visit
      cost                         the brief's whole-life cost (taught items before the running mean first reaches `level`);
                                   descriptive only: a task met late has fewer visits left, so "reached" is not comparable over n
      kept                         per return visit and task: the share right in the visit's FIRST update (answered before any
                                   update of the visit), the same over the first `window` items, both split by wrap, the absence
      relearn                      taught items within the visit before the running mean (over the visit's own items) reaches `level`
      late_level                   mean score_by_age over the last quarter of the visits
      score_trend                  least-squares slope of score_by_age on ln(visit + 1) over the second half
      learning_trend               least-squares slope of the first-visit level on ln(n), n from 3 (after the first group)
      cost_trend                   the brief's slope of ln(cost) on ln(n) over tasks that reached, n from 2 (biased; see above)
    A running mean exists only once `window` items are there; a shorter window is never used.
    """
    keep = np.asarray(ev["visit"]) < visits
    upd, vis, task, cor, blind, wrap, seen = (np.asarray(ev[k])[keep] for k in ("update", "visit", "task", "correct", "blind", "wrap", "seen"))
    out = {}
    out["score_by_age"] = np.array([cor[vis == v].mean() if (vis == v).any() else np.nan for v in range(visits)])
    out["blind_by_age"] = np.array([blind[vis == v].mean() if (vis == v).any() else np.nan for v in range(visits)])
    firstv = {}
    for t, v in zip(task.tolist(), vis.tolist()):
        firstv.setdefault(t, v)
    order = sorted(firstv, key=lambda t: (firstv[t], t))

    def reach(c, sn):
        """Taught items seen at the first asked item where the mean of the last `window` items is at least `level`."""
        if len(c) < window:
            return None
        run = np.convolve(c, np.ones(window) / window, mode="valid")
        hit = np.nonzero(run >= level - 1e-12)[0]
        return int(sn[hit[0] + window - 1]) if len(hit) else None

    first_visit, cost, kept, relearn = [], [], [], []
    for n, t in enumerate(order, 1):
        m = task == t
        c, sn, v_t, w_t, u_t = cor[m].astype(float), seen[m], vis[m], wrap[m], upd[m]
        f = v_t == firstv[t]
        first_visit.append(dict(n=n, task=t, level=float(c[f][-window:].mean()), cost=reach(c[f], sn[f])))
        r = reach(c, sn)
        cost.append(dict(n=n, task=t, cost=r, level=None if r is not None else float(c[-window:].mean())))
        vs = sorted(set(v_t.tolist()))
        for prev, v in zip(vs, vs[1:]):
            mv = v_t == v
            cv, sv, wv, uv = c[mv], sn[mv], w_t[mv], u_t[mv]
            fu = uv == uv.min()                                             # the visit's first update: nothing relearned yet

            def part(sel):
                return float(cv[sel].mean()) if sel.any() else None
            head = np.arange(len(cv)) < window
            kept.append(dict(task=t, visit=v, absence=v - prev, kept=part(fu), kept_wrap=part(fu & (wv == 1)), kept_nowrap=part(fu & (wv == 0)),
                             kept32=part(head), kept32_wrap=part(head & (wv == 1)), kept32_nowrap=part(head & (wv == 0))))
            rr = reach(cv, sv)
            relearn.append(dict(task=t, visit=v, absence=v - prev, items=None if rr is None else int(rr - sv[0])))
    out.update(first_visit=first_visit, cost=cost, kept=kept, relearn=relearn)
    q = visits - max(1, visits // 4)
    out["late_level"] = float(np.nanmean(out["score_by_age"][q:]))
    half = np.arange(visits // 2, visits)
    out["score_trend"] = float(np.polyfit(np.log(half + 1), out["score_by_age"][half], 1)[0]) if len(half) > 1 else float("nan")
    fv = [(d["n"], d["level"]) for d in first_visit if d["n"] >= 3]
    out["learning_trend"] = float(np.polyfit(np.log([n for n, _ in fv]), [l for _, l in fv], 1)[0]) if len(fv) > 1 else float("nan")
    rc = [(d["n"], d["cost"]) for d in cost if d["cost"] and d["n"] >= 2]
    out["cost_trend"] = float(np.polyfit(np.log([n for n, _ in rc]), np.log([c for _, c in rc]), 1)[0]) if len(rc) > 1 else float("nan")
    return out


def events_without_network(world, visits):
    """Every column of the event file that does not need a network: update, visit, task id, the blind predictor's result,
    needs-wrap, the task's stream taught items presented before this update, and whether the mirrored pair is taught."""
    tid = getattr(world, "tid", None) or {t: i for i, t in enumerate(world.tasks)}
    taught_set = {t: set(world.taught[t]) for t in world.tasks}
    last, n_seen = {}, {}
    ev = dict(update=[], visit=[], task=[], correct=[], blind=[], wrap=[], seen=[], mirror=[])
    for v, u, taught, asked in world.batches(visits):
        for t, pair, prompt, target in asked:                              # asked items are answered before the update
            ev["update"].append(u); ev["visit"].append(v); ev["task"].append(tid[t]); ev["correct"].append(0)
            ev["blind"].append(int(last.get(t) == target))
            ev["wrap"].append(int(needs_wrap(t[0], t[1], *pair))); ev["seen"].append(n_seen.get(t, 0))
            ev["mirror"].append(int(symmetric(t[0]) and pair[0] != pair[1] and (pair[1], pair[0]) in taught_set[t]))
        for t, pair, prompt, target in taught:
            last[t] = target; n_seen[t] = n_seen.get(t, 0) + 1
    return {k: np.array(x) for k, x in ev.items()}


def hand_made():
    """Two tasks, two visits of 16 updates, 4 asked items per task per update, 24 taught items per task per update.
    Task 0 is right from update 6 on, wrong again in the first update of visit 1, right after; task 1 is never right."""
    ev = dict(update=[], visit=[], task=[], correct=[], blind=[], wrap=[], seen=[])
    for u in range(32):
        for t in (0, 1):
            for j in range(4):
                right = t == 0 and u >= 6 and u != 16
                ev["update"].append(u); ev["visit"].append(u // 16); ev["task"].append(t); ev["correct"].append(int(right))
                ev["blind"].append(0); ev["wrap"].append(j % 2); ev["seen"].append(u * 24)
    r = score({k: np.array(v) for k, v in ev.items()}, visits=2, visit_updates=16)
    # task 0: items of updates 0-5 (24 items) are wrong.  A window of 32 ending at item i holds 55 - i wrong items for i <= 55;
    # 0.8 needs at most 6 wrong, so i = 49, the second item of update 12: 12 * 24 = 288 taught items seen.
    assert r["first_visit"][0] == dict(n=1, task=0, level=1.0, cost=288), r["first_visit"][0]
    assert r["first_visit"][1] == dict(n=2, task=1, level=0.0, cost=None)
    assert r["cost"][0]["cost"] == 288 and r["cost"][1] == dict(n=2, task=1, cost=None, level=0.0)
    k0 = [k for k in r["kept"] if k["task"] == 0][0]
    assert k0["absence"] == 1 and k0["kept"] == 0.0 and k0["kept32"] == 28 / 32, k0      # first update wrong; the first 32 items hold 4 wrong
    assert [x for x in r["relearn"] if x["task"] == 0][0]["items"] == 7 * 24               # the window is 28/32 = 0.875 at its first full position, update 23
    assert abs(r["score_by_age"][0] - 40 / 128) < 1e-12 and abs(r["score_by_age"][1] - 60 / 128) < 1e-12
    assert abs(r["late_level"] - 60 / 128) < 1e-12
    print("hand-made event file: every quantity equals its hand-computed value")


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
        for sd in (1, 2):
            ev = events_without_network(mk(sd), 16)
            print(f"world {sd}, 16 visits: blind predictor right on {100 * ev['blind'].mean():.1f}% of {len(ev['blind']):,} asked items; needs-wrap share {ev['wrap'].mean():.2f}; asked items whose mirrored pair is taught {100 * ev['mirror'].mean():.1f}%")
    elif what == "scorer":
        hand_made()
    elif what == "events":
        # compare a runner's event file (.npz with arrays update, visit, task, correct, blind, wrap, seen[, mirror]) with the generator
        f = np.load(sys.argv[3]); seed = int(sys.argv[4]); visits = int(f["visit"].max()) + 1
        ev = events_without_network(mk(seed), visits)
        for k in ("update", "visit", "task", "blind", "wrap", "seen", "mirror"):
            if k in f.files:
                print(f"   {k}: {'equal' if np.array_equal(np.asarray(f[k]).astype(np.int64), ev[k]) else 'DIFFERENT'}")
    elif what == "score":
        f = np.load(sys.argv[3]); visits = int(sys.argv[4]) if len(sys.argv) > 4 else int(f["visit"].max()) + 1
        r = score({k: f[k] for k in f.files}, visits)
        print({k: r[k] for k in ("late_level", "score_trend", "learning_trend", "cost_trend")})
        print("first-visit level by task:", [round(d["level"], 3) for d in r["first_visit"]])
