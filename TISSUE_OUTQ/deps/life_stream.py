"""The stream of one life (task 26 of the reviewer's MESSAGES_TO_WORKERS_20261118.md, 19 November 2026).

One unbroken stream of bytes for a seed, in phases, with a label for every position: its phase, its class in the first family, and what,
if anything, is due there. Python 3 standard library and numpy only; nothing of the repository is imported.

    python3 life_stream.py            # the test: counts of each kind of position for seeds 73129, 93683 and 116089, and the assertions
    from life_stream import life      # life(seed) -> dict of numpy arrays (see LABELS below); iter_life(seed) yields one position at a time

The phases, in order (lengths in bytes; every number is a default of LifeConfig and can be changed):

 1. FIRST    (262,144)  the items of the first family as in the brief of step 1, section 4: the record's split, order, letters and render,
                        with the end mark byte 10 and no start symbol. Nothing is due (step 1 has no correction).
 2. KNOWABLE (131,072, then the item in progress is completed)  the same item stream, continued without a break, as in the brief of
                        step 2: positions 262,144 to 393,215 are its stream. Due: the bytes that the last three bytes fix (SIGN after
                        the first operand, EQ after the second, END after the answer).
    QUIET    (4,096)    the byte 32 (space), repeated; nothing due. Between every two of the later phases.
 3. DELAY    (131,072)  items  # X F... = X \\n : the byte X shown, then a gap of L bytes (L - 1 capital letters and '='), then X due again,
                        then the end mark. X is a digit, L is 1, 2, 4 or 8 (uniform). Due: the second X.
 4. TABLES   two small tables of taught answers, items  @ k1 k2 = a1 a2 \\n : a key of two lower-case letters and its fixed answer of two
                        digits, drawn once for the seed. 4a: table 1 (65,536); quiet; 4b: table 2 (65,536); quiet; 4c: table 1 returns
                        (32,768). Each phase cycles through its table in shuffled passes. Due: the two answer digits.

No pair of the first family's reserve (the second half of each task's split, as in check_step1_brief.py) is ever rendered: phases 1 and 2
use only the first half, and phases 3 and 4 use no task byte (p, q, v, w are excluded from the keys) and no sign byte (+ - > <).

Choices of this program (none is fixed by a brief; each is a default that can be changed): the lengths above; phases 1 and 2 without a
quiet stretch between them, so that positions 0 to 393,215 are the streams of steps 1 and 2 byte for byte; the item in progress at the end
of phase 2 is completed before the quiet stretch, so that no item is cut; the quiet byte (space, 32); the delay item's form, its marks '#'
and '=' and its digits; the table items' form and mark '@'; 16 keys a table; the generators Random(seed * 1000 + 3) for phase 3 and
Random(seed * 1000 + 4) for the tables. The test prints, for each gap L, whether the last three bytes before the due position contain X
(they do for L = 1 and L = 2, so a table of the last three bytes can answer those).
"""
import hashlib
from dataclasses import dataclass
from random import Random

import numpy as np

SEEDS = (73129, 93683, 116089)

# ------------------------------------------------------------------------------------------------ the first family (check_step1_brief.py)
SYM = {"add": "+", "sub": "-", "max": ">", "min": "<"}
DEV = [("add", 89, b"p"), ("sub", 89, b"q"), ("max", 89, b"v"), ("min", 89, b"w")]
DISTRACT = list(b"ABCDEFGHIJKLMNOPQRSTUVWXYZ")
END = 10
KMAX = 12


def apply(op, m, a, b):
    return {"add": (a + b) % m, "sub": (a - b) % m, "max": max(a, b), "min": min(a, b)}[op]


def render(task, a, b, k, rng):           # the record's render, with the end mark as a byte (check_step1_brief.py, lines 49 to 53)
    op, m, code = task
    d = [rng.choice(DISTRACT) for _ in range(k)]
    prompt = list(code) + list(f"{a:02d}".encode()) + [ord(SYM[op])] + d + list(f"{b:02d}".encode()) + [ord("=")]
    return prompt + list(f"{apply(op, m, a, b):02d}".encode()) + [END]


def split(task, frac, seed):               # the record's split
    op, m, code = task
    pairs = [(a, b) for a in range(m) for b in range(m)]
    Random(seed * 1000 + round(frac * 100) + m * 7 + code[0]).shuffle(pairs)
    k = round(frac * len(pairs))
    return pairs[:k], pairs[k:]


def reserve(seed):
    """The first family's reserve: for each task, the pairs of the second half of its split. Never rendered in a life."""
    return {t[0]: set(split(t, .5, seed)[1]) for t in DEV}


def items(seed):                           # the record's stream order, item by item (check_step1_brief.py, lines 64 to 73)
    rng = Random(seed + 77)
    flat = [(t, pr) for t in DEV for pr in split(t, .5, seed)[0]]
    while True:
        order = list(range(len(flat))); rng.shuffle(order)
        for i in order:
            t, (a, b) = flat[i]
            k = rng.randint(0, KMAX)
            yield render(t, a, b, k, rng), t, a, b, k


# ------------------------------------------------------------------------------------------------ labels
PHASES = {0: "FIRST", 1: "KNOWABLE", 2: "QUIET", 3: "DELAY", 4: "TABLE1", 5: "TABLE2", 6: "TABLE1_RETURNS"}
CLASSES = {0: "-", 1: "TASK", 2: "A1", 3: "A2", 4: "SIGN", 5: "LET", 6: "B1", 7: "B2", 8: "EQ", 9: "R1", 10: "R2", 11: "END"}
DUE = {0: "none", 1: "SIGN", 2: "EQ", 3: "END", 4: "RECALL", 5: "ANSWER1", 6: "ANSWER2"}
LABELS = """life(seed) returns: bytes (uint8), phase (uint8, PHASES), cls (uint8, CLASSES: the role of the byte in a first-family item, 0
elsewhere), due (uint8, DUE: what is due at the position, 0 for nothing), gap (int8: the gap L of a delay item at its positions, else 0),
table (int8: 1 or 2 at the positions of a table item, else 0), key (int16: the key's index in its table, else -1), first (bool: the due
position of a table item whose key has not been asked before in the life), spans (list of (phase name, start, end)), tables (the two
tables, key -> answer), config."""
QUIET_BYTE = 32
DIGITS = list(b"0123456789")
KEY_LETTERS = [c for c in b"abcdefghijklmnopqrstuvwxyz" if c not in b"pqvw"]


@dataclass(frozen=True)
class LifeConfig:
    first: int = 262144
    knowable: int = 131072
    quiet: int = 4096
    delay: int = 131072
    table1: int = 65536
    table2: int = 65536
    table1_returns: int = 32768
    keys_per_table: int = 16
    gaps: tuple = (1, 2, 4, 8)


def first_family_positions(cfg, seed):
    """Phases 1 and 2: the item stream, labelled by class; cut after `first + knowable` bytes at the end of the item in progress."""
    n12 = cfg.first + cfg.knowable
    out, cls = [], []
    for byts, t, a, b, k in items(seed):
        c = [1, 2, 3, 4] + [5] * k + [6, 7, 8, 9, 10, 11]
        assert len(c) == len(byts)
        out.extend(byts); cls.extend(c)
        if len(out) >= n12:
            break
    return out, cls


def tables_of(seed, nkeys):
    rng = Random(seed * 1000 + 4)
    keys = [bytes([x, y]) for x in KEY_LETTERS for y in KEY_LETTERS]
    rng.shuffle(keys)
    t1 = {kk: f"{rng.randint(0, 99):02d}".encode() for kk in keys[:nkeys]}
    t2 = {kk: f"{rng.randint(0, 99):02d}".encode() for kk in keys[nkeys:2 * nkeys]}
    return t1, t2, rng


def life(seed, cfg=LifeConfig()):
    b, ph, cl, du, gp, tb, ky, fs = [], [], [], [], [], [], [], []
    spans = []

    def put(byte, phase, c=0, d=0, g=0, t=0, k=-1, f=False):
        b.append(byte); ph.append(phase); cl.append(c); du.append(d); gp.append(g); tb.append(t); ky.append(k); fs.append(f)

    def quiet():
        s = len(b)
        for _ in range(cfg.quiet):
            put(QUIET_BYTE, 2)
        spans.append(("QUIET", s, len(b)))

    # phases 1 and 2
    out, cls = first_family_positions(cfg, seed)
    for i, (x, c) in enumerate(zip(out, cls)):
        if i < cfg.first:
            put(x, 0, c)
        else:
            put(x, 1, c, {4: 1, 8: 2, 11: 3}.get(c, 0))
    spans.append(("FIRST", 0, cfg.first)); spans.append(("KNOWABLE", cfg.first, len(b)))
    quiet()
    # phase 3
    rng = Random(seed * 1000 + 3)
    s = len(b)
    while len(b) - s < cfg.delay:
        x = rng.choice(DIGITS)
        L = rng.choice(cfg.gaps)
        put(ord("#"), 3, g=L); put(x, 3, g=L)
        for _ in range(L - 1):
            put(rng.choice(DISTRACT), 3, g=L)
        put(ord("="), 3, g=L); put(x, 3, d=4, g=L); put(END, 3, g=L)
    spans.append(("DELAY", s, len(b)))
    quiet()
    # phase 4
    t1, t2, rt = tables_of(seed, cfg.keys_per_table)
    asked = {1: set(), 2: set()}

    def table_phase(name, phase, tid, tab, length):
        s = len(b)
        keys = list(tab)
        while len(b) - s < length:
            order = keys[:]; rt.shuffle(order)
            for kk in order:
                ki = keys.index(kk)
                first = kk not in asked[tid]
                asked[tid].add(kk)
                ans = tab[kk]
                put(ord("@"), phase, t=tid, k=ki); put(kk[0], phase, t=tid, k=ki); put(kk[1], phase, t=tid, k=ki)
                put(ord("="), phase, t=tid, k=ki)
                put(ans[0], phase, d=5, t=tid, k=ki, f=first); put(ans[1], phase, d=6, t=tid, k=ki, f=first)
                put(END, phase, t=tid, k=ki)
                if len(b) - s >= length:
                    break
        spans.append((name, s, len(b)))

    table_phase("TABLE1", 4, 1, t1, cfg.table1)
    quiet()
    table_phase("TABLE2", 5, 2, t2, cfg.table2)
    quiet()
    table_phase("TABLE1_RETURNS", 6, 1, t1, cfg.table1_returns)
    return dict(bytes=np.array(b, np.uint8), phase=np.array(ph, np.uint8), cls=np.array(cl, np.uint8), due=np.array(du, np.uint8),
                gap=np.array(gp, np.int8), table=np.array(tb, np.int8), key=np.array(ky, np.int16), first=np.array(fs, bool),
                spans=spans, tables={1: t1, 2: t2}, config=cfg)


def iter_life(seed, cfg=LifeConfig()):
    """One position at a time: (position, byte, phase name, class name, due name)."""
    L = life(seed, cfg)
    for i in range(len(L["bytes"])):
        yield i, int(L["bytes"][i]), PHASES[int(L["phase"][i])], CLASSES[int(L["cls"][i])], DUE[int(L["due"][i])]


# ------------------------------------------------------------------------------------------------ the test
STEP_DIGESTS = {   # the briefs' digests: step 1, positions 0 to 262,143 (first 16 hex digits); step 2, positions 262,144 to 393,215
    73129: ("7772341c01ab0954", "7ea85f8bf128b75a807402b51299716dc92402260c2bd80e53fd9c8928808266"),
    93683: ("a6363a43f56c1cdc", "70ca5578ce500e33c4a2a7954a96b93754bddc87ac3c4e8c1c819d6089bb61ce"),
    116089: ("0473b978c0a8db0d", "a3ef9af083f568fd65beb9d97764f75b5689b52a68b675a3e614a51c3c4f87de"),
}


def scan_first_family(by):
    """Every first-family item pattern in a byte array: (start, op, a, b). Searched in the whole life, every phase."""
    s = by.tobytes()
    import re
    found = []
    for mt in re.finditer(rb"([pqvw])(\d\d)([+\-><])[A-Z]*(\d\d)=", s):
        op = {b"p": "add", b"q": "sub", b"v": "max", b"w": "min"}[mt.group(1)]
        found.append((mt.start(), op, int(mt.group(2)), int(mt.group(4)), mt.group(3)))
    return found


def test(seeds=SEEDS):
    ok = True
    for seed in seeds:
        L = life(seed)
        by, ph, cl, du = L["bytes"], L["phase"], L["cls"], L["due"]
        n = len(by)
        print("=" * 110)
        print(f"seed {seed}: {n:,} positions; SHA-256 of the life {hashlib.sha256(by.tobytes()).hexdigest()}")
        print("  phases:", "; ".join(f"{nm} {a:,}-{e - 1:,} ({e - a:,})" for nm, a, e in L["spans"]))
        d1 = hashlib.sha256(by[:262144].tobytes()).hexdigest()[:16]
        d2 = hashlib.sha256(by[262144:393216].tobytes()).hexdigest()
        same = (d1, d2) == STEP_DIGESTS[seed]
        ok &= same
        print(f"  positions 0-262,143 and 262,144-393,215 equal the streams of steps 1 and 2 (their briefs' digests): {same}")
        print("  positions by phase:", ", ".join(f"{PHASES[p]} {int((ph == p).sum()):,}" for p in PHASES))
        print("  positions by first-family class (phases 1 and 2):", ", ".join(f"{CLASSES[c]} {int((cl == c).sum()):,}" for c in CLASSES if c))
        print("  positions by what is due:", ", ".join(f"{DUE[d]} {int((du == d).sum()):,}" for d in DUE))
        for p in (1,):
            print(f"  due in KNOWABLE:", ", ".join(f"{DUE[d]} {int(((ph == p) & (du == d)).sum()):,}" for d in (1, 2, 3)))
        g = L["gap"]
        rec = np.nonzero(du == 4)[0]
        print("  DELAY, recall positions by gap L:", ", ".join(f"L={x} {int((g[rec] == x).sum()):,}" for x in LifeConfig().gaps))
        right = all(by[i] == by[i - L_ - 1] for i, L_ in zip(rec.tolist(), g[rec].tolist()))
        within3 = {x: all(by[i] in by[i - 3:i] for i in rec[g[rec] == x].tolist()) for x in LifeConfig().gaps}
        print(f"  DELAY: the due byte is the byte shown L + 1 positions before: {right}; "
              f"the last three bytes before the due position contain it, by L: {within3}")
        ok &= right and within3 == {1: True, 2: True, 4: False, 8: False}
        tb, ky, fs = L["table"], L["key"], L["first"]
        for tid, pset in ((1, (4, 6)), (2, (5,))):
            for p in pset:
                a1 = (ph == p) & (du == 5)
                print(f"  {PHASES[p]}: items {int(a1.sum()):,}; due positions {int(((ph == p) & (du >= 5)).sum()):,}; "
                      f"first asking of a key {int((a1 & fs).sum())}; keys asked {len(set(ky[a1].tolist()))}")
        tabs = L["tables"]
        good = True
        for i in np.nonzero(du == 5)[0].tolist():
            ans = tabs[int(tb[i])][bytes(by[i - 3:i - 1].tolist())]
            good &= by[i] == ans[0] and by[i + 1] == ans[1]
        print(f"  TABLES: every answer is the table's: {good}; table 1 {sorted(k.decode() + '=' + v.decode() for k, v in tabs[1].items())[:4]}...; "
              f"table 2 {sorted(k.decode() + '=' + v.decode() for k, v in tabs[2].items())[:4]}...")
        ok &= good
        fam = scan_first_family(by)
        res = reserve(seed)
        in_res = sum(1 for _, op, a, b_, _s in fam if (a, b_) in res[op])
        outside = sum(1 for st, *_ in fam if ph[st] not in (0, 1))
        print(f"  first-family items found by scanning the whole life: {len(fam):,}; of them in the reserve: {in_res}; "
              f"outside phases 1 and 2: {outside}")
        ok &= in_res == 0 and outside == 0
        print(f"  byte values used: {len(set(by.tolist()))}; values new to the life after phase 2: "
              f"{sorted(set(by[393216:].tolist()) - set(by[:393216].tolist()))}")
    print("=" * 110)
    print("all assertions hold:", ok)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if test() else 1)
