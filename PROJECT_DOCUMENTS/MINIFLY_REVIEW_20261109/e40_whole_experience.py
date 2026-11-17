#!/usr/bin/env python3
"""E40: the lived learner's saving, counted in its whole experience (17 November 2026).

Read-only, standard library only. Every figure is read from a receipt or a source file of the record through
`git show <ref>:<path>`, or computed here from the code's own rules (the split of the pairs, the composition triples), and each
such computation is checked against a count the receipts hold. Nothing is checked out, imported from the record or executed
from it, and nothing is written: the tables go to standard output as Markdown.

    cd <repository>
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e40_whole_experience.py
        [--ref origin/codex/minifly-consolidated-20261023]   the ref every file is read from
        [--paths]                                             also list every path read

Terms. A *family* is four operations (add, sub, max, min) under one modulus. A *pair* is an ordered pair of operands (a, b).
A *distinct taught pair* is one entry of a family's teaching list, the triple (operation of that family, a, b); the same operands
under the same operation in another family are another entry, because the code byte and the modulus differ. A *composition* is
one entry (operator pair, a, b, c) of the life's second family. An *item* is one presentation of an entry, rendered with 0 to 12
capital letters drawn afresh, so one entry gives many different items. An *update* is one optimizer step on a batch of 64 items.
A *row* is a point at which a run was scored. A *reduction pair* is a pair of add with a + b >= modulus or of sub with a - b < 0.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from random import Random

REPO = Path(__file__).resolve().parents[2]
M = "MOD97_LOCAL_RULE_20261106"
SEEDS = (73129, 93683, 116089)
THRESHOLDS = (50, 80, 90)
REF = "origin/codex/minifly-consolidated-20261023"
READ = []


# ---------------------------------------------------------------- reading the record (git show only)
def git(*args):
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, check=True).stdout


def show(path):
    READ.append(path)
    return git("show", f"{REF}:{path}")


def J(path):
    return json.loads(show(path))


def n_(v):
    return "none" if v is None else f"{v:,}"


def span(vals, fmt="{:,}"):
    lo, hi = fmt.format(min(vals)), fmt.format(max(vals))
    return lo if lo == hi else f"{lo} to {hi}"


def table(head, rows):
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for r in rows:
        print("| " + " | ".join(str(c) for c in r) + " |")
    print()


# ---------------------------------------------------------------- the code's rules, restated and checked against the source text
SYM = {"add": "+", "sub": "-", "max": ">", "min": "<"}
DEV = [("add", 89, b"p"), ("sub", 89, b"q"), ("max", 89, b"v"), ("min", 89, b"w")]      # the first family (the birth)
F1 = [("add", 97, b"r"), ("sub", 97, b"t"), ("max", 97, b"x"), ("min", 97, b"n")]       # life, blocks 1 and 2 onward
F3 = [("add", 73, b"s"), ("sub", 73, b"u"), ("max", 73, b"y"), ("min", 73, b"z")]       # life, blocks 5 and 6
F4 = [("add", 83, b"e"), ("sub", 83, b"f"), ("max", 83, b"g"), ("min", 83, b"h")]       # E40's new family
SCHEDULE = [("F1", 1), ("F1", 2), ("F2", 1), ("F2", 2), ("F3", 1), ("F3", 2)]
PAIRS = [(a, b) for a in SYM.values() for b in SYM.values()]
HELD_PAIRS = [("+", "-"), ("-", "+"), (">", "<"), ("<", ">")]
TAUGHT_PAIRS = [p for p in PAIRS if p not in HELD_PAIRS]
SYM_OP = {v: k for k, v in SYM.items()}
BATCH, DEV_UPDATES, STAGE_UPDATES = 64, 24576, 6144
E40_ROWS = (0, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
E41_DOSE, E41_ROW = 12288, 256

SOURCE_LINES = [   # (file, a line of it that states the rule used here)
    ("exp45.py", "BATCH, BLOCK = 64, 512"),
    ("exp45.py", "DEV_UPDATES = 24576"),
    ("exp11_cell.py", 'DEV_UPDATES, STAGE_UPDATES = X.DEV_UPDATES, int(os.environ.get("E11_STAGE_UPDATES", "6144"))'),
    ("exp11_cell.py", 'DEV = [("add", 89, b"p"), ("sub", 89, b"q"), ("max", 89, b"v"), ("min", 89, b"w")]'),
    ("exp11_cell.py", 'LIFE = [("add", 97, b"r"), ("sub", 97, b"t"), ("max", 97, b"x"), ("min", 97, b"n")]'),
    ("exp11_cell.py", "rng = Random(seed * 1000 + round(frac * 100) + m * 7 + code[0])"),
    ("exp11_cell.py", "k = round(frac * len(pairs))"),
    ("exp11_cell.py", "rng.shuffle(order)"),
    ("exp11_cell.py", "yield render(t, a, b, rng.randint(0, kmax), rng)"),
    ("exp16_integrated.py", "F1 = E.LIFE"),
    ("exp16_integrated.py", 'F3 = [("add", 73, b"s"), ("sub", 73, b"u"), ("max", 73, b"y"), ("min", 73, b"z")]'),
    ("exp16_integrated.py", 'HELD_PAIRS = [("+", "-"), ("-", "+"), (">", "<"), ("<", ">")]'),
    ("exp16_integrated.py", "rng = Random(seed * 31 + ord(pair[0]) * 7 + ord(pair[1])) if split == \"taught\""),
    ("exp16_integrated.py", 'return [("comp", o1, o2, a, b, c) for o1, o2 in TAUGHT_PAIRS for a, b, c in triples((o1, o2), seed, 2000)]'),
    ("exp16_integrated.py", 'return [("op", t, a, b) for t in (F1 if fam == "F1" else F3) for a, b in E.split(t, .25, seed)[0]]'),
    ("exp16_integrated.py", "yield render(entries[i], rng)"),
    ("exp16_integrated.py", "yield next(streams[rng.randrange(len(streams))])"),
    ("exp16_integrated.py", 'SCHEDULE = [("F1", 1), ("F1", 2), ("F2", 1), ("F2", 2), ("F3", 1), ("F3", 2)]'),
    ("exp26_candidate.py", "tp = [(t, E.split(t, .5, seed)[0]) for t in E.DEV]"),
    ("exp26_candidate.py", "for stop in range(X.BLOCK, E.DEV_DOSE + 1, X.BLOCK):"),
    ("exp26_candidate.py", "entries = {f: X16.family_entries(f, seed) for f in (\"F1\", \"F2\", \"F3\")}"),
    ("exp26_candidate.py", "seen_streams.append(X16.entry_stream(entries[fam], seed + 100 + si))"),
    ("exp26_candidate.py", "for u in range(E.STAGE_UPDATES):"),
    ("e40_new_family.py", "ROWS = (0, 64, 128, 256, 512, 1024, 2048, 4096, 8192)"),
    ("e40_new_family.py", 'F4 = [("add", 83, b"e"), ("sub", 83, b"f"), ("max", 83, b"g"), ("min", 83, b"h")]'),
    ("e40_new_family.py", "FRAC = .10"),
    ("e40_new_family.py", "return a + b >= 83 if op == \"add\" else a - b < 0"),
    ("exp41_scale.py", "DOSE = {256: 40960, 512: 12288, 1024: 8192}"),
    ("exp41_scale.py", "ROW = 256"),
    ("exp41_scale.py", 'return Random(f"e41|{seed}|{round(f * 100)}|{t[0]}").sample(sorted(pairs), 1000)'),
]


def check_source():
    """Returns the appendix's rows; stops if a line is missing."""
    text, rows = {}, []
    for f, line in SOURCE_LINES:
        if f not in text:
            text[f] = show(f"{M}/{f}").decode().splitlines()
        hits = [i + 1 for i, s in enumerate(text[f]) if line in s]
        assert hits, f"{f}: line not found: {line}"
        rows.append((f"`{f}`", hits[0], f"`{line}`"))
    return rows


def split(task, frac, seed):
    """exp11_cell.split: the first round(frac * m * m) pairs of a seeded shuffle are taught, the rest are never taught."""
    op, m, code = task
    pairs = [(a, b) for a in range(m) for b in range(m)]
    rng = Random(seed * 1000 + round(frac * 100) + m * 7 + code[0])
    rng.shuffle(pairs)
    k = round(frac * len(pairs))
    return pairs[:k], pairs[k:]


def apply(op, m, a, b):
    return {"add": (a + b) % m, "sub": (a - b) % m, "max": max(a, b), "min": min(a, b)}[op]


def triples(pair, seed, n):
    """exp16_integrated.triples, taught split: n distinct operand triples under modulus 97 for one operator pair."""
    rng = Random(seed * 31 + ord(pair[0]) * 7 + ord(pair[1]))
    out, seen = [], set()
    while len(out) < n:
        tr = (rng.randrange(97), rng.randrange(97), rng.randrange(97))
        if tr not in seen:
            seen.add(tr); out.append(tr)
    return out


def needs_reduction(op, m, a, b):
    return a + b >= m if op == "add" else a - b < 0


def name(t):
    return f"{t[0]}{t[1]}"


def strict_counts(seed):
    """exp16_integrated.strict_panels' counts (held pairs of the life's families with no equivalent supervision by any route),
    rebuilt here only to check this file's split and triples against the life receipts."""
    alone = {name(t): set(split(t, .25, seed)[0]) for t in F1 + F3}
    alone.update({name(t): set(split(t, .5, seed)[0]) for t in DEV})
    sub = {}
    for o1, o2 in TAUGHT_PAIRS:
        for a, b, c in triples((o1, o2), seed, 2000):
            sub.setdefault(SYM_OP[o1], set()).add((a, b))
            sub.setdefault(SYM_OP[o2], set()).add((apply(SYM_OP[o1], 97, a, b), c))
    other = {}
    for t in F1 + F3 + DEV:
        if t[0] in ("max", "min"):
            other[name(t)] = set().union(*[alone[name(u)] for u in F1 + F3 + DEV if u[0] == t[0] and u[1] != t[1]])
    out = {}
    for t in F1 + F3:
        comm = t[0] in ("add", "max", "min"); held = split(t, .25, seed)[1]; s = sub.get(t[0], ()) if t[1] == 97 else ()
        om = other.get(name(t), ())
        ok = 0
        for a, b in held:
            flagged = (comm and a != b and (b, a) in alone[name(t)]) or (a, b) in s or (comm and (b, a) in s) or (a, b) in om or (b, a) in om
            ok += not flagged
        out[name(t)] = dict(held=len(held), strict=ok)
    return out, sub


# ---------------------------------------------------------------- 1. before the new family: the birth and the life
def before_the_new_family():
    print("## 4. What a lived network was given before the new family\n")
    print("### 4.1 Updates and presented items\n")
    births = {(a, s): J(f"{M}/exp45/e26/e26_dev_{a}_{s}.json") for a in "SB" for s in SEEDS}
    lives = {(a, s): J(f"{M}/exp45/e26/e26_life_{a}_{s}.json") for a in "SB" for s in SEEDS}
    lives.update({("K", s): J(f"{M}/exp45/e31/e31_life_K_{s}.json") for s in SEEDS})
    # the chain: an E40 lived run starts from a life file, which starts from a birth file
    for (a, s), lf in lives.items():
        st = lf["stages"][0]["settings"]
        b = births[("S" if a == "K" else a, s)]
        assert st["birth_sha256"] == b["weights_sha256"], (a, s)
        assert st["stage_updates"] == STAGE_UPDATES and st["dev_updates"] == DEV_UPDATES and st["batch"] == BATCH
        assert [tuple(x) for x in lf["schedule"]] == SCHEDULE and lf["complete"]["optimizer_steps"] == [STAGE_UPDATES * 6]
        assert lf["stages"][0]["entries"] == {"F1": 9408, "F2": 24000, "F3": 5328}
    for (a, s), b in births.items():
        assert b["complete"] and b["dev_updates"] == DEV_UPDATES and b["batch"] == BATCH and b["optimizer_steps"] == [DEV_UPDATES]
        assert b["data_digest"]["batches"] == DEV_UPDATES and [c["step"] for c in b["curve"]] == list(range(4096, DEV_UPDATES + 1, 4096))
        assert b["model"]["params"] == 1124113
    # the data are the same for every rule of a seed
    for s in SEEDS:
        assert births[("S", s)]["data_digest"] == births[("B", s)]["data_digest"]
        for i in range(1, 7):
            d = {a: lives[(a, s)]["stages"][i]["data_digest"] for a in "SBK"}
            e = {a: lives[(a, s)]["stages"][i]["exposure_rows_tokens"] for a in "SBK"}
            assert d["S"] == d["B"] == d["K"] and e["S"] == e["B"] == e["K"], (s, i)
    print("Receipts: `exp45/e26/e26_dev_<S|B>_<seed>.json` (births), `exp45/e26/e26_life_<S|B>_<seed>.json` and "
          "`exp45/e31/e31_life_K_<seed>.json` (lives). Checked here: each life names its birth file by sha256; the sweep's "
          "life and the life of arm K (the sweep with learned feedback) both start from the sweep's birth; the batch digest "
          "of the birth is equal under both rules, and the batch digest and the count of items by family of every life "
          "block are equal under the three rules. So the counts below hold for every rule and differ only by seed.\n")
    print("Rows the receipts have: a birth is scored at 4,096, 8,192, 12,288, 16,384, 20,480 and 24,576 updates; a life "
          "before its first block and after each of its six blocks.\n")
    rows, tot = [], {}
    rows.append(("Birth", "first family (modulus 89) alone", f"{DEV_UPDATES:,}", f"{DEV_UPDATES * BATCH:,}",
                 *[f"89: {DEV_UPDATES * BATCH:,}"] * 3))
    for i, (fam, blk) in enumerate(SCHEDULE, 1):
        cells = []
        for s in SEEDS:
            e = lives[("S", s)]["stages"][i]["exposure_rows_tokens"]
            assert sum(v[0] for v in e.values()) == STAGE_UPDATES * BATCH
            assert lives[("S", s)]["stages"][i]["task"] == f"{fam}.{blk}"
            for k, v in e.items():
                tot.setdefault(s, {}).setdefault(k[-2:], 0); tot[s][k[-2:]] += v[0]
            cells.append("; ".join(f"{k[-2:]}: {e['joint/' + k[-2:]][0]:,}" for k in sorted(e)))
        mix = {1: "F1 alone", 2: "F1 alone", 3: "F1 and F2, an item from either with equal chance", 4: "as block 3",
               5: "F1, F2 and F3, an item from any with equal chance", 6: "as block 5"}[i]
        rows.append((f"Life block {i} ({fam}.{blk})", mix, f"{STAGE_UPDATES:,}", f"{STAGE_UPDATES * BATCH:,}", *cells))
    life_u, life_i = STAGE_UPDATES * 6, STAGE_UPDATES * 6 * BATCH
    rows.append(("Life, six blocks", "", f"{life_u:,}", f"{life_i:,}",
                 *["; ".join(f"{k}: {tot[s][k]:,}" for k in ("F1", "F2", "F3")) for s in SEEDS]))
    rows.append(("**Birth and life**", "", f"**{DEV_UPDATES + life_u:,}**", f"**{(DEV_UPDATES + life_u) * BATCH:,}**", "", "", ""))
    table(("Part", "Stream", "Updates", "Presented items", "Items by family, seed 73129", "seed 93683", "seed 116089"), rows)
    print("F1 is the family of modulus 97, F2 the compositions, F3 the family of modulus 73. The first family is not in a "
          "life's stream.\n")
    return births, lives, tot, (DEV_UPDATES + life_u) * BATCH


# ---------------------------------------------------------------- 2. distinct taught samples
def distinct_samples(lives, tot):
    print("### 4.2 Distinct taught samples\n")
    fams = [("First family (birth)", DEV, .50, 89), ("F1 (life)", F1, .25, 97), ("F3 (life)", F3, .25, 73), ("New family (E40)", F4, .10, 83)]
    counts = {}
    for label, fam, frac, m in fams:
        per = {len(split(t, frac, s)[0]) for t in fam for s in SEEDS}
        assert per == {round(frac * m * m)}
        counts[label] = (m, frac, m * m, per.pop())
    # checks of this file's split and triples against counts the receipts hold
    chk = []
    c5 = J(f"{M}/exp45/e40/checks/E40_check5.json")["result"]["seeds"]
    for s in SEEDS:
        assert c5[str(s)]["taught_pairs"] == {t[0]: 689 for t in F4} and c5[str(s)]["items"] == 8192 * BATCH
        sc, sub = strict_counts(s)
        for a in "SB":
            assert lives[(a, s)]["stages"][0]["settings"]["strict_panel_counts"] == sc, (a, s)
        r40 = J(f"{M}/exp45/e40/e40_lived_S_{s}.json")["rows"][0]["scores"]["held_plain_k0"]
        for t in F4[:2]:
            held = split(t, .10, s)[1]
            red = sum(needs_reduction(t[0], 83, a, b) for a, b in held)
            assert r40[name(t)]["reduction"]["n"] == red and r40[name(t)]["no_reduction"]["n"] == len(held) - red, (s, t)
        for fp, frac in ((10, .10), (25, .25), (50, .50)):
            want = J(f"{M}/exp45/e41/e41_S_h512_f{fp}_{s}.json")["reduction_sample_sha256"]
            for t in DEV[:2]:
                held = split(t, frac, s)[1]
                pr = [(a, b) for a, b in held if needs_reduction(t[0], 89, a, b)]
                smp = Random(f"e41|{s}|{round(frac * 100)}|{t[0]}").sample(sorted(pr), 1000)
                got = hashlib.sha256(json.dumps([list(p) for p in smp]).encode()).hexdigest()
                assert got == want[name(t)], (s, fp, t)
        chk.append(s)
    g = J(f"{M}/exp45/e35a/e35a_dev_G_f10_73129.json")["split_counts"]["add89"]
    assert g == dict(taught=792, held=7129)
    print("Counting rule. A family's teaching list is `exp11_cell.split(task, fraction, seed)[0]` for each of its four "
          "operations: the first `round(fraction × modulus²)` pairs of a seeded shuffle (Python's `round`, which sends a "
          "half to the even number). The compositions' list is 2,000 operand triples for each of the 12 taught operator "
          "pairs. The count of distinct samples is the length of these lists. It is the same for every seed and rule. "
          "Every entry of every list was presented at least once, because a stream passes through its whole list in "
          "shuffled order before it repeats and each stream ran for more items than its list is long (last column).\n")
    print(f"Checked for seeds {', '.join(map(str, chk))}: this file's split gives E40's count of taught pairs "
          "(`exp45/e40/checks/E40_check5.json`), the counts of never-taught pairs with and without the reduction at row 0 "
          "of `e40_lived_S_<seed>.json`, the sha256 of E41's two samples of 1,000 reduction pairs at the three fractions "
          "(`e41_S_h512_f<10|25|50>_<seed>.json`), and, with the composition triples, the eight `strict_panel_counts` in "
          "the life receipts; and E35-A's split count at a tenth (792 taught, 7,129 not).\n")
    rows = []
    items = {"First family (birth)": [DEV_UPDATES * BATCH] * 3, "F1 (life)": [tot[s]["F1"] for s in SEEDS],
             "F3 (life)": [tot[s]["F3"] for s in SEEDS], "New family (E40)": [8192 * BATCH] * 3}
    for label, fam, frac, m in fams[:3]:
        mm, fr, all_, per = counts[label]
        it = items[label]
        assert min(it) >= 4 * per
        rows.append((label, mm, f"{all_:,}", f"{fr:.2f}", f"{per:,}", f"{4 * per:,}", span(it), span([v / (4 * per) for v in it], "{:.1f}")))
    f2 = [tot[s]["F2"] for s in SEEDS]
    assert min(f2) >= 24000
    single = sum(4 * counts[l][3] for l in ("First family (birth)", "F1 (life)", "F3 (life)"))
    rows.insert(2, ("F2 (life): compositions under modulus 97", 97, "97³ = 912,673 triples for each of 12 operator pairs", "",
                    "2,000 triples per operator pair", "**24,000 compositions**", span(f2), span([v / 24000 for v in f2], "{:.1f}")))
    rows.append(("**Before the new family**", "", "", "", "", f"**{single:,} pairs and 24,000 compositions**", "", ""))
    mm, fr, all_, per = counts["New family (E40)"]
    rows.append(("New family (E40), whole run of 8,192 updates", mm, f"{all_:,}", f"{fr:.2f}", f"{per:,}", f"**{4 * per:,}**",
                 f"{8192 * BATCH:,}", f"{8192 * BATCH / (4 * per):.1f}"))
    table(("Family", "Modulus", "Pairs per operation", "Taught fraction", "Taught pairs per operation",
           "Distinct taught samples (four operations)", "Items presented (three seeds)", "Times each sample was shown, on average"), rows)
    new = 4 * counts["New family (E40)"][3]
    # what the same operands under the same operation had already been (not used in the accounts; stated for the definition)
    rows = []
    for s in SEEDS:
        _, sub = strict_counts(s)
        earlier = {op: set() for op in SYM}
        per_fam = 0
        for fam, frac in ((DEV, .5), (F1, .25), (F3, .25)):
            for t in fam:
                tp = split(t, frac, s)[0]; per_fam += len(tp); earlier[t[0]] |= set(tp)
        uni = sum(len(v) for v in earlier.values())
        subn = sum(len(v) for v in sub.values())
        f1 = {t[0]: set(split(t, .25, s)[0]) for t in F1}
        sub_new = sum(len(sub[op] - f1[op]) for op in sub)
        seen = seen_or_sub = 0
        for t in F4:
            for p in split(t, .10, s)[0]:
                seen += p in earlier[t[0]]; seen_or_sub += p in earlier[t[0]] or p in sub.get(t[0], ())
        rows.append((s, f"{per_fam:,}", f"{uni:,}", f"{subn:,}", f"{sub_new:,}", f"{seen:,} of {new:,}", f"{seen_or_sub:,} of {new:,}"))
    print("The same operands under the same operation recur across families (for max and min, and for add and sub without "
          "reduction, with the same answer). The accounts below do not merge them. For the record, by seed:\n")
    table(("Seed", "Earlier single-operation entries, by family", "The same, merged over families by (operation, a, b)",
           "Distinct single-operation steps (operation, a, b) inside the 24,000 compositions",
           "of these, not in F1's taught quarter", "New family's taught pairs whose (operation, a, b) an earlier family taught",
           "the same, or a composition step"), rows)
    return single, new


# ---------------------------------------------------------------- thresholds
def first_at(rows, thr):
    return next((u for u, v in rows if v >= thr), None)


def first_three(rows, thr):
    for i in range(len(rows) - 2):
        if all(rows[j][1] >= thr for j in (i, i + 1, i + 2)):
            return rows[i][0]
    return None


def cell(u):
    return "not reached" if u is None else f"{u:,} ({u * BATCH:,})"


# ---------------------------------------------------------------- 3. the new family (E40)
def new_family(lives):
    print("## 5. The new family (E40): updates and items to a level\n")
    runs = {}
    for kind, rules in (("lived", "SBK"), ("fresh", "SB")):
        for a in rules:
            for s in SEEDS:
                r = J(f"{M}/exp45/e40/e40_{kind}_{a}_{s}.json")
                assert r["complete"] and r["updates"] == 8192 and tuple(x["updates"] for x in r["rows"]) == E40_ROWS
                assert all(x["presented"]["items"] == x["updates"] * BATCH and x["presented"]["held_out_presented"] == 0 for x in r["rows"])
                if kind == "lived":
                    assert r["start"]["weights_sha256"] == lives[(a, s)]["complete"]["weights_sha256"], (a, s)
                runs[(kind, a, s)] = r
    for s in SEEDS:      # the same batches for every run of a seed
        for i in range(len(E40_ROWS)):
            assert len({runs[k]["rows"][i]["data_digest"]["sha256"] for k in runs if k[2] == s}) == 1
    print("Receipts: `exp45/e40/e40_<lived|fresh>_<S|B|K>_<seed>.json`. Rows the receipts have: 0, 64, 128, 256, 512, "
          "1,024, 2,048, 4,096 and 8,192 updates (0 to 524,288 items), and no row between them, so a level reached at row "
          "64 was reached at or before 4,096 items. Checked here: each lived run starts from the life file its receipt "
          "names (sha256 equal to the life receipt's); every row's presented items equal 64 times its updates and none is a "
          "never-taught pair; the batch digest is equal across the five runs of a seed at every row.\n")
    nred = {}
    for s in SEEDS:
        sc = runs[("lived", "S", s)]["rows"][0]["scores"]
        nred[s] = (sc["held_plain_k0"]["add83"]["reduction"]["n"], sc["held_plain_k0"]["sub83"]["reduction"]["n"])
        assert all(sc["taught_k12"][n]["n"] == 300 for n in ("add83", "sub83"))
    allred = [v for pr in nred.values() for v in pr]
    print("A level is *reached* at the first row at which the score is at or above it. Cells give updates, with "
          "presented items in brackets. The reduction score is the share of whole answers right on all never-taught "
          f"reduction pairs of add and of sub ({span(allred)} pairs each), no letters, mean of the two. The taught panel is "
          "300 taught pairs of add and of sub at 12 letters, mean of the two. A lived network answers part of the new "
          "family before any update, so its row 0 is not zero.\n")
    out = {}
    for what, key in (("never-taught reduction pairs", "reduction_score"), ("the taught panel", "taught_add_sub_mean")):
        rows = []
        for (kind, a, s), r in runs.items():
            cur = [(x["updates"], x["scores"]["summary"][key]) for x in r["rows"]]
            th = {t: first_at(cur, t) for t in THRESHOLDS}
            best = max(cur, key=lambda c: c[1])
            after = [v for u, v in cur if th[80] is not None and u > th[80]]
            rows.append((f"{kind} {a}", s, f"{cur[0][1]:.1f}", *[cell(th[t]) for t in THRESHOLDS],
                         f"{min(after):.1f} to {max(after):.1f}" if after else "—", f"{cur[-1][1]:.1f}", f"{best[1]:.1f} ({best[0]:,})"))
            if key == "reduction_score":
                out[(kind, a, s)] = dict(th=th, last=cur[-1][1], row0=cur[0][1], best=best[1])
        print(f"**On {what}**\n")
        table(("Run", "Seed", "Row 0", "50%", "80%", "90%", "Range over the rows after the 80% row", "Row 8,192", "Highest (row)"), rows)
    new = sum(len(split(t, .10, SEEDS[0])[0]) for t in F4)
    assert E40_ROWS[1] * BATCH >= new
    print(f"The first row after row 0 is at {E40_ROWS[1] * BATCH:,} items, more than the {new:,} taught pairs of the new family, "
          "and the stream shows every taught pair once before it repeats any. So at every row after row 0 the run had "
          f"been shown all {new:,} distinct taught pairs: the count of distinct pairs to a level is "
          f"{new:,} for every lived run, and the receipts cannot show whether fewer would do.\n")
    print("Never-taught reduction pairs scored (add, sub): " + "; ".join(f"seed {s}: {a:,}, {b:,}" for s, (a, b) in nred.items()) + ".\n")
    lv = [v["last"] for k, v in out.items() if k[0] == "lived"]; fr = [v["last"] for k, v in out.items() if k[0] == "fresh"]
    print(f"At row 8,192 the nine lived runs are at {min(lv):.1f} to {max(lv):.1f} and the six fresh runs at {min(fr):.1f} to "
          f"{max(fr):.1f} on the reduction score.\n")
    # what happened to the earlier families while the new one was learned
    rows, fall = [], {}
    for (kind, a, s), r in runs.items():
        if kind != "lived":
            continue
        e0, e1 = r["rows"][0]["scores"]["earlier_k12"], r["rows"][-1]["scores"]["earlier_k12"]
        mean = lambda d, m: 50.0 * (d["add" + m]["exact"] / d["add" + m]["n"] + d["sub" + m]["exact"] / d["sub" + m]["n"])
        for m in ("89", "97", "73"):
            fall.setdefault(m, []).append((mean(e0, m), mean(e1, m)))
        rows.append((f"lived {a}", s, *[f"{mean(e0, m):.1f} → {mean(e1, m):.1f}" for m in ("89", "97", "73")]))
    print("The earlier families during the new one (never-taught add and sub at 12 letters, mean of the two, row 0 → row 8,192):\n")
    table(("Run", "Seed", "Modulus 89", "Modulus 97", "Modulus 73"), rows)
    print("Over the nine lived runs: " + "; ".join(
        f"modulus {m}, {span([v[0] for v in fall[m]], '{:.1f}')} at row 0 and {span([v[1] for v in fall[m]], '{:.1f}')} at row 8,192"
        for m in ("97", "73", "89")) + ".")
    b0, b6 = [], []
    for lf in lives.values():      # the first family before and after the life, on the same panel (life receipts, stages 0 and 6)
        for st, acc in ((lf["stages"][0], b0), (lf["stages"][6], b6)):
            sc = st["scores"]
            acc.append(50.0 * (sc["add89"]["12"]["exact"] / sc["add89"]["12"]["n"] + sc["sub89"]["12"]["exact"] / sc["sub89"]["12"]["n"]))
    assert sorted(round(v, 6) for v in b6) == sorted(round(v[0], 6) for v in fall["89"])
    print(f"The first family (modulus 89) had fallen during the life, in which it is not taught: {span(b0, '{:.1f}')} before "
          f"the life and {span(b6, '{:.1f}')} after it on the same panel (life receipts, stages 0 and 6).\n")
    return out


# ---------------------------------------------------------------- 4. a fresh network on one family
def fresh_references(births):
    print("## 6. A fresh network that learns one family\n")
    print("### 6.1 The first family at three taught fractions (E41, present width)\n")
    print("Receipts: `exp45/e41/e41_<S|B>_h512_f<50|25|10>_<seed>.json` (feedforward width 512, 256 recurrent units, "
          "1,124,113 parameters: the width of every network of E26 and E40). Rows the receipts have: every 256 updates from "
          "0 to 12,288 (0 to 786,432 items). Accuracy R is the share of whole answers right on a fixed sample of 1,000 "
          "never-taught reduction pairs of add and of sub, no letters, mean of the two.\n")
    out, rows = {}, []
    fam = {50: .50, 25: .25, 10: .10}
    for fp in (50, 25, 10):
        for a in "SB":
            for s in SEEDS:
                r = J(f"{M}/exp45/e41/e41_{a}_h512_f{fp}_{s}.json")
                assert r["complete"] and r["H"] == 512 and r["U"] == 256 and r["dose"] == E41_DOSE and r["model"]["params"] == 1124113
                assert [c["step"] for c in r["curve"]] == list(range(0, E41_DOSE + 1, E41_ROW)) and r["batch"] == BATCH
                pairs = sum(len(split(t, fam[fp], s)[0]) for t in DEV)
                R = [(c["step"], c["R"]) for c in r["curve"]]; TA = [(c["step"], c["taught"]) for c in r["curve"]]
                th = {t: first_at(R, t) for t in THRESHOLDS}
                assert th[50] is None or th[50] * BATCH >= pairs      # every taught pair had been shown by then
                out[(fp, a, s)] = dict(th=th, th3=first_three(R, 80), pairs=pairs, last=R[-1][1], best=max(v for _, v in R))
                if fp == 50:     # this run is the lived network's own birth, scored more often
                    b = births[(a, s)]
                    assert next(c for c in r["curve"] if c["step"] == 4096)["data_digest"] == b["curve"][0]["data_digest"]
                    for step, bi in ((4096, 0), (8192, 1)):
                        sealed = next(c for c in r["curve"] if c["step"] == step)["e26_sealed"]
                        assert sealed == b["curve"][bi]["sealed"], (a, s, step)
                rows.append((f"{fam[fp]:.2f}", f"{pairs:,}", a, s, *[cell(th[t]) for t in THRESHOLDS], cell(out[(fp, a, s)]["th3"]),
                             f"{R[-1][1]:.1f}", f"{max(v for _, v in R):.1f}", cell(first_at(TA, 80)), cell(first_at(TA, 90))))
    table(("Taught fraction", "Distinct taught pairs", "Rule", "Seed", "R at 50%", "R at 80%", "R at 90%",
           "R at 80% on three rows running", "R at row 12,288", "Highest R", "Taught panel at 80%", "Taught panel at 90%"), rows)
    print("Checked here: at fraction 0.50 each run's batch digest at row 4,096 equals that of the E26 birth of the same rule "
          "and seed, and its counts on E26's own panels at rows 4,096 and 8,192 equal the birth receipt's. So the fresh "
          "reference at a half is the lived network's own birth, scored every 256 updates instead of every 4,096.\n")
    tenth = [v for (fp, a, s), v in out.items() if fp == 10]
    assert all(v["th"][50] is None for v in tenth)
    print(f"At a tenth no run reaches 50% on R within 12,288 updates; the highest R of the six runs is "
          f"{min(v['best'] for v in tenth):.1f} to {max(v['best'] for v in tenth):.1f}.\n")

    print("### 6.2 Fresh references on other families, and longer ones\n")
    rows = []
    for kind, m, upd in (("F3", 73, (1024, 2048, 4096)), ("F1", 97, (2048, 4096, 6144))):
        for a in "SB":
            for s in SEEDS:
                r = J(f"{M}/exp45/e37/e37_{kind}_{a}_{s}.json")
                assert r["complete"] and tuple(c["step"] for c in r["curve"]) == upd
                assert all(c["scores_k12"][f"{o}{m}"]["n"] == 400 for c in r["curve"] for o in ("add", "sub"))
                fam_ = F3 if kind == "F3" else F1
                pairs = sum(len(split(t, .25, s)[0]) for t in fam_)
                cur = [(c["step"], c["mean_add_sub"]) for c in r["curve"]]
                rows.append((f"E37, {kind} (modulus {m}) alone", "0.25", f"{pairs:,}", a, s,
                             "; ".join(f"{u:,} ({u * BATCH:,}): {v:.1f}" for u, v in cur), cell(first_at(cur, 80))))
    for fp, frac in ((25, .25), (10, .10)):
        r = J(f"{M}/exp45/e35a/e35a_dev_G_f{fp}_73129.json")
        assert r["complete"] and r["dev_updates"] == DEV_UPDATES and r["fraction"] == frac and r["local"] is False
        assert all(c["sealed"][n]["12"]["n"] == 300 for c in r["curve"] for n in ("add89", "sub89"))
        cur = [(c["step"], 100.0 * (c["sealed"]["add89"]["12"]["exact"] + c["sealed"]["sub89"]["12"]["exact"]) / 600) for c in r["curve"]]
        rows.append(("E35-A arm G, first family (modulus 89), a whole birth", f"{frac:.2f}", f"{4 * r['split_counts']['add89']['taught']:,}", "B", 73129,
                     "; ".join(f"{u:,} ({u * BATCH:,}): {v:.1f}" for u, v in cur), cell(first_at(cur, 80))))
    table(("Run", "Taught fraction", "Distinct taught pairs", "Rule", "Seed",
           "Never-taught add and sub at 12 letters, mean, by updates (items)", "80% on that panel"), rows)
    print("These runs have the panel at 12 letters only (300 or 400 never-taught pairs of add and of sub, reduction and "
          "no-reduction pairs together), not the reduction pairs apart. Receipts: `exp45/e37/e37_<F3|F1>_<S|B>_<seed>.json`, "
          "`exp45/e35a/e35a_dev_G_f<25|10>_73129.json`.\n")
    rows = []
    for a in "SB":
        for s in SEEDS:
            r = J(f"{M}/exp45/e36/e36_reference_{a}_{s}.json")
            assert r["complete"] and r["rows_planned"] == [0, 16, 32, 64, 128, 256, 512, 1024]
            ap = r["rows"][-1]["scores"]["birth_all_pairs_k0"]
            assert all(ap[n]["wrap"]["n"] + ap[n]["nowrap"]["n"] == 89 * 89 for n in ("add89", "sub89"))
            rows.append((a, s, f"{r['rows'][-1]['scores']['summary']['wrap_score']:.1f}", f"{r['rows'][-1]['scores']['summary']['held_add_sub_mean']:.1f}"))
    print("E36's references are fresh networks on the first family at a half for 1,024 updates (65,536 items), not on a "
          "later family; their reduction score is over all 7,921 pairs, taught ones included. At their last row "
          "(`exp45/e36/e36_reference_<S|B>_<seed>.json`):\n")
    table(("Rule", "Seed", "Reduction score at 1,024 updates", "Never-taught add and sub at 12 letters"), rows)
    return out


# ---------------------------------------------------------------- 5. the accounts
def breakeven(P0, c, F):
    """Smallest whole number n of new families with P0 + n * c < n * F."""
    if F <= c:
        return None
    return int(P0 // (F - c)) + 1


def accounts(single, new, before_items, e40, e41):
    print("## 7. The accounts\n")
    comp = 24000
    lv = [v["last"] for k, v in e40.items() if k[0] == "lived"]; fr = [v["last"] for k, v in e40.items() if k[0] == "fresh"]
    print(f"What the record states (E40's verdict) is account (a) at equal experience of the new family: with the same {new:,} "
          f"taught pairs and the same {8192 * BATCH:,} items, the nine lived runs are at {min(lv):.1f} to {max(lv):.1f}% on "
          f"never-taught reduction pairs and the six fresh runs at {min(fr):.1f} to {max(fr):.1f}%. The tables below restate it "
          "as the experience needed to reach a level, which is what \"needs fewer samples\" asks.\n")
    print("### 7.1 In distinct taught samples\n")
    q, h = e41[(25, "B", SEEDS[0])]["pairs"], e41[(50, "B", SEEDS[0])]["pairs"]
    q83 = 4 * round(.25 * 83 * 83); h83 = 4 * round(.50 * 83 * 83)
    rows = [
        ("(a) the new family alone", f"{new:,} pairs", f"{new:,} pairs; does not reach the level", f"{q:,} pairs", f"{h:,} pairs",
         f"{q / new:.1f} and {h / new:.1f} times fewer for the lived learner"),
        ("(b) the new family and everything before it", f"{single + new:,} pairs and {comp:,} compositions ({single + new + comp:,} samples)",
         "as (a)", f"{q:,} pairs", f"{h:,} pairs",
         f"{(single + new) / q:.1f} and {(single + new) / h:.1f} times more for the lived learner without the compositions; "
         f"{(single + new + comp) / q:.1f} and {(single + new + comp) / h:.1f} with them"),
    ]
    table(("Account", "Lived learner", "Fresh, a tenth of the new family (E40)", "Fresh, a quarter of the first family (E41)",
           "Fresh, a half of the first family (E41)", "Lived against fresh at a quarter and at a half"), rows)
    rows = []
    for label, P0 in (("pairs only", single), ("pairs and compositions", single + comp)):
        for fl, F in ((f"a quarter of modulus 89 ({q:,})", q), (f"a half of modulus 89 ({h:,})", h),
                      (f"a quarter of modulus 83 ({q83:,}; no run)", q83), (f"a half of modulus 83 ({h83:,}; no run)", h83)):
            n = breakeven(P0, new, F)
            rows.append((label, f"{P0:,}", f"{new:,}", fl, f"{P0 / (F - new):.2f}", n, n - 1))
    print("(c) If each further new family cost the lived learner the tenth that E40's did, and a fresh network is spent on "
          "each family: the lived learner's whole count P + n × c falls below the fresh networks' n × F at the smallest whole "
          "n above P / (F − c). n counts new families from the modulus-83 family on; the further ones after it are n − 1.\n")
    table(("What is counted before the new family", "P", "c (lived, each new family)", "F (fresh, each family)", "P / (F − c)",
           "n, new families in all", "Further families after E40's"), rows)

    print("### 7.2 In presented items, to 80% on never-taught reduction pairs\n")
    print("The lived learner's figure is the first row of its E40 run at or above 80%; the fresh figure is the first row of "
          "the E41 run of the same seed at or above 80% on R. Arm K is set beside the fresh sweep network, as in E40.\n")
    rows, rows_c, summ = [], [], {}
    for a in "SBK":
        fa = "S" if a == "K" else a
        for s in SEEDS:
            u = e40[("lived", a, s)]["th"][80]
            c = u * BATCH
            f10 = e40[("fresh", fa, s)]
            fq, fh = e41[(25, fa, s)]["th"][80], e41[(50, fa, s)]["th"][80]
            Fq = None if fq is None else fq * BATCH; Fh = fh * BATCH
            ratio = lambda F, x: "—" if F is None else f"{F / x:.0f}" if F >= x else f"1/{x / F:.1f}"
            rows.append((a, s, f"{c:,}", f"not reached in {8192 * BATCH:,} (highest {f10['best']:.1f}%)",
                         f"more than {E41_DOSE * BATCH:,}" if Fq is None else f"{Fq:,}", f"{Fh:,}",
                         ("more than " + f"{E41_DOSE * BATCH / c:.0f}") if Fq is None else ratio(Fq, c), ratio(Fh, c),
                         f"{before_items + c:,}",
                         ("fewer than " + f"{(before_items + c) / (E41_DOSE * BATCH):.1f}") if Fq is None else f"{(before_items + c) / Fq:.1f}",
                         f"{(before_items + c) / Fh:.1f}"))
            nq = breakeven(before_items, c, E41_DOSE * BATCH if Fq is None else Fq)
            nh = breakeven(before_items, c, Fh)
            summ.setdefault(a, []).append(dict(c=c, Fq=Fq, Fh=Fh, nq=nq, nh=nh))
            rows_c.append((a, s, f"{before_items:,}", f"{c:,}",
                           (f"more than {E41_DOSE * BATCH:,}" if Fq is None else f"{Fq:,}"),
                           (f"at most {nq} ({nq - 1})" if Fq is None else f"{nq} ({nq - 1})"), f"{Fh:,}", f"{nh} ({nh - 1})"))
    table(("Lived rule", "Seed", "(a) Lived: items of the new family", "Fresh, a tenth of the new family (E40)",
           "Fresh, a quarter of the first family (E41)", "Fresh, a half of the first family (E41)",
           "(a) fresh ÷ lived, a quarter", "(a) fresh ÷ lived, a half", "(b) Lived: birth, life and new family",
           "(b) lived ÷ fresh, a quarter", "(b) lived ÷ fresh, a half"), rows)
    print("(c) The same rule as in 7.1, in items: P is the birth and the life, c the lived learner's items to 80% on the new "
          "family, F the fresh network's items to 80% on the first family. Cells give n, the new families in all, with the "
          "further families after E40's in brackets.\n")
    table(("Lived rule", "Seed", "P", "c", "F, a quarter", "n (further), a quarter", "F, a half", "n (further), a half"), rows_c)
    dose = 8192 * BATCH
    reach = [v["th"][80] * BATCH for (fp, a, s), v in e41.items() if fp in (25, 50) and v["th"][80] is not None]
    print(f"c is the lived learner's items to the level, not the dose E40 gave. E40 gave {dose:,} items of the new family "
          f"to every run; {sum(v < dose for v in reach)} of the {len(reach)} fresh runs at a quarter or a half that reach 80% "
          "do so in fewer items than that, so with the dose in place of c the lived learner's whole count would not fall "
          "below theirs after any number of families.\n")
    for s in SEEDS:      # the sweep beside back-propagation, seed by seed
        assert e40[("lived", "S", s)]["th"][80] >= e40[("lived", "B", s)]["th"][80]
        for fp in (25, 50):
            fs, fb = e41[(fp, "S", s)]["th"][80], e41[(fp, "B", s)]["th"][80]
            assert fb is not None and (fs is None or fs > fb), (fp, s)
    print("The sweep beside back-propagation (checked here, seed by seed): the lived sweep run reaches 80% at the same row "
          "as the lived back-propagation run or a later one, and the fresh sweep run reaches it later than the fresh "
          "back-propagation run or not at all, at a quarter and at a half, in all three seeds. The taught pairs are the same "
          "under both rules.\n")
    print("### 7.3 The three seeds together, by rule\n")
    rows = []
    for a, v in summ.items():
        cs = [x["c"] for x in v]; fq = [x["Fq"] for x in v if x["Fq"] is not None]; fh = [x["Fh"] for x in v]
        miss = sum(x["Fq"] is None for x in v)
        qa = [x["Fq"] / x["c"] for x in v if x["Fq"] is not None]; ha = [x["Fh"] / x["c"] for x in v]
        qb = [(before_items + x["c"]) / x["Fq"] for x in v if x["Fq"] is not None]; hb = [(before_items + x["c"]) / x["Fh"] for x in v]
        rows.append((a, span(cs), span(fq) + (f"; {miss} of 3 not reached in {E41_DOSE * BATCH:,}" if miss else ""), span(fh),
                     f"{span(qa, '{:.0f}')}; {span(ha, '{:.0f}')}", span([before_items + x for x in cs]),
                     f"{span(qb, '{:.1f}')}; {span(hb, '{:.1f}')}",
                     f"{span([x['nq'] for x in v])}; {span([x['nh'] for x in v])}"))
    table(("Lived rule", "(a) Lived: items of the new family to 80%", "Fresh: items to 80%, a quarter of the first family",
           "Fresh: items to 80%, a half", "(a) fresh ÷ lived (a quarter; a half)", "(b) Lived: all items to 80%",
           "(b) lived ÷ fresh (a quarter; a half)", "(c) n, new families in all (a quarter; a half)"), rows)
    print("In the quarter columns the ratios and n leave out the seed whose fresh run did not reach 80%, except n, which "
          "uses the dose as the least the fresh run could have needed.\n")


def main():
    global REF
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=REF)
    ap.add_argument("--paths", action="store_true")
    a = ap.parse_args()
    REF = a.ref
    print("# E40 in the learner's whole experience: sections 4 to 7 and the appendix of the document (output of `e40_whole_experience.py`)\n")
    commit = git("rev-parse", REF).decode().strip()
    print(f"Every file is read from ref `{REF}`, commit `{commit}`.\n")
    source_rows = check_source()
    births, lives, tot, before_items = before_the_new_family()
    single, new = distinct_samples(lives, tot)
    e40 = new_family(lives)
    e41 = fresh_references(births)
    accounts(single, new, before_items, e40, e41)
    print("## Appendix. The rules, as the code states them\n")
    print(f"Each line below is in the named file of `{M}/` at commit `{commit}`; the script stops if one is missing.\n")
    table(("File", "Line", "Text"), source_rows)
    print(f"Files read with `git show {REF}:<path>`: {len(set(READ))}; every assertion passed.")
    if a.paths:
        for p in sorted(set(READ)):
            print(" ", p)


if __name__ == "__main__":
    sys.exit(main())
