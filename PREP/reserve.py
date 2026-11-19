"""The reserve of never-taught instances of the first family, divided beforehand (roadmap, section 5, condition 4; section 6, steps 5 and 6a).

The reserve of a task is the second half of the record's split of its pairs (`split(task, .5, seed)[1]`, as in
`PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/COORDINATION_20261117/check_step1_brief.py`, itself the record's `exp11_cell.split`); the taught half is
the first half, which the stream of step 1 shows. For each seed and each of the four tasks (max and min, whose answer is one of the two
numbers; addition and subtraction modulo 89) this program

  1. puts every reserve pair into a group by which of its parts occur among the taught pairs of the same task:
       T  the two tens digits together: some taught pair (a', b') has a' // 10 == a // 10 and b' // 10 == b // 10
       U  the two units digits together: some taught pair has a' % 10 == a % 10 and b' % 10 == b % 10
       R  the reversed pair (b, a) is a taught pair (never for a == b, whose reverse is itself and is in the reserve)
     The group is the set of the three marks that hold, written for example "T U R" or "T U -".
  2. divides the reserve into three parts fixed before any stream is made: asked early (E), asked after a delay (D), asked after other
     learning (O). Within each task and group the pairs are shuffled by Random(f"reserve|{seed}|{task}|{group}") and dealt to E, D, O in
     turn, so that every group is spread over the three parts as evenly as its count allows.
  3. computes what two countermodels, which know only the taught pairs, score on every task, group and part:
       table      a table of the taught pairs and their answers; the key is (a, b), or {a, b} where the operation is symmetric (addition, max,
                  min), so that a reserve pair whose reverse was taught is answered from the table; a pair whose key is absent is answered by
                  the frequencies below
       frequency  the most frequent tens digit and the most frequent units digit among the taught answers of the task, position by position
     The score is the share of reserve pairs whose two-digit answer is exactly right; the digit scores are given beside it.

    python3 reserve.py                  prints the tables for seeds 73129, 93683, 116089 and writes reserve_<seed>.json (the division)
    python3 reserve.py --check          also asserts the counts of the record (3,960 taught and 3,961 reserve pairs a task)

Standard library and numpy only; nothing is trained and nothing reads a network.
"""
import json
import sys
from collections import Counter
from pathlib import Path
from random import Random

import numpy as np

HERE = Path(__file__).resolve().parent
SEEDS = (73129, 93683, 116089)
DEV = [("add", 89, b"p"), ("sub", 89, b"q"), ("max", 89, b"v"), ("min", 89, b"w")]
SYMMETRIC = {"add", "max", "min"}
PARTS = ("E", "D", "O")
PART_NAMES = {"E": "asked early", "D": "asked after a delay", "O": "asked after other learning"}
ORDER = ("max", "min", "add", "sub")


def apply(op, m, a, b):                     # check_step1_brief.py, line 45
    return {"add": (a + b) % m, "sub": (a - b) % m, "max": max(a, b), "min": min(a, b)}[op]


def split(task, frac, seed):                # the record's split (exp11_cell.split), check_step1_brief.py, line 56
    op, m, code = task
    pairs = [(a, b) for a in range(m) for b in range(m)]
    rng = Random(seed * 1000 + round(frac * 100) + m * 7 + code[0]); rng.shuffle(pairs)
    k = round(frac * len(pairs))
    return pairs[:k], pairs[k:]


def group_of(a, b, tens, units, taught):
    t = "T" if (a // 10, b // 10) in tens else "-"
    u = "U" if (a % 10, b % 10) in units else "-"
    r = "R" if (b, a) in taught and a != b else "-"
    return f"{t} {u} {r}"


def reduction(op, m, a, b):
    return (op == "add" and a + b >= m) or (op == "sub" and a - b < 0)


def build(seed):
    out = {}
    for task in DEV:
        op, m, code = task
        taught, reserve = split(task, .5, seed)
        tset = set(taught)
        tens = Counter((a // 10, b // 10) for a, b in taught)
        units = Counter((a % 10, b % 10) for a, b in taught)
        groups = {}
        for a, b in reserve:
            groups.setdefault(group_of(a, b, tens, units, tset), []).append((a, b))
        part = {}
        for g in sorted(groups):
            ps = sorted(groups[g]); Random(f"reserve|{seed}|{op}|{g}").shuffle(ps)
            for i, pr in enumerate(ps):
                part[pr] = PARTS[i % 3]
        # countermodels
        ans = {pr: apply(op, m, *pr) for pr in taught}
        key = (lambda a, b: (min(a, b), max(a, b))) if op in SYMMETRIC else (lambda a, b: (a, b))
        table = {key(a, b): v for (a, b), v in ans.items()}
        tens_digit = Counter(v // 10 for v in ans.values()).most_common(1)[0][0]
        units_digit = Counter(v % 10 for v in ans.values()).most_common(1)[0][0]
        freq_answer = 10 * tens_digit + units_digit
        rows = []
        for a, b in reserve:
            right = apply(op, m, a, b)
            k = key(a, b)
            in_table = k in table
            guess = table[k] if in_table else freq_answer
            rows.append(dict(a=a, b=b, answer=right, group=group_of(a, b, tens, units, tset), part=part[(a, b)], reduction=reduction(op, m, a, b),
                             in_table=in_table, table=guess == right, table_tens=guess // 10 == right // 10, table_units=guess % 10 == right % 10,
                             freq=freq_answer == right, freq_tens=tens_digit == right // 10, freq_units=units_digit == right % 10))
        out[op] = dict(task=task, taught=len(taught), reserve=len(reserve), rows=rows, freq_answer=freq_answer,
                       tens_support=(min(tens.values()), int(np.median(list(tens.values()))), max(tens.values()), len(tens)),
                       units_support=(min(units.values()), int(np.median(list(units.values()))), max(units.values()), len(units)))
    return out


def pct(n, d):
    return f"{100 * n / d:5.1f}" if d else "    -"


def report(seed, data, check=False):
    lines = [f"=== seed {seed} ==="]
    for op in ORDER:
        d = data[op]; rows = d["rows"]
        if check:
            assert d["taught"] == 3960 and d["reserve"] == 3961, (op, d["taught"], d["reserve"])
        lines.append(f"-- {op} (code {d['task'][2].decode()}): {d['taught']} taught, {d['reserve']} in the reserve; the frequency answer is {d['freq_answer']:02d}; "
                     f"taught pairs sharing a tens pair: min {d['tens_support'][0]}, median {d['tens_support'][1]}, max {d['tens_support'][2]} over {d['tens_support'][3]} tens pairs; "
                     f"sharing a units pair: min {d['units_support'][0]}, median {d['units_support'][1]}, max {d['units_support'][2]} over {d['units_support'][3]} units pairs")
        diag = [r for r in rows if r["a"] == r["b"]]
        lines.append(f"   pairs with a == b in the reserve (their reverse is themselves, so they are never R): {len(diag)}, all in group "
                     f"{sorted({r['group'] for r in diag})}; table {pct(sum(r['table'] for r in diag), len(diag)).strip()}%")
        hdr = f"   {'group':7} {'part':4} {'count':>6} {'reduct':>6} {'in tbl':>6} {'table%':>7} {'tens%':>6} {'units%':>6} {'freq%':>6} {'tens%':>6} {'units%':>6}"
        lines.append(hdr)
        groups = sorted({r["group"] for r in rows})
        for g in groups + ["all"]:
            for p in PARTS + ("all",):
                sel = [r for r in rows if (g == "all" or r["group"] == g) and (p == "all" or r["part"] == p)]
                n = len(sel)
                if n == 0:
                    continue
                s = lambda k: sum(r[k] for r in sel)
                lines.append(f"   {g:7} {p:4} {n:6d} {s('reduction'):6d} {s('in_table'):6d} {pct(s('table'), n):>7} {pct(s('table_tens'), n):>6} {pct(s('table_units'), n):>6} "
                             f"{pct(s('freq'), n):>6} {pct(s('freq_tens'), n):>6} {pct(s('freq_units'), n):>6}")
        if op in ("add", "sub"):
            red = [r for r in rows if r["reduction"]]
            s = lambda k: sum(r[k] for r in red)
            lines.append(f"   reduction pairs of the reserve: {len(red)}; table {pct(s('table'), len(red)).strip()}%, frequency {pct(s('freq'), len(red)).strip()}%")
    return lines


def main(argv):
    check = "--check" in argv
    allout = ["The reserve of never-taught instances of the first family: groups, parts and the scores of two countermodels.",
              "Columns: group (T: the tens digits of the pair occur together in a taught pair; U: the units digits; R: the reversed pair is taught; - where not),",
              "part (E asked early, D asked after a delay, O asked after other learning), count, reduct (pairs needing the modulus), in tbl (key in the table),",
              "table% and its tens% and units% (the table of taught pairs, falling back on the frequencies), freq% and its tens% and units% (the frequencies alone).", ""]
    for seed in SEEDS:
        data = build(seed)
        allout += report(seed, data, check) + [""]
        division = {op: {p: sorted([[r["a"], r["b"]] for r in data[op]["rows"] if r["part"] == p]) for p in PARTS} for op in ORDER}
        groups = {op: {f"{r['a']},{r['b']}": r["group"] for r in data[op]["rows"]} for op in ORDER}
        (HERE / f"reserve_{seed}.json").write_text(json.dumps(dict(seed=seed, source="split(task, .5, seed)[1] of check_step1_brief.py", parts=PART_NAMES,
                                                                   division=division, groups=groups), indent=0))
    text = "\n".join(allout)
    print(text)
    (HERE / "reserve_output.txt").write_text(text + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
