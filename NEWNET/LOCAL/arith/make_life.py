"""English + addition life, design 2 (25 Nov 2026). Data only: writes a life, its control life, test sets and an exposure log.
Design 2 removes, in the data, the confounds of design 1 (stage tied to delay; size tied to lag and carry; exposure tied to stage;
a control that did not match the answer distribution; data amount tied to content; one held-out split).

Line:   {a in words} plus {b in words}{filler} {verb} {sum in words} - in equation {a}+{b}={s}
        verb: "is" | "equal to" (taught); filler: "" | ", we say," | ", as everyone knows," | ", as every child in school knows,"
        e.g. "twenty eight plus five, we say, equal to thirty three - in equation 28+5=33"
E2 only (never in the life): "{a} and {b}{filler} make {sum}". Numbers 0..1000, sums <= 1000, words like "three hundred forty seven".

Cells: (digits of the larger operand d = 1, 2, 3) x (carries in column addition c = 0, 1, 2+), the 8 that exist:
        d1c0 d1c1 d2c0 d2c1 d2c2 d3c0 d3c1 d3c2. Units are unordered pairs; each presentation draws the order.
Held out first (by the seed): HN = 10% of the numbers 20..999 (never in any arithmetic line of the life); then per cell 16 held-out
        units (8 in d1c0 and d1c1, which have only 30 and 25 pairs) and 16 taught units, all from numbers outside HN. The held-out sets do not depend on the number of cycles.
Balance: every arithmetic block shows every taught unit of every cell exactly once, so every cell has the same exposure (= number of
        cycles) and the same delays. Within each cell and block the 8 (verb, filler) combinations are used twice each and rotate
        across blocks, so the lag (bytes between the operands and the answer) varies inside every cell independently of the cell.
Life:   Eng(shak) then CYCLES x [Arith, Eng(emma/shak alternating)]. Arithmetic blocks carry 10% English lines (no review lines).
        Data scaling = CYCLES: the same pools and proportions, more presentations and more new English.
Control: the same lines in the same places; within each cell and block the sums are permuted among the cell's lines (no line keeps
        its own sum), so per cell the answer words and their frequencies are identical and only the operand -> sum relation is broken.
Tests (JSONL, all with byte spans word / word_first / digit, cell, lag, verb, filler, level):
        taught (E1) all 16 units per cell; e2 (E2) the taught units in the untaught template; heldpair (E3) 16 (or 8) units per cell;
        heldop (E3) 16 pairs per d2/d3 cell with an HN operand and a non-HN sum; heldsum (E3) 16 pairs per d2/d3 cell with an HN sum
        and non-HN operands. Verb x filler balanced within each test and cell.
        word = the addition answer; word_first = its first word; digit = the sum in digits (after the word sum: translation, not addition).
Usage: python3 make_life.py [SEED] [CYCLES] [OUTDIR]"""
import sys, os, json, random
from collections import Counter, defaultdict

D = "/Users/pencilbard/Downloads/digitalbaby_newnet/"
ENG = dict(shak=D + "PROJECT_DOCUMENTS/MINIFLY_OUTSIDE_ADVICE_20261118/nano/data/tinyshakespeare.txt",
           emma=D + "minifly_v155_keys_dev_v1_20261026/data/austen-emma.txt")
ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
VERBS = ["is", "equal to"]; FILLERS = ["", ", we say,", ", as everyone knows,", ", as every child in school knows,"]
COMBOS = [(v, f) for v in range(2) for f in range(4)]
CELLS = ["d1c0", "d1c1", "d2c0", "d2c1", "d2c2", "d3c0", "d3c1", "d3c2"]
NTAUGHT, NHELD, NHN = 16, 16, 16     # held-out units per cell: 16, or 8 where the cell is too small (d1c0 has 30 pairs, d1c1 25)
ENG_BLOCK, P_ENG_IN_ADD = 32000, 0.10


def words(n):
    if n == 1000: return "one thousand"
    if n < 20: return ONES[n]
    if n < 100: return TENS[n // 10] + ("" if n % 10 == 0 else " " + ONES[n % 10])
    return ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + words(n % 100))


def carries(a, b):
    c = k = 0
    while a or b or k:
        k = 1 if a % 10 + b % 10 + k >= 10 else 0; c += k; a //= 10; b //= 10
    return c


def cell(a, b):
    d = len(str(max(a, b))); return f"d{d}c{min(carries(a, b), 2)}"


def line(a, b, s, verb, fil, e2=False):
    pre = f"{words(a)} {'and' if e2 else 'plus'} {words(b)}{FILLERS[fil]} {'make' if e2 else VERBS[verb]} "
    t = pre + f"{words(s)} - in equation {a}+{b}={s}"
    i = len(pre); ws = words(s); j = t.rindex("=") + 1
    sp = dict(word=[i, i + len(ws)], word_first=[i, i + len(ws.split()[0])], digit=[j, len(t)])
    lag = dict(from_a=i - len(words(a)), from_b=i - (len(words(a)) + 1 + len("and" if e2 else "plus") + 1 + len(words(b))))
    assert t[sp["word"][0]:sp["word"][1]] == ws and t[j:] == str(s)
    return t, sp, lag


def derange(xs, rng):
    if len(set(xs)) < 2: return list(xs)
    while True:
        p = list(xs); rng.shuffle(p)
        if all(u != v for u, v in zip(p, xs)) or len(xs) < 2: return p


def main(seed=1, cycles=3, out=None):
    out = out or os.path.join(os.path.dirname(os.path.abspath(__file__)), f"data_seed{seed}_c{cycles}")
    os.makedirs(out, exist_ok=True)
    rng = random.Random(seed)
    HN = set(rng.sample(range(20, 1000), round(0.10 * 980)))
    allp = defaultdict(list)                                      # every unordered pair (a <= b, a + b <= 1000, b <= 999) by cell
    for a in range(1000):
        for b in range(a, min(999, 1000 - a) + 1):
            allp[cell(a, b)].append((a, b))
    clean = {c: [p for p in allp[c] if p[0] not in HN and p[1] not in HN and sum(p) not in HN] for c in CELLS}
    TP, HP = {}, {}
    for c in CELLS:
        ps = list(clean[c]); rng.shuffle(ps)
        nh = NHELD if len(ps) >= NTAUGHT + NHELD else 8
        assert len(ps) >= NTAUGHT + nh, (c, len(ps))
        HP[c] = ps[:nh]; TP[c] = ps[nh:nh + NTAUGHT]
    eng = {k: open(v, "rb").read().decode("latin-1").replace("\r", "") for k, v in ENG.items()}
    englines = {k: v.split("\n") for k, v in eng.items()}; limit = {k: len(eng[k]) // 2 for k in eng}
    epos = {k: 0 for k in eng}; ebyte = {k: 0 for k in eng}
    rl = random.Random(seed + 1000)

    def eng_line(k):
        t = englines[k][epos[k]]; epos[k] += 1; ebyte[k] += len(t) + 1
        assert ebyte[k] < limit[k], "English would reach the held-out middle stretch; use fewer cycles"
        return t

    life, ctrl, meta = [], [], []
    plan = [("eng", "shak")] + [x for k in range(cycles) for x in (("add", k), ("eng", "emma" if k % 2 == 0 else "shak"))]
    for blk, (kind, arg) in enumerate(plan):
        if kind == "eng":
            n = 0
            while n < ENG_BLOCK:
                t = eng_line(arg); life.append(t); ctrl.append(t); meta.append(("eng", blk, arg)); n += len(t) + 1
            continue
        k = arg; rows = []
        for c in CELLS:
            sums = [a + b for a, b in TP[c]]; csums = derange(sums, rl)
            for j, (a, b) in enumerate(TP[c]):
                v, f = COMBOS[(j + 3 * k) % 8]
                if (j + k) % 2: a, b = b, a
                rows.append((c, a, b, a + b, csums[j], v, f))
        rl.shuffle(rows)
        for c, a, b, s, sc, v, f in rows:
            if rl.random() < P_ENG_IN_ADD:
                src = "shak" if rl.random() < 0.5 else "emma"; t = eng_line(src); life.append(t); ctrl.append(t); meta.append(("eng", blk, src))
            life.append(line(a, b, s, v, f)[0]); ctrl.append(line(a, b, sc, v, f)[0]); meta.append(("add", blk, c, a, b, s, v, f))
    open(os.path.join(out, "life.txt"), "w").write("\n".join(life) + "\n")
    open(os.path.join(out, "control.txt"), "w").write("\n".join(ctrl) + "\n")
    rt = random.Random(seed + 2000); tests = {}

    def emit(name, c, units, level, e2=False):
        rows = []
        for j, (a, b) in enumerate(units):
            v, f = COMBOS[j % 8]
            if rt.random() < 0.5: a, b = b, a
            t, sp, lag = line(a, b, a + b, v, f, e2)
            rows.append(dict(text=t, spans=sp, lag=lag, a=a, b=b, s=a + b, cell=c, verb=VERBS[v] if not e2 else "make", filler=f, level=level))
        tests.setdefault(name, []).extend(rows)

    for c in CELLS:
        emit("taught", c, TP[c], "E1"); emit("e2", c, TP[c], "E2", e2=True); emit("heldpair", c, HP[c], "E3")
        if c.startswith("d1"): continue
        op = [p for p in allp[c] if (p[0] in HN or p[1] in HN) and sum(p) not in HN]; sm = [p for p in allp[c] if sum(p) in HN and p[0] not in HN and p[1] not in HN]
        rt.shuffle(op); rt.shuffle(sm)
        assert len(op) >= NHN and len(sm) >= NHN, (c, len(op), len(sm))
        emit("heldop", c, op[:NHN], "E3"); emit("heldsum", c, sm[:NHN], "E3")
    for name, rows in tests.items():
        with open(os.path.join(out, f"test_{name}.jsonl"), "w") as fh:
            for r in rows: fh.write(json.dumps(r) + "\n")
    # audit
    adds = [m for m in meta if m[0] == "add"]
    shown = Counter((min(m[3], m[4]), max(m[3], m[4])) for m in adds)
    nums = set(x for m in adds for x in (m[3], m[4], m[5]))
    life_set = set(life); ctrl_add = [ctrl[i] for i, m in enumerate(meta) if m[0] == "add"]
    per_cell = {c: dict(units_shown=len([u for u in TP[c] if u in shown]), times_each=sorted(set(shown[u] for u in TP[c])),
                        combos={f"{VERBS[v]}|{FILLERS[f]}": n for (v, f), n in sorted(Counter((m[6], m[7]) for m in adds if m[2] == c).items())}) for c in CELLS}
    real_sums = {c: Counter(m[5] for m in adds if m[2] == c) for c in CELLS}
    ctl_sums = {c: Counter() for c in CELLS}; fixed = 0
    for i, m in enumerate(meta):
        if m[0] != "add": continue
        sc = int(ctrl[i].rsplit("=", 1)[1]); ctl_sums[m[2]][sc] += 1; fixed += sc == m[5]
    checks = dict(held_numbers_in_life=sorted(nums & HN),
                  held_pairs_in_life=sum(u in shown for c in CELLS for u in HP[c]),
                  held_test_lines_in_life=sum(r["text"] in life_set for n in ("heldpair", "heldop", "heldsum") for r in tests.get(n, [])),
                  e2_lines_in_life=sum(" make " in t and " - in equation " in t for t in life),
                  control_sum_multiset_equal_per_cell=all(real_sums[c] == ctl_sums[c] for c in CELLS),
                  control_lines_keeping_true_sum=fixed,
                  exposure_equal_across_cells=len({tuple(v["times_each"]) for v in per_cell.values()}) == 1)
    lagv = defaultdict(list)
    for r in tests["taught"]: lagv[r["cell"]].append(r["lag"]["from_b"])
    log = dict(seed=seed, cycles=cycles, bytes_life=sum(len(t) + 1 for t in life), bytes_control=sum(len(t) + 1 for t in ctrl),
               add_lines=len(adds), eng_lines=len(meta) - len(adds), held_numbers=sorted(HN), per_cell=per_cell,
               lag_from_b_in_taught_tests={c: [min(v), max(v)] for c, v in lagv.items()},
               english_bytes_read=ebyte, english_limit=limit, checks=checks, tests={k: len(v) for k, v in tests.items()},
               blocks=[dict(block=b, kind=k, arg=a, lines=sum(1 for m in meta if m[1] == b)) for b, (k, a) in enumerate(plan)])
    json.dump(log, open(os.path.join(out, "exposure.json"), "w"), indent=1, default=str)
    print(json.dumps(dict((k, log[k]) for k in ("seed", "cycles", "bytes_life", "add_lines", "checks", "tests", "lag_from_b_in_taught_tests")), default=str))


if __name__ == "__main__":
    a = sys.argv[1:]
    main(int(a[0]) if a else 1, int(a[1]) if len(a) > 1 else 3, a[2] if len(a) > 2 else None)
