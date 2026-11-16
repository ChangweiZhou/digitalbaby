"""Checks behind the audit of the E41 brief at 213c38f4 (16 November 2026).

Read-only: receipts, saved tensors, and a pure-Python rebuild of the splits and panels.  Nothing trains and no harness code is run.

A. Parameter counts of the three widths from the layer shapes, and their ratios.
B. The brief's table "What is on file": held-out add and sub at 12 letters (mean of the two) by row, from the E26 birth receipts
   (half the pairs taught, both rules, three seeds) and E35-A's arm G (a quarter and a tenth, backpropagation, seed 73129); the first
   row at or above 80 and 90, and whether a later row falls back below.
C. How often a first crossing is reversed at finer spacing: the same question on E36's nine reminders and on the new family in E38's
   ten lives, for thresholds 80 and 90, beside the rule "three consecutive scored rows".
D. The splits at the three taught fractions: generator seeds, taught and held-out counts, held-out reduction pairs per task, and the
   sample of 1,000 as the audit words it.
E. The reviewer's reader for the brief's three scores (A, R, taught) on files that already exist: the six E26 births (f = 0.50) and
   E35-A's arm G at row 24,576 (f = 0.25 and 0.10, seed 73129).  This is the table for the added check.
F. Batch digests on file that E41's runs can be compared with: E26 at rows 4,096 and 8,192 (both rules), the dry births at 2,048,
   and E35-A's arm G.
G. Sampling error of A and R near 80, what the doses can and cannot decide, and the cost forecast.

Run from the repository root:  python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e41_brief_checks.py [A B C D E F G]
"""
import hashlib, json, math, os, sys
from random import Random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from np_gru import git_bytes, load_weights, forward, split, panel, render, exact_teacher_forced, DEV, BOS, P

REF = '213c38f4'; SEEDS = (73129, 93683, 116089); FRACS = (0.10, 0.25, 0.50)
E35A = 'origin/claude/nice-rubin-y5de8k'; E38 = 'origin/claude/peaceful-hopper-8ofq4i'
want = {a for a in sys.argv[1:] if len(a) == 1} or set("ABCDEFG")
js = lambda path, ref=REF: json.loads(git_bytes(P + path, ref))
pc = lambda d: 100 * d['exact'] / d['n']

def counts(H, U):
    book = 258 * 8; l1 = 8 * H + H; gru = 3 * U * H + 3 * U * U + 6 * U; lm = U * H + H; l2 = H * H + H; l3 = 257 * H + 257
    return dict(book=book, l1=l1, rnn=gru, lm=lm, l2=l2, l3=l3, total=book + l1 + gru + lm + l2 + l3)

def crossing(series, thr, consecutive=1):
    """series: list of (row, value). First row from which `consecutive` scored rows in a row are at or above thr; None if none."""
    for i in range(len(series) - consecutive + 1):
        if all(v >= thr for _, v in series[i:i + consecutive]):
            return series[i][0]
    return None

def reversed_after(series, thr):
    first = crossing(series, thr)
    return first is not None and any(v < thr for r, v in series if r > first)

if "A" in want:
    print("A. parameter counts (feed-forward H, GRU units U)")
    tot = {}
    for name, (H, U) in (("half", (256, 128)), ("present", (512, 256)), ("double", (1024, 512))):
        c = counts(H, U); tot[name] = c['total']
        print(f"   {name:8s} H={H:5d} U={U:4d}: " + ", ".join(f"{k} {v:,}" for k, v in c.items()) + f"; without the embedding table {c['total'] - c['book']:,}")
    print(f"   ratios: present / half {tot['present'] / tot['half']:.3f}; double / present {tot['double'] / tot['present']:.3f}")

def held_mean(sealed):
    return (pc(sealed['add89']['12']) + pc(sealed['sub89']['12'])) / 2

if "B" in want:
    print("B. held-out add and sub at 12 letters, mean of the two, by row (present width)")
    runs = [(f"f=0.50 {'sweep' if a == 'S' else 'backprop'} {s}", js(f'e26/e26_dev_{a}_{s}.json')) for a in 'SB' for s in SEEDS]
    runs += [(f"f={f} backprop 73129 (E35-A arm G)", js(f'e35a/e35a_dev_G_f{t}_73129.json', E35A)) for f, t in ((0.25, '25'), (0.10, '10'))]
    for name, d in runs:
        ser = [(c['step'], held_mean(c['sealed'])) for c in d['curve']]
        txt = ", ".join(f"{r}: {v:.1f}" for r, v in ser)
        print(f"   {name:36s} {txt} | first row >= 80: {crossing(ser, 80)}; >= 90: {crossing(ser, 90)}; falls back below 80 / 90 later: {reversed_after(ser, 80)} / {reversed_after(ser, 90)}")

if "C" in want:
    print("C. reversals of a first crossing at finer spacing, and the three-consecutive rule")
    series = []
    for arm in 'SBK':
        for s in SEEDS:
            d = js(f'e36/e36_reminder_{arm}_{s}.json')
            series.append((f"E36 reminder {arm} {s}", [(r['updates'], (pc(r['scores']['birth_held_k12']['add89']) + pc(r['scores']['birth_held_k12']['sub89'])) / 2) for r in d['rows']]))
    for arm in ('zero', 'none', 'anchor', 'anchor10', 'rehearse'):
        for rule in 'SB':
            d = js(f'e38/e38_{arm}_{rule}_73129.json', E38)
            series.append((f"E38 {arm} {rule}, new family", [(r['updates'], (pc(r['new_heldout_k12']['add97']) + pc(r['new_heldout_k12']['sub97'])) / 2) for r in d['rows']]))
    for thr in (80, 90):
        reach = [(n, s) for n, s in series if crossing(s, thr) is not None]
        rev = [n for n, s in reach if reversed_after(s, thr)]
        later = [n for n, s in reach if crossing(s, thr, 3) != crossing(s, thr)]
        print(f"   threshold {thr}: {len(reach)} of {len(series)} series reach it; a later scored row falls back below in {len(rev)}; the three-consecutive rule gives a later row (or none) in {len(later)}")
        for n in rev:
            s = dict(series)[n]
            print(f"      {n}: " + ", ".join(f"{r}: {v:.1f}" for r, v in s) + f" | single {crossing(s, thr)}, three in a row {crossing(s, thr, 3)}")

def reduction(task, a, b):
    op, m, _ = task
    return (a + b >= m) if op == "add" else (a - b < 0)

def sample_r(task, held, seed, f):
    """The audit's wording of the brief's sample: the fraction enters the generator's seed as round(f * 100)."""
    pairs = [p for p in held if reduction(task, *p)]
    return Random(f"e41|{seed}|{round(f * 100)}|{task[0]}").sample(sorted(pairs), 1000), len(pairs)

if "D" in want:
    print("D. splits of the birth family at the three taught fractions")
    for f in FRACS:
        for s in SEEDS:
            row = []
            for t in DEV:
                taught, held = split(t, f, s)
                gen = s * 1000 + round(f * 100) + t[1] * 7 + t[2][0]
                extra = ""
                if t[0] in ("add", "sub"):
                    smp, n = sample_r(t, held, s, f)
                    assert len(set(smp)) == 1000 and not set(smp) & set(taught)
                    extra = f", held-out reduction pairs {n}"
                row.append(f"{t[0]} taught {len(taught)} held {len(held)} (generator {gen - s * 1000}){extra}")
            print(f"   f={f:.2f} seed {s}: " + "; ".join(row))
    print("   sha256 of json.dumps([list(p) for p in sample]), first 16 hex digits, add then sub:")
    for s in SEEDS:
        print(f"   | {s} | " + " | ".join(", ".join(hashlib.sha256(json.dumps([list(p) for p in sample_r(t, split(t, f, s)[1], s, f)[0]]).encode()).hexdigest()[:16]
                                                     for t in DEV[:2]) for f in FRACS) + " |")
    a = sample_r(DEV[0], split(DEV[0], 0.5, 73129)[1], 73129, 0.5)[0][:4]
    print(f"   first four of the add sample at f = 0.50, seed 73129: {a}; f-string forms of the three fractions: {[f'{f}' for f in FRACS]} (0.10 prints as '0.1', 0.50 as '0.5')")

def count(W, items):
    x = np.array([[BOS] + p + a[:-1] for p, a in items], dtype=np.int64); tgt = np.array([a for _, a in items], dtype=np.int64)
    return int(exact_teacher_forced(forward(W, x)[3], tgt).sum())

def scores(W, seed, f):
    """Whole answers right, add then sub: A on 300 held-out pairs at 12 letters, R on 1,000 held-out reduction pairs without letters,
    taught on 300 taught pairs at 12 letters (the generator of seed + 1)."""
    out = {}
    for t in DEV[:2]:
        taught, held = split(t, f, seed)
        out['A', t[0]] = count(W, panel(t, held, seed, (12,), 300)[12])
        out['R', t[0]] = count(W, [render(t, a, b, 0, None) for a, b in sample_r(t, held, seed, f)[0]])
        out['taught', t[0]] = count(W, panel(t, taught, seed + 1, (12,), 300)[12])
        out['E26 panel', t[0]] = count(W, panel(t, held, seed, (0, 4, 8, 12, 20), 300)[12])
    return out

if "E" in want:
    print("E. the reviewer's reader on existing files: whole answers right for add, sub; A of 300 each, R of 1,000 each, taught of 300 each")
    files = [(f"e26/e26_dev_{a}_{s}.pt", REF, s, 0.50) for a in 'SB' for s in SEEDS]
    files += [(f"e35a/e35a_dev_G_f{t}_73129_row24576.pt", E35A, 73129, f) for f, t in ((0.25, '25'), (0.10, '10'))]
    for path, ref, s, f in files:
        r = scores(load_weights(P + path, ref)[0], s, f)
        last = js(path.replace('_row24576', '').replace('.pt', '.json'), ref)['curve'][-1]; assert last['step'] == 24576
        if ref == REF:      # E26's own panel (five letter counts, so other letters at 12 than E41's panel): the reader against the receipt
            rc = [last['sealed'][n]['12']['exact'] for n in ('add89', 'sub89')]; mine = [r['E26 panel', 'add'], r['E26 panel', 'sub']]; what = "E26's sealed panel at 12 letters"
        else:               # E35-A's taught panel is the call E41 prescribes: the reader's taught counts against the receipt
            rc = [last['taught'][n]['12']['exact'] for n in ('add89', 'sub89')]; mine = [r['taught', 'add'], r['taught', 'sub']]; what = "E35-A's taught panel"
        assert rc == mine, (path, rc, mine)
        mean = lambda k, n: 100 * (r[k, 'add'] + r[k, 'sub']) / (2 * n)
        print(f"   | `{path}` | {f:.2f} | {r['A', 'add']}, {r['A', 'sub']} | {mean('A', 300):.2f} | {r['R', 'add']}, {r['R', 'sub']} | {mean('R', 1000):.2f} | {r['taught', 'add']}, {r['taught', 'sub']} | {mean('taught', 300):.2f} |"
              f"   (reader equals the receipt on {what}: {mine})", flush=True)

if "F" in want:
    print("F. batch digests on file")
    for s in SEEDS:
        d = {a: js(f'e26/e26_dev_{a}_{s}.json') for a in 'SB'}
        for row in (0, 1):
            dg = {a: d[a]['curve'][row]['data_digest'] for a in 'SB'}
            print(f"   E26 f=0.50 seed {s}, row {d['S']['curve'][row]['step']}: {dg['S']['batches']} batches, sweep {dg['S']['sha256'][:16]}, backprop {dg['B']['sha256'][:16]}, equal {dg['S'] == dg['B']}")
    for a in 'SB':
        d = js(f'e26_qual/e26_dev_{a}_dry2048_73129.json')
        print(f"   E26 dry birth {a} 73129: rows {[(c['step'], c['data_digest']['sha256'][:16]) for c in d['curve']]}; keys with a hash: {[k for k in d if 'sha' in k]}")
    for t in ('25', '10'):
        d = js(f'e35a/e35a_dev_G_f{t}_73129.json', E35A)
        print(f"   E35-A arm G f=0.{t} 73129: " + ", ".join(f"row {c['step']} {c['data_digest']['sha256'][:16]}" for c in d['curve'][:2]) + f"; CPU-seconds of updates to 4,096: {d['curve'][0]['cpu_updates_so_far']:.0f}")

if "G" in want:
    print("G. sampling error, doses and cost")
    for name, n in (("A (2 x 300 items)", 600), ("R (2 x 1,000 items)", 2000)):
        se = 100 * math.sqrt(.8 * .2 / n)
        print(f"   {name}: standard error at 80% = {se:.2f} points; 95% interval +/- {1.96 * se:.1f}")
    PAR = dict(half=317457, present=1124113, double=4211985)
    print("   80% on A at the present width, by straight line between the rows on file (4,096 apart):")
    for name, path, ref in [(f"f=0.50 {a} {s}", f'e26/e26_dev_{a}_{s}.json', REF) for a in 'SB' for s in SEEDS] + [("f=0.25 B 73129", 'e35a/e35a_dev_G_f25_73129.json', E35A)]:
        ser = [(0, 0.0)] + [(c['step'], held_mean(c['sealed'])) for c in js(path, ref)['curve']]
        i = next(i for i, (_, v) in enumerate(ser) if v >= 80); (r0, v0), (r1, v1) = ser[i - 1], ser[i]
        print(f"      {name}: {r0 + (80 - v0) / (v1 - v0) * (r1 - r0):,.0f}")
    doses = {"as briefed": dict(half=16384, present=8192, double=8192), "as the audit asks": dict(half=40960, present=12288, double=8192)}
    upd = {w: 453.4 / 4096 * PAR[w] / PAR['present'] for w in PAR}            # E35-A arm G: 453.4 CPU-s for 4,096 updates at the present width
    unit = 13.2 / (1200 * sum(9.5 + k for k in (0, 4, 8, 12, 20, 40)) + 1200 * 21.5)   # E35-A: 13.2 CPU-s per scored row; an item costs about (9.5 + letters)
    row = {w: unit * (2400 * 21.5 + 2000 * 9.5) * PAR[w] / PAR['present'] for w in PAR}
    print("   CPU-seconds per update: " + ", ".join(f"{w} {v:.3f}" for w, v in upd.items()) + "; per scored row of E41 (4,400 items): " + ", ".join(f"{w} {v:.1f}" for w, v in row.items()))
    for label, d in doses.items():
        print(f"   doses {label}:")
        for w in PAR:
            print(f"      {w}: {d[w]:,} updates, {d[w] // 256} scored rows, {d[w] * PAR[w]:.3e} weight-updates (x 64 for weight-items); a crossing of three rows can begin no later than {d[w] - 512:,}")
        for n, w in (("half", "present"), ("present", "double")):
            k = PAR[w] / PAR[n]
            print(f"      {n} -> {w}, the narrower not reaching within its dose: 'falls' is decided in updates if the wider reaches by {d[n] / 1.5:,.0f}, in weight-items if by {d[n] / (1.5 * k):,.0f}"
                  f" (three-row version: {(d[n] - 512) / 1.5:,.0f} and {(d[n] - 512) / (1.5 * k):,.0f})")
        u = sum(d[w] * upd[w] for w in PAR); sc = sum(d[w] // 256 * row[w] for w in PAR)
        print(f"      cost: per fraction, rule and seed {u:,.0f} CPU-s of updates and {sc:,.0f} of scoring; per worker (6 such) {6 * (u + sc) / 3600:.1f} CPU-hours; all 54 runs {18 * u / 3600:.1f} + {18 * sc / 3600:.1f} = {18 * (u + sc) / 3600:.1f};"
              f" the double width's share of the updates {d['double'] * upd['double'] / u:.0%}")
    print("   final weights, float32: " + ", ".join(f"{w} {PAR[w] * 4 / 1e6:.1f} MB" for w in PAR) + f"; 18 files of each: {18 * sum(PAR.values()) * 4 / 1e6:.0f} MB")
