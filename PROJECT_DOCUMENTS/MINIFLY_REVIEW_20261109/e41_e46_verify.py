"""Verification of E41 on three seeds and of E46 (17 November 2026).  Read-only.

E41: 54 births from random weights (rules S = the sweep and B = back-propagation; taught fractions 0.10, 0.25, 0.50; widths
half (H 256, U 128), present (512, 256) and double (1,024, 512); seeds 73129, 93683, 116089).  E46: 18 births at the present
width and half of the pairs taught, with weight decay 0.25, 0.05 and 0.  Nothing is trained and no code of another branch is
executed: receipts and weight files are read with `git show` at fixed commits, the scores are recomputed from the receipts'
counts, and the weight files are scored by the numpy reader `np_gru.py`.

  A  E41: the 54 receipts (complete, dose, runner hash, the default branch's copies); updates to 80 and 90 on A and on R,
     three-consecutive-rows version, with the three-seed median and range; single-row version where it differs
  B  multiply-adds per item and position at the three widths, counted from the layer shapes; computation to 80 in forward
     multiply-adds and in weight-items; the cheapest width per rule, fraction and seed
  C  the four-case description of every width step (updates and weight-items; A and R; thresholds 50, 80, 90; both versions),
     compared cell by cell with the worker's report E41_REPORT.md and run by run with E41_SUMMARY_all.json
  D  statements: the sweep against back-propagation; a tenth of the pairs; R against A
  E  the births named as learners for the long lives: committed files, file hashes against the receipts, and the numpy
     reader's counts on A, R and the taught panel against the receipts' last rows
  F  E46 from its 18 receipts: decay recorded, thresholds, levels, norms, the three cells and the sentence per rule;
     the reference runs against E41's and E26's
  G  E46 weight files through the numpy reader: last-row counts (A, R, taught, length test) and the norms of the seven groups
  H  E46: how the weights grow without decay (norms by row), and the sizes of the differences the 15-point band hides

Run from the repository root:  python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e41_e46_verify.py [A B C D E F G H]
"""
import hashlib, json, os, re, statistics, subprocess, sys, time
from random import Random
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from np_gru import git_bytes, load_weights, forward, split, panel, render, exact_teacher_forced, DEV, BOS

M = 'MOD97_LOCAL_RULE_20261106/'
DREF = '7cabb13a'                                    # origin/codex/minifly-consolidated-20261023 on 17 November, 19:06 UTC
SEED_REF = {73129: '2722d1c1', 93683: 'd4d95951', 116089: '79ea3575'}     # the three workers' branches at the commits the report names
REPORT_REF = '2722d1c1'                              # claude/wonderful-edison-8hx6i6: the three-seed report
E46REF = 'b76f5979'                                  # the same branch at its tip: E46 complete
SEEDS = (73129, 93683, 116089); FR = (10, 25, 50); HS = (256, 512, 1024); RULES = 'SB'
U_OF = {256: 128, 512: 256, 1024: 512}
PARAMS = {256: 317457, 512: 1124113, 1024: 4211985}
DOSE = {256: 40960, 512: 12288, 1024: 8192}
RUNNER = '559a7ee9978639d7'
want = {a for a in sys.argv[1:] if len(a) == 1} or set("ABCDEFGH")
T0 = time.time()


def js(path, ref):
    return json.loads(git_bytes(path, ref))


def blob(ref, path):
    r = subprocess.run(['git', 'rev-parse', f'{ref}:{path}'], capture_output=True, text=True, cwd=HERE)
    return r.stdout.strip() if r.returncode == 0 else None


def crossing(rows, key, thr, consecutive):
    """First scored row from which `consecutive` scored rows in a row are at or above thr; None if there is none."""
    for i in range(len(rows) - consecutive + 1):
        if all(r[key] >= thr for r in rows[i:i + consecutive]):
            return rows[i]['step']
    return None


def describe(xn, xw, Dn, Dw, kn=1.0, kw=1.0):
    """E41's table of four cases.  xn, xw: the narrower and the wider run's updates to threshold (None = not reached);
    Dn, Dw: the largest count at which each could have been recorded; kn, kw: factors for the unit (1 for updates)."""
    if xn is not None and xw is not None:
        q = (xw * kw) / (xn * kn)
        return 'falls' if q < 2 / 3 else 'rises' if q > 1.5 else 'stays'
    if xn is not None:
        return 'rises' if (Dw * kw) / (xn * kn) >= 1.5 else 'not decided'
    if xw is not None:
        return 'falls' if (xw * kw) / (Dn * kn) <= 2 / 3 else 'not decided'
    return 'neither reaches'


def cell(descs):
    for d in set(descs):
        if descs.count(d) >= 2:
            return d
    return 'mixed'


def fmt_u(v):
    return 'n.r.' if v is None else f"{v:,}"


def med_range(vals, f=lambda v: f"{v:,}"):
    """Median and range of three values; a run that does not reach the threshold is written n.r. and listed."""
    if all(v is not None for v in vals):
        return f"{f(sorted(vals)[1])} ({f(min(vals))}-{f(max(vals))})"
    return "; ".join('n.r.' if v is None else f(v) for v in vals)


def madds(H, U):
    """Multiply-adds of one forward pass per item and position: every matrix entry is used once."""
    parts = dict(l1=8 * H, input_map=3 * U * H, recurrent=3 * U * U, lm=U * H, l2=H * H, l3=257 * H)
    return parts, sum(parts.values())


# ------------------------------------------------------------------------------------------------ E41 receipts
RUNS = {}
if want & set("ABCDE"):
    for seed in SEEDS:
        for rule in RULES:
            for H in HS:
                for f in FR:
                    path = f"{M}exp45/e41/e41_{rule}_h{H}_f{f}_{seed}.json"
                    d = js(path, SEED_REF[seed])
                    rows = d['curve']
                    for r in rows:      # the scores are recomputed from the counts
                        a = 100 * (r['A_counts']['add89'] + r['A_counts']['sub89']) / 600
                        rr = 100 * (r['R_counts']['add89'] + r['R_counts']['sub89']) / 2000
                        tt = 100 * (r['taught_counts']['add89'] + r['taught_counts']['sub89']) / 600
                        assert abs(a - r['A']) < 1e-9 and abs(rr - r['R']) < 1e-9 and abs(tt - r['taught']) < 1e-9, path
                    assert [r['step'] for r in rows] == list(range(0, DOSE[H] + 1, 256)), path
                    RUNS[rule, H, f, seed] = dict(d=d, rows=rows, path=path)
    THR = {}
    for key, R in RUNS.items():
        for m in 'AR':
            for thr in (50, 80, 90):
                THR[key + (m, thr, 3)] = crossing(R['rows'], m, thr, 3)
                THR[key + (m, thr, 1)] = crossing(R['rows'], m, thr, 1)

if "A" in want:
    print("A. E41: the 54 receipts")
    bad = []
    for (rule, H, f, seed), R in RUNS.items():
        d = R['d']
        ok = (d['complete'] is True and d['dose'] == DOSE[H] and d['H'] == H and d['U'] == U_OF[H] and d['seed'] == seed and d['rule'] == rule
              and round(d['fraction'] * 100) == f and d['model']['params'] == PARAMS[H] and d['provenance']['runner_sha256'].startswith(RUNNER)
              and d['optimizer']['weight_decay'] == 0.25 and d['optimizer']['lr'] == 0.001 and d['batch'] == 64 and d['kmax'] == 12
              and d['optimizer_steps'] == [DOSE[H]] and all(r['finite_losses'] for r in R['rows']))
        if not ok:
            bad.append((rule, H, f, seed))
    print(f"   {len(RUNS)} receipts read at the workers' commits {SEED_REF}; complete, at the full dose, rows every 256 updates, runner {RUNNER}..., "
          f"decay 0.25, rate 0.001, batch 64, losses finite: {'all' if not bad else 'NOT ' + str(bad)}")
    same = sum(blob(SEED_REF[s], R['path']) == blob(DREF, R['path']) for (_, _, _, s), R in RUNS.items())
    print(f"   the same files on the default branch at {DREF}: {same} of {len(RUNS)} are identical blobs")
    dg = {}
    for (rule, H, f, seed), R in RUNS.items():
        dg.setdefault((seed, f), set()).add(R['rows'][16]['data_digest']['sha256'])
    print(f"   batch digest at row 4,096 equal across the six runs of a seed and fraction: {all(len(v) == 1 for v in dg.values())} ({len(dg)} groups)")
    print("   CPU-hours of the complete runs (updates + scoring), per seed: " + "; ".join(
        f"{s}: {sum(R['d']['cpu_updates'] for k, R in RUNS.items() if k[3] == s) / 3600:.2f} + {sum(R['d']['cpu_scoring'] for k, R in RUNS.items() if k[3] == s) / 3600:.2f}" for s in SEEDS))
    for m in 'AR':
        for thr in (80, 90):
            print(f"   updates to {thr} on {m}, three-row version: median (range) of the three seeds; n.r. = not reached, then the seeds in the order {SEEDS}")
            print("      rule  f     half (H 256)                     present (H 512)                  double (H 1,024)")
            for rule in RULES:
                for f in FR:
                    print(f"      {rule}     0.{f:02d}  " + "  ".join(f"{med_range([THR[rule, H, f, s, m, thr, 3] for s in SEEDS]):31s}" for H in HS))
    diff = [(k[:4], k[4], k[5], THR[k[:6] + (1,)], THR[k]) for k in THR if k[6] == 3 and THR[k] != THR[k[:6] + (1,)]]
    print(f"   the single-row version gives another row than the three-row version in {len(diff)} of {len(THR) // 2} (run, score, threshold) cases")
    fs = [100 * (1 - (v3 is None and v1 is not None)) for *_, v1, v3 in diff]
    lost = sum(1 for *_, v1, v3 in diff if v3 is None)
    print(f"      of these, the three-row version records no crossing at all in {lost}; where both record one, the three-row row is later by a median of "
          f"{statistics.median([v3 - v1 for *_, v1, v3 in diff if v3 is not None]):,.0f} updates (largest {max(v3 - v1 for *_, v1, v3 in diff if v3 is not None):,})")

if "B" in want:
    print("B. computation")
    for H in HS:
        parts, tot = madds(H, U_OF[H])
        print(f"   H {H:5d}, U {U_OF[H]:4d}: forward multiply-adds per item and position {tot:,} ({', '.join(f'{k} {v:,}' for k, v in parts.items())}); parameters {PARAMS[H]:,}")
    mp = {H: madds(H, U_OF[H])[1] for H in HS}
    print(f"   ratios half : present : double, multiply-adds {mp[256] / mp[512]:.4f} : 1 : {mp[1024] / mp[512]:.4f}; parameters {PARAMS[256] / PARAMS[512]:.4f} : 1 : {PARAMS[1024] / PARAMS[512]:.4f}")
    own = 64 * sum(10 + k for k in range(13)) / 13
    pmax = 1 - (12 / 13) ** 64
    print(f"   an item with k letters has 10 + k input positions; k is uniform on 0..12, so a batch of 64 holds {own:,.0f} positions of its own items on average;")
    print(f"   exp45.batch_tensor pads every item to the longest of its batch, which has 12 letters in {100 * pmax:.1f}% of batches, so about {64 * 22:,} positions are computed")
    print("   forward multiply-adds per update = 1,024 x the count above (items' own positions); a backward pass under either rule uses matrices of the same shapes")
    for m in 'AR':
        print(f"   computation to 80 on {m} (three-row): median (range) over seeds, in forward multiply-adds [and in weight-items = parameters x 64 x updates]")
        for rule in RULES:
            for f in (25, 50):
                out = []
                for H in HS:
                    v = [THR[rule, H, f, s, m, 80, 3] for s in SEEDS]
                    fm = med_range([None if x is None else x * 1024 * mp[H] for x in v], lambda x: f"{x:.2e}")
                    wi = med_range([None if x is None else x * 64 * PARAMS[H] for x in v], lambda x: f"{x:.2e}")
                    out.append(f"{'half' if H == 256 else 'present' if H == 512 else 'double'} {fm} [{wi}]")
                print(f"      {rule} 0.{f}: " + " | ".join(out))
    print("   the width with the least computation to threshold (weight-items), per seed in the order of SEEDS; a width that does not reach the threshold is left out")
    tally = {}
    for thr in (80, 90):
        for ver in (3, 1):
            for m in 'AR':
                out = []
                for rule in RULES:
                    for f in (25, 50):
                        best = []
                        for s in SEEDS:
                            c = {H: THR[rule, H, f, s, m, thr, ver] * PARAMS[H] for H in HS if THR[rule, H, f, s, m, thr, ver] is not None}
                            best.append(min(c, key=c.get) if c else None)
                            tally.setdefault((thr, ver), []).append((best[-1], len(c)))
                        out.append(f"{rule} 0.{f}: {best}")
                print(f"      threshold {thr}, {'three-row' if ver == 3 else 'single-row'}, {m}: " + " | ".join(out))
    for (thr, ver), v in tally.items():
        print(f"      threshold {thr}, {'three-row' if ver == 3 else 'single-row'}: of {len(v)} (score, rule, fraction, seed) cases, the half width is cheapest in {sum(b == 256 for b, _ in v)}, "
              f"the present in {sum(b == 512 for b, _ in v)}, the double in {sum(b == 1024 for b, _ in v)}, no width reaches it in {sum(b is None for b, _ in v)}; "
              f"cases in which the half width does not reach it but a wider one does: {sum(b in (512, 1024) for b, _ in v)}")
    print("   the E44 width rule (R at f 0.50 under back-propagation): weight-items to 80 at the three widths, per seed")
    for ver in (3, 1):
        print(f"      {'three-row' if ver == 3 else 'single-row'}: " + " | ".join(
            f"{s}: " + ", ".join(f"{THR['B', H, 50, s, 'R', 80, ver] * 64 * PARAMS[H]:.3e}" for H in HS) for s in SEEDS))
    print("   ratio of updates to 80, present / half and double / present (three-row), per seed")
    for m in 'AR':
        for rule in RULES:
            for f in (25, 50):
                out = []
                for n, w in ((256, 512), (512, 1024)):
                    q = []
                    for s in SEEDS:
                        a, b = THR[rule, n, f, s, m, 80, 3], THR[rule, w, f, s, m, 80, 3]
                        q.append('n.r.' if a is None or b is None else f"{b / a:.2f}")
                    out.append(f"{n}->{w}: " + ", ".join(q))
                print(f"      {m} {rule} 0.{f}: " + " | ".join(out))
    print("   ratio of weight-items to 80, present / half and double / present (three-row), per seed")
    for m in 'AR':
        for rule in RULES:
            for f in (25, 50):
                out = []
                for n, w in ((256, 512), (512, 1024)):
                    q = []
                    for s in SEEDS:
                        a, b = THR[rule, n, f, s, m, 80, 3], THR[rule, w, f, s, m, 80, 3]
                        q.append('n.r.' if a is None or b is None else f"{b * PARAMS[w] / (a * PARAMS[n]):.2f}")
                    out.append(f"{n}->{w}: " + ", ".join(q))
                print(f"      {m} {rule} 0.{f}: " + " | ".join(out))

if "C" in want:
    print("C. the description of every width step, against the worker's report")
    NAME = {'S': 'sweep', 'B': 'backprop'}
    mine, mine_any = {}, {}
    for thr in (80, 50, 90):
        for m in 'RA':
            for rule in RULES:
                for f in FR:
                    row, row_any = [], []
                    for unit in ('updates', 'weight-items'):
                        for n, w in ((256, 512), (512, 1024)):
                            d3, d1 = [], []
                            for s in SEEDS:
                                kn, kw = (1, 1) if unit == 'updates' else (PARAMS[n], PARAMS[w])
                                d3.append(describe(THR[rule, n, f, s, m, thr, 3], THR[rule, w, f, s, m, thr, 3], DOSE[n] - 512, DOSE[w] - 512, kn, kw))
                                d1.append(describe(THR[rule, n, f, s, m, thr, 1], THR[rule, w, f, s, m, thr, 1], DOSE[n], DOSE[w], kn, kw))
                            row.append(cell(d3) + ('*' if cell(d1) != cell(d3) else ''))
                            row_any.append(cell(d3) + ('*' if d1 != d3 else ''))
                            mine[thr, m, rule, f, unit, n, 'seeds'] = d3
                    mine[thr, m, rule, f] = row; mine_any[thr, m, rule, f] = row_any
    rep = git_bytes(M + 'E41_REPORT.md', REPORT_REF).decode()
    theirs, thr = {}, None
    for line in rep.splitlines():
        h = re.match(r'\*\*Threshold (\d+)\*\*', line)
        if h:
            thr = int(h.group(1)); continue
        c = [x.strip() for x in line.strip().strip('|').split('|')]
        if thr and len(c) == 7 and c[0] in ('A', 'R') and c[1] in ('sweep', 'backprop'):
            theirs[thr, c[0], {'sweep': 'S', 'backprop': 'B'}[c[1]], round(float(c[2]) * 100)] = c[3:]
        if line.startswith('## Predictions'):
            thr = None
    strip = lambda row: [c.rstrip('*') for c in row]
    agree = sum(strip(mine[k]) == strip(theirs[k]) for k in theirs)
    print(f"   the report's three description tables parsed: {len(theirs)} rows of four cells; the descriptions (asterisks set aside) are identical to mine in {agree} rows")
    for k in theirs:
        if strip(mine[k]) != strip(theirs[k]):
            print(f"      DIFFERS threshold {k[0]}, {k[1]}, {NAME[k[2]]}, f 0.{k[3]}: report {theirs[k]}; mine {mine[k]}")
    a_cell = sum(mine[k] == theirs[k] for k in theirs); a_any = sum(mine_any[k] == theirs[k] for k in theirs)
    print(f"   the asterisk: rows identical to the report when it marks a cell whose two-of-three description differs between the versions: {a_cell};"
          f" when it marks a cell in which any seed's description differs: {a_any}")
    for k in theirs:
        if mine_any[k] != theirs[k]:
            print(f"      asterisks differ under the second reading: threshold {k[0]}, {k[1]}, {NAME[k[2]]}, f 0.{k[3]}: report {theirs[k]}; mine {mine_any[k]}")
    stars = sum(c.endswith('*') for k in theirs for c in theirs[k]); real = sum(c.endswith('*') for k in mine for c in (mine[k] if len(k) == 4 else []))
    print(f"   cells marked in the report: {stars} of {4 * len(theirs)}; cells whose two-of-three description itself differs between the versions: {real}")
    summ = js(M + 'exp45/e41/E41_SUMMARY_all.json', REPORT_REF)
    nd = 0
    for (rule, H, f, seed), R in RUNS.items():
        t = summ[f"{rule} h{H} f{f} {seed}"]['thresholds']
        for m in 'AR':
            for th in (50, 80, 90):
                for ver, name in ((1, 'single'), (3, 'three')):
                    v = t[f"{m}{th}_{name}"]; v = None if v is None else v['updates']
                    if v != THR[rule, H, f, seed, m, th, ver]:
                        nd += 1; print(f"      DIFFERS run {rule} h{H} f{f} {seed} {m}{th} {name}: report {v}, mine {THR[rule, H, f, seed, m, th, ver]}")
    print(f"   thresholds of E41_SUMMARY_all.json against mine: {len(RUNS) * 12} values compared, {nd} differ")
    print("   threshold 80, three-row version, the three seeds' descriptions (order 73129, 93683, 116089):")
    for m in 'RA':
        for rule in RULES:
            for f in (25, 50):
                print(f"      {m} {NAME[rule]:8s} 0.{f}: " + " | ".join(f"{unit} {n}->{n * 2}: {', '.join(mine[80, m, rule, f, unit, n, 'seeds'])}"
                                                                   for unit in ('updates', 'weight-items') for n in (256, 512)))

if "D" in want:
    print("D. statements on three seeds")
    print("   (a) the sweep's updates to threshold over back-propagation's, same seed, fraction and width (equal to the ratio of computation, since the")
    print("       two rules use matrices of the same shapes at a given width); three-row version; n.r. = one of the two does not reach it")
    allq = {}
    for m, thr in (('A', 80), ('R', 80), ('A', 90), ('R', 90)):
        for f in (25, 50):
            out = []
            for H in HS:
                q = []
                for s in SEEDS:
                    a, b = THR['S', H, f, s, m, thr, 3], THR['B', H, f, s, m, thr, 3]
                    q.append(None if a is None or b is None else a / b)
                allq[m, thr, f, H] = q
                out.append(f"H {H}: " + ", ".join('n.r.' if x is None else f"{x:.2f}" for x in q))
            print(f"      {m}{thr} f 0.{f}: " + " | ".join(out))
    for m, thr in (('A', 80), ('R', 80)):
        v = [x for (mm, tt, f, H), q in allq.items() if mm == m and tt == thr for x in q if x is not None]
        nn = sum(1 for (mm, tt, f, H), q in allq.items() if mm == m and tt == thr for x in q if x is None)
        print(f"      {m}{thr}, the 18 (seed, fraction, width) cases at f 0.25 and 0.50: both rules reach it in {len(v)}, ratio median {statistics.median(v):.2f}, range {min(v):.2f}-{max(v):.2f}; "
              f"the sweep slower (ratio above 1) in {sum(x > 1 for x in v)}, equal in {sum(x == 1 for x in v)}, faster in {sum(x < 1 for x in v)}; not both reaching in {nn}")
        for f in (25, 50):
            w = [x for (mm, tt, ff, H), q in allq.items() if mm == m and tt == thr and ff == f for x in q if x is not None]
            print(f"         f 0.{f}: median {statistics.median(w):.2f}, range {min(w):.2f}-{max(w):.2f} over {len(w)} cases")
    print("       which rule does not reach 80 within the dose (three-row), f 0.25 and 0.50:")
    for m in 'AR':
        miss = [(r, H, f, s) for r in RULES for H in HS for f in (25, 50) for s in SEEDS if THR[r, H, f, s, m, 80, 3] is None]
        print(f"         {m}: {miss}")
    print("       CPU-seconds of updates per update (median of the last row's cpu_updates / dose over the nine runs of a rule and width):")
    for H in HS:
        c = {r: statistics.median(RUNS[r, H, f, s]['rows'][-1]['cpu_updates'] / DOSE[H] for f in FR for s in SEEDS) for r in RULES}
        print(f"         H {H}: sweep {c['S']:.4f}, back-propagation {c['B']:.4f}, ratio {c['S'] / c['B']:.3f}")
    print("   (b) a quarter and a half of the pairs taught: R at the last row and its highest row, and the mean of R over rows 7,680 to 8,192, per seed")
    for f in (25, 50):
        for rule in RULES:
            for H in HS:
                out = []
                for s in SEEDS:
                    rows = RUNS[rule, H, f, s]['rows']; mr = max(rows, key=lambda r: r['R'])
                    out.append(f"last {rows[-1]['R']:.1f}, max {mr['R']:.1f} at {mr['step']:,}, rows 7,680-8,192 {statistics.mean(r['R'] for r in rows if r['step'] in (7680, 7936, 8192)):.1f}")
                print(f"      f 0.{f} {rule} H {H:4d} (dose {DOSE[H]:,}): " + " || ".join(out))
    print("   (c) a tenth of the pairs taught: the highest and the last score, and the mean over rows 7,680 to 8,192, per seed")
    for rule in RULES:
        for H in HS:
            out = []
            for s in SEEDS:
                rows = RUNS[rule, H, 10, s]['rows']
                mx = max(rows, key=lambda r: r['A']); mr = max(rows, key=lambda r: r['R'])
                lvl = statistics.mean(r['A'] for r in rows if r['step'] in (7680, 7936, 8192))
                out.append(f"A max {mx['A']:.1f} at {mx['step']:,}, last {rows[-1]['A']:.1f}, rows 7,680-8,192 {lvl:.1f}; R max {mr['R']:.1f}, last {rows[-1]['R']:.1f}; taught last {rows[-1]['taught']:.1f}")
            print(f"      {rule} H {H:4d}: " + " || ".join(out))
    reach = [(k[:4], k[4], k[5], k[6]) for k in THR if k[2] == 10 and THR[k] is not None]
    print(f"      thresholds reached at f 0.10 (any score, threshold or version): {sorted(reach)}")
    print("      the half width under back-propagation, A by updates (three seeds):")
    for step in (4096, 8192, 12288, 16384, 24576, 32768, 40960):
        print(f"         {step:6,}: B " + ", ".join(f"{RUNS['B', 256, 10, s]['rows'][step // 256]['A']:.1f}" for s in SEEDS)
              + " | S " + ", ".join(f"{RUNS['S', 256, 10, s]['rows'][step // 256]['A']:.1f}" for s in SEEDS))
    print("      least-squares slope of A against updates over the last 8,192 updates of the half-width runs at f 0.10 (points per 1,000 updates):")
    for rule in RULES:
        sl = []
        for s in SEEDS:
            rows = [r for r in RUNS[rule, 256, 10, s]['rows'] if r['step'] >= 40960 - 8192]
            x = np.array([r['step'] for r in rows], float); y = np.array([r['A'] for r in rows])
            sl.append(float(np.polyfit(x, y, 1)[0]) * 1000)
        print(f"         {rule}: " + ", ".join(f"{v:+.2f}" for v in sl))
    print("   (e) R against A at the present width, f 0.50 (the report's P41.8), three-row: rows of R80 and A80 per seed")
    for rule in RULES:
        print(f"      {rule}: " + ", ".join(f"{s}: R {fmt_u(THR[rule, 512, 50, s, 'R', 80, 3])}, A {fmt_u(THR[rule, 512, 50, s, 'A', 80, 3])}" for s in SEEDS))


def sample_r(task, held, seed, f):
    op, m, _ = task
    pairs = [p for p in held if ((p[0] + p[1] >= m) if op == "add" else (p[0] - p[1] < 0))]
    return Random(f"e41|{seed}|{round(f * 100)}|{task[0]}").sample(sorted(pairs), 1000)


def count(W, items):
    x = np.array([[BOS] + p + a[:-1] for p, a in items], dtype=np.int64); tgt = np.array([a for _, a in items], dtype=np.int64)
    return int(exact_teacher_forced(forward(W, x)[3], tgt).sum())


def reader_counts(W, seed, f):
    """Whole answers right: A on 300 held-out pairs at 12 letters, R on 1,000 held-out reduction pairs without letters,
    taught on 300 taught pairs at 12 letters (generator of seed + 1); for the four tasks where the receipt has four."""
    out = {'A': {}, 'R': {}, 'taught': {}}
    for t in DEV:
        taught, held = split(t, f, seed); name = f"{t[0]}{t[1]}"
        out['A'][name] = count(W, panel(t, held, seed, (12,), 300)[12])
        out['taught'][name] = count(W, panel(t, taught, seed + 1, (12,), 300)[12])
        if t[0] in ('add', 'sub'):
            out['R'][name] = count(W, [render(t, a, b, 0, None) for a, b in sample_r(t, held, seed, f)])
    return out


if "E" in want:
    print("E. the births named as learners for the long lives (roster note of 17 November: E41's half-width births at f 0.50, back-propagation's")
    print("   for release 1, the sweep's for release 2)")
    r = subprocess.run(['git', 'ls-tree', '--full-tree', '-r', '-l', DREF, '--', M + 'exp45/e41/'], capture_output=True, text=True, cwd=HERE)
    listed = {ln.split('\t')[1]: int(ln.split()[3]) for ln in r.stdout.splitlines() if ln.endswith('.pt')}
    byk = {}
    for p in listed:
        mm = re.search(r'e41_([SB])_h(\d+)_f(\d+)_(\d+)\.pt$', p)
        byk[mm.group(1), int(mm.group(2)), int(mm.group(3)), int(mm.group(4))] = p
    for s in SEEDS:
        print(f"   seed {s}: weight files on the default branch at {DREF}: half {sum(1 for k in byk if k[3] == s and k[1] == 256)}, present {sum(1 for k in byk if k[3] == s and k[1] == 512)}, "
              f"double {sum(1 for k in byk if k[3] == s and k[1] == 1024)}")
    ids = {}
    for ref in [DREF] + list(SEED_REF.values()):
        rr = subprocess.run(['git', 'ls-tree', '--full-tree', '-r', ref, '--', M + 'exp45/e41/'], capture_output=True, text=True, cwd=HERE)
        ids[ref] = {ln.split('\t')[1]: ln.split()[2] for ln in rr.stdout.splitlines() if ln.endswith('.pt')}
    same_blob = sum(ids[DREF][p] == ids[SEED_REF[k[3]]].get(p) for k, p in byk.items())
    okh = 0
    for k, p in sorted(byk.items()):
        d = RUNS[k]['d']
        okh += hashlib.sha256(git_bytes(p, DREF)).hexdigest() == d['weights_sha256'] and d['weights'] == os.path.basename(p)
    print(f"   all {len(byk)} committed weight files: the same blob on the worker's branch as on the default branch in {same_blob}; sha256 of the file equal to its receipt's weights_sha256 in {okh}")
    extra = [a for a in sys.argv[1:] if a.startswith('files=')]
    todo = [(rule, 256, 50, s) for s in SEEDS for rule in 'BS']
    if extra and extra[0] == 'files=all':
        todo = sorted(byk)
    for k in todo:
        rule, H, f, s = k
        if k not in byk:
            print(f"   {k}: NOT COMMITTED on the default branch"); continue
        t = time.time()
        W, sha = load_weights(byk[k], DREF)
        d = RUNS[k]['d'] if k in RUNS else js(byk[k].replace('.pt', '.json'), DREF)
        last = d['curve'][-1]
        got = reader_counts(W, s, f / 100)
        same = (got['A'] == last['A_counts'] and got['R'] == last['R_counts'] and got['taught'] == last['taught_counts'])
        print(f"   e41_{rule}_h{H}_f{f}_{s}.pt: file hash equals the receipt's weights_sha256 {sha == d['weights_sha256']}; shapes U {W['rnn.weight_hh_l0'].shape[1]}, H {W['l2.weight'].shape[0]}; "
              f"reader A {got['A']['add89']}, {got['A']['sub89']} R {got['R']['add89']}, {got['R']['sub89']} taught {got['taught']['add89']}, {got['taught']['sub89']}"
              f" (A {last['A']:.2f}, R {last['R']:.2f}); all ten counts equal the receipt's last row: {same}  [{time.time() - t:.0f} s]", flush=True)
        if not same:
            print(f"      receipt: {last['A_counts']} {last['R_counts']} {last['taught_counts']}; reader: {got}")

# ------------------------------------------------------------------------------------------------ E46
E46 = {}
if want & set("FGH"):
    for rule in RULES:
        for wd in ('025', '005', '000'):
            for s in SEEDS:
                path = f"{M}exp45/e46/e46_{rule}_wd{wd}_{s}.json"
                d = js(path, E46REF)
                for r in d['curve']:
                    assert abs(100 * (r['R_counts']['add89'] + r['R_counts']['sub89']) / 2000 - r['R']) < 1e-9
                    assert abs(100 * (r['A_counts']['add89'] + r['A_counts']['sub89']) / 600 - r['A']) < 1e-9
                    assert abs(100 * r['length_counts']['40'] / 300 - r['len40']) < 1e-9
                assert [r['step'] for r in d['curve']] == list(range(0, 24577, 256))
                E46[rule, wd, s] = dict(d=d, rows=d['curve'], path=path)
    WDV = {'025': 0.25, '005': 0.05, '000': 0.0}
    end = lambda rows, key: statistics.mean(r[key] for r in rows if r['step'] in (24064, 24320, 24576))
    mid = lambda rows, key: statistics.mean(r[key] for r in rows if r['step'] in (11776, 12032, 12288))
    level = lambda ref, new: 'lower' if new - ref <= -15 else 'higher' if new - ref >= 15 else 'about the same'

if "F" in want:
    print("F. E46 from its 18 receipts")
    bad = []
    for (rule, wd, s), R in E46.items():
        d = R['d']
        g0 = [g['weight_decay'] for g in d['optimizer_groups_at_start']] if isinstance(d['optimizer_groups_at_start'], list) else d['optimizer_groups_at_start']
        g1 = [g['weight_decay'] for g in d['optimizer_groups_at_end']] if isinstance(d['optimizer_groups_at_end'], list) else d['optimizer_groups_at_end']
        ok = (d['complete'] is True and d['dose'] == 24576 and d['H'] == 512 and d['U'] == 256 and d['fraction'] == 0.5 and d['decay'] == WDV[wd]
              and d['optimizer']['weight_decay'] == WDV[wd] and d['optimizer_read_back']['weight_decay'] == WDV[wd]
              and [g['weight_decay'] for g in d['optimizer']['groups']] == [WDV[wd], 0.0]
              and [g['weight_decay'] for g in d['optimizer_read_back']['groups']] == [WDV[wd], 0.0]
              and d['optimizer']['groups'][1]['params'] == ['book.weight'] and d['reduction_sample_matches_e41_check7'] is True
              and d['optimizer_steps'] == [24576] and all(r['finite_losses'] for r in R['rows']))
        if not ok:
            bad.append((rule, wd, s, g0, g1))
    print(f"   18 receipts at {E46REF}: complete at 24,576 updates, present width, f 0.50, the receipt's `decay`, `optimizer.weight_decay` and the groups read back "
          f"all equal to the run's decay, byte table without decay, sample digests asserted: {'all' if not bad else 'NOT ' + str(bad)}")
    r = subprocess.run(['git', 'ls-tree', '--full-tree', '-r', '-l', E46REF, '--', M + 'exp45/e46/'], capture_output=True, text=True, cwd=HERE)
    pts = [ln for ln in r.stdout.splitlines() if ln.endswith('.pt')]
    print(f"   weight files committed at {E46REF}: {len(pts)}; runner hashes: {sorted({R['d']['provenance']['runner_sha256'][:16] for R in E46.values()})}")
    print(f"   CPU-hours of the 18 complete runs: updates {sum(R['d']['cpu_updates'] for R in E46.values()) / 3600:.2f}, scoring {sum(R['d']['cpu_scoring'] for R in E46.values()) / 3600:.2f}")
    dg = {s: {R['rows'][16]['data_digest']['sha256'] for (_, _, ss), R in E46.items() if ss == s} for s in SEEDS}
    e41dg = {s: js(f"{M}exp45/e41/e41_S_h512_f50_{s}.json", SEED_REF[s])['curve'][16]['data_digest']['sha256'] for s in SEEDS}
    print(f"   batch digest at row 4,096: one per seed {all(len(v) == 1 for v in dg.values())}; equal to E41's at f 0.50 {all(dg[s] == {e41dg[s]} for s in SEEDS)}")
    # the reference runs against E41 (present width, f 0.50) row by row to 12,288, and against E26's final tensors
    for rule in RULES:
        out = []
        for s in SEEDS:
            e41 = js(f"{M}exp45/e41/e41_{rule}_h512_f50_{s}.json", SEED_REF[s])['curve']
            e46 = E46[rule, '025', s]['rows']
            eq = sum(all(a[k] == b[k] for k in ('A_counts', 'R_counts', 'taught_counts')) for a, b in zip(e41, e46))
            e26 = js(f"{M}exp45/e26/e26_dev_{rule}_{s}.json", DREF)
            out.append(f"{s}: counts equal E41's at {eq} of {len(e41)} rows; final tensors_sha256 equals E26's birth {E46[rule, '025', s]['d']['tensors_sha256'] == e26.get('tensors_sha256')}")
        print(f"   decay 0.25, rule {rule}: " + "; ".join(out))
    print("   per run: updates to 80 on R and on A, three-row [single-row]; levels = means over rows 24,064 to 24,576; len40 = the length test at 40 letters")
    print("      rule decay seed    R80             A80             R90             end R   end A   end len40  len40 last 8  mid R   mid len40  end taught")
    for rule in RULES:
        for wd in ('025', '005', '000'):
            for s in SEEDS:
                rows = E46[rule, wd, s]['rows']
                c = lambda m, th: f"{fmt_u(crossing(rows, m, th, 3))} [{fmt_u(crossing(rows, m, th, 1))}]"
                l8 = statistics.mean(r['len40'] for r in rows[-8:])
                print(f"      {rule}    {WDV[wd]:4.2f}  {s:6d}  {c('R', 80):15s} {c('A', 80):15s} {c('R', 90):15s} {end(rows, 'R'):6.2f}  {end(rows, 'A'):6.2f}  {end(rows, 'len40'):6.2f}     {l8:6.2f}"
                      f"        {mid(rows, 'R'):6.2f}  {mid(rows, 'len40'):6.2f}     {end(rows, 'taught'):6.2f}")
    print("   the three cells per rule (decay 0 and 0.05 against 0.25): per seed, then the cell (two of three seeds)")
    for rule in RULES:
        for wd in ('000', '005'):
            cells = {}
            for name in ('updates to 80 on R', 'updates to 80 on R, single-row', 'updates to 80 on A', 'level R', 'level A', 'level len40'):
                ds = []
                for s in SEEDS:
                    ref, new = E46[rule, '025', s]['rows'], E46[rule, wd, s]['rows']
                    if name.startswith('updates'):
                        m = 'R' if ' on R' in name else 'A'; ver = 1 if 'single' in name else 3; D = 24576 - (512 if ver == 3 else 0)
                        a, b = crossing(ref, m, 80, ver), crossing(new, m, 80, ver)
                        ds.append((describe(a, b, D, D), f"{fmt_u(a)} -> {fmt_u(b)}" + ("" if a is None or b is None else f", x{b / a:.2f}")))
                    else:
                        key = {'level R': 'R', 'level A': 'A', 'level len40': 'len40'}[name]
                        a, b = end(ref, key), end(new, key)
                        ds.append((level(a, b), f"{a:.2f} -> {b:.2f} ({b - a:+.2f})"))
                cells[name] = cell([d for d, _ in ds])
                print(f"      {rule} decay {WDV[wd]:4.2f} {name:30s}: {cells[name]:15s} | " + "; ".join(f"{d} ({t})" for d, t in ds))
            thr_c, lr, l40 = cells['updates to 80 on R'], cells['level R'], cells['level len40']
            needs = [n for n, c in (('updates to 80 on R', thr_c), ('level R', lr), ('level len40', l40)) if c in ('rises', 'lower')]
            sent = ('needs decay for ' + ', '.join(needs)) if needs else ('does not need decay' if thr_c in ('falls', 'stays') and lr in ('about the same', 'higher') and l40 in ('about the same', 'higher') else 'not decided')
            print(f"      => {rule}, decay {WDV[wd]:4.2f} against 0.25: learning the first family {sent}" + ("" if wd == '000' else "  (the runs at 0.05 enter no sentence)"))
    summ = js(M + 'exp45/e46/E46_SUMMARY.json', E46REF)
    nd = 0
    for (rule, wd, s), R in E46.items():
        t = summ['runs'][f"{rule} wd{wd} {s}"]
        for m in 'AR':
            for th in (50, 80, 90):
                for ver, name in ((1, 'single'), (3, 'three')):
                    nd += t['thresholds'][f"{m}{th}_{name}"] != crossing(R['rows'], m, th, ver)
        for key, mine in (('end_R', end(R['rows'], 'R')), ('end_A', end(R['rows'], 'A')), ('end_len40', end(R['rows'], 'len40')), ('mid_R', mid(R['rows'], 'R'))):
            nd += abs(t[key] - mine) > 1e-9
        nd += abs(t['norms_end']['recurrent'] - R['rows'][-1]['norms']['recurrent']) > 1e-9
    print(f"   the worker's E46_SUMMARY.json against mine: 18 runs x 17 values compared, {nd} differ; its sentences: "
          + json.dumps(summ['sentences'])[:600])
    print("   norms of the seven weight groups at row 24,576 (Frobenius), three seeds, and the ratio decay 0 / decay 0.25 (median)")
    for rule in RULES:
        for g in ('book', 'l1', 'input_map', 'recurrent', 'lm', 'l2', 'l3'):
            v = {wd: [E46[rule, wd, s]['rows'][-1]['norms'][g] for s in SEEDS] for wd in WDV}
            q = [a / b for a, b in zip(v['000'], v['025'])]
            print(f"      {rule} {g:10s} start {E46[rule, '025', SEEDS[0]]['rows'][0]['norms'][g]:6.2f} | decay 0.25: " + ", ".join(f"{x:6.2f}" for x in v['025'])
                  + " | 0.05: " + ", ".join(f"{x:6.2f}" for x in v['005']) + " | 0: " + ", ".join(f"{x:6.2f}" for x in v['000']) + f" | ratio 0 / 0.25: {', '.join(f'{x:.2f}' for x in q)}")

GROUPS = dict(book=['book.weight'], l1=['l1.weight', 'l1.bias'], input_map=['rnn.weight_ih_l0', 'rnn.bias_ih_l0'], recurrent=['rnn.weight_hh_l0', 'rnn.bias_hh_l0'],
              lm=['lm.weight', 'lm.bias'], l2=['l2.weight', 'l2.bias'], l3=['l3.weight', 'l3.bias'])

if "G" in want:
    print("G. E46 weight files through the numpy reader (last row of each run)")
    extra = [a for a in sys.argv[1:] if a.startswith('files=')]
    todo = [(rule, wd, s) for s in SEEDS for rule, wd in (('S', '000'), ('B', '000'))]
    if extra and extra[0] == 'files=all':
        todo = sorted(E46)
    for k in todo:
        rule, wd, s = k; t = time.time()
        W, sha = load_weights(f"{M}exp45/e46/e46_{rule}_wd{wd}_{s}.pt", E46REF)
        d = E46[k]['d']; last = d['curve'][-1]
        got = reader_counts(W, s, 0.5)
        lp = panel(("add", 89, b"p"), split(("add", 89, b"p"), .5, s)[1], s, (12, 40), 300)
        ln = {str(kk): count(W, lp[kk]) for kk in (12, 40)}
        norms = {g: float(np.sqrt(sum(float((W[n].astype(np.float64) ** 2).sum()) for n in names))) for g, names in GROUPS.items()}
        same = got['A'] == last['A_counts'] and got['R'] == last['R_counts'] and got['taught'] == last['taught_counts'] and ln == last['length_counts']
        nsame = all(abs(norms[g] - last['norms'][g]) < 1e-3 * last['norms'][g] for g in GROUPS)
        print(f"   e46_{rule}_wd{wd}_{s}.pt: file hash equals the receipt's {sha == d['weights_sha256']}; A {got['A']['add89']}, {got['A']['sub89']}; R {got['R']['add89']}, {got['R']['sub89']}; "
              f"length test 12 / 40 letters {ln['12']}, {ln['40']} of 300; all twelve counts equal the receipt's last row: {same}; the seven norms equal the receipt's to 0.1%: {nsame}"
              f" (recurrent {norms['recurrent']:.2f})  [{time.time() - t:.0f} s]", flush=True)
        if not same:
            print(f"      receipt {last['A_counts']} {last['R_counts']} {last['taught_counts']} {last['length_counts']}; reader {got} {ln}")

if "H" in want:
    print("H. E46: growth of the weights without decay, and what the 15-point band leaves unsaid")
    for rule in RULES:
        for g in ('recurrent', 'l2', 'input_map'):
            out = []
            for wd in ('025', '000'):
                v = [statistics.mean(E46[rule, wd, s]['rows'][st // 256]['norms'][g] for s in SEEDS) for st in (0, 6144, 12288, 18432, 24576)]
                out.append(f"decay {WDV[wd]:4.2f}: " + ", ".join(f"{x:.1f}" for x in v))
            print(f"   {rule} {g:10s} norm at 0 / 6,144 / 12,288 / 18,432 / 24,576 updates (mean of three seeds): " + " | ".join(out))
    for rule in RULES:
        v = [statistics.mean(E46[rule, '000', s]['rows'][st // 256]['norms']['recurrent'] for s in SEEDS) for st in (12288, 18432, 24576)]
        print(f"   {rule} recurrent norm without decay: growth per 1,000 updates {1000 * (v[1] - v[0]) / 6144:.2f} over 12,288-18,432 and {1000 * (v[2] - v[1]) / 6144:.2f} over 18,432-24,576")
    print("   differences decay 0 minus decay 0.25 at the end (means over rows 24,064 to 24,576), per seed, in points:")
    for rule in RULES:
        for key in ('R', 'A', 'len40', 'taught'):
            dd = [end(E46[rule, '000', s]['rows'], key) - end(E46[rule, '025', s]['rows'], key) for s in SEEDS]
            print(f"      {rule} {key:6s}: " + ", ".join(f"{x:+.2f}" for x in dd) + f"   (mean {statistics.mean(dd):+.2f})")
    print("   the same at decay 0.05:")
    for rule in RULES:
        for key in ('R', 'len40'):
            dd = [end(E46[rule, '005', s]['rows'], key) - end(E46[rule, '025', s]['rows'], key) for s in SEEDS]
            print(f"      {rule} {key:6s}: " + ", ".join(f"{x:+.2f}" for x in dd) + f"   (mean {statistics.mean(dd):+.2f})")
    print("   R over the last 4,096 updates (17 rows): mean and lowest row, per seed")
    for rule in RULES:
        for wd in ('025', '000'):
            print(f"      {rule} decay {WDV[wd]:4.2f}: " + "; ".join(f"mean {statistics.mean(r['R'] for r in E46[rule, wd, s]['rows'][-17:]):.2f}, lowest {min(r['R'] for r in E46[rule, wd, s]['rows'][-17:]):.2f}" for s in SEEDS))
    print("   updates to 80 and 90 on R, decay 0 over decay 0.25, three-row: " + " | ".join(
        f"{rule} R{th}: " + ", ".join(f"{crossing(E46[rule, '000', s]['rows'], 'R', th, 3) / crossing(E46[rule, '025', s]['rows'], 'R', th, 3):.2f}" for s in SEEDS)
        for rule in RULES for th in (80, 90)))

print(f"done in {time.time() - T0:.0f} s")
