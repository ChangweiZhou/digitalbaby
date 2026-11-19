# An independent check of the claim "A7" (E57's worker), written from the reviewer's description of 15:28 UTC, 19 November, alone (the other
# program, ADDITION/addsys.py on claude/e63-tissue-event-017dSE, was not read before this program and its first results were written).
#
# Text: lines "a+b=c\n", a and b in 0..D-1, c = a + b in decimal; of the D*D pairs a fixed set (drawn by the seed) is never shown: 20 of 100
# for D = 10, 2,000 of 10,000 for D = 100; the lines are drawn at random, with replacement, from the shown pairs. The system reads one byte at
# a time and, before each byte arrives, names it (the next byte); it is told yes or no.
#
# The cells (choices where the description is open are marked [C]):
#   value of a byte: the digit's value for '0'..'9', 0 for every other byte [C]; in the control "perm" the ten digit values are replaced by a
#     fixed random permutation of 0..9 (drawn by the seed) before they enter the magnitude cells.
#   magnitude cell 1: drive m1(t) = value(byte t-1) + value(byte t-3); magnitude cell 2: m2(t) = value(t-2) + value(t-4) (unit connections
#     with delays of whole bytes; bytes before the start count as 0) [C: t is the present byte, the one just read].
#   tuned cells: for every total T (0 .. 2(D-1)... only the totals a drive can take): excited when the drive is >= T, inhibited when >= T+1,
#     so active exactly when the drive equals T. Kind 1 (from m1) is gated by the present byte: one cell for each (T, present byte); kind 2
#     (from m2) by the present byte and the byte 1 back: one cell for each (T, present byte, byte 1 back) [C: a gate is a product, so a gated
#     cell is active only for its own gating bytes; cells are made for every combination, not only those that occur]. In the control "nogate"
#     one cell for each T of each kind, no gate.
#   byte cells: one for each value of the present byte. And a constant 1.
#   So U has exactly four (three without gates... also four: kind 1, kind 2, byte cell, constant) active entries of value 1; n = |U|^2 = 4.
# The answering cell: for each byte value v an estimate a[v] = W[v] U (W starts at 0); it names the value of largest estimate (ties to the
#   lowest value) or, with chance eps = 1/16 while learning, a value seen so far at random [C: eps and its form, as rule B/Q in measure_q.py].
#   Rule Q (as measure_q.py, the reviewer's message of 13:05 UTC) [C: eta = 1/16]: the named value e: W[e] += eta (ok - a[e]) U / n; on yes
#   every other value v: W[v] += eta (0 - a[v]) U / n; on no the others do nothing.
# Score, with learning frozen and no random choice: for every pair, the bytes "\n a + b =" are read (the "\n" ends a previous line; the
#   state is only the last four bytes, so nothing else carries over) and then, with each true byte read after it is named (teacher forcing)
#   [C], every byte after '=' up to and including the end of line must be named right. Share of pairs right, shown and never shown.
# usage: addcheck.py <label> D=10 lines=60000 seeds=73129,93683,116089 orders=5 variant=A7|perm|nogate   -> res/add_<label>.json
import sys, os, json, time, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
label = sys.argv[1]; kw = dict(a.split('=') for a in sys.argv[2:]); t0 = time.time()
D = int(kw.get('D', 10)); LINES = int(kw.get('lines', 60000)); SEEDS = [int(s) for s in kw.get('seeds', '73129,93683,116089').split(',')]
ORDERS = int(kw.get('orders', 5)); VAR = kw.get('variant', 'A7'); ETA = float(eval(kw.get('eta', '1/16'))); EPS = float(eval(kw.get('eps', '1/16')))
NHIDE = 20 if D == 10 else 2000
ALPH = [ord(c) for c in '0123456789+=\n']; AI = {c: i for i, c in enumerate(ALPH)}; NA = len(ALPH)
TMAX = 2 * 9                                      # a drive is the sum of two digit values, 0..18
NT = TMAX + 1
def feats(hist, val):
    """Active feature indices for the present byte hist[-1] (hist holds at least the last five bytes, oldest first)."""
    p, p1 = hist[-1], hist[-2]
    m1 = val[hist[-2]] + val[hist[-4]]; m2 = val[hist[-3]] + val[hist[-5]]
    if VAR == 'nogate':
        f1 = m1; f2 = NT + m2; base = 2 * NT
    else:
        f1 = m1 * NA + AI[p]; f2 = NT * NA + (m2 * NA + AI[p]) * NA + AI[p1]; base = NT * NA + NT * NA * NA
    return (f1, f2, base + AI[p], base + NA)      # kind 1, kind 2, byte cell, constant
NF = (2 * NT + NA + 1) if VAR == 'nogate' else (NT * NA + NT * NA * NA + NA + 1)
def line(a, b): return [ord(c) for c in f'{a}+{b}={a + b}\n']
runs = []
for seed in SEEDS:
    rs = np.random.default_rng(seed)
    pairs = [(a, b) for a in range(D) for b in range(D)]
    hide = set(map(tuple, np.array(pairs)[rs.choice(len(pairs), NHIDE, replace=False)].tolist()))
    shown = [p for p in pairs if p not in hide]
    val = {c: 0 for c in ALPH}
    perm = rs.permutation(10) if VAR == 'perm' else np.arange(10)
    for d in range(10): val[ord('0') + d] = int(perm[d])
    for order in range(ORDERS):
        ro = np.random.default_rng(seed * 100 + order)
        W = np.zeros((256, NF)); seen = np.zeros(256, bool)
        hist = [ord('\n')] * 5; yes = 0; steps = 0
        for li in range(LINES):
            a, b = shown[ro.integers(len(shown))]
            for nxt in line(a, b):
                f = feats(hist, val); A = W[:, f].sum(1)
                e = int(A.argmax())
                if ro.random() < EPS and seen.any():
                    sv = np.flatnonzero(seen); e = int(sv[ro.integers(len(sv))])
                ok = e == nxt; yes += ok; steps += 1
                if ok:
                    d = ETA * (0.0 - A) / 4.0; d[e] = ETA * (1.0 - A[e]) / 4.0
                    W[:, f] += d[:, None]
                else:
                    W[e, f] += ETA * (0.0 - A[e]) / 4.0
                seen[nxt] = True; hist = hist[1:] + [nxt]
        def score(ps):
            right, wrong = 0, []
            for a, b in ps:
                h = [ord('\n')] * 5
                for c in [ord(x) for x in f'{a}+{b}=']: h = h[1:] + [c]
                good = True
                for nxt in [ord(x) for x in f'{a + b}\n']:
                    f = feats(h, val); e = int(W[:, f].sum(1).argmax())
                    if e != nxt: good = False
                    h = h[1:] + [nxt]
                if good: right += 1
                else: wrong.append(f'{a}+{b}')
            return right / len(ps), wrong
        s_sh, w_sh = score(shown); s_hd, w_hd = score(sorted(hide))
        runs.append(dict(seed=seed, order=order, shown=s_sh, never_shown=s_hd, wrong_shown=w_sh[:50], wrong_never_shown=w_hd[:50],
                         yes_share_learning=yes / steps, hidden=sorted(f'{a}+{b}' for a, b in hide)[:50]))
        print(f'{label} seed {seed} order {order}: shown {100 * s_sh:.1f}, never shown {100 * s_hd:.1f}; yes while learning {100 * yes / steps:.1f}%; '
              f'wrong never shown: {" ".join(w_hd[:12])}', flush=True)
q = lambda k: [100 * min(r[k] for r in runs), 100 * float(np.median([r[k] for r in runs])), 100 * max(r[k] for r in runs)]
out = dict(label=label, args=kw, features=NF, runs=runs, shown_min_median_max=q('shown'), never_shown_min_median_max=q('never_shown'), seconds=time.time() - t0)
os.makedirs(os.path.join(HERE, 'res'), exist_ok=True); json.dump(out, open(os.path.join(HERE, 'res', f'add_{label}.json'), 'w'), indent=1)
print(f'== {label}: shown {" / ".join(f"{v:.1f}" for v in q("shown"))}, never shown {" / ".join(f"{v:.1f}" for v in q("never_shown"))} (smallest / median / largest over {len(runs)} runs; {time.time() - t0:.0f} s)', flush=True)
