# E57's worker, 19 November 2026: the reviewer's task of 15:13 UTC. The text and the score are those of the reviewer's yesno/addtext.py
# (claude/vibrant-cerf-sje726 at d6808869f): lines "a+b=s\n", a and b from 0 to MAXN, HELD of the pairs kept out of learning; the system
# names the next byte and is told yes or no; scored: the answer's bytes, the other bytes, and the WHOLE answer (every byte after '=' up
# to the end mark right), for sums shown (no learning) and for sums never shown.
#
# Own code. A system is a set of cells, each with binary output, feeding output units (one a byte value) that learn by rule Q of
# THE_TISSUE_AS_RUN section 3b (eta 1/16, z1; emission the largest, with chance 1/16 in learning a byte value seen so far).
#
# The cells (sys= a comma list of parts):
#   byte    : one cell per byte value, firing when that byte is the present byte (a cell that takes the byte as input, tuned to one value).
#   trace   : the reviewer's slow copies s_2, s_4, s_8 of the byte cells (graded, scaled by sqrt(2k-1)); with 'byte' this is addtext.py.
#   pair:D1-D2 : a cell for each pair of byte values (x, y), firing when the byte D1 back is x and the byte D2 back is y (a conjunction
#             of two delayed byte cells); with gate=1 also of the present byte (three bytes together).
#   mag:D1-D2 : a magnitude cell whose activity is the byte value D1 back plus the byte value D2 back (a cell that takes the byte itself as
#             a graded input, over two connections of unit strength with delays D1 and D2), and threshold cells reading it: cell T fires
#             when the magnitude is at least T, for every integer T over the range of the magnitude. With gate=1 each threshold cell also
#             needs the present byte: one cell per (T, present byte) (a conjunction of a threshold cell and a byte cell).
#             gate=2: also of the byte 1 back (the threshold cell, the present byte and the byte before it together).
#             mag:D1-D2:G sets the gate of that part alone.
#             With tuned=1 the cell for T fires only when the magnitude equals T: excited by threshold cell T and inhibited by T + 1.
#   magall:K : mag:D1-D2:G for every 1 <= D1 < D2 <= K and G = 1, 2.
#   D: 0 is the present byte, 1 the byte before it, and so on (delays along the stream, which runs on across lines).
#
#   python3 -I addsys.py <label> sys=byte,trace [seed=1] [order=0] [gate=0] [tuned=0] [wjit=0] [maxn=9] [held=20] [items=60000]
import json
import os
import sys

import numpy as np

label = sys.argv[1]; kw = dict(a.split('=', 1) for a in sys.argv[2:])
SEED = int(kw.get('seed', 1)); ORDER = int(kw.get('order', 0)); MAXN = int(kw.get('maxn', 9)); HELD = int(kw.get('held', 20))
NIT = int(kw.get('items', 60000)); GATE = int(kw.get('gate', 0)); TUNED = int(kw.get('tuned', 0)); PARTS = kw.get('sys', 'byte,trace').split(',')
_P = []
for _p in PARTS:                                       # magall:K = mag:D1-D2:G for every 1 <= D1 < D2 <= K and gates 1 and 2: delays not chosen
    if _p.startswith('magall:'):
        K = int(_p[7:]); _P += [f'mag:{d1}-{d2}:{g}' for d1 in range(1, K + 1) for d2 in range(d1 + 1, K + 1) for g in (1, 2)]
    else:
        _P.append(_p)
PARTS = _P
rng = np.random.default_rng(SEED)
pairs = [(a, b) for a in range(MAXN + 1) for b in range(MAXN + 1)]; rng.shuffle(pairs); held = pairs[:HELD]; shown = pairs[HELD:]
if ORDER > 0:                                          # order 0: the stream of addtext.py for this seed; orders 1-4: other random orders
    rng = np.random.default_rng([SEED, ORDER])
HMAX = 8
hist = [10] * HMAX                                      # the last bytes, hist[0] the present byte
TR = [2.0, 4.0, 8.0]; tr = [np.zeros(256) for _ in TR]
fid = {}                                                # feature (cell) name -> column
MAXF = 60000
W = np.zeros((256, MAXF)); CONST = 0; fid['const'] = 0
seen = np.zeros(256, bool)
WJIT = float(kw.get('wjit', 0)); WJ = {}; _wr = np.random.default_rng(20261119)   # wjit: strengths 1 +- wjit, drawn once per part


def col(name):
    c = fid.get(name)
    if c is None:
        c = len(fid); fid[name] = c
        assert c < MAXF
    return c


def cells():
    """The active binary cells (columns) and the graded ones (columns, values) at the present byte."""
    act = [CONST]; gc = []; gv = []
    p = hist[0]
    for part in PARTS:
        if part == 'byte':
            act.append(col(('b', p)))
        elif part == 'trace':
            for j, k in enumerate(TR):
                nz = np.flatnonzero(tr[j])
                for v in nz:
                    gc.append(col(('t', k, int(v)))); gv.append(tr[j][v] * np.sqrt(2 * k - 1))
        elif part.startswith('pair:'):
            d1, d2 = (int(x) for x in part[5:].split('-'))
            act.append(col(('p', d1, d2, hist[d1], hist[d2]) + tuple(hist[:GATE])))
        elif part.startswith('mag:'):
            spec = part[4:].split(':'); d1, d2 = (int(x) for x in spec[0].split('-'))
            G = int(spec[1]) if len(spec) > 1 else GATE  # mag:D1-D2:G gives this part its own gate
            w1, w2 = WJ.setdefault(part, tuple(1.0 + WJIT * (2 * _wr.random(2) - 1)))
            m = int(np.floor(w1 * hist[d1] + w2 * hist[d2]))   # wjit=0: unit strengths, m the plain sum
            if TUNED:                                   # tuned=1: the cell for T fires only when the magnitude is T (excited by threshold
                act.append(col(('m', d1, d2, G, m) + tuple(hist[:G])))   # cell T, inhibited by threshold cell T + 1)
            else:
                for T in range(20, m + 1):              # threshold cells T = 20 .. (2 x the largest byte of the text) that fire: T <= m
                    act.append(col(('m', d1, d2, G, T) + tuple(hist[:G])))
        else:
            raise SystemExit('unknown part ' + part)
    return act, gc, gv


def push(byte):
    hist.insert(0, byte); del hist[HMAX:]
    x = np.zeros(256); x[byte] = 1.0
    for j, k in enumerate(TR):
        tr[j] += (x - tr[j]) / k


def run(ps, n_items, learn):
    ans = []; oth = []; whole = []
    for _ in range(n_items):
        a, b = ps[rng.integers(len(ps))]; line = f"{a}+{b}={a + b}\n".encode(); eq = line.index(b'='); allok = True
        for t in range(len(line) - 1):
            push(line[t]); seen[line[t]] = True; yy = line[t + 1]
            act, gc, gv = cells()
            act = np.array(act); gc = np.array(gc, int); gv = np.array(gv)
            est = W[:, act].sum(1) + (W[:, gc] @ gv if len(gc) else 0.0)
            n = float(len(act) + (gv @ gv if len(gv) else 0.0))
            e = int(est.argmax())
            if learn and rng.random() < 1 / 16:
                sv = np.flatnonzero(seen); e = int(sv[rng.integers(len(sv))])
            ok = e == yy
            if t >= eq and yy != 10:
                ans.append(ok); allok = allok and ok
            else:
                oth.append(ok)
            if learn:
                if ok:
                    d = -est / 16; d[e] = (1.0 - est[e]) / 16
                else:
                    d = np.zeros(256); d[e] = (0.0 - est[e]) / 16
                nz = np.flatnonzero(d)
                if len(nz):
                    W[np.ix_(nz, act)] += (d[nz] / n)[:, None]
                    if len(gc):
                        W[np.ix_(nz, gc)] += np.outer(d[nz] / n, gv)
        push(line[-1])                                  # the end mark is followed by the first byte of the next line: not scored
        whole.append(allok)
    return float(np.mean(ans)), float(np.mean(oth)), float(np.mean(whole))


rows = []
for blk in range(6):
    rows.append(('learning block %d' % (blk + 1),) + run(shown, NIT // 6, True))
rows.append(('sums shown, no learning',) + run(shown, 2000, False))
rows.append(('sums NEVER shown',) + run(held, 2000, False))
# per held pair: whole answer right on a fresh pass of each held pair 20 times (no learning), and whether its sum occurs among shown pairs
per = {}
shown_sums = sorted({a + b for a, b in shown})
for a, b in held:
    r = run([(a, b)], 20, False)[2]
    per[f'{a}+{b}'] = dict(whole=r, sum_shown=(a + b) in shown_sums)
os.makedirs('res', exist_ok=True)
out = dict(label=label, args=kw, rows=rows, cells=len(fid) - 1, held=[list(p) for p in held], per_held=per)
json.dump(out, open(f'res/{label}.json', 'w'), indent=1)
print(f"== {label} sys={','.join(PARTS)} gate={GATE} tuned={TUNED} seed={SEED} order={ORDER}: cells used {len(fid) - 1}; "
      f"answer bytes % | other bytes % | whole answer %")
for r_ in rows:
    print(f"   {r_[0]:26s} {100 * r_[1]:5.1f} | {100 * r_[2]:5.1f} | {100 * r_[3]:5.1f}")
nos = [k for k, v in per.items() if not v['sum_shown']]
print('   held pairs whose sum is never shown: ' + (', '.join(nos) if nos else 'none'))
