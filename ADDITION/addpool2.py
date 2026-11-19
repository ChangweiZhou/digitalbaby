# E57's worker, 19 November 2026: the reviewer's task of 15:42 UTC (items 1-3). addpool2.py: addpool.py with general gates and
# inhibitory second connections (pool=fixed:D1-D2-G[-pm], G = 1, 2 or g<positions>), for item 3. Own code; the text, the score and rule Q as in addsys.py
# (which follows the reviewer's addtext.py). This program has a POOL of magnitude cells and can change it while it learns.
#
# Text: lines "a+b=s\n" (op=+) or, with op=+-, also "a-b=d\n" with a >= b (results never negative), the two mixed at random line by line.
#   Held out of learning: 20 of the 100 sum pairs; and 11 of the 55 difference pairs. Score: the whole answer (every byte after '=' up
#   to the end mark right), for each operation, on pairs shown (no learning) and on pairs never shown.
# Cells:
#   byte cells: one a byte value, firing when that byte is the present byte.
#   magnitude cell c: delays D1 < D2 (1 = the byte before the present one), strengths w1, w2, gate G (1 or 2); its activity is
#     w1 x(D1) + w2 x(D2), x the byte's value; its value-tuned cells fire when floor(activity) = T and the last G bytes (the present
#     byte, and with G = 2 the byte before it) are as they are: one tuned cell for each (T, those bytes) that ever occurs.
#   output units: rule Q (eta 1/16, z1, exploration 1/16 in learning) on the byte cells and every tuned cell.
# Pool:
#   pool=fixed:D1-D2-G,... : the cells given; pool=N : N cells drawn at random (D1 < D2 from 1 to dmax, G 1 or 2; default_rng([seed, order, 7])).
# Item 1, strengths: w0=a-b: each strength drawn uniformly from a to b at the start (default 1-1: both 1).
#   eq=drive : each connection keeps a running mean of its drive, r_i += (w_i x_i - r_i) / tau (tau=4096 steps by default); after each step
#              w_i *= 1 + eta_w (rbar / r_i - 1), rbar the mean of the cell's two r_i (eta_w=1/1024 by default); then w1 + w2 is scaled back to its
#              value at the start (the cell keeps its total strength). Uses only the cell's own two inputs and strengths.
#              At every byte, the end mark included (sall=0: not at the end mark, as in runs B2 and B3, which leaves a bias).
# Item 2, withdrawal: every `every` lines (default 1000), among the cells older than 2 x every lines, the share `frac` (default 1/8) with
#   the lowest keep score is withdrawn: its tuned cells and their weights go, and it is remade with new random delays and gate.
#   keep=w  : score = the summed absolute weight from its tuned cells to the output units (how strong its connection to the answer is).
#   keep=wb : that score times the cell's breadth, the mean number of firings per tuned cell that fired since it was made (a cell that adds
#             fires a tuned cell for several pairs; a table cell for one).
#
#   python3 -I addpool.py <label> [seed=1] [order=0] [op=+] [pool=fixed:1-3-1,2-4-2 | pool=300] [dmax=6] [w0=1-1] [eq=none|drive]
#                        [keep=none|w|wb] [every=1000] [frac=1/8] [items=60000]
import json
import os
import sys
import time

import numpy as np

label = sys.argv[1]; kw = dict(a.split('=', 1) for a in sys.argv[2:])
SEED = int(kw.get('seed', 1)); ORDER = int(kw.get('order', 0)); OPS = kw.get('op', '+'); NIT = int(kw.get('items', 60000))
POOL = kw.get('pool', 'fixed:1-3-1,2-4-2'); DMAX = int(kw.get('dmax', 6)); W0 = [float(x) for x in kw.get('w0', '1-1').split('-')]
EQ = kw.get('eq', 'none'); KEEP = kw.get('keep', 'none'); EVERY = int(kw.get('every', 1000)); FRAC = float(eval(kw.get('frac', '1/8')))
BIN = 0.5 if kw.get('bin', 'floor') == 'round' else 0.0; WTOT = float(kw['wtot']) if 'wtot' in kw else None; STRENGTH_ALL = int(kw.get('sall', 1)); TAU = float(kw.get('tau', 4096)); ETA_W = float(eval(kw.get('eta_w', '1/1024')))
t0 = time.process_time()
rng = np.random.default_rng(SEED)
pairs_add = [(a, b) for a in range(10) for b in range(10)]; rng.shuffle(pairs_add)
held = {'+': pairs_add[:20]}; shown = {'+': pairs_add[20:]}
if '-' in OPS:
    rs = np.random.default_rng([SEED, 5]); pairs_sub = [(a, b) for a in range(10) for b in range(a + 1)]; rs.shuffle(pairs_sub)
    held['-'] = pairs_sub[:11]; shown['-'] = pairs_sub[11:]
if ORDER > 0:
    rng = np.random.default_rng([SEED, ORDER])
prng = np.random.default_rng([SEED, ORDER, 7])

UNITS = np.array(sorted(set(b'0123456789+-=\n')))       # the byte values of the text; every other unit's output stays 0 under rule Q
NU = len(UNITS); uidx = {int(u): i for i, u in enumerate(UNITS)}
HMAX = max(DMAX, 4) + 3
hist = np.full(HMAX, 10, np.int64)                        # hist[0] the present byte

# pool
if POOL.startswith('fixed:'):
    spec = []
    for s_ in POOL[6:].split(','):                     # D1-D2-G[-pm]: G = 1, 2 (the last G bytes) or g<positions>, e.g. g02 = the present
        f_ = s_.split('-')                              # byte and the byte 2 back; pm: the second connection inhibitory (w2 < 0)
        gp = tuple(range(int(f_[2]))) if f_[2].isdigit() else tuple(int(ch) for ch in f_[2][1:])
        spec.append((int(f_[0]), int(f_[1]), gp, -1.0 if len(f_) > 3 and f_[3] == 'pm' else 1.0))
else:
    spec = [None] * int(POOL)
M = len(spec)
D1 = np.zeros(M, np.int64); D2 = np.zeros(M, np.int64); G = np.zeros(M, np.int64); GEN = np.zeros(M, np.int64)
GP = np.full((M, 3), HMAX, np.int64); SG = np.ones(M); GNAME = [''] * M      # gate positions (HMAX: unused, reads 0); sign of the second connection
WS = np.zeros((M, 2)); R = np.zeros((M, 2)); WSUM = np.zeros(M); BORN = np.zeros(M, np.int64)
gen_counter = [0]


def draw(c, s=None):
    if s is None:
        d1, d2 = sorted(prng.choice(np.arange(1, DMAX + 1), 2, replace=False)); g = tuple(range(int(prng.integers(1, 3)))); sg = 1.0
    else:
        d1, d2, g, sg = s
    D1[c], D2[c] = d1, d2; GP[c] = HMAX; GP[c, :len(g)] = g; SG[c] = sg
    GNAME[c] = f"{d1}-{d2}-g{''.join(str(x) for x in g)}" + ('-pm' if sg < 0 else '')
    WS[c] = prng.uniform(W0[0], W0[1], 2) if W0[0] != W0[1] else W0[0]
    WSUM[c] = WS[c].sum() if WTOT is None else WTOT; R[c] = 0.0     # wtot=2: every cell's total strength is 2 (its two strengths still differ)
    WS[c] *= WSUM[c] / WS[c].sum()
    gen_counter[0] += 1; GEN[c] = gen_counter[0]


for c in range(M):
    draw(c, spec[c])

# features: rows of W (feature x unit); row 0 the constant; rows 1..256 the byte cells; tuned cells get rows as they first fire
MAXR = max(1200000, 3000 * M + 1000)
W = np.zeros((MAXR, NU)); free = list(range(MAXR - 1, 256, -1))
row_of = {}                                               # code -> row
rows_of_cell = [dict() for _ in range(M)]                 # cell -> {row: firings}
codes_of_cell = [[] for _ in range(M)]
seen = np.zeros(256, bool)


def tuned_rows():
    x1 = hist[D1].astype(float); x2 = hist[D2].astype(float)
    m = np.floor(WS[:, 0] * x1 + SG * WS[:, 1] * x2 + BIN).astype(np.int64)     # bin=round: the tuned cell's band centred on T
    he = np.append(hist, 0)
    ctx = he[GP[:, 0]] + 256 * he[GP[:, 1]] + 65536 * he[GP[:, 2]]
    codes = (GEN << 40) + (np.clip(m + 2048, 0, 4095) << 28) + ctx
    out = []
    for c, code in enumerate(codes.tolist()):
        r = row_of.get(code)
        if r is None:
            r = free.pop(); row_of[code] = r; codes_of_cell[c].append(code)
        d = rows_of_cell[c]; d[r] = d.get(r, 0) + 1
        out.append(r)
    return out, x1, x2


def learn_strengths(x1, x2):
    if EQ != 'drive':
        return
    drive = WS * np.stack([x1, x2], 1)
    R[:] += (drive - R) / TAU
    ok = (R > 0).all(1)
    rbar = R.mean(1, keepdims=True)
    WS[ok] *= 1 + ETA_W * (rbar[ok] / R[ok] - 1)
    WS[:] *= (WSUM / WS.sum(1))[:, None]


def withdraw(line_no):
    if KEEP == 'none' or M == 0:
        return 0
    old = np.flatnonzero(line_no - BORN >= 2 * EVERY)
    if len(old) == 0:
        return 0
    score = np.zeros(len(old))
    for j, c in enumerate(old):
        rows = list(rows_of_cell[c].keys())
        w = np.abs(W[rows]).sum() if rows else 0.0
        if KEEP == 'wb':
            w *= (sum(rows_of_cell[c].values()) / max(1, len(rows)))
        score[j] = w
    k = int(round(FRAC * len(old)))
    for c in old[np.argsort(score, kind='stable')[:k]]:
        for r in rows_of_cell[c]:
            W[r] = 0.0; free.append(r)
        for code in codes_of_cell[c]:
            del row_of[code]
        rows_of_cell[c] = {}; codes_of_cell[c] = []
        draw(c); BORN[c] = line_no
    return k


line_count = [0]; withdrawn = [0]


def push(byte):
    hist[1:] = hist[:-1]; hist[0] = byte


def run(op_pairs, n_items, learn):
    res = {op: dict(whole=[]) for op in op_pairs}
    ops = list(op_pairs)
    for _ in range(n_items):
        op = ops[int(rng.integers(len(ops)))] if len(ops) > 1 else ops[0]
        ps = op_pairs[op]; a, b = ps[rng.integers(len(ps))]
        line = f"{a}{op}{b}={a + b if op == '+' else a - b}\n".encode(); eq = line.index(b'='); allok = True
        for t in range(len(line) - 1):
            push(line[t]); seen[line[t]] = True; yy = line[t + 1]
            rows, x1, x2 = tuned_rows()
            act = [0, 1 + int(line[t])] + rows
            est_u = W[act].sum(0)
            est = np.zeros(256); est[UNITS] = est_u
            e = int(est.argmax())
            if learn and rng.random() < 1 / 16:
                sv = np.flatnonzero(seen); e = int(sv[rng.integers(len(sv))])
            ok = e == yy
            if t >= eq and yy != 10:
                allok = allok and ok
            if learn:
                n = float(len(act))
                if ok:
                    d = -est_u / 16; d[uidx[e]] = (1.0 - est_u[uidx[e]]) / 16
                elif e in uidx:
                    d = np.zeros(NU); d[uidx[e]] = (0.0 - est_u[uidx[e]]) / 16
                else:
                    d = None
                if d is not None:
                    W[act] += d / n                       # rows may repeat only if two cells share a code: they do not (GEN differs)
                learn_strengths(x1, x2)
        push(line[-1])
        if learn and EQ != 'none' and STRENGTH_ALL:          # the cell sees every byte, the end mark too (before 16:20 UTC: not the end mark)
            learn_strengths(hist[D1].astype(float), hist[D2].astype(float))
        res[op]['whole'].append(allok)
        if learn:
            line_count[0] += 1
            if KEEP != 'none' and line_count[0] % EVERY == 0:
                withdrawn[0] += withdraw(line_count[0])
    return {op: float(np.mean(v['whole'])) for op, v in res.items()}


rows_out = []
for blk in range(6):
    rows_out.append(('learning block %d' % (blk + 1), run(shown, NIT // 6, True)))
rows_out.append(('shown, no learning', run(shown, 2000, False)))
rows_out.append(('NEVER shown', run(held, 2000, False)))
per = {}
for op in held:
    sums_shown = {(a + b if op == '+' else a - b) for a, b in shown[op]}
    for a, b in held[op]:
        r = run({op: [(a, b)]}, 20, False)[op]
        per[f'{a}{op}{b}'] = dict(whole=r, result_shown=(a + b if op == '+' else a - b) in sums_shown)
# the pool at the end: kinds of cells, with their keep scores
kinds = {}
for c in range(M):
    rows = list(rows_of_cell[c].keys()); w = float(np.abs(W[rows]).sum()) if rows else 0.0
    k = GNAME[c]
    kinds.setdefault(k, dict(n=0, weight=0.0)); kinds[k]['n'] += 1; kinds[k]['weight'] += w
out = dict(label=label, args=kw, rows=rows_out, per_held=per, pool_kinds=kinds, withdrawn=withdrawn[0],
           strengths=dict(first=WS[:min(M, 8)].tolist(), largest_relative_difference=float(np.abs(WS[:, 0] - WS[:, 1]).max() / WS.mean()) if M else None),
           tuned_cells=len(row_of), cpu_s=time.process_time() - t0)
os.makedirs('res', exist_ok=True)
json.dump(out, open(f'res/{label}.json', 'w'), indent=1)
print(f"== {label} {' '.join(sys.argv[2:])}: cpu {out['cpu_s']:.0f} s; tuned cells {len(row_of)}; withdrawn {withdrawn[0]}")
for nm, r in rows_out:
    print(f"   {nm:22s} " + '  '.join(f"{op} whole {100 * v:5.1f}" for op, v in r.items()))
top = sorted(kinds.items(), key=lambda kv: -kv[1]['weight'])[:6]
print('   pool, strongest kinds (D1-D2-G: cells, weight): ' + '; '.join(f"{k}: {v['n']}, {v['weight']:.1f}" for k, v in top))
if M:
    print(f"   strengths: largest |w1 - w2| / mean w = {out['strengths']['largest_relative_difference']:.4f}")
