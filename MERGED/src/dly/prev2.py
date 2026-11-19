# Bytes one, two and more back at the reading step, on bases that cannot depend on their past (the reviewer's messages of 19 November 2026,
# 06:06 and 06:14 UTC). One tissue of tissue.c (the reviewer's program at d4250a84, copied; additions marked "TISSUE_PREV2").
#
#   python3 -I prev2.py <label> rule=fixed [gain=2 rho=1 gmax=4] [seed=..] [cand=none|delay|delaydist|delayrand] [share=1/4] [delay=8]
#                       [dmax=32] [full=1]
#
#   cand=delay      a share of the connections between cells, drawn once by numpy's default_rng(20261119), delayed by `delay` steps
#   cand=delaydist  every connection between cells delayed by round(dmax x (distance of its two cells' coordinates) / (1/8)) steps
#   cand=delayrand  every connection between cells delayed by an integer drawn uniformly from 0 to dmax (default_rng(20261119))
#   A delayed connection carries the sender's output of 1 + delay steps before (no delay: of the step before, as in the definition).
#
# Writes prev2/<label>.json:
#   S(d, L) for d = 1, 2, 4 and L = 0, 1, 2 with the readings tool as it is (as drive.py); c0; shares;
#   divergence: two copies, the second with the potential of its most active cell changed by 1e-9 at byte NB - 4096 - 512 (diag.py's
#   method); the largest difference of outputs at the reading step after 1, 4, 16, 64, 256, 512 bytes;
#   full=1: for every distance and for the pool of all cells that are not feedback cells, lags 0 to 4, (a) the tool as it is and (b) no
#   floor with each cell's output divided by its own standard deviation (the helper's where.py, 4bda7aa1, method copied here).
import ctypes
import json
import os
import sys
import time
from random import Random

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tis  # noqa: E402

ts, RD = tis.ts, tis.readings
label = sys.argv[1]; kw = dict(a.split('=') for a in sys.argv[2:])
cand = kw.pop('cand', 'none'); share = float(eval(kw.pop('share', '1/4'))); delay = int(kw.pop('delay', '8'))
dmax = int(kw.pop('dmax', '32')); full = int(kw.pop('full', '0'))
LEVEL = {'0': 0.1103, '1': 0.090, '2': 0.076}
LAGS = (0, 1, 2, 3, 4)


def make():
    T = tis.make(kw)
    t, A, B, N = T['t'], T['A'], T['B'], T['N']
    rng = np.random.default_rng(20261119)
    maxd = A['c_src'].shape[1]
    valid = np.arange(maxd)[None, :] < A['deg'][:, None]
    cc = np.argwhere(valid & (A['c_src'] < N))                      # connections between cells, (receiver, slot)
    dl = None
    if cand == 'delay':
        pick = cc[rng.choice(len(cc), int(round(share * len(cc))), replace=False)]
        dl = np.zeros((N, maxd), np.int32); dl[pick[:, 0], pick[:, 1]] = delay
    elif cand == 'delaydist':
        P = np.array(B['P'])
        s = A['c_src'][cc[:, 0], cc[:, 1]].astype(np.int64)
        dist_xy = np.sqrt(((P[cc[:, 0]] - P[s]) ** 2).sum(1))
        dl = np.zeros((N, maxd), np.int32); dl[cc[:, 0], cc[:, 1]] = np.rint(dmax * dist_xy / (1 / 8)).astype(np.int32)
    elif cand == 'delayrand':
        dl = np.zeros((N, maxd), np.int32); dl[cc[:, 0], cc[:, 1]] = rng.integers(0, dmax + 1, len(cc))
    else:
        assert cand == 'none'
    T['extra'] = {}
    if dl is not None:
        hl = int(dl.max()) + 2
        zh = np.zeros((hl, N))
        T['extra'] = dict(c_del=dl, zh=zh)
        t.c_del = dl.ctypes.data; t.zh = zh.ctypes.data; t.hlen = hl
        T['delay_stats'] = dict(connections_delayed=int((dl[cc[:, 0], cc[:, 1]] > 0).sum()), connections_between_cells=int(len(cc)),
                                mean_delay=float(dl[cc[:, 0], cc[:, 1]].mean()), max_delay=int(dl.max()))
    return T


def run(T, lo, hi, rec_lo=None):
    t, L, b = T['t'], T['L'], T['b']
    n = 0 if rec_lo is None else hi - rec_lo
    rec = np.zeros((max(n, 1), T['N']), np.float32)
    r0 = hi if rec_lo is None else rec_lo
    L.tis_run(ctypes.byref(t), ctypes.c_void_p(b.ctypes.data), ctypes.c_int64(lo), ctypes.c_int64(hi), ctypes.c_int64(r0), ctypes.c_int64(hi),
              ctypes.c_void_p(rec.ctypes.data))
    return rec


c0t = time.process_time()
T1 = make(); T2 = make(); NB, N, SEED = T1['NB'], T1['N'], T1['SEED']
P0 = NB - 4096 - 512
run(T1, 0, P0); run(T2, 0, P0)
assert np.array_equal(T1['A']['v'], T2['A']['v'])
j = int(np.argmax(T2['A']['xp'])); T2['A']['v'][j] += 1e-9
d1 = run(T1, P0, P0 + 512, P0); d2 = run(T2, P0, P0 + 512, P0)
diff = np.abs(d1.astype(np.float64) - d2).max(axis=1)
divergence = {str(n): float(diff[n - 1]) for n in (1, 4, 16, 64, 256, 512)}
del T2
lo, hi = NB - 4096, NB
rec = run(T1, P0 + 512, hi, lo)
cpu_run = time.process_time() - c0t
b, A, B = T1['b'], T1['A'], T1['B']
lines = set(ts.LINE0 + int(v) for v in set(b[lo:hi].tolist()))
deg = A['deg']; src = A['c_src']; isfb = B['kind'] == ts.FBK
dist = np.zeros(N, np.int16)
fr = np.array([(not isfb[i]) and any(int(s) in lines for s in src[i, :deg[i]]) for i in range(N)]); dist[fr] = 1; d = 1
while fr.any():
    nxt = np.array([(dist[i] == 0) and (not isfb[i]) and bool(fr[src[i, :deg[i]][src[i, :deg[i]] < N]].any()) for i in range(N)])
    d += 1; dist[nxt] = d; fr = nxt
X = rec.astype(np.float64)
x = np.where(X < 0.01, 0.0, X)
r = RD.read_window(x, b[lo:hi], dist, SEED, stream=b, start=lo, lags=(0, 1, 2))
S = {str(dd): {str(Lg): (r['scores'][dd]['by_lag'][Lg] or {}).get('S') for Lg in (0, 1, 2)} for dd in (1, 2, 4)}
out = dict(label=label, args=kw, cand=cand, share=share if cand == 'delay' else None, delay=delay if cand == 'delay' else None,
           dmax=dmax if cand in ('delaydist', 'delayrand') else None, delay_stats=T1.get('delay_stats'), seed=SEED,
           S=S, above_level={dd: {Lg: (v is not None and v >= LEVEL[Lg]) for Lg, v in row.items()} for dd, row in S.items()},
           c0={str(k): (v.get('S') if isinstance(v, dict) else v) for k, v in r['c0'].items()},
           share_informative={str(dd): r['shares'][dd]['share'] if r['shares'][dd]['n'] else None for dd in (1, 2, 4)},
           divergence=divergence, gain_median=float(np.median(A['gain'])), gain_at_max=int((A['gain'] >= T1['t'].gmax).sum()),
           max_distance=int(dist.max()), unreached=int(((dist == 0) & ~isfb).sum()))
if full:                                                            # the helper's where.py (4bda7aa1), its method copied
    sd = X.std(axis=0); wl = hi - lo
    Xb = np.where(sd > 0, X / np.where(sd > 0, sd, 1.0), 0.0)

    def score(F, cells, L_):
        idx, lab = RD._samples(wl, L_, b[lo:hi], b, lo)
        if not cells:
            return None
        s_, se, k = RD.classify(F[idx][:, cells], lab, idx, wl)
        return s_

    def sample(cells):
        cells = sorted(int(c) for c in cells)
        return Random(SEED + 900000).sample(cells, 128) if len(cells) > 128 else cells
    groups = [(str(dd), np.nonzero(dist == dd)[0]) for dd in range(1, int(dist.max()) + 1)] + [("pool", np.nonzero(~isfb)[0])]
    rows = []
    for name, cells in groups:
        if len(cells) == 0:
            continue
        samp = sample(cells); samp_b = [c for c in samp if sd[c] > 0]
        rows.append(dict(d=name, cells=int(len(cells)), mean_output=float(X[:, cells].mean()),
                         a={str(L_): score(x, samp, L_) for L_ in LAGS}, b={str(L_): score(Xb, samp_b, L_) for L_ in LAGS}))
    out['full'] = dict(rows=rows, c0={str(k): v['S'] for k, v in RD.c0(b[lo:hi], b, lo, LAGS).items()})
out['cpu_s'] = time.process_time() - c0t
os.makedirs(os.path.join(HERE, 'prev2'), exist_ok=True)
json.dump(out, open(os.path.join(HERE, 'prev2', f'{label}.json'), 'w'), indent=1, default=float)
f = lambda v, L: '  —  ' if v is None else (f'{v:.3f}*' if v >= LEVEL[L] else f'{v:.3f} ')
print(f"{label:26s} cpu {out['cpu_s']:4.0f}s | S0 " + ' '.join(f(S[dd]['0'], '0') for dd in ('1', '2', '4')) + ' | S1 '
      + ' '.join(f(S[dd]['1'], '1') for dd in ('1', '2', '4')) + ' | S2 ' + ' '.join(f(S[dd]['2'], '2') for dd in ('1', '2', '4'))
      + f" | div 64/512 {divergence['64']:.1e} {divergence['512']:.1e} | gain med {out['gain_median']:.2f}", flush=True)
