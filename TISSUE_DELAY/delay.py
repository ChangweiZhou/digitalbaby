# Delays in the tissue that carries the present byte far (the reviewer's message of 19 November 2026, 06:42 UTC). One tissue of tissue.c
# (the reviewer's program at 6d805e04, copied; additions marked "TISSUE_DELAY").
#
#   python3 -I delay.py <label> rule=timing norm=1 lines_learn=1 eta_s=1 no_make=1 [we=1 wi=1/4] [seed=..] [nbytes=16384]
#                       [cand=none|rand|share|dist] [dmax=32] [share=1/4] [delay=32] [windows=1]
#
#   cand=rand   every connection between cells delayed by an integer drawn uniformly from 0 to dmax (numpy default_rng(20261119))
#   cand=share  a share of the connections between cells, drawn once by default_rng(20261119), delayed by `delay` steps
#   cand=mix    random 0 to dmax on every connection between cells, then `delay` on a share of them (drawn after)
#   cand=dist   every connection between cells delayed by round(dmax x (distance of its two cells' coordinates) / (1/8)) steps
#
# Readings, in each window of 4,096 bytes (windows=1: the last 4,096 bytes; windows=4: the last 4,096 bytes of each quarter of the run),
# for every distance d (from the lines of the byte values present in the window, along the connections present at its end), for the pool
# of all cells that are not feedback cells, and for the cells near the output lines (the 256 drivers of the output lines and every cell
# in their neighbourhoods, feedback cells left out: reach.py's set, 5ddfbaad on claude/e55-output-reach-017dSEpW), lags 0 to 4:
#   (a) S by the readings tool as it is (outputs below 0.01 set to 0); (b) no floor, each cell divided by its own standard deviation over
#   the window (the helper's where.py, 4bda7aa1); at most 128 cells sampled by Random(seed + 900000).sample(sorted(cells), 128);
#   the mean and the largest raw output; c0 of the window.
# Also: connections a cell (cells that are not feedback cells); divergence: two copies, the second with the potential of its most active
# cell changed by 1e-9 at byte NB - 4096 - 512; the largest difference of outputs at the reading step after 1, 4, 16, 64, 256, 512 bytes.
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
cand = kw.pop('cand', 'none'); dmax = int(kw.pop('dmax', '32')); share = float(eval(kw.pop('share', '1/4')))
delay = int(kw.pop('delay', '32')); nwin = int(kw.pop('windows', '1'))
LAGS = (0, 1, 2, 3, 4)
WL = 4096


def make():
    T = tis.make(kw)
    t, A, B, N = T['t'], T['A'], T['B'], T['N']
    rng = np.random.default_rng(20261119)
    maxd = A['c_src'].shape[1]
    valid = np.arange(maxd)[None, :] < A['deg'][:, None]
    cc = np.argwhere(valid & (A['c_src'] < N))
    dl = None
    if cand == 'rand':
        dl = np.zeros((N, maxd), np.int32); dl[cc[:, 0], cc[:, 1]] = rng.integers(0, dmax + 1, len(cc))
    elif cand == 'share':
        pick = cc[rng.choice(len(cc), int(round(share * len(cc))), replace=False)]
        dl = np.zeros((N, maxd), np.int32); dl[pick[:, 0], pick[:, 1]] = delay
    elif cand == 'mix':                                             # random 0..dmax on every connection, then `delay` on a share of them
        dl = np.zeros((N, maxd), np.int32); dl[cc[:, 0], cc[:, 1]] = rng.integers(0, dmax + 1, len(cc))
        pick = cc[rng.choice(len(cc), int(round(share * len(cc))), replace=False)]
        dl[pick[:, 0], pick[:, 1]] = delay
    elif cand == 'dist':
        P = np.array(B['P']); s = A['c_src'][cc[:, 0], cc[:, 1]].astype(np.int64)
        dxy = np.sqrt(((P[cc[:, 0]] - P[s]) ** 2).sum(1))
        dl = np.zeros((N, maxd), np.int32); dl[cc[:, 0], cc[:, 1]] = np.rint(dmax * dxy / (1 / 8)).astype(np.int32)
    else:
        assert cand == 'none'
    T['extra'] = {}
    if dl is not None:
        hl = int(dl.max()) + 2
        zh = np.zeros((hl, N)); Fh = np.zeros((hl, N))
        T['extra'] = dict(c_del=dl, zh=zh, Fh=Fh)
        t.c_del = dl.ctypes.data; t.zh = zh.ctypes.data; t.Fh = Fh.ctypes.data; t.hlen = hl
        T['delay_stats'] = dict(mean_delay=float(dl[cc[:, 0], cc[:, 1]].mean()), connections_delayed=int((dl[cc[:, 0], cc[:, 1]] > 0).sum()),
                                connections_between_cells=int(len(cc)), max_delay=int(dl.max()))
    return T


def run(T, lo, hi, rec_lo=None):
    t, L, b = T['t'], T['L'], T['b']
    n = 0 if rec_lo is None else hi - rec_lo
    rec = np.zeros((max(n, 1), T['N']), np.float32)
    r0 = hi if rec_lo is None else rec_lo
    L.tis_run(ctypes.byref(t), ctypes.c_void_p(b.ctypes.data), ctypes.c_int64(lo), ctypes.c_int64(hi), ctypes.c_int64(r0), ctypes.c_int64(hi),
              ctypes.c_void_p(rec.ctypes.data))
    return rec


def distances(T, lo, hi):
    b, A, B, N = T['b'], T['A'], T['B'], T['N']
    lines = set(ts.LINE0 + int(v) for v in set(b[lo:hi].tolist()))
    deg = A['deg']; src = A['c_src']; isfb = B['kind'] == ts.FBK
    dist = np.zeros(N, np.int16)
    fr = np.array([(not isfb[i]) and any(int(s) in lines for s in src[i, :deg[i]]) for i in range(N)]); dist[fr] = 1; d = 1
    while fr.any():
        nxt = np.array([(dist[i] == 0) and (not isfb[i]) and bool(fr[src[i, :deg[i]][src[i, :deg[i]] < N]].any()) for i in range(N)])
        d += 1; dist[nxt] = d; fr = nxt
    return dist


def read(T, rec, lo, hi, dist, near):
    b, B, N, SEED = T['b'], T['B'], T['N'], T['SEED']
    isfb = B['kind'] == ts.FBK
    X = rec.astype(np.float64); Xa = np.where(X < 0.01, 0.0, X)
    sd = X.std(axis=0); Xb = np.where(sd > 0, X / np.where(sd > 0, sd, 1.0), 0.0)
    wl = hi - lo

    def score(F, cells, L_):
        if not cells:
            return None
        idx, lab = RD._samples(wl, L_, b[lo:hi], b, lo)
        return RD.classify(F[idx][:, cells], lab, idx, wl)[0]

    def sample(cells):
        cells = sorted(int(c) for c in cells)
        return Random(SEED + 900000).sample(cells, 128) if len(cells) > 128 else cells
    groups = [(str(dd), np.nonzero(dist == dd)[0]) for dd in range(1, int(dist.max()) + 1)] + [('pool', np.nonzero(~isfb)[0]), ('near', near)]
    rows = []
    for name, cells in groups:
        if len(cells) == 0:
            continue
        samp = sample(cells); samp_b = [c for c in samp if sd[c] > 0]
        rows.append(dict(d=name, cells=int(len(cells)), mean_output=float(X[:, cells].mean()), max_output=float(X[:, cells].max()),
                         share_above_001=float((X[:, cells] > 0.01).mean()),
                         a={str(L_): score(Xa, samp, L_) for L_ in LAGS}, b={str(L_): score(Xb, samp_b, L_) for L_ in LAGS}))
    c0 = {str(k): v['S'] for k, v in RD.c0(b[lo:hi], b, lo, LAGS).items()}
    near_dist = {str(k): int(v) for k, v in zip(*np.unique(dist[near], return_counts=True))}
    return dict(window=[lo, hi - 1], rows=rows, c0=c0, max_distance=int(dist.max()), unreached=int(((dist == 0) & ~isfb).sum()), near_by_distance=near_dist)


c0t = time.process_time()
T1 = make(); T2 = make(); NB, N = T1['NB'], T1['N']
B = T1['B']; isfb = B['kind'] == ts.FBK
near = set(int(i) for i in B['driver'])
for i in B['driver']:
    near.update(int(m) for m in B['members'][i] if m < N)
near = np.array(sorted(i for i in near if not isfb[i]))
P0 = NB - WL - 512
wins = [((q + 1) * NB // nwin - WL, (q + 1) * NB // nwin) for q in range(nwin)]
out = dict(label=label, args=kw, cand=cand, dmax=dmax if cand in ('rand', 'dist', 'mix') else None, share=share if cand in ('share', 'mix') else None,
           delay=delay if cand in ('share', 'mix') else None, seed=T1['SEED'], nbytes=NB, windows=[], near_cells=int(len(near)))
pos = 0
for (lo, hi) in wins:
    if lo <= P0 < hi or (pos <= P0 < lo):                          # the divergence is taken before the last window
        pass
    if hi == NB:                                                    # last window: the two copies from P0
        run(T1, pos, P0); run(T2, 0, P0)
        assert np.array_equal(T1['A']['v'], T2['A']['v'])
        j = int(np.argmax(T2['A']['xp'])); T2['A']['v'][j] += 1e-9
        d1 = run(T1, P0, P0 + 512, P0); d2 = run(T2, P0, P0 + 512, P0)
        diff = np.abs(d1.astype(np.float64) - d2).max(axis=1)
        out['divergence'] = {str(n): float(diff[n - 1]) for n in (1, 4, 16, 64, 256, 512)}
        del T2
        pos = P0 + 512
    rec = run(T1, pos, hi, lo)
    pos = hi
    dist = distances(T1, lo, hi)
    w = read(T1, rec, lo, hi, dist, near)
    A = T1['A']
    w['connections_a_cell'] = float(A['deg'][~isfb].mean())
    w['largest_output_by_distance'] = {str(dd): float(rec[:, dist == dd].max()) for dd in range(1, int(dist.max()) + 1) if (dist == dd).any()}
    out['windows'].append(w)
out['delay_stats'] = T1.get('delay_stats')
out['cpu_s'] = time.process_time() - c0t
os.makedirs(os.path.join(HERE, 'delay'), exist_ok=True)
json.dump(out, open(os.path.join(HERE, 'delay', f'{label}.json'), 'w'), indent=1, default=float)
w = out['windows'][-1]
g = {r['d']: r for r in w['rows']}
f = lambda v: '  —  ' if v is None else f'{v:.3f}'
print(f"{label:28s} cpu {out['cpu_s']:4.0f}s conns {w['connections_a_cell']:.1f} div64/512 {out['divergence']['64']:.1e}/{out['divergence']['512']:.1e} | "
      f"(a) S0 d1/2/4 " + ' '.join(f(g[d]['a']['0']) if d in g else '  —  ' for d in ('1', '2', '4'))
      + ' | (b) best lag1..3 ' + ' '.join(f(max((r['b'][L] or 0) for r in w['rows'] if r['d'] not in ('pool', 'near'))) for L in ('1', '2', '3'))
      + ' | near (b) lag1..3 ' + ' '.join(f(g['near']['b'][L]) for L in ('1', '2', '3')), flush=True)
