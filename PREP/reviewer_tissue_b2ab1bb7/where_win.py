# where_win.py: where_tau.py over several windows of 4,096 bytes (option windows=s1,s2,...; default 0,20480,40960,61440), each read as
# where_tau.py reads its one window, with C<L of each window; the divergence after 512 bytes at NB - 4608 as before.
# where_tau.py (reviewer folder at b2ab1bb7, its own tau option; taudraw=own replaces the draw with the helper's): where.py on tissue_tau.c (through tis_tau.py), with the divergence of two copies after 512 bytes (diag.py's method).
# Where, and at what amplitude, the tissue carries each of the last bytes (the reviewer's task of 19 November, 05:57 UTC).
# usage: python3 -I where.py <label> seed=.. rule=fixed [options of drive.py]
# One tissue of tissue.c (tis.make), 16,384 bytes; window A2 (bytes 12,288 to 16,383), the output of every cell after each byte.
# For every distance d from 1 to the largest (distances as drive.py computes them, from the lines of the byte values present in A2),
# and for the pool of all cells that are not feedback cells, lags 0..4:
#   (a) S(d, lag) by the readings tool as it is: outputs below 0.01 set to 0 (readings.scores)
#   (b) S(d, lag) with no floor and each cell's output divided by its own standard deviation: the raw outputs, each cell divided by its
#       standard deviation over the window (a cell whose outputs are all equal is left out), then the tool's classifier, which divides
#       each feature by its standard deviation in the fitting folds; the classifier is unchanged by a positive scale of a feature, so this
#       is the tool's classifier on the raw outputs without its floor and without its dropping of deviations below 1e-9 that the raw
#       amplitudes would trip. Same folds, the same sample of at most 128 cells: Random(seed + 900000).sample(sorted(cells), 128).
#   (c) C0(lag): the tool's present-byte reference
#   (d) the mean and the largest raw output of the cells at d over the window
import sys, os, json, ctypes, time
from random import Random
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import tis
import readings_tool as RD


def own_tau(spec, seed, B):
    """The helper's own draw (taudraw=own): Random(seed + 700000), one draw per cell in cell order, from {2, 4, ..., top}; for spec ending
    in 'e' a cell whose neighbourhood holds an input line keeps 2 (the helper's definition, which differs from the reviewer's: kind ENC)."""
    from random import Random
    top = int(spec[3:].rstrip('e')); ch = [2.0 ** k for k in range(1, top.bit_length())]
    r = Random(seed + 700000); tau = np.array([ch[r.randrange(len(ch))] for _ in range(tis.ts.N)])
    if spec.endswith('e'):
        for i in range(tis.ts.N):
            if any(tis.ts.LINE0 <= m < tis.ts.LINE0 + tis.ts.NIN for m in B['members'][i]):
                tau[i] = 2.0
    return tau


def make_kw(kw):
    TT = tis.make(kw)
    if kw.get('taudraw') == 'own':
        TT['A']['tauv'][:] = own_tau(kw['tau'], TT['SEED'], TT['B'])
    return TT
label = sys.argv[1]; kw = dict(a.split('=') for a in sys.argv[2:])
LAGS = (0, 1, 2, 3, 4)
t0 = time.process_time()
T = make_kw(kw); ts = tis.ts; N, NB, SEED, b, A, B, L = T['N'], T['NB'], T['SEED'], T['b'], T['A'], T['B'], T['L']
STARTS = [int(x) for x in kw.get('windows', '0,20480,40960,61440').split(',')]
WL = 4096
P0 = NB - 4096 - 512
def run(TT, a, c, r0=None):
    n = 0 if r0 is None else c - r0
    rr = np.zeros((max(n, 1), N), np.float32)
    TT['L'].tis_run(ctypes.byref(TT['t']), ctypes.c_void_p(TT['b'].ctypes.data), ctypes.c_int64(a), ctypes.c_int64(c), ctypes.c_int64(c if r0 is None else r0), ctypes.c_int64(c), ctypes.c_void_p(rr.ctypes.data))
    return rr
isfb = B['kind'] == ts.FBK


def distances(lo, hi):
    lines = set(ts.LINE0 + int(v) for v in set(b[lo:hi].tolist()))
    deg = A['deg']; src = A['c_src']
    dist = np.zeros(N, np.int16)
    fr = np.array([(not isfb[i]) and any(int(s) in lines for s in src[i, :deg[i]]) for i in range(N)]); dist[fr] = 1; d = 1
    while fr.any():
        nxt = np.array([(dist[i] == 0) and (not isfb[i]) and bool(fr[src[i, :deg[i]][src[i, :deg[i]] < N]].any()) for i in range(N)])
        d += 1; dist[nxt] = d; fr = nxt
    return dist


def sample(cells):
    cells = sorted(int(c) for c in cells)
    return Random(SEED + 900000).sample(cells, 128) if len(cells) > 128 else cells


def analyse(rec, lo, hi):
    dist = distances(lo, hi)
    X = rec.astype(np.float64); Xa = np.where(X < 0.01, 0.0, X); sd = X.std(axis=0); wl = hi - lo
    Xb = np.where(sd > 0, X / np.where(sd > 0, sd, 1.0), 0.0)
    def score(F, cells, L_):
        idx, lab = RD._samples(wl, L_, b[lo:hi], b, lo)
        if not cells:
            return None
        S, se, k = RD.classify(F[idx][:, cells], lab, idx, wl)
        return dict(S=S, se=se)
    groups = [(str(dd), np.nonzero(dist == dd)[0]) for dd in range(1, int(dist.max()) + 1)] + [("pool", np.nonzero(~isfb)[0]), ("unreached", np.nonzero((dist == 0) & ~isfb)[0])]
    rows = []
    for name, cells in groups:
        if len(cells) == 0:
            continue
        samp = sample(cells); samp_b = [c for c in samp if sd[c] > 0]
        row = dict(d=name, cells=int(len(cells)), sampled=len(samp), sampled_varying=len(samp_b), mean_output=float(X[:, cells].mean()), max_output=float(X[:, cells].max()),
                   share_above_001=float((X[:, cells] > 0.01).mean()), cells_with_output=int((X[:, cells] > 0).any(axis=0).sum()), cells_above_001=int((X[:, cells] > 0.01).any(axis=0).sum()), a={}, b={})
        for L_ in LAGS:
            row["a"][str(L_)] = score(Xa, samp, L_); row["b"][str(L_)] = score(Xb, samp_b, L_)
        rows.append(row)
    c0 = RD.c0(b[lo:hi], b, lo, LAGS)
    clt = RD.c_lt(b[lo:hi], b, lo, (1, 2, 3, 4))
    return dict(window=[lo, hi - 1], rows=rows, c0={str(k): v["S"] for k, v in c0.items()}, c_lt={str(k): v["S"] for k, v in clt.items()}, max_distance=int(dist.max()),
                conns_per_cell=float(A['deg'].sum()) / N, conns_max=int(A['deg'].max()), made=int(T['t'].made), given=int(T['t'].given))


pos = 0; wins = {}; divergence = None
for ws in sorted(STARTS) + [None]:
    target = ws if ws is not None else NB
    if divergence is None and P0 <= target:
        run(T, pos, P0); pos = P0
        T2 = make_kw(kw); run(T2, 0, P0)
        assert np.array_equal(A['v'], T2['A']['v'])
        jj = int(np.argmax(T2['A']['xp'])); T2['A']['v'][jj] += 1e-9
        d1 = run(T, P0, P0 + 512, P0); d2 = run(T2, P0, P0 + 512, P0); pos = P0 + 512
        divergence = {str(n): float(np.abs(d1[n - 1].astype(np.float64) - d2[n - 1]).max()) for n in (1, 4, 16, 64, 256, 512)}
        del T2
        assert ws is None or ws >= pos, "a window overlaps the divergence check"
    if ws is None:
        break
    if pos < ws:
        run(T, pos, ws); pos = ws
    rec = run(T, ws, ws + WL, ws); pos = ws + WL
    wins[str(ws)] = analyse(rec, ws, ws + WL)
    print(label, 'window', ws, 'done', flush=True)
last = wins[str(max(STARTS))]
out = dict(label=label, args=kw, divergence=divergence, windows=wins, rows=last['rows'], c0=last['c0'], conns_per_cell=last['conns_per_cell'], conns_max=last['conns_max'],
           made=last['made'], given=last['given'], max_distance=last['max_distance'], seed=SEED, total_cpu_s=time.process_time() - t0,
           tau_draw=('own Random(seed + 700000)' if kw.get('taudraw') == 'own' else 'reviewer default_rng(seed + 20261119)' if kw.get('tau') else 'none'),
           tau_counts={str(k): int((A['tauv'] == k).sum()) for k in sorted(set(A['tauv'].tolist()))})
os.makedirs(os.path.join(HERE, 'where_tau'), exist_ok=True)
json.dump(out, open(os.path.join(HERE, 'where_tau', label + '.json'), 'w'), default=float)
print(label, 'done', round(out['total_cpu_s'], 1), 's', flush=True)
