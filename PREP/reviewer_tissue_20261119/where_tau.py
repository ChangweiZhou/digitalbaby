# where_tau.py: where.py on tissue_tau.c (through tis_tau.py), with the divergence of two copies after 512 bytes (diag.py's method).
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
import tis_tau as tis
import readings_tool as RD
label = sys.argv[1]; kw = dict(a.split('=') for a in sys.argv[2:])
LAGS = (0, 1, 2, 3, 4)
t0 = time.process_time()
T = tis.make(kw); ts = tis.ts; N, NB, SEED, b, A, B, L = T['N'], T['NB'], T['SEED'], T['b'], T['A'], T['B'], T['L']
lo, hi = NB - 4096, NB
rec = np.zeros((hi - lo, N), np.float32)
# the divergence of two copies (diag.py's method): a second tissue, one cell's potential changed by 1e-9 at byte NB - 4096 - 512
P0 = NB - 4096 - 512
T2 = tis.make(kw)
def run(TT, a, c, r0=None):
    n = 0 if r0 is None else c - r0
    rr = np.zeros((max(n, 1), N), np.float32)
    TT['L'].tis_run(ctypes.byref(TT['t']), ctypes.c_void_p(TT['b'].ctypes.data), ctypes.c_int64(a), ctypes.c_int64(c), ctypes.c_int64(c if r0 is None else r0), ctypes.c_int64(c), ctypes.c_void_p(rr.ctypes.data))
    return rr
run(T, 0, P0); run(T2, 0, P0)
assert np.array_equal(T['A']['v'], T2['A']['v'])
jj = int(np.argmax(T2['A']['xp'])); T2['A']['v'][jj] += 1e-9
d1 = run(T, P0, P0 + 512, P0); d2 = run(T2, P0, P0 + 512, P0)
divergence = {str(n): float(np.abs(d1[n - 1].astype(np.float64) - d2[n - 1]).max()) for n in (1, 4, 16, 64, 256, 512)}
del T2
L.tis_run(ctypes.byref(T['t']), ctypes.c_void_p(b.ctypes.data), ctypes.c_int64(P0 + 512), ctypes.c_int64(hi), ctypes.c_int64(lo), ctypes.c_int64(hi), ctypes.c_void_p(rec.ctypes.data))
run_cpu = time.process_time() - t0
lines = set(ts.LINE0 + int(v) for v in set(b[lo:hi].tolist()))
deg = A['deg']; src = A['c_src']; isfb = B['kind'] == ts.FBK
dist = np.zeros(N, np.int16)
fr = np.array([(not isfb[i]) and any(int(s) in lines for s in src[i, :deg[i]]) for i in range(N)]); dist[fr] = 1; d = 1
while fr.any():
    nxt = np.array([(dist[i] == 0) and (not isfb[i]) and bool(fr[src[i, :deg[i]][src[i, :deg[i]] < N]].any()) for i in range(N)])
    d += 1; dist[nxt] = d; fr = nxt
X = rec.astype(np.float64)
Xa = np.where(X < 0.01, 0.0, X)
sd = X.std(axis=0)
wl = hi - lo


def score(F, cells, L_):
    idx, lab = RD._samples(wl, L_, b[lo:hi], b, lo)
    if not cells:
        return None
    S, se, k = RD.classify(F[idx][:, cells], lab, idx, wl)
    return dict(S=S, se=se)


def sample(cells):
    cells = sorted(int(c) for c in cells)
    return Random(SEED + 900000).sample(cells, 128) if len(cells) > 128 else cells


Xb = np.where(sd > 0, X / np.where(sd > 0, sd, 1.0), 0.0)
groups = [(str(dd), np.nonzero(dist == dd)[0]) for dd in range(1, int(dist.max()) + 1)] + [("pool", np.nonzero(~isfb)[0]), ("unreached", np.nonzero((dist == 0) & ~isfb)[0])]
rows = []
for name, cells in groups:
    if len(cells) == 0:
        continue
    samp = sample(cells)
    samp_b = [c for c in samp if sd[c] > 0]
    row = dict(d=name, cells=int(len(cells)), sampled=len(samp), sampled_varying=len(samp_b), mean_output=float(X[:, cells].mean()), max_output=float(X[:, cells].max()),
               share_above_001=float((X[:, cells] > 0.01).mean()), a={}, b={})
    for L_ in LAGS:
        row["a"][str(L_)] = score(Xa, samp, L_)
        row["b"][str(L_)] = score(Xb, samp_b, L_)
    rows.append(row)
c0 = RD.c0(b[lo:hi], b, lo, LAGS)
out = dict(label=label, args=kw, divergence=divergence, tau_counts={str(k): int((T['A']['tauv'] == k).sum()) for k in sorted(set(T['A']['tauv'].tolist()))}, tau_by_distance={str(dd): float(T['A']['tauv'][dist == dd].mean()) for dd in range(1, int(dist.max()) + 1)}, seed=SEED, window=[lo, hi - 1], run_cpu_s=run_cpu, total_cpu_s=time.process_time() - t0, c0={str(k): v["S"] for k, v in c0.items()},
           max_distance=int(dist.max()), rows=rows, gain_median=float(np.median(A['gain'])), gain_at_max=int((A['gain'] >= T['t'].gmax).sum()))
os.makedirs(os.path.join(HERE, 'where_tau'), exist_ok=True)
json.dump(out, open(os.path.join(HERE, 'where_tau', label + '.json'), 'w'), default=float)
print(label, 'done', round(out['total_cpu_s'], 1), 's; max distance', out['max_distance'], flush=True)
