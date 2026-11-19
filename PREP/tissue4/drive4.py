# A second driver of the reviewer's tissue.c (copied unchanged from 3402ce3b beside this file), for the task "a rule of making and
# withdrawing". It uses tis.make(kw) (the reviewer's set-up, unchanged) and the helper's readings tool as it is (deps/), and reads more
# than drive.py: S(d, L) and the share at d = 1, 2, 4, 8, 10; connections a cell (mean and largest) at the end of every window; made,
# withdrawn, withdrawn by use. For a life of more than 16,384 bytes it reads four windows: the last 4,096 bytes of each quarter.
#   python3 -I drive4.py <label> <options as drive.py> [nbytes=..]
import sys, os, time, json, ctypes, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tis
from tis import ts, readings

DS = (1, 2, 4, 8, 10)
label = sys.argv[1]; kw = dict(a.split('=') for a in sys.argv[2:])
M = tis.make(kw)
t, A, B, L, b, N, NB, SEED = (M[k] for k in ('t', 'A', 'B', 'L', 'b', 'N', 'NB', 'SEED'))
if NB > 16384:
    q = NB // 4
    WIN = {f'Q{j + 1}': ((j + 1) * q - 4096, (j + 1) * q) for j in range(4)}
else:
    WIN = {'A1': (2048, 6144), 'A2': (NB - 4096, NB)}
isfb = B['kind'] == ts.FBK


def distances(present):
    lines = set(ts.LINE0 + int(v) for v in present)
    deg = A['deg']; src = A['c_src']
    dist = np.zeros(N, np.int16)
    fr = np.array([(not isfb[i]) and any(int(s) in lines for s in src[i, :deg[i]]) for i in range(N)])
    dist[fr] = 1; d = 1
    while fr.any():
        nxt = np.array([(dist[i] == 0) and (not isfb[i]) and bool(fr[src[i, :deg[i]][src[i, :deg[i]] < N]].any()) for i in range(N)])
        d += 1; dist[nxt] = d; fr = nxt
    return dist


def conns():
    deg = A['deg'][~isfb]
    return dict(mean=float(deg.mean()), max=int(deg.max()), mean_all=float(A['deg'].sum()) / N)


out = dict(label=label, args=kw, windows=list(WIN)); pos = 0; t0 = time.process_time()
recs = {}; dists = {}; cw = {}
for w, (lo, hi) in WIN.items():
    rec = np.zeros((hi - lo, N), np.float32)
    L.tis_run(ctypes.byref(t), ctypes.c_void_p(b.ctypes.data), ctypes.c_int64(pos), ctypes.c_int64(hi), ctypes.c_int64(lo), ctypes.c_int64(hi), ctypes.c_void_p(rec.ctypes.data))
    recs[w] = rec; dists[w] = distances(sorted(set(b[lo:hi].tolist()))); pos = hi
    cw[w] = dict(conns(), made=int(t.made), given=int(t.given), given_by_use=int(t.given_by_use))
cpu = time.process_time() - t0
out.update(ms_per_byte=cpu / NB * 1e3, conns_per_cell=float(A['deg'].sum()) / N, conns_end=conns(), made=int(t.made), given=int(t.given),
           given_by_use=int(t.given_by_use), active_per_step=t.active_sum / t.k, gain_median=float(np.median(A['gain'])))
for w, (lo, hi) in WIN.items():
    x = np.where(recs[w] < 0.01, 0.0, recs[w])
    r = readings.read_window(x, b[lo:hi], dists[w], SEED, stream=b, start=lo, distances=DS, lags=(0, 1, 2))
    out[w] = dict(at_end=cw[w], cells={str(d): int((dists[w] == d).sum()) for d in DS},
                  share={str(d): r['shares'][d]['share'] if r['shares'][d]['n'] else None for d in DS}, n={str(d): r['shares'][d]['n'] for d in DS},
                  S={str(d): {str(Lg): (r['scores'][d]['by_lag'][Lg] or {}).get('S') for Lg in (0, 1, 2)} for d in DS},
                  c0={str(k): (v.get('S') if isinstance(v, dict) else v) for k, v in r['c0'].items()})
os.makedirs(os.path.join(HERE, 'out'), exist_ok=True)
json.dump(out, open(os.path.join(HERE, 'out', label + '.json'), 'w'), default=float)
lw = list(WIN)[-1]; a = out[lw]; f = lambda v: '  -  ' if v is None else f'{v:.3f}'
print(f"{label:34s} {out['ms_per_byte']:6.2f} ms/byte conns {out['conns_end']['mean']:5.1f} max {out['conns_end']['max']:3d} made {out['made']} given {out['given']} | {lw} S0 d1 2 4 8 10 "
      + " ".join(f(a['S'][str(d)]['0']) for d in DS), flush=True)
