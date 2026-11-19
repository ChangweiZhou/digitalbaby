# The tool's readings at every distance and lags 0 to 4, as it is (floor 0.01) and with no floor and each cell divided by its own
# deviation; and the same for the cells within one neighbourhood radius of the output lines (x >= 7/8).
# usage: far.py <label> rule=.. [options of drive.py]
import sys, os, json, ctypes, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import tis
readings = tis.readings; ts = tis.ts
label = sys.argv[1]; kw = dict(a.split('=') for a in sys.argv[2:])
T = tis.make(kw); t, L, b, N, NB = T['t'], T['L'], T['b'], T['N'], T['NB']
lo, hi = NB - 4096, NB
rec = np.zeros((hi - lo, N), np.float32)
L.tis_run(ctypes.byref(t), ctypes.c_void_p(b.ctypes.data), ctypes.c_int64(0), ctypes.c_int64(hi), ctypes.c_int64(lo), ctypes.c_int64(hi), ctypes.c_void_p(rec.ctypes.data))
A = T['A']; B = T['B']; wb = b[lo:hi]; SEED = T['SEED']
lines = set(ts.LINE0 + int(v) for v in set(wb.tolist()))
deg = A['deg']; src = A['c_src']; isfb = np.asarray(B['kind']) == ts.FBK
dist = np.zeros(N, np.int16)
fr = np.array([(not isfb[i]) and any(int(s) in lines for s in src[i, :deg[i]]) for i in range(N)]); dist[fr] = 1; d = 1
while fr.any():
    nxt = np.array([(dist[i] == 0) and (not isfb[i]) and bool(fr[src[i, :deg[i]][src[i, :deg[i]] < N]].any()) for i in range(N)])
    d += 1; dist[nxt] = d; fr = nxt
LAGS = (0, 1, 2, 3, 4); DS = tuple(range(1, int(dist.max()) + 1))
def table(x, dd, ds):
    r = readings.read_window(x, wb, dd, SEED, stream=b, start=lo, distances=ds, lags=LAGS)
    return {str(d): {str(Lg): (r['scores'][d]['by_lag'][Lg] or {}).get('S') for Lg in LAGS} for d in ds}, r
xa = np.where(rec < 0.01, 0.0, rec).astype(np.float64)
sd = rec.std(axis=0).astype(np.float64); xb = rec.astype(np.float64) / np.where(sd > 1e-30, sd, 1.0)
ta, r = table(xa, dist, DS); tb, _ = table(xb, dist, DS)
c0 = {str(k): (v.get('S') if isinstance(v, dict) else v) for k, v in r['c0'].items()}
# the cells near the output lines: all with a driver of an output line in their neighbourhood are not in the birth record; use the
# cells that the drivers can receive from, i.e. the members of the drivers' neighbourhoods, the drivers themselves included
near = np.zeros(N, bool); kind = np.asarray(B['kind'])
for i in np.where(kind == ts.OUTE)[0]:
    near[i] = True
    for m in B['members'][int(i)]:
        if m < N: near[m] = True
near &= ~isfb
dn = np.where(near, 1, 0).astype(np.int16)
na, _ = table(xa, dn, (1,)); nb, _ = table(xb, dn, (1,))
out = dict(label=label, args=kw, c0=c0, tool=ta, nofloor=tb, near_cells=int(near.sum()), near_tool=na['1'], near_nofloor=nb['1'],
           near_distances={str(k): int(((dist == k) & near).sum()) for k in DS}, cells={str(k): int((dist == k).sum()) for k in DS},
           max_output={str(k): float(rec[:, dist == k].max()) if (dist == k).any() else None for k in DS})
os.makedirs('far', exist_ok=True); json.dump(out, open(f'far/{label}.json', 'w'), default=float)
f = lambda v: ' nan ' if v is None else f'{v:.3f}'
print(f"== {label}   c0 by lag: " + ' '.join(f(c0.get(str(Lg))) for Lg in LAGS))
for d in DS:
    print(f"   d={d:2d} cells {out['cells'][str(d)]:4d} | tool " + ' '.join(f(ta[str(d)][str(Lg)]) for Lg in LAGS) + " | no floor " + ' '.join(f(tb[str(d)][str(Lg)]) for Lg in LAGS))
print(f"   near the output lines ({out['near_cells']} cells) | tool " + ' '.join(f(na['1'][str(Lg)]) for Lg in LAGS) + " | no floor " + ' '.join(f(nb['1'][str(Lg)]) for Lg in LAGS))
sys.stdout.flush()
