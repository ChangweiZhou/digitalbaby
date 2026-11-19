# The cells a driver of an output line can receive from, in two groups: (a) by the enlarged neighbourhoods (with long-axon cells
# within their reach), as far.py takes them; (b) by the neighbourhoods of 1/8 alone (the cells that lie near the output lines).
# Also (c): the long-axon cells among (a) that are not in (b). The tool as it is, lags 0 to 4.
# usage: near2.py <label> rule=.. [options of drive.py, with long=K-R]
import sys, os, json, ctypes, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import tis
readings = tis.readings; ts = tis.ts
label = sys.argv[1]; kw = dict(a.split('=') for a in sys.argv[2:])
T = tis.make(kw); t, L, b, N, NB = T['t'], T['L'], T['b'], T['N'], T['NB']
lo, hi = NB - 4096, NB
rec = np.zeros((hi - lo, N), np.float32)
L.tis_run(ctypes.byref(t), ctypes.c_void_p(b.ctypes.data), ctypes.c_int64(0), ctypes.c_int64(hi), ctypes.c_int64(lo), ctypes.c_int64(hi), ctypes.c_void_p(rec.ctypes.data))
B = T['B']; wb = b[lo:hi]; SEED = T['SEED']; kind = np.asarray(B['kind']); isfb = kind == ts.FBK
P = np.asarray(B['P']); drivers = np.where(kind == ts.OUTE)[0]
ga = np.zeros(N, bool); gb = np.zeros(N, bool)
for i in drivers:
    ga[i] = gb[i] = True
    for m in B['members'][int(i)]:
        if m < N: ga[m] = True
    d2 = ((P - P[i]) ** 2).sum(1); gb[d2 <= ts.R2] = True
ga &= ~isfb; gb &= ~isfb; gc = ga & ~gb
LAGS = (0, 1, 2, 3, 4)
xa = np.where(rec < 0.01, 0.0, rec).astype(np.float64)
def score(g):
    if g.sum() == 0: return None, None
    r = readings.read_window(xa, wb, np.where(g, 1, 0).astype(np.int16), SEED, stream=b, start=lo, distances=(1,), lags=LAGS)
    return {str(Lg): (r['scores'][1]['by_lag'][Lg] or {}).get('S') for Lg in LAGS}, r
sa, r = score(ga); sb, _ = score(gb); sc, _ = score(gc)
c0 = {str(k): (v.get('S') if isinstance(v, dict) else v) for k, v in r['c0'].items()}
# connections of the drivers at the end: how many come from long-axon cells
A = T['A']; longs = set(int(j) for j in B.get('longs', []))
nd = sum(int(A['deg'][i]) for i in drivers); nl = sum(int(s) in longs for i in drivers for s in A['c_src'][i, :A['deg'][i]])
out = dict(label=label, args=kw, c0=c0, enlarged=dict(cells=int(ga.sum()), S=sa, x_median=float(np.median(P[ga, 0])), max_output=float(rec[:, ga].max()), share_above_half=float((rec[:, ga].max(0) > 0.5).mean())),
           by_eighth=dict(cells=int(gb.sum()), S=sb, x_median=float(np.median(P[gb, 0])), max_output=float(rec[:, gb].max()), share_above_half=float((rec[:, gb].max(0) > 0.5).mean())),
           long_only=dict(cells=int(gc.sum()), S=sc, x_median=float(np.median(P[gc, 0])) if gc.any() else None),
           driver_connections=nd, driver_connections_from_long=nl)
os.makedirs('near2', exist_ok=True); json.dump(out, open(f'near2/{label}.json', 'w'), default=float)
f = lambda d: ' '.join(' nan ' if d is None or d[str(k)] is None else f"{d[str(k)]:.3f}" for k in LAGS)
print(f"== {label}  c0 lags 1-4 " + ' '.join(f"{c0[str(k)]:.3f}" for k in (1, 2, 3, 4)))
print(f"   (a) enlarged neighbourhoods {out['enlarged']['cells']:4d} cells, x median {out['enlarged']['x_median']:.2f}, above 1/2: {out['enlarged']['share_above_half']:.2f} | {f(sa)}")
print(f"   (b) within 1/8 of a driver  {out['by_eighth']['cells']:4d} cells, x median {out['by_eighth']['x_median']:.2f}, above 1/2: {out['by_eighth']['share_above_half']:.2f} | {f(sb)}")
print(f"   (c) long-axon cells only    {out['long_only']['cells']:4d} cells, x median {out['long_only']['x_median']} | {f(sc)}")
print(f"   connections of the drivers: {nd}, of which from long-axon cells {nl}"); sys.stdout.flush()
