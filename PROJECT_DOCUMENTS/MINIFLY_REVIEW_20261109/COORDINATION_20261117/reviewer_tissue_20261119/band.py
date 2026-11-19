# What the cells of a band of the tissue carry, by the place of the band: if the output lines stood at x = x0 in place of x = 1, the
# cells within one neighbourhood radius (1/8) of them would be the cells with |x - x0| <= 1/8. For each x0: the tool's score for lags
# 0 to 4 on those cells (as one class; the tool samples at most 128 of them), their number, and their largest output.
# usage: band.py <label> rule=.. [options of drive.py]
import sys, os, json, ctypes, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import tis
readings = tis.readings; ts = tis.ts
label = sys.argv[1]; kw = dict(a.split('=') for a in sys.argv[2:])
T = tis.make(kw); t, L, b, N, NB = T['t'], T['L'], T['b'], T['N'], T['NB']
lo, hi = NB - 4096, NB
rec = np.zeros((hi - lo, N), np.float32)
L.tis_run(ctypes.byref(t), ctypes.c_void_p(b.ctypes.data), ctypes.c_int64(0), ctypes.c_int64(hi), ctypes.c_int64(lo), ctypes.c_int64(hi), ctypes.c_void_p(rec.ctypes.data))
B = T['B']; wb = b[lo:hi]; SEED = T['SEED']; isfb = np.asarray(B['kind']) == ts.FBK
X0 = (1/8, 1/4, 3/8, 1/2, 5/8, 3/4, 7/8, 1.0); LAGS = (0, 1, 2, 3, 4)
px = np.array([p[0] for p in B['P']])
xa = np.where(rec < 0.01, 0.0, rec).astype(np.float64)
out = dict(label=label, args=kw, bands={})
c0 = None
for x0 in X0:
    m = (np.abs(px - x0) <= 1/8) & ~isfb
    dn = np.where(m, 1, 0).astype(np.int16)
    r = readings.read_window(xa, wb, dn, SEED, stream=b, start=lo, distances=(1,), lags=LAGS)
    if c0 is None: c0 = {str(k): (v.get('S') if isinstance(v, dict) else v) for k, v in r['c0'].items()}
    out['bands'][str(x0)] = dict(cells=int(m.sum()), max_output=float(rec[:, m].max()), S={str(Lg): (r['scores'][1]['by_lag'][Lg] or {}).get('S') for Lg in LAGS})
out['c0'] = c0
os.makedirs('band', exist_ok=True); json.dump(out, open(f'band/{label}.json', 'w'), default=float)
f = lambda v: ' nan ' if v is None else f'{v:.3f}'
print(f"== {label}   c0 by lag: " + ' '.join(f(c0.get(str(Lg))) for Lg in LAGS) + "   (levels: 0.1103; c0 + 0.02 for lags 1 to 4)")
for x0 in X0:
    o = out['bands'][str(x0)]
    print(f"   x0 = {x0:.3f}  cells {o['cells']:4d}  largest output {o['max_output']:.2f} | lags 0-4: " + ' '.join(f(o['S'][str(Lg)]) for Lg in LAGS))
sys.stdout.flush()
