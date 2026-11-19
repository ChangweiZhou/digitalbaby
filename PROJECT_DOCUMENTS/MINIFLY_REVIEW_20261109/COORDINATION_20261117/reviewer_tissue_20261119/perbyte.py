# Which byte values the cells near the output lines tell apart: for each byte value, how often a nearest-centroid reader (centroids
# from the first half of the window, read on the second half; outputs below 0.01 set to 0) names it right when it is the present
# byte, and when it is the byte one back. Window: the last 8,192 bytes. Group: the drivers and their (enlarged) neighbourhoods.
# usage: perbyte.py <label> rule=.. [options]
import sys, os, json, ctypes, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import tis
ts = tis.ts
label = sys.argv[1]; kw = dict(a.split('=') for a in sys.argv[2:])
T = tis.make(kw); t, L, b, N, NB = T['t'], T['L'], T['b'], T['N'], T['NB']
W = 8192; lo, hi = NB - W, NB
rec = np.zeros((W, N), np.float32)
L.tis_run(ctypes.byref(t), ctypes.c_void_p(b.ctypes.data), ctypes.c_int64(0), ctypes.c_int64(hi), ctypes.c_int64(lo), ctypes.c_int64(hi), ctypes.c_void_p(rec.ctypes.data))
B = T['B']; kind = np.asarray(B['kind']); isfb = kind == ts.FBK
g = np.zeros(N, bool)
for i in np.where(kind == ts.OUTE)[0]:
    g[i] = True
    for m in B['members'][int(i)]:
        if m < N: g[m] = True
g &= ~isfb
x = np.where(rec[:, g] < 0.01, 0.0, rec[:, g]).astype(np.float64)
out = dict(label=label, args=kw, cells=int(g.sum()), lags={})
h = W // 2
for lag in (0, 1):
    y = b[lo - lag:hi - lag].astype(int)
    vals = sorted(set(y[:h].tolist()))
    C = np.stack([x[:h][y[:h] == v].mean(0) for v in vals])
    xt = x[h:]; yt = y[h:]
    d = (xt ** 2).sum(1)[:, None] - 2 * xt @ C.T + (C ** 2).sum(1)[None, :]
    pred = np.array(vals)[d.argmin(1)]
    res = {}
    for v in vals:
        m = yt == v
        if m.sum(): res[v] = (float((pred[m] == v).mean()), int(m.sum()), float((pred == v).sum()))
    out['lags'][str(lag)] = dict(overall=float((pred == yt).mean()), by_value={str(v): r for v, r in res.items()})
os.makedirs('perbyte', exist_ok=True); json.dump(out, open(f'perbyte/{label}.json', 'w'))
name = lambda v: repr(chr(v)) if 32 <= v < 127 else f'#{v}'
for lag in (0, 1):
    r = out['lags'][str(lag)]; bv = {int(k): v for k, v in r['by_value'].items()}
    order = sorted(bv, key=lambda v: -bv[v][1])
    print(f"== {label} lag {lag}: overall {r['overall']:.3f}; by value (most frequent first) value:recall(count)")
    print('   ' + ' '.join(f"{name(v)}:{bv[v][0]:.2f}({bv[v][1]})" for v in order))
    print(f"   values never named right: {sum(1 for v in bv if bv[v][0] == 0)} of {len(bv)}; named right more than half the time: {sum(1 for v in bv if bv[v][0] > 0.5)}")
sys.stdout.flush()
