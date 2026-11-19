# Diagnosis by distance from the input lines: what a cell's output at the reading step depends on.
# usage: diag.py <label> rule=.. [options of drive.py]
# (1) by distance d: cells, mean output, share of positions above 0.01, and the share of a cell's output variance explained by the
#     present byte and by the previous byte (median and upper decile over cells; raw outputs, no floor of the reading);
# (2) dependence on the tissue's own past: two copies of one tissue, the second with one cell's potential changed by 1e-9 at byte
#     NB-4096-512; the largest difference of outputs after 1, 4, 16, 64, 256, 512 bytes.
import sys, os, json, ctypes, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import tis
label = sys.argv[1]; kw = dict(a.split('=') for a in sys.argv[2:])
def run(T, lo, hi, rec_lo=None):
    t, L, b = T['t'], T['L'], T['b']
    n = 0 if rec_lo is None else hi - rec_lo
    rec = np.zeros((max(n, 1), T['N']), np.float32)
    r0 = hi if rec_lo is None else rec_lo
    L.tis_run(ctypes.byref(t), ctypes.c_void_p(b.ctypes.data), ctypes.c_int64(lo), ctypes.c_int64(hi), ctypes.c_int64(r0), ctypes.c_int64(hi), ctypes.c_void_p(rec.ctypes.data))
    return rec
T1 = tis.make(kw); T2 = tis.make(kw); NB = T1['NB']; N = T1['N']; ts = tis.ts
P0 = NB - 4096 - 512
run(T1, 0, P0); run(T2, 0, P0)
assert np.array_equal(T1['A']['v'], T2['A']['v'])
j = int(np.argmax(T2['A']['xp'])); T2['A']['v'][j] += 1e-9
d1 = run(T1, P0, P0 + 512, P0); d2 = run(T2, P0, P0 + 512, P0)
diff = np.abs(d1.astype(np.float64) - d2).max(axis=1)
out = dict(label=label, args=kw, divergence={str(n): float(diff[n - 1]) for n in (1, 4, 16, 64, 256, 512)}, mean_output=float(d1.mean()))
lo, hi = NB - 4096, NB
rec = run(T1, P0 + 512, hi, lo).astype(np.float64)
b = T1['b']; A = T1['A']; B = T1['B']
lines = set(ts.LINE0 + int(v) for v in set(b[lo:hi].tolist()))
deg = A['deg']; src = A['c_src']; isfb = B['kind'] == ts.FBK
dist = np.zeros(N, np.int16)
fr = np.array([(not isfb[i]) and any(int(s) in lines for s in src[i, :deg[i]]) for i in range(N)]); dist[fr] = 1; d = 1
while fr.any():
    nxt = np.array([(dist[i] == 0) and (not isfb[i]) and bool(fr[src[i, :deg[i]][src[i, :deg[i]] < N]].any()) for i in range(N)])
    d += 1; dist[nxt] = d; fr = nxt
def eta2(x, lab):
    tot = x.var(axis=0); m = x.mean(axis=0); bet = np.zeros(x.shape[1])
    for v in np.unique(lab):
        s = lab == v; bet += s.mean() * (x[s].mean(axis=0) - m) ** 2
    return np.where(tot > 1e-30, bet / np.maximum(tot, 1e-30), np.nan)
e0 = eta2(rec[1:], b[lo + 1:hi]); e1 = eta2(rec[1:], b[lo:hi - 1])
rows = []
for dd in range(1, int(dist.max()) + 1):
    c = dist == dd
    if not c.any(): continue
    q = lambda e, p: float(np.nanpercentile(e[c], p)) if np.isfinite(e[c]).any() else None
    rows.append(dict(d=dd, cells=int(c.sum()), mean_output=float(rec[:, c].mean()), max_output=float(rec[:, c].max()), share_above_half=float((rec[:, c].max(axis=0) > 0.5).mean()), share_above_001=float((rec[:, c] > 0.01).mean()),
                     present_median=q(e0, 50), present_p90=q(e0, 90), previous_median=q(e1, 50), previous_p90=q(e1, 90)))
out['by_distance'] = rows; out['unreached'] = int(((dist == 0) & ~isfb).sum())
os.makedirs('diag', exist_ok=True); json.dump(out, open(f'diag/{label}.json', 'w'), default=float)
f = lambda v: '  nan' if v is None else f'{v:.3f}'
print(f"== {label}  divergence after 1/4/16/64/256/512 bytes: " + ' '.join(f"{out['divergence'][str(n)]:.1e}" for n in (1, 4, 16, 64, 256, 512)))
for r in rows:
    print(f"   d={r['d']:2d} cells {r['cells']:4d} mean {r['mean_output']:.5f} max {r['max_output']:.3f} cells>1/2 {r['share_above_half']:.2f} >0.01 {r['share_above_001']:.3f} | present med {f(r['present_median'])} p90 {f(r['present_p90'])} | previous med {f(r['previous_median'])} p90 {f(r['previous_p90'])}")
sys.stdout.flush()
