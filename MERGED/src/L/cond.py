# How much the outputs of the cells near the output lines move together (E55's worker found that this, not the amount of information,
# keeps a one-pass rule below a fitted reader). For the cells that the drivers of the output lines can receive from: each cell divided
# by its own deviation; the share of the variance in the first and in the first four principal components; the ratio of the largest
# to the median eigenvalue; the median absolute correlation between two cells; and the same for each driver's own neighbourhood (median over drivers).
# usage: cond.py <label> rule=.. [options of drive.py]
import sys, os, json, ctypes, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import tis
ts = tis.ts
label = sys.argv[1]; kw = dict(a.split('=') for a in sys.argv[2:])
T = tis.make(kw); t, L, b, N, NB = T['t'], T['L'], T['b'], T['N'], T['NB']
lo, hi = NB - 4096, NB
rec = np.zeros((hi - lo, N), np.float32)
L.tis_run(ctypes.byref(t), ctypes.c_void_p(b.ctypes.data), ctypes.c_int64(0), ctypes.c_int64(hi), ctypes.c_int64(lo), ctypes.c_int64(hi), ctypes.c_void_p(rec.ctypes.data))
B = T['B']; kind = np.asarray(B['kind']); isfb = kind == ts.FBK
def stats(cells):
    X = rec[:, cells].astype(np.float64); sd = X.std(axis=0); X = X[:, sd > 1e-30]
    if X.shape[1] < 3: return None
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    C = np.corrcoef(X.T); ev = np.sort(np.linalg.eigvalsh(C))[::-1]; ev = np.maximum(ev, 0)
    iu = np.triu_indices(C.shape[0], 1)
    return dict(cells=int(X.shape[1]), pc1=float(ev[0] / ev.sum()), pc4=float(ev[:4].sum() / ev.sum()),
                ratio=float(ev[0] / max(np.median(ev), 1e-300)), abs_corr_median=float(np.median(np.abs(C[iu]))))
drivers = np.where(kind == ts.OUTE)[0]
near = np.zeros(N, bool)
per = []
for i in drivers:
    mem = [m for m in B['members'][int(i)] if m < N and not isfb[m]]
    near[mem] = True; near[i] = True
    s = stats(np.array(mem, int)) if len(mem) >= 3 else None
    if s: per.append(s)
allnear = stats(np.where(near & ~isfb)[0])
med = lambda k: float(np.median([p[k] for p in per])) if per else None
out = dict(label=label, args=kw, near_all=allnear, per_driver=dict(drivers=len(per), cells=med('cells'), pc1=med('pc1'), pc4=med('pc4'), ratio=med('ratio'), abs_corr_median=med('abs_corr_median')))
os.makedirs('cond', exist_ok=True); json.dump(out, open(f'cond/{label}.json', 'w'), default=float)
a = allnear; p = out['per_driver']
print(f"== {label}: near the output lines, {a['cells']} cells with output: first component {a['pc1']:.3f}, first four {a['pc4']:.3f}, largest/median eigenvalue {a['ratio']:.3g}, median |correlation| {a['abs_corr_median']:.3f}")
print(f"   each driver's own neighbourhood (median of {p['drivers']} drivers, {p['cells']:.0f} cells): first component {p['pc1']:.3f}, first four {p['pc4']:.3f}, largest/median {p['ratio']:.3g}, median |correlation| {p['abs_corr_median']:.3f}")
sys.stdout.flush()
