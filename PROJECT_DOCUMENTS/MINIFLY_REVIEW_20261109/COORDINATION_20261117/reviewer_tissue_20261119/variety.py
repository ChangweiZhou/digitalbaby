# What the cells at each distance answer to: the variety of bytes.
# usage: variety.py <label> rule=.. [options of drive.py]
# For every cell: its mean output for each byte value in window A2 (raw outputs at the reading step); its preferred byte (largest mean);
# the number of byte values at which its mean is above half of its largest. By distance from the input lines: cells, cells with any
# output, number of different preferred bytes (of the 46 in the stream), share of cells preferring the commonest preferred byte, median
# number of bytes a cell answers to, and the frequency in the stream of the preferred bytes (median) against the median frequency of a byte.
import sys, os, json, ctypes, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import tis
label = sys.argv[1]; kw = dict(a.split('=') for a in sys.argv[2:])
T = tis.make(kw); t, L, b, N, NB, ts = T['t'], T['L'], T['b'], T['N'], T['NB'], tis.ts
lo, hi = NB - 4096, NB
rec = np.zeros((hi - lo, N), np.float32)
L.tis_run(ctypes.byref(t), ctypes.c_void_p(b.ctypes.data), ctypes.c_int64(0), ctypes.c_int64(hi), ctypes.c_int64(lo), ctypes.c_int64(hi), ctypes.c_void_p(rec.ctypes.data))
rec = rec.astype(np.float64); A = T['A']; B = T['B']; wb = b[lo:hi]
vals = np.unique(wb); freq = np.array([(wb == v).mean() for v in vals])
lines = set(ts.LINE0 + int(v) for v in vals)
deg = A['deg']; src = A['c_src']; isfb = B['kind'] == ts.FBK
dist = np.zeros(N, np.int16)
fr = np.array([(not isfb[i]) and any(int(s) in lines for s in src[i, :deg[i]]) for i in range(N)]); dist[fr] = 1; d = 1
while fr.any():
    nxt = np.array([(dist[i] == 0) and (not isfb[i]) and bool(fr[src[i, :deg[i]][src[i, :deg[i]] < N]].any()) for i in range(N)])
    d += 1; dist[nxt] = d; fr = nxt
Mn = np.stack([rec[wb == v].mean(axis=0) for v in vals])          # byte values x cells
mx = Mn.max(axis=0); pref = Mn.argmax(axis=0); nb = (Mn > 0.5 * mx[None, :]).sum(axis=0)
def s_std(cells):
    """Nearest-centroid score of the present byte from the cells given, each divided by its own deviation, no floor:
    centroids from the first half of the window, score (mean over byte values of the share right) on the second half."""
    X = rec[:, cells]; h = X.shape[0] // 2
    sd = X[:h].std(axis=0); keep = sd > 1e-30
    if keep.sum() < 2: return None
    X = X[:, keep] / sd[keep]
    lab1, lab2 = wb[:h], wb[h:]
    vs = [v for v in vals if (lab1 == v).any() and (lab2 == v).any()]
    C = np.stack([X[:h][lab1 == v].mean(axis=0) for v in vs])
    D = ((X[h:, None, :] - C[None, :, :]) ** 2).sum(axis=2); pred = D.argmin(axis=1)
    return float(np.mean([(pred[lab2 == v] == k).mean() for k, v in enumerate(vs)]))
rows = []
for dd in range(1, int(dist.max()) + 1):
    c = (dist == dd); a = c & (mx > 1e-12)
    if not c.any(): continue
    if a.any():
        cnt = np.bincount(pref[a], minlength=len(vals))
        rows.append(dict(d=dd, cells=int(c.sum()), s_std=s_std(np.where(a)[0]), answering=int(a.sum()), preferred_bytes=int((cnt > 0).sum()), top_share=float(cnt.max() / a.sum()),
                         bytes_a_cell_median=float(np.median(nb[a])), preferred_freq_median=float(np.median(freq[pref[a]])), largest_mean_median=float(np.median(mx[a]))))
    else:
        rows.append(dict(d=dd, cells=int(c.sum()), answering=0))
out = dict(label=label, args=kw, byte_values=int(len(vals)), byte_freq_median=float(np.median(freq)), byte_freq_max=float(freq.max()), by_distance=rows)
os.makedirs('variety', exist_ok=True); json.dump(out, open(f'variety/{label}.json', 'w'), default=float)
print(f"== {label}: {len(vals)} byte values; frequency of a byte: median {np.median(freq):.4f}, largest {freq.max():.4f}")
for r in rows:
    if r['answering']:
        print(f"   d={r['d']:2d} cells {r['cells']:4d} answering {r['answering']:4d} | preferred bytes {r['preferred_bytes']:2d} of {len(vals)}, commonest holds {r['top_share']:.2f} | bytes a cell {r['bytes_a_cell_median']:4.1f} | freq of preferred {r['preferred_freq_median']:.4f} | largest mean {r['largest_mean_median']:.4f} | S no floor {(r['s_std'] if r['s_std'] is not None else float('nan')):.3f}")
    else: print(f"   d={r['d']:2d} cells {r['cells']:4d} answering 0")
sys.stdout.flush()
