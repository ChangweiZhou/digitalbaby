# Where in the tissue is the next byte readable? The fitted ridge reader of measure.py (part A) on the cells at each distance d from the
# input lines (diag.py's distances, from the connections at the end of the run), on the cells of set (a) at that distance, and on the
# cells at d or more. Share right overall and at R1, R2 on the last quarter.
# usage: by_distance.py <label> [tissue options] seed=.. nbytes=..   -> res/dist_<label>.json
import sys, os, json, time, ctypes, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, 'deps', 'PREP'))
import tis, life_stream as LS, score_emissions as SE
label = sys.argv[1]; kw = dict(a.split('=') for a in sys.argv[2:]); t0 = time.time()
T = tis.make(kw); NB = T['NB']; N = T['N']; ts = tis.ts; B = T['B']; A_ = T['A']; b = T['b'].astype(np.int64); isfb = B['kind'] == ts.FBK
L = LS.life(T['SEED'])
allc = np.flatnonzero(~isfb)
drivers = list(B['driver']); near = set(drivers)
for i in drivers: near.update(int(m) for m in B['members'][i] if m < N)
innear = np.zeros(N, bool); innear[[i for i in near if not isfb[i]]] = True
R = np.zeros((NB, N), np.float32)
T['L'].tis_run(ctypes.byref(T['t']), ctypes.c_void_p(T['b'].ctypes.data), ctypes.c_int64(0), ctypes.c_int64(NB), ctypes.c_int64(0), ctypes.c_int64(NB), ctypes.c_void_p(R.ctypes.data))
lines = set(ts.LINE0 + int(v) for v in set(b.tolist())); deg = A_['deg']; src = A_['c_src']
dist = np.zeros(N, np.int16)
fr = np.array([(not isfb[i]) and any(int(s) in lines for s in src[i, :deg[i]]) for i in range(N)]); dist[fr] = 1; d = 1
while fr.any():
    nxt = np.array([(dist[i] == 0) and (not isfb[i]) and bool(fr[src[i, :deg[i]][src[i, :deg[i]] < N]].any()) for i in range(N)])
    d += 1; dist[nxt] = d; fr = nxt
M = NB - 1; y = b[1:]; nf = (3 * M) // 4; nin = (4 * nf) // 5
Y = np.zeros((M, 256)); Y[np.arange(M), y] = 1
K_ = SE.kinds(L); pos = np.arange(nf, M) + 1
km = {'all': np.ones(len(pos), bool), 'R1': K_['R1 (first family, phase FIRST)'][:NB][pos], 'R2': K_['R2 (first family, phase FIRST)'][:NB][pos]}
def ridge(cs):
    X = R[:M][:, cs].astype(np.float64); sd = X[:nf].std(0); X = X[:, sd > 0] / sd[sd > 0]
    if X.shape[1] == 0: return None
    def fit(n):
        mu = X[:n].mean(0); my = Y[:n].mean(0); Xc = X[:n] - mu; return mu, my, Xc.T @ Xc, Xc.T @ (Y[:n] - my)
    mu, my, G, Rr = fit(nin); best, bl = -1, 0
    for e in range(-6, 3):
        Gl = G.copy(); Gl[np.diag_indices_from(Gl)] += 10.0 ** e * nin
        s = float((((X[nin:nf] - mu) @ np.linalg.solve(Gl, Rr) + my).argmax(1) == y[nin:nf]).mean())
        if s >= best: best, bl = s, e
    mu, my, G, Rr = fit(nf); G[np.diag_indices_from(G)] += 10.0 ** bl * nf
    g = ((X[nf:] - mu) @ np.linalg.solve(G, Rr) + my).argmax(1)
    return {k: float((g[m] == b[pos][m]).mean()) for k, m in km.items()}
rows = []
for dd in range(0, int(dist.max()) + 1):
    c = np.flatnonzero((dist == dd) & ~isfb)
    if len(c) == 0: continue
    ca = c[innear[c]]; cge = np.flatnonzero((dist >= dd) & ~isfb) if dd > 0 else None
    rows.append(dict(d=dd, cells=int(len(c)), cells_in_set_a=int(len(ca)), mean_output=float(R[:, c].mean()), at_d=ridge(c),
                     set_a_at_d=ridge(ca) if len(ca) >= 8 else None, at_d_or_more=ridge(cge) if cge is not None else None))
    print(f"d={dd:2d} cells {len(c):4d} (set a {len(ca):3d}) mean {rows[-1]['mean_output']:.3g} | at d " +
          (' / '.join(f"{100 * v:.1f}" for v in rows[-1]['at_d'].values()) if rows[-1]['at_d'] else '-') +
          ' | set a at d ' + (' / '.join(f"{100 * v:.1f}" for v in rows[-1]['set_a_at_d'].values()) if rows[-1]['set_a_at_d'] else '-') +
          ' | at d or more ' + (' / '.join(f"{100 * v:.1f}" for v in rows[-1]['at_d_or_more'].values()) if rows[-1]['at_d_or_more'] else '-'), flush=True)
json.dump(dict(label=label, args=kw, bytes=NB, note='d = 0: cells not reached from the input lines', rows=rows, seconds=time.time() - t0),
          open(os.path.join(HERE, 'res', f'dist_{label}.json'), 'w'), indent=1)
print(f'== {label} ({time.time() - t0:.0f} s)', flush=True)
