# The output end placed at x0 in place of x = 1. One tissue of tis.make is run once, as it is (only the place of the reading end changes;
# at step 1 a driver is an ordinary cell); then, for each x0 in X0:
#   reading end: 256 points (x0, y_v), y_v drawn by numpy.random.default_rng(seed * 1000 + round(1000 * x0)).random(256); for each point in
#     turn (v = 0 .. 255) the nearest non-feedback cell not yet taken is its driver; the driver's inputs are the non-feedback cells within
#     1/8 of the point (the driver included when it lies within 1/8).
#   A. fitted ridge readers of the next byte (as measure.py): on all non-feedback cells of the band |x - x0| <= 1/8; on each driver's own
#      inputs (output v on driver v's inputs only); fitted on the first three quarters, scored on the last quarter; overall and at R1, R2.
#      The one-hot readers of one, two and three bytes and ref_1, ref_2, ref_3, the end mark are computed once (they do not depend on x0).
#   B. online one-pass rules on each driver's own inputs, the outputs as they are (no standardisation) and a constant 1: supervised D
#      (eta 1/64, step eta (target - activity) u / |u|^2), yes/no B (eps 1/16), yes/no A (kappa 1/8; eps 0 and 1/16), as in measure.py.
#   C. for each online rule, the end of the first window of 4,096 whose share reached the end mark's share, ref_1 and ref_2 of that window.
# usage: measure_x0b.py <label> [tissue options] seed=.. nbytes=.. [x0s=3/8,1/2]   (measure_x0.py with the places given; default all five)   -> res/x0_<label>.json (rewritten after each x0)
import sys, os, json, time, ctypes, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, 'deps', 'PREP'))
import tis, life_stream as LS, score_emissions as SE
label = sys.argv[1]; kw = dict(a.split('=') for a in sys.argv[2:]); t0 = time.time()
X0 = tuple(float(eval(v)) for v in kw.pop('x0s', '3/8,1/2,5/8,3/4,1').split(','))
ONLINE_ONLY = kw.pop('online_only', '0') == '1'      # skip the fitted readers (parts A and the one-hot readers): for long runs
T = tis.make(kw); NB = T['NB']; N = T['N']; ts = tis.ts; B = T['B']; b = T['b'].astype(np.int64); isfb = B['kind'] == ts.FBK; SEED = T['SEED']
L = LS.life(SEED); assert np.array_equal(L['bytes'][:NB], T['b'])
Pxy = np.array(B['P']); allc = np.flatnonzero(~isfb)
R = np.zeros((NB, N), np.float32)
for lo in range(0, NB, 4096):
    hi = min(NB, lo + 4096)
    T['L'].tis_run(ctypes.byref(T['t']), ctypes.c_void_p(T['b'].ctypes.data), ctypes.c_int64(lo), ctypes.c_int64(hi), ctypes.c_int64(lo), ctypes.c_int64(hi), ctypes.c_void_p(R[lo:hi].ctypes.data))
t_tissue = time.time() - t0
M = NB - 1; y = b[1:]; nf = (3 * M) // 4; nin = (4 * nf) // 5; q0 = nf + 1
Y = np.zeros((M, 256)) if 'online_only=1' not in sys.argv else None
if Y is not None: Y[np.arange(M), y] = 1
refs = SE.references(L['bytes'][:NB]); K_ = SE.kinds(L); pos = np.arange(nf, M) + 1
kmask = {'all': np.ones(len(pos), bool), 'R1': K_['R1 (first family, phase FIRST)'][:NB][pos], 'R2': K_['R2 (first family, phase FIRST)'][:NB][pos]}
shares = lambda g: {k: float((g[m] == b[pos][m]).mean()) for k, m in kmask.items()}
def ridge_all(X):
    def fit(n):
        mu = X[:n].mean(0); my = Y[:n].mean(0); Xc = X[:n] - mu; return mu, my, Xc.T @ Xc, Xc.T @ (Y[:n] - my)
    mu, my, G, Rr = fit(nin); best, bl = -1, 0
    for e in range(-6, 3):
        Gl = G.copy(); Gl[np.diag_indices_from(Gl)] += 10.0 ** e * nin
        s = float((((X[nin:nf] - mu) @ np.linalg.solve(Gl, Rr) + my).argmax(1) == y[nin:nf]).mean())
        if s >= best: best, bl = s, e
    mu, my, G, Rr = fit(nf); G[np.diag_indices_from(G)] += 10.0 ** bl * nf
    return ((X[nf:] - mu) @ np.linalg.solve(G, Rr) + my).argmax(1), 10.0 ** bl
def stdz(X):
    sd = X[:nf].std(0); keep = sd > 0; return X[:, keep] / sd[keep]
common = {}
tt = np.arange(M)
for h in ((1, 2, 3) if not ONLINE_ONLY else ()):
    OH = np.zeros((M, 256 * h))
    for j in range(h):
        q = tt[tt >= j]; OH[q, 256 * j + b[q - j]] = 1
    g, lam = ridge_all(OH); common[f'{h} one-hot(s) of the last bytes'] = dict(shares=shares(g), ridge=lam)
for k in ('ref_1', 'ref_2', 'ref_3'): common[k] = dict(shares=shares(refs[k][pos].astype(np.int64)))
common['end mark'] = dict(shares={k: float((b[pos][m] == 10).mean()) for k, m in kmask.items()})
wins = [(a_, min(a_ + 4096, NB)) for a_ in range(0, NB, 4096)]
wsh = lambda ok: [float(ok[max(a_, 1):e_].mean()) for a_, e_ in wins]
lv = {'end mark': [float((b[a_:e_] == 10).mean()) for a_, e_ in wins], 'ref_1': wsh(refs['ref_1'] == b), 'ref_2': wsh(refs['ref_2'] == b)}
out = dict(label=label, args=kw, bytes=NB, seconds_tissue=t_tissue, common=common, levels=lv, windows=wins, x0={})
RULES = [('D', 0.0, 1 / 64), ('B', 1 / 16, 0.0), ('A', 0.0, 1 / 8), ('A', 1 / 16, 1 / 8)]
for x0 in X0:
    t1 = time.time()
    yv = np.random.default_rng(SEED * 1000 + round(1000 * x0)).random(256)
    taken = np.zeros(N, bool); taken[isfb] = True; drv, own = [], []
    for v in range(256):
        d2 = (Pxy[:, 0] - x0) ** 2 + (Pxy[:, 1] - yv[v]) ** 2
        d2m = np.where(taken, np.inf, d2); i = int(d2m.argmin()); taken[i] = True; drv.append(i)
        own.append(np.flatnonzero((d2 <= 1 / 64) & ~isfb))
    band = np.flatnonzero((np.abs(Pxy[:, 0] - x0) <= 1 / 8) & ~isfb)
    near = np.unique(np.concatenate(own)); col = {c: k for k, c in enumerate(near)}; K = len(near)
    ownk = [np.array([col[c] for c in o], np.int64) for o in own]
    G_ = max(len(o) for o in own); IDX = np.full((256, G_), K, np.int64)
    for v in range(256): IDX[v, :len(ownk[v])] = ownk[v]
    P = np.array([len(o) + 1.0 for o in own])
    A = {}
    Xa = R[:M][:, near].astype(np.float64)
    if not ONLINE_ONLY:
      Xb = stdz(R[:M][:, band].astype(np.float64)); g, lam = ridge_all(Xb); A['all cells of the band'] = dict(shares=shares(g), ridge=lam, cells=int(len(band))); del Xb
      sda = Xa[:nf].std(0); Xs = np.where(sda > 0, Xa / np.where(sda > 0, sda, 1), 0.0)
      def pred(n, lam):
        o = np.zeros((M, 256))
        for v in range(256):
            Xv = Xs[:, ownk[v]]; mu = Xv[:n].mean(0); my = Y[:n, v].mean(); Xc = Xv[:n] - mu
            Gm = Xc.T @ Xc; Gm[np.diag_indices_from(Gm)] += lam * n; o[:, v] = (Xv - mu) @ np.linalg.solve(Gm, Xc.T @ (Y[:n, v] - my)) + my
        return o
      inner = {e: float((pred(nin, 10.0 ** e)[nin:nf].argmax(1) == y[nin:nf]).mean()) for e in range(-6, 3)}
      be = max(inner, key=lambda e: (inner[e], e)); A["each driver's own inputs"] = dict(shares=shares(pred(nf, 10.0 ** be)[nf:].argmax(1)), ridge=10.0 ** be)
      del Xs
    C = len(RULES); Wt = np.zeros((C, 256, G_ + 1)); rbar = np.zeros((C, 256)); rng = np.random.default_rng(SEED)
    right = np.zeros((C, NB), bool); seen = np.zeros(256, bool)
    for t in range(M):
        seen[b[t]] = True
        U = np.ones((256, G_ + 1)); U[:, :-1] = np.append(Xa[t], 0.0)[IDX]; yy = y[t]
        for c, (rule, eps, par) in enumerate(RULES):
            a = (Wt[c] * U).sum(1); e = int(a.argmax())
            if eps and rng.random() < eps:
                sv = np.flatnonzero(seen); e = int(sv[rng.integers(0, len(sv))])
            ok = e == yy; right[c, t + 1] = ok
            if rule == 'A':
                Wt[c, e] += (1.0 if ok else -par) * (0.25 / P[e]) * U[e]
            elif rule == 'B':
                Wt[c, e] += (float(ok) - rbar[c, e]) * (0.25 / P[e]) * U[e]; rbar[c, e] += (float(ok) - rbar[c, e]) / 64
            else:
                tg = np.zeros(256); tg[yy] = 1; nr = (U * U).sum(1); nr[nr == 0] = 1; Wt[c] += (par * (tg - a) / nr)[:, None] * U
    Bres = {}
    for c, (rule, eps, par) in enumerate(RULES):
        nm = f'{rule}' + (f' eps={eps:g}' if rule != 'D' else '') + (f' kappa={par:g}' if rule == 'A' else '') + (f' eta={par:g}' if rule == 'D' else '')
        w = wsh(right[c])
        Bres[nm] = dict(windows=w, last_window=w[-1], last_quarter=float(right[c, q0:].mean()),
                        first_window_reaching={k: next((wins[i][1] for i in range(len(w)) if w[i] >= l[i]), None) for k, l in lv.items()})
    out['x0'][f'{x0:g}'] = dict(band_cells=int(len(band)), drivers=drv, inputs_per_driver=dict(min=int(P.min() - 1), median=float(np.median(P) - 1), max=int(P.max() - 1)),
                                mean_output_inputs=float(Xa.mean()), A=A, B=Bres, seconds=time.time() - t1)
    del Xa
    json.dump(out, open(os.path.join(HERE, 'res', f'x0_{label}.json'), 'w'), indent=0)
    print(f"== {label} x0={x0:g} (band {len(band)} cells; inputs/driver median {np.median(P) - 1:.0f}; mean input {out['x0'][f'{x0:g}']['mean_output_inputs']:.3g}; {time.time() - t1:.0f} s)")
    for k, v in A.items(): print(f'   A {k:28s} ' + ' / '.join(f"{100 * v['shares'][q]:.2f}" for q in ('all', 'R1', 'R2')))
    for k, v in Bres.items(): print(f"   B {k:22s} last window {100 * v['last_window']:5.2f}  last quarter {100 * v['last_quarter']:5.2f}  first reaching {v['first_window_reaching']}")
    sys.stdout.flush()
print(f'== {label} common: ' + ', '.join(f"{k} {100 * v['shares']['all']:.2f}" for k, v in common.items()) + f'; total {time.time() - t0:.0f} s', flush=True)
