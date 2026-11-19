# One tissue of tis.make, run once; then, on its recorded outputs at the reading step:
#   A. fitted ridge readers of the next byte (as reach.py and reach_nbhd.py of output_reach_20261119): set (a), the cells near the
#      output lines; each byte on its own driver's neighbourhood; set (b), every non-feedback cell; and the same reader on one, two and
#      three one-hots of the last bytes. Fitted on the first three quarters of the positions, scored on the last quarter; inputs
#      standardised over the fitting part (no floor; a cell that does not vary is left out); ridge constant from 10^-6 .. 10^2 (times the
#      number of fitting positions) chosen on the last fifth of the fitting part. Share right overall and at R1, R2 (the scorer's kinds),
#      with the scorer's ref_1, ref_2, ref_3 on the same positions.
#   B. online one-pass rules on the cells near each driver (driver v: the set-(a) cells in its own neighbourhood, the driver included,
#      and a constant 1), as yesno3.py of output_reach2_20261119: yes/no A (kappa 1/8; eps 0 and 1/16), yes/no B (eps 1/16), supervised
#      D (eta 1/64 and 1/16, step eta (target - activity) u / |u|^2). Two preparations of the inputs:
#        run  running standardisation (m += r (x - m), s2 += r ((x - m)^2 - s2), r = 1 / min(n, 1024)); lms_diag.py found the fixed
#             standardisation no better and eta 1/64 the best step of D;
#        raw  the outputs as they are (no standardisation).
#   C. for every online rule, the share right in windows of 4,096 life positions, and the end of the first window whose share reached
#      the end mark's share, ref_1 and ref_2 of that window.
# usage: measure.py <label> [tissue options of drive.py] seed=.. nbytes=..   -> res/<label>.json
import sys, os, json, time, ctypes, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, 'deps', 'PREP'))
import tis, life_stream as LS, score_emissions as SE
label = sys.argv[1]; kw = dict(a.split('=') for a in sys.argv[2:]); t0 = time.time()
T = tis.make(kw); NB = T['NB']; N = T['N']; ts = tis.ts; B = T['B']; b = T['b'].astype(np.int64); isfb = B['kind'] == ts.FBK
L = LS.life(T['SEED']); assert np.array_equal(L['bytes'][:NB], T['b'])
allc = np.flatnonzero(~isfb)
drivers = list(B['driver']); near = set(drivers)
for i in drivers: near.update(int(m) for m in B['members'][i] if m < N)
near = np.array(sorted(i for i in near if not isfb[i])); K = len(near)
pa = {c: k for k, c in enumerate(allc)}; ia = np.array([pa[c] for c in near])         # set (a) inside set (b)
colk = {c: k for k, c in enumerate(near)}
own = [sorted({colk[c] for c in [i] + [int(m) for m in B['members'][i] if m < N] if c in colk}) for i in drivers]
G_ = max(map(len, own)); IDX = np.full((256, G_), K, np.int64)
for v in range(256): IDX[v, :len(own[v])] = own[v]
P = np.array([len(o) + 1.0 for o in own])
# ---- run the tissue once
R = np.zeros((NB, len(allc)), np.float32)
for lo in range(0, NB, 4096):
    hi = min(NB, lo + 4096); rec = np.zeros((hi - lo, N), np.float32)
    T['L'].tis_run(ctypes.byref(T['t']), ctypes.c_void_p(T['b'].ctypes.data), ctypes.c_int64(lo), ctypes.c_int64(hi), ctypes.c_int64(lo), ctypes.c_int64(hi), ctypes.c_void_p(rec.ctypes.data))
    R[lo:hi] = rec[:, allc]
t_tissue = time.time() - t0
M = NB - 1; y = b[1:]; nf = (3 * M) // 4; nin = (4 * nf) // 5; q0 = nf + 1
Y = np.zeros((M, 256)); Y[np.arange(M), y] = 1
refs = SE.references(L['bytes'][:NB]); K_ = SE.kinds(L)
pos = np.arange(nf, M) + 1
kmask = {'all': np.ones(len(pos), bool), 'R1': K_['R1 (first family, phase FIRST)'][:NB][pos], 'R2': K_['R2 (first family, phase FIRST)'][:NB][pos]}
shares = lambda g: {k: float((g[m] == b[pos][m]).mean()) for k, m in kmask.items()}
# ---- A. fitted readers
def ridge_all(X):
    """256 outputs on all columns of X (positions 0 .. M-1); returns the scored-quarter bytes and the chosen constant."""
    def fit(n):
        mu = X[:n].mean(0); my = Y[:n].mean(0); Xc = X[:n] - mu; G = Xc.T @ Xc; R_ = Xc.T @ (Y[:n] - my); return mu, my, G, R_
    best, bl = -1, None
    mu, my, G, R_ = fit(nin)
    for e in range(-6, 3):
        Gl = G.copy(); Gl[np.diag_indices_from(Gl)] += 10.0 ** e * nin
        s = float((((X[nin:nf] - mu) @ np.linalg.solve(Gl, R_) + my).argmax(1) == y[nin:nf]).mean())
        if s >= best: best, bl = s, e
    mu, my, G, R_ = fit(nf); G[np.diag_indices_from(G)] += 10.0 ** bl * nf
    return ((X[nf:] - mu) @ np.linalg.solve(G, R_) + my).argmax(1), 10.0 ** bl
def stdz(X):
    sd = X[:nf].std(0); keep = sd > 0; return X[:, keep] / sd[keep]
A = {}
Xb = stdz(R[:M].astype(np.float64)); g, lam = ridge_all(Xb); A['set (b), every non-feedback cell'] = dict(shares=shares(g), ridge=lam, features=int(Xb.shape[1])); del Xb
Xa = R[:M][:, ia].astype(np.float64); sda = Xa[:nf].std(0); Xa = np.where(sda > 0, Xa / np.where(sda > 0, sda, 1), 0.0)
g, lam = ridge_all(Xa[:, sda > 0]); A['set (a), near the output lines'] = dict(shares=shares(g), ridge=lam, features=int((sda > 0).sum()))
def ridge_own():
    def pred(n, lam):
        out = np.zeros((M, 256))
        for v in range(256):
            Xv = Xa[:, own[v]]; mu = Xv[:n].mean(0); my = Y[:n, v].mean(); Xc = Xv[:n] - mu
            G = Xc.T @ Xc; G[np.diag_indices_from(G)] += lam * n; out[:, v] = (Xv - mu) @ np.linalg.solve(G, Xc.T @ (Y[:n, v] - my)) + my
        return out
    inner = {e: float((pred(nin, 10.0 ** e)[nin:nf].argmax(1) == y[nin:nf]).mean()) for e in range(-6, 3)}
    be = max(inner, key=lambda e: (inner[e], e)); return pred(nf, 10.0 ** be)[nf:].argmax(1), 10.0 ** be
g, lam = ridge_own(); A["each byte on its own driver's neighbourhood"] = dict(shares=shares(g), ridge=lam)
tt = np.arange(M)
for h in (1, 2, 3):
    OH = np.zeros((M, 256 * h))
    for j in range(h):
        q = tt[tt >= j]; OH[q, 256 * j + b[q - j]] = 1
    g, lam = ridge_all(OH); A[f'{h} one-hot(s) of the last bytes'] = dict(shares=shares(g), ridge=lam)
for k in ('ref_1', 'ref_2', 'ref_3'): A[k] = dict(shares=shares(refs[k][pos].astype(np.int64)))
A['end mark'] = dict(shares={k: float((b[pos][m] == 10).mean()) for k, m in kmask.items()})
t_fit = time.time() - t0
# ---- B, C. online rules
RULES = [('A', 0.0, 1 / 8), ('A', 1 / 16, 1 / 8), ('B', 1 / 16, 0.0), ('D', 0.0, 1 / 64), ('D', 0.0, 1 / 16)]
CFG = [(p, r) for p in ('run', 'raw') for r in RULES]; C = len(CFG)
Wt = np.zeros((C, 256, G_ + 1)); rbar = np.zeros((C, 256)); rng = np.random.default_rng(T['SEED'])
right = np.zeros((C, NB), bool); seen = np.zeros(256, bool)
m1 = np.zeros(K); s1 = np.zeros(K)
for t in range(M):
    x = R[t, ia].astype(np.float64); r = 1.0 / min(t + 1, 1024); seen[b[t]] = True
    m1 += r * (x - m1); s1 += r * ((x - m1) ** 2 - s1); sd = np.sqrt(s1)
    u1 = np.where(sd > 0, (x - m1) / np.where(sd > 0, sd, 1), 0.0)
    Um = {}
    for nm, z in (('run', u1), ('raw', x)):
        U = np.ones((256, G_ + 1)); U[:, :-1] = np.append(z, 0.0)[IDX]; Um[nm] = U
    yy = y[t]
    for c, (prep, (rule, eps, par)) in enumerate(CFG):
        U = Um[prep]; a = (Wt[c] * U).sum(1); e = int(a.argmax())
        if eps and rng.random() < eps:
            sv = np.flatnonzero(seen); e = int(sv[rng.integers(0, len(sv))])
        ok = e == yy; right[c, t + 1] = ok
        if rule == 'A':
            Wt[c, e] += (1.0 if ok else -par) * (0.25 / P[e]) * U[e]
        elif rule == 'B':
            Wt[c, e] += (float(ok) - rbar[c, e]) * (0.25 / P[e]) * U[e]; rbar[c, e] += (float(ok) - rbar[c, e]) / 64
        else:
            tg = np.zeros(256); tg[yy] = 1; Wt[c] += (par * (tg - a) / (U * U).sum(1))[:, None] * U
wins = [(a_, min(a_ + 4096, NB)) for a_ in range(0, NB, 4096)]
wsh = lambda ok: [float(ok[max(a_, 1):e_].mean()) for a_, e_ in wins]
lv = {'end mark': [float((b[a_:e_] == 10).mean()) for a_, e_ in wins], 'ref_1': wsh(refs['ref_1'] == b), 'ref_2': wsh(refs['ref_2'] == b)}
Bres = {}
for c, (prep, (rule, eps, par)) in enumerate(CFG):
    nm = f'{prep} {rule}' + (f' eps={eps:g}' if rule != 'D' else '') + (f' kappa={par:g}' if rule == 'A' else '') + (f' eta={par:g}' if rule == 'D' else '')
    w = wsh(right[c])
    Bres[nm] = dict(windows=w, last_window=w[-1], last_quarter=float(right[c, q0:].mean()),
                    first_window_reaching={k: next((wins[i][1] for i in range(len(w)) if w[i] >= l[i]), None) for k, l in lv.items()})
out = dict(label=label, args=kw, bytes=NB, cells_a=int(K), cells_b=int(len(allc)), inputs_per_driver=dict(min=int(P.min() - 1), median=float(np.median(P) - 1), max=int(P.max() - 1)),
           mean_output_a=float(R[:, ia].mean()), max_output_a=float(R[:, ia].max()), A=A, B=Bres, levels=lv, windows=wins,
           seconds=dict(tissue=t_tissue, fitted=t_fit - t_tissue, all=time.time() - t0))
os.makedirs(os.path.join(HERE, 'res'), exist_ok=True); json.dump(out, open(os.path.join(HERE, 'res', f'{label}.json'), 'w'), indent=0)
print(f'== {label} ({NB} bytes; set a {K} cells, mean output {out["mean_output_a"]:.3g}; {time.time() - t0:.0f} s)')
for k, v in A.items(): print(f'   A {k:45s} ' + ' / '.join(f"{100 * v['shares'][q]:.2f}" for q in ('all', 'R1', 'R2')))
for k, v in Bres.items(): print(f"   B {k:30s} last window {100 * v['last_window']:5.2f}  last quarter {100 * v['last_quarter']:5.2f}  first reaching {v['first_window_reaching']}")
sys.stdout.flush()
