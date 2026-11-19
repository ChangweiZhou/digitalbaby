# Check of the standardisation used by measure.py (asked by the reviewer at 10:59 UTC: on L8-256 seed 93683 the fitted reader on set (a)
# gave R1 0.5 % and R2 0.0 %, below what naming the commonest byte gives). The tissue is run once (65,536 bytes); on the cells of set (a)
# (the birth's drivers and their neighbourhoods, as measure.py):
#  (1) the standardised inputs: with the deviation of the fitting part (as measure.py's fitted readers) and with the running deviation
#      (as measure.py's online "run"; it starts with m = 0 and s2 = 0 and rate r = 1/min(n, 1024), so that after the first byte m = x and
#      s2 = 0, and a cell silent at first has a deviation of 0, then a tiny one when it first answers): the largest |input|, the number of
#      cells with |input| > 10 at some position, the number of entries above 10, and how many of these lie in the last quarter;
#  (2) the fitted readers on set (a) and on each driver's own neighbourhood: the ridge constant chosen and the share right on the
#      fitting part itself and on the scored quarter, overall and at R1, R2;
#  (3) the same fitted readers and online rules with the deviation floored: sd' = max(sd, sd_max / 64), sd_max the largest deviation among
#      the cells of set (a) (fitted: over the fitting part; online: the largest running deviation at that step); and the online rules with
#      the running-standardised inputs clipped at +-8; beside them the raw inputs and the unfloored standardisation.
# usage: diag_std.py <label> [tissue options] seed=.. nbytes=..   -> res/diagstd_<label>.json
import sys, os, json, time, ctypes, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, 'deps', 'PREP'))
import tis, life_stream as LS, score_emissions as SE
label = sys.argv[1]; kw = dict(a.split('=') for a in sys.argv[2:]); t0 = time.time()
T = tis.make(kw); NB = T['NB']; N = T['N']; ts = tis.ts; B = T['B']; b = T['b'].astype(np.int64); isfb = B['kind'] == ts.FBK
L = LS.life(T['SEED'])
drivers = list(B['driver']); near = set(drivers)
for i in drivers: near.update(int(m) for m in B['members'][i] if m < N)
near = np.array(sorted(i for i in near if not isfb[i])); K = len(near); colk = {c: k for k, c in enumerate(near)}
own = [sorted({colk[c] for c in [i] + [int(m) for m in B['members'][i] if m < N] if c in colk}) for i in drivers]
G_ = max(map(len, own)); IDX = np.full((256, G_), K, np.int64)
for v in range(256): IDX[v, :len(own[v])] = own[v]
P = np.array([len(o) + 1.0 for o in own])
R = np.zeros((NB, N), np.float32)
for lo in range(0, NB, 4096):
    hi = min(NB, lo + 4096)
    T['L'].tis_run(ctypes.byref(T['t']), ctypes.c_void_p(T['b'].ctypes.data), ctypes.c_int64(lo), ctypes.c_int64(hi), ctypes.c_int64(lo), ctypes.c_int64(hi), ctypes.c_void_p(R[lo:hi].ctypes.data))
X = R[:NB - 1][:, near].astype(np.float64); del R
M = NB - 1; y = b[1:]; nf = (3 * M) // 4; nin = (4 * nf) // 5; q0 = nf + 1
Y = np.zeros((M, 256)); Y[np.arange(M), y] = 1
K_ = SE.kinds(L); refs = SE.references(L['bytes'][:NB])
def kmask(lo, hi):
    p = np.arange(lo, hi) + 1
    return {'all': np.ones(len(p), bool), 'R1': K_['R1 (first family, phase FIRST)'][:NB][p], 'R2': K_['R2 (first family, phase FIRST)'][:NB][p]}
KF, KT = kmask(0, nf), kmask(nf, M)
sh = lambda g, km, yy: {k: float((g[m] == yy[m]).mean()) for k, m in km.items()}
out = dict(label=label, args=kw, bytes=NB, cells_a=int(K), inputs_per_driver=dict(median=float(np.median(P) - 1)))
# (1) the standardised inputs
mu = X[:nf].mean(0); sd = X[:nf].std(0)
def zstat(Z, name):
    a = np.abs(Z); big = a > 10
    out[name] = dict(largest=float(a.max()), cells_above_10=int(big.any(0).sum()), entries_above_10=int(big.sum()),
                     entries_above_10_in_last_quarter=int(big[nf:].sum()), cells_with_zero_deviation=int((sd == 0).sum()) if 'fitting' in name else None)
Zf = np.where(sd > 0, (X - mu) / np.where(sd > 0, sd, 1), 0.0); zstat(Zf, 'fitting-part standardisation')
sdf = np.maximum(sd, sd.max() / 64); Zff = (X - mu) / sdf; zstat(Zff, 'fitting-part standardisation, floored')
# running standardisation (as measure.py), floored and clipped forms
Zr = np.zeros_like(X); Zrf = np.zeros_like(X); m1 = np.zeros(K); s1 = np.zeros(K)
for t in range(M):
    r = 1.0 / min(t + 1, 1024); x = X[t]; m1 += r * (x - m1); s1 += r * ((x - m1) ** 2 - s1); s = np.sqrt(s1)
    Zr[t] = np.where(s > 0, (x - m1) / np.where(s > 0, s, 1), 0.0)
    fl = max(s.max() / 64, 1e-300); Zrf[t] = (x - m1) / np.maximum(s, fl)
zstat(Zr, 'running standardisation'); zstat(Zrf, 'running standardisation, floored')
# (2), (3) fitted readers
def ridge_all(Z):
    def fit(n):
        m_ = Z[:n].mean(0); my = Y[:n].mean(0); Zc = Z[:n] - m_; return m_, my, Zc.T @ Zc, Zc.T @ (Y[:n] - my)
    m_, my, G, Rr = fit(nin); best, bl = -1, 0
    for e in range(-6, 3):
        Gl = G.copy(); Gl[np.diag_indices_from(Gl)] += 10.0 ** e * nin
        s = float((((Z[nin:nf] - m_) @ np.linalg.solve(Gl, Rr) + my).argmax(1) == y[nin:nf]).mean())
        if s >= best: best, bl = s, e
    m_, my, G, Rr = fit(nf); G[np.diag_indices_from(G)] += 10.0 ** bl * nf; W = np.linalg.solve(G, Rr)
    g = ((Z - m_) @ W + my).argmax(1)
    return dict(ridge=10.0 ** bl, fitting_part=sh(g[:nf], KF, y[:nf]), scored_quarter=sh(g[nf:], KT, y[nf:]))
def ridge_own(Z):
    def pred(n, lam):
        o = np.zeros((M, 256))
        for v in range(256):
            Zv = Z[:, own[v]]; m_ = Zv[:n].mean(0); my = Y[:n, v].mean(); Zc = Zv[:n] - m_
            Gm = Zc.T @ Zc; Gm[np.diag_indices_from(Gm)] += lam * n; o[:, v] = (Zv - m_) @ np.linalg.solve(Gm, Zc.T @ (Y[:n, v] - my)) + my
        return o
    inner = {e: float((pred(nin, 10.0 ** e)[nin:nf].argmax(1) == y[nin:nf]).mean()) for e in range(-6, 3)}
    be = max(inner, key=lambda e: (inner[e], e)); g = pred(nf, 10.0 ** be).argmax(1)
    return dict(ridge=10.0 ** be, fitting_part=sh(g[:nf], KF, y[:nf]), scored_quarter=sh(g[nf:], KT, y[nf:]))
out['fitted'] = {}
for nm, Z in (('fitting-part standardisation (measure.py)', Zf), ('fitting-part standardisation, deviation floored at 1/64 of the largest', Zff)):
    out['fitted'][nm] = {'set (a)': ridge_all(Z), "own driver's neighbourhood": ridge_own(Z)}
    print(f'== {label} fitted, {nm}: ' + '; '.join(f"{k}: ridge {v['ridge']:.0e}, fitting part " + ' / '.join(f"{100 * v['fitting_part'][q]:.1f}" for q in ('all', 'R1', 'R2'))
          + ', scored ' + ' / '.join(f"{100 * v['scored_quarter'][q]:.1f}" for q in ('all', 'R1', 'R2')) for k, v in out['fitted'][nm].items()), flush=True)
del Zf, Zff
# online rules
PREPS = {'run (measure.py)': Zr, 'run, floored': Zrf, 'run, clipped at 8': np.clip(Zr, -8, 8), 'raw': X}
RULES = [('D', 0.0, 1 / 64), ('B', 1 / 16, 0.0), ('A', 0.0, 1 / 8), ('A', 1 / 16, 1 / 8)]
CFG = [(p, r) for p in PREPS for r in RULES]; C = len(CFG)
Wt = np.zeros((C, 256, G_ + 1)); rbar = np.zeros((C, 256)); rng = np.random.default_rng(T['SEED'])
right = np.zeros((C, NB), bool); seen = np.zeros(256, bool)
for t in range(M):
    seen[b[t]] = True; yy = y[t]; Um = {}
    for nm, Z in PREPS.items():
        U = np.ones((256, G_ + 1)); U[:, :-1] = np.append(Z[t], 0.0)[IDX]; Um[nm] = U
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
            tg = np.zeros(256); tg[yy] = 1; nr = (U * U).sum(1); nr[nr == 0] = 1; Wt[c] += (par * (tg - a) / nr)[:, None] * U
wins = [(a_, min(a_ + 4096, NB)) for a_ in range(0, NB, 4096)]
wsh = lambda ok: [float(ok[max(a_, 1):e_].mean()) for a_, e_ in wins]
lv = {'end mark': [float((b[a_:e_] == 10).mean()) for a_, e_ in wins], 'ref_1': wsh(refs['ref_1'] == b), 'ref_2': wsh(refs['ref_2'] == b)}
out['online'] = {}
for c, (prep, (rule, eps, par)) in enumerate(CFG):
    nm = f'{prep} | {rule}' + (f' eps={eps:g}' if rule != 'D' else '') + (f' kappa={par:g}' if rule == 'A' else '') + (f' eta={par:g}' if rule == 'D' else '')
    w = wsh(right[c]); pos = np.arange(nf, M) + 1
    out['online'][nm] = dict(windows=w, last_quarter=float(right[c, q0:].mean()), last_quarter_R=sh(np.where(right[c, pos], b[pos], -1), KT, b[pos]),
                             first_window_reaching={k: next((wins[i][1] for i in range(len(w)) if w[i] >= l[i]), None) for k, l in lv.items()})
out['levels_last_quarter'] = {k: float((refs[k][q0:] == b[q0:]).mean()) for k in ('ref_1', 'ref_2')}
out['seconds'] = time.time() - t0
json.dump(out, open(os.path.join(HERE, 'res', f'diagstd_{label}.json'), 'w'), indent=0)
for k in ('fitting-part standardisation', 'fitting-part standardisation, floored', 'running standardisation', 'running standardisation, floored'):
    print(f'== {label} {k}: ' + ', '.join(f'{a} {v}' for a, v in out[k].items() if v is not None), flush=True)
for k, v in out['online'].items():
    print(f"   {k:44s} last quarter {100 * v['last_quarter']:5.2f}  R1/R2 {100 * v['last_quarter_R']['R1']:.1f}/{100 * v['last_quarter_R']['R2']:.1f}  first reaching {v['first_window_reaching']}", flush=True)
print(f'== {label} done ({time.time() - t0:.0f} s)', flush=True)
