# What differs near the output lines between seeds (asked by the reviewer at 11:19 UTC): the fitted ridge reader of the next byte on
# set (a) and on each driver's own neighbourhood (as measure.py), scored on the last quarter by the class of the predicted position
# (life_stream.CLASSES: operands A1, A2, B1, B2; answer digits R1, R2; other: TASK, SIGN, LET, EQ, END), in two forms of input:
#   std    the outputs standardised over the fitting part (measure.py's fitted readers);
#   tool   the readings tool's inputs: outputs below 0.01 set to 0, then standardised over the fitting part (cells that never pass 0.01 in
#          the fitting part left out).
# Beside them, on the same positions: linear readers on one, two and three one-hots of the last bytes, and ref_1, ref_2, ref_3.
# usage: by_class.py <label> [tissue options] seed=.. nbytes=..   -> res/byclass_<label>.json
import sys, os, json, time, ctypes, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, 'deps', 'PREP'))
import tis, life_stream as LS, score_emissions as SE
label = sys.argv[1]; kw = dict(a.split('=') for a in sys.argv[2:]); t0 = time.time()
T = tis.make(kw); NB = T['NB']; N = T['N']; ts = tis.ts; B = T['B']; b = T['b'].astype(np.int64); isfb = B['kind'] == ts.FBK
L = LS.life(T['SEED'])
drivers = list(B['driver']); near = set(drivers)
for i in drivers: near.update(int(m) for m in B['members'][i] if m < N)
near = np.array(sorted(i for i in near if not isfb[i])); colk = {c: k for k, c in enumerate(near)}
own = [sorted({colk[c] for c in [i] + [int(m) for m in B['members'][i] if m < N] if c in colk}) for i in drivers]
R = np.zeros((NB, N), np.float32)
for lo in range(0, NB, 4096):
    hi = min(NB, lo + 4096)
    T['L'].tis_run(ctypes.byref(T['t']), ctypes.c_void_p(T['b'].ctypes.data), ctypes.c_int64(lo), ctypes.c_int64(hi), ctypes.c_int64(lo), ctypes.c_int64(hi), ctypes.c_void_p(R[lo:hi].ctypes.data))
X = R[:NB - 1][:, near].astype(np.float64); del R
M = NB - 1; y = b[1:]; nf = (3 * M) // 4; nin = (4 * nf) // 5
Y = np.zeros((M, 256)); Y[np.arange(M), y] = 1
pos = np.arange(nf, M) + 1; cl = L['cls'][:NB][pos]
GROUPS = {'all': np.ones(len(pos), bool), 'operands (A1 A2 B1 B2)': np.isin(cl, (2, 3, 6, 7)), 'R1': cl == 9, 'R2': cl == 10,
          'other (TASK SIGN LET EQ END)': np.isin(cl, (1, 4, 5, 8, 11))}
for c, nm in LS.CLASSES.items():
    if c and (cl == c).any(): GROUPS[nm] = cl == c
sh = lambda g: {k: dict(n=int(m.sum()), right=float((g[m] == b[pos][m]).mean())) for k, m in GROUPS.items()}
def stdz(Z):
    sd = Z[:nf].std(0); keep = sd > 0; return Z[:, keep] / sd[keep], keep
def ridge_all(Z):
    def fit(n):
        m_ = Z[:n].mean(0); my = Y[:n].mean(0); Zc = Z[:n] - m_; return m_, my, Zc.T @ Zc, Zc.T @ (Y[:n] - my)
    m_, my, G, Rr = fit(nin); best, bl = -1, 0
    for e in range(-6, 3):
        Gl = G.copy(); Gl[np.diag_indices_from(Gl)] += 10.0 ** e * nin
        s = float((((Z[nin:nf] - m_) @ np.linalg.solve(Gl, Rr) + my).argmax(1) == y[nin:nf]).mean())
        if s >= best: best, bl = s, e
    m_, my, G, Rr = fit(nf); G[np.diag_indices_from(G)] += 10.0 ** bl * nf
    return ((Z[nf:] - m_) @ np.linalg.solve(G, Rr) + my).argmax(1)
def ridge_own(Z, keep):
    idx = np.cumsum(keep) - 1
    ow = [idx[[o for o in ov if keep[o]]] for ov in own]
    def pred(n, lam):
        o = np.zeros((M, 256))
        for v in range(256):
            if len(ow[v]) == 0: o[:, v] = Y[:n, v].mean(); continue
            Zv = Z[:, ow[v]]; m_ = Zv[:n].mean(0); my = Y[:n, v].mean(); Zc = Zv[:n] - m_
            Gm = Zc.T @ Zc; Gm[np.diag_indices_from(Gm)] += lam * n; o[:, v] = (Zv - m_) @ np.linalg.solve(Gm, Zc.T @ (Y[:n, v] - my)) + my
        return o
    inner = {e: float((pred(nin, 10.0 ** e)[nin:nf].argmax(1) == y[nin:nf]).mean()) for e in range(-6, 3)}
    be = max(inner, key=lambda e: (inner[e], e)); return pred(nf, 10.0 ** be)[nf:].argmax(1)
out = dict(label=label, args=kw, bytes=NB, cells_a=int(len(near)), readers={})
for form, Z0 in (('std', X), ('tool', np.where(X >= 0.01, X, 0.0))):
    Z, keep = stdz(Z0)
    out['readers'][f'{form}: set (a)'] = sh(ridge_all(Z))
    out['readers'][f"{form}: own driver's neighbourhood"] = sh(ridge_own(Z, keep))
    out[f'{form}: cells used'] = int(keep.sum())
tt = np.arange(M)
for h in (1, 2, 3):
    OH = np.zeros((M, 256 * h))
    for j in range(h):
        q = tt[tt >= j]; OH[q, 256 * j + b[q - j]] = 1
    out['readers'][f'{h} one-hot(s)'] = sh(ridge_all(OH))
refs = SE.references(L['bytes'][:NB])
for k in ('ref_1', 'ref_2', 'ref_3'): out['readers'][k] = sh(refs[k][pos].astype(np.int64))
out['seconds'] = time.time() - t0
json.dump(out, open(os.path.join(HERE, 'res', f'byclass_{label}.json'), 'w'), indent=1)
ks = ['all', 'operands (A1 A2 B1 B2)', 'R1', 'R2', 'other (TASK SIGN LET EQ END)']
print(f"== {label}: share right by class of the predicted position, last quarter, %: " + ' / '.join(ks) + f"  (n {' / '.join(str(GROUPS[k].sum()) for k in ks)})")
for k, v in out['readers'].items(): print(f'   {k:36s} ' + ' / '.join(f"{100 * v[q]['right']:.1f}" for q in ks))
print(f"   classes: " + ', '.join(f"{c} " + '/'.join(f"{100 * out['readers'][r][c]['right']:.0f}" for r in ('std: set (a)', 'tool: set (a)', '2 one-hot(s)')) for c in LS.CLASSES.values() if c in GROUPS) + '  (std set (a) / tool set (a) / two one-hots)', flush=True)
