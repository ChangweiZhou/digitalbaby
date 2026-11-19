# reach.py's reader with each output restricted to its own driver's neighbourhood: the output for byte v is a ridge regression of
# 1[next byte = v] on the set-(a) cells in driver v's neighbourhood (the driver included; the same inputs as yesno.py's units), each cell
# standardised over the fitting part as in reach.py. The same 256 outputs on all of set (a) (reach.py's reader) are recomputed beside it.
# Fitted on the first three quarters, scored on the last quarter; ridge constant chosen on the last fifth of the fitting part.
# usage: reach_nbhd.py <label> rule=fixed [gain=2 rho=1 gmax=4] seed=.. nbytes=..   -> reach/nbhd_<label>.json
import sys, os, json, time, ctypes, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, 'deps', 'PREP'))
import tis, life_stream as LS, score_emissions as SE
label = sys.argv[1]; kw = dict(a.split('=') for a in sys.argv[2:]); t0 = time.time()
T = tis.make(kw); NB = T['NB']; N = T['N']; ts = tis.ts; B = T['B']; b = T['b']; isfb = B['kind'] == ts.FBK
drivers = list(B['driver']); near = set(drivers)
for i in drivers: near.update(int(m) for m in B['members'][i] if m < N)
near = np.array(sorted(i for i in near if not isfb[i])); col = {c: k for k, c in enumerate(near)}
own = [sorted({col[c] for c in [i] + [int(m) for m in B['members'][i] if m < N] if c in col}) for i in drivers]
rec = np.zeros((NB, N), np.float32)
T['L'].tis_run(ctypes.byref(T['t']), ctypes.c_void_p(b.ctypes.data), ctypes.c_int64(0), ctypes.c_int64(NB), ctypes.c_int64(0), ctypes.c_int64(NB), ctypes.c_void_p(rec.ctypes.data))
X = rec[:NB - 1][:, near].astype(np.float64); del rec
tt = np.arange(NB - 1); y = b[1:].astype(np.int64); nf = (3 * len(tt)) // 4; nin = (4 * nf) // 5
sd = X[:nf].std(0); X = np.where(sd > 0, X / np.where(sd > 0, sd, 1), 0.0)
Y = np.zeros((len(tt), 256)); Y[tt, y] = 1
def fitpred(lo, hi, lam, cols_of):
    S = np.zeros((len(tt) - lo if hi is None else 0, 256))
    out = np.zeros((len(tt), 256))
    for v in range(256):
        cs = cols_of(v); Xa = X[:hi][:, cs] if hi else X[:nf][:, cs]
        n = Xa.shape[0]; mu = Xa.mean(0); my = Y[:n, v].mean(); Xc = Xa - mu
        G = Xc.T @ Xc; G[np.diag_indices_from(G)] += lam * n
        w = np.linalg.solve(G, Xc.T @ (Y[:n, v] - my)); out[:, v] = (X[:, cs] - mu) @ w + my
    return out
res = {}
for name, cols_of in (('own_neighbourhood', lambda v: own[v]), ('all_of_set_a', lambda v: np.arange(len(near)))):
    if name == 'all_of_set_a':
        def solve(n, lam):
            Xa = X[:n]; mu = Xa.mean(0); my = Y[:n].mean(0); Xc = Xa - mu; G = Xc.T @ Xc; G[np.diag_indices_from(G)] += lam * n
            return (X - mu) @ np.linalg.solve(G, Xc.T @ (Y[:n] - my)) + my
    else:
        def solve(n, lam):
            return fitpred(0, n, lam, cols_of)
    inner = {e: float((solve(nin, 10.0 ** e)[nin:nf].argmax(1) == y[nin:nf]).mean()) for e in range(-6, 3)}
    best = max(inner, key=lambda e: (inner[e], e)); g = solve(nf, 10.0 ** best)[nf:].argmax(1)
    res[name] = dict(ridge=10.0 ** best, inner_check=inner, right=float((g == y[nf:]).mean()))
refs = SE.references(b); pos = tt[nf:] + 1
for k in ('ref_1', 'ref_2', 'ref_3'): res[k] = dict(right=float((refs[k][pos] == b[pos]).mean()))
res['end mark share'] = dict(right=float((b[pos] == 10).mean()))
out = dict(label=label, args=kw, bytes=NB, own_inputs=dict(min=min(map(len, own)), median=float(np.median(list(map(len, own)))), max=max(map(len, own))), readers=res, seconds=time.time() - t0)
json.dump(out, open(os.path.join(HERE, 'reach', f'nbhd_{label}.json'), 'w'), indent=1)
print(label, ' '.join(f"{k} {100 * v['right']:.2f}" for k, v in res.items()), f"({time.time() - t0:.0f} s)"); sys.stdout.flush()
