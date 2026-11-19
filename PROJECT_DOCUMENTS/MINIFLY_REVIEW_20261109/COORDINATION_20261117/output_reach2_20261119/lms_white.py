# Is the gap between the supervised one-pass rule and the fitted reader (lms_diag.py) the conditioning of each driver's inputs?
# For each driver: its own cells' outputs, standardised over the fitting part (as lms_diag.py's fixed standardisation).
#   (1) the spread of the inputs' covariance over the fitting part: the share of variance in the first principal component, and the
#       ratio of the largest eigenvalue to the median one (median over drivers);
#   (2) the same delta rule (one pass in order through all positions, step eta (target - activity) u / |u|^2, eta 1/64 and 1/16) on the
#       inputs whitened per driver by the covariance of the fitting part (a transformation no cell has: a diagnostic only, it tells
#       whether a one-pass rule could reach the fitted reader if its inputs were uncorrelated); scored on the last quarter.
# usage: lms_white.py <label> [tissue options] seed=.. nbytes=..   -> reach/white_<label>.json
import sys, os, json, time, ctypes, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import tis
label = sys.argv[1]; kw = dict(a.split('=') for a in sys.argv[2:]); t0 = time.time()
T = tis.make(kw); NB = T['NB']; N = T['N']; ts = tis.ts; B = T['B']; b = T['b'].astype(np.int64); isfb = B['kind'] == ts.FBK
drivers = list(B['driver']); near = set(drivers)
for i in drivers: near.update(int(m) for m in B['members'][i] if m < N)
near = np.array(sorted(i for i in near if not isfb[i])); col = {c: k for k, c in enumerate(near)}; K = len(near)
own = [sorted({col[c] for c in [i] + [int(m) for m in B['members'][i] if m < N] if c in col}) for i in drivers]
X = np.zeros((NB, K), np.float32)
for lo in range(0, NB, 4096):
    hi = min(NB, lo + 4096); rec = np.zeros((hi - lo, N), np.float32)
    T['L'].tis_run(ctypes.byref(T['t']), ctypes.c_void_p(T['b'].ctypes.data), ctypes.c_int64(lo), ctypes.c_int64(hi), ctypes.c_int64(lo), ctypes.c_int64(hi), ctypes.c_void_p(rec.ctypes.data))
    X[lo:hi] = rec[:, near]
X = X[:NB - 1].astype(np.float64); y = b[1:]; M = len(y); nf = (3 * M) // 4
mu = X[:nf].mean(0); sd = X[:nf].std(0); X = np.where(sd > 0, (X - mu) / np.where(sd > 0, sd, 1), 0.0)
G_ = max(map(len, own)); IDX = np.full((256, G_), K, np.int64); WH = np.zeros((256, G_, G_))
first, ratio = [], []
for v in range(256):
    cs = own[v]; C = np.cov(X[:nf][:, cs].T); ev, V = np.linalg.eigh(C); ev = np.clip(ev, 0, None)
    first.append(float(ev[-1] / ev.sum())); ratio.append(float(ev[-1] / max(np.median(ev), 1e-300)))
    keep = ev > ev[-1] * 1e-9; Wh = V[:, keep] / np.sqrt(ev[keep])          # whitening of the fitting part, rank-limited
    IDX[v, :len(cs)] = cs; WH[v, :len(cs), :keep.sum()] = Wh
Xp = np.concatenate([X, np.zeros((M, 1))], 1)
def inp(t):
    U = np.ones((256, G_ + 1)); U[:, :-1] = np.einsum('vg,vgh->vh', Xp[t][IDX], WH); return U
res = dict(first_component_share_median=float(np.median(first)), first_component_share_p90=float(np.percentile(first, 90)),
           eig_ratio_largest_to_median_median=float(np.median(ratio)))
for eta in (1 / 64, 1 / 16):
    W = np.zeros((256, G_ + 1)); ok = np.zeros(M, bool)
    for t in range(M):
        U = inp(t); a = (W * U).sum(1); ok[t] = a.argmax() == y[t]
        tg = np.zeros(256); tg[y[t]] = 1; W += (eta * (tg - a) / (U * U).sum(1))[:, None] * U
    res[f'one pass, whitened inputs, eta {eta:g}'] = float(ok[nf:].mean())
res['end mark share'] = float((y[nf:] == 10).mean())
json.dump(dict(label=label, args=kw, results=res, seconds=time.time() - t0), open(os.path.join(HERE, 'reach', f'white_{label}.json'), 'w'), indent=1)
print(f'== {label} ({time.time() - t0:.0f} s)'); [print(f'   {k:50s} {v:.4f}') for k, v in res.items()]; sys.stdout.flush()
