# drive11.py, by E52's worker: part A of the reviewer's message of 13:40. One run of a tissue of the reviewer's tissue.c and tis.py at
# 97ba8349 (beside this file, unchanged; -ffp-contract=off). The linear reader of perbyte3.py (ridge 0.01 of the mean diagonal, a constant
# 1, fitted on the first 4,096 of the last 8,192 bytes, read on the last 4,096; each input column standardised over the fitting half,
# columns with deviation at most 1e-9 left out, clipped to [-8, 8]) near the output lines (perbyte3's two groups), for the bytes 0 to 3
# back, with two input sets:
#   x          the outputs at the reading step of each byte, as perbyte3.py;
#   x,s2,s4    the same and, for every cell, slow copies once a byte: s_k(t) = s_k(t-1) + (x(t) - s_k(t-1)) / k, k = 2 and 4, started at
#              0 at byte NB - 8,192 - 64 (64 bytes before the window, so that they have settled when it starts).
# The reading for a byte value: share of its positions named right; "digits above 0.9": digit values with share above 0.9.
#   python3 -I drive11.py <label> <options as drive.py>
import sys, os, json, time, ctypes, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tis
ts = tis.ts
label = sys.argv[1]; kw = dict(a.split('=') for a in sys.argv[2:])
T = tis.make(kw); t, L, b, N, NB = T['t'], T['L'], T['b'], T['N'], T['NB']
W = 8192; PRE = 64; lo, hi = NB - W, NB; r0 = lo - PRE
rec = np.zeros((W + PRE, N), np.float32)
t0 = time.process_time()
L.tis_run(ctypes.byref(t), ctypes.c_void_p(b.ctypes.data), ctypes.c_int64(0), ctypes.c_int64(hi), ctypes.c_int64(r0), ctypes.c_int64(hi), ctypes.c_void_p(rec.ctypes.data))
cpu = time.process_time() - t0
B = T['B']; kind = np.asarray(B['kind']); isfb = kind == ts.FBK; P = np.asarray(B['P'])
near = np.zeros(N, bool); near8 = np.zeros(N, bool)
for i in np.where(kind == ts.OUTE)[0]:
    near[i] = near8[i] = True
    for m in B['members'][int(i)]:
        if m < N: near[m] = True
    near8[((P - P[i]) ** 2).sum(1) <= ts.R2] = True
near &= ~isfb; near8 &= ~isfb
xall = rec.astype(np.float64)
s2 = np.zeros_like(xall); s4 = np.zeros_like(xall); a2 = np.zeros(N); a4 = np.zeros(N)
for k in range(xall.shape[0]):
    a2 += (xall[k] - a2) / 2; a4 += (xall[k] - a4) / 4; s2[k] = a2; s4[k] = a4
xall, s2, s4 = xall[PRE:], s2[PRE:], s4[PRE:]
h = W // 2


def classes(v):
    c = chr(v) if 32 <= v < 127 else ''
    return 'end' if v == 10 else 'eq' if c == '=' else 'digit' if c.isdigit() else 'letter' if c.isalpha() else 'sign'


def summ(pred, yt, vals):
    rc = {v: float((pred[yt == v] == v).mean()) for v in vals if (yt == v).any()}
    by = {}
    for v, r in rc.items(): by.setdefault(classes(v), []).append(r)
    m = lambda k: float(np.mean(by[k])) if k in by else float('nan')
    return dict(overall=float((pred == yt).mean()), end=m('end'), eq=m('eq'), digits=m('digit'), digits_above_09=int(sum(r > 0.9 for r in by.get('digit', []))),
                letters=m('letter'), never=int(sum(r == 0 for r in rc.values())), values=len(rc))


out = dict(label=label, args=kw, ms_per_byte=cpu / NB * 1e3, groups={})
for name, g in (('near output', near), ('within 1/8 of a driver', near8)):
    res = {}
    for inp, X in (('x', xall[:, g]), ('x,s2,s4', np.hstack([xall[:, g], s2[:, g], s4[:, g]]))):
        sd = X[:h].std(0); keep = sd > 1e-9
        xs = np.clip((X[:, keep] - X[:h][:, keep].mean(0)) / sd[keep], -8, 8); xs = np.hstack([xs, np.ones((W, 1))])
        G = xs[:h].T @ xs[:h]; lam = 0.01 * np.trace(G) / G.shape[0]; Gi = np.linalg.solve(G + lam * np.eye(G.shape[0]), xs[:h].T)
        rr = {}
        for lag in (0, 1, 2, 3):
            y = b[lo - lag:hi - lag].astype(int); vals = sorted(set(y[:h].tolist())); yt = y[h:]
            Y = (y[:h, None] == np.array(vals)[None, :]).astype(np.float64)
            pred = np.array(vals)[(xs[h:] @ (Gi @ Y)).argmax(1)]
            rr[str(lag)] = summ(pred, yt, vals)
        res[inp] = rr
    out['groups'][name] = dict(cells=int(g.sum()), inputs=res)
os.makedirs(os.path.join(HERE, 'outA'), exist_ok=True)
json.dump(out, open(os.path.join(HERE, 'outA', label + '.json'), 'w'))
gg = out['groups']['near output']['inputs']
print(f"{label:20s} {out['ms_per_byte']:5.1f} ms/byte | near, all values 0-3 back: x " + " ".join(f"{gg['x'][str(l)]['overall']:.3f}" for l in range(4))
      + " | x,s2,s4 " + " ".join(f"{gg['x,s2,s4'][str(l)]['overall']:.3f}" for l in range(4)), flush=True)
