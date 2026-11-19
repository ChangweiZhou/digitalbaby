# What a LINEAR reader can take from the cells near the output lines (and from all cells), byte value by byte value: a ridge
# reader (one output a byte value, least squares with ridge 0.01 of the mean diagonal, inputs divided by their deviation and
# clipped at 8) fitted on the first half of the last 8,192 bytes and read on the second half, for the present byte and the
# bytes 1, 2, 3 back. Beside it the nearest-centroid reader of perbyte.py. A slow cell holds a sum of several bytes; a linear
# reader can take such a sum apart where the nearest-centroid reader cannot.
# Exactly (asked by E62's worker): the window is the last 8,192 bytes; its first 4,096 fit, its last 4,096 are read. Inputs: the
# outputs as they are (no floor); mean and deviation over the fitting half; (x - mean)/sd; cells with deviation at most 1e-9 left
# out; clipped to [-8, 8]; then a constant 1 appended (under the ridge like the rest). G = X'X on the fitting half; ridge = 0.01 *
# trace(G) / columns on every diagonal entry. One output for each byte value seen in the fitting half as the target (the byte at
# t - lag for the output read at byte t); reading = the largest output, a tie to the lowest value. "digits", "letters" = the mean
# of the per-value shares; "end", "eq" = the share of that value; "never" = values with share 0. The nearest-centroid reader
# beside it sets outputs below 0.01 to zero.
# usage: perbyte3.py <label> rule=.. [options]
import sys, os, json, ctypes, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import tis
ts = tis.ts
label = sys.argv[1]; kw = dict(a.split('=') for a in sys.argv[2:])
T = tis.make(kw); t, L, b, N, NB = T['t'], T['L'], T['b'], T['N'], T['NB']
W = 8192; lo, hi = NB - W, NB
rec = np.zeros((W, N), np.float32)
L.tis_run(ctypes.byref(t), ctypes.c_void_p(b.ctypes.data), ctypes.c_int64(0), ctypes.c_int64(hi), ctypes.c_int64(lo), ctypes.c_int64(hi), ctypes.c_void_p(rec.ctypes.data))
B = T['B']; kind = np.asarray(B['kind']); isfb = kind == ts.FBK; P = np.asarray(B['P'])
near = np.zeros(N, bool); near8 = np.zeros(N, bool)
for i in np.where(kind == ts.OUTE)[0]:
    near[i] = near8[i] = True
    for m in B['members'][int(i)]:
        if m < N: near[m] = True
    near8[((P - P[i]) ** 2).sum(1) <= ts.R2] = True
near &= ~isfb; near8 &= ~isfb
h = W // 2
def classes(v):
    c = chr(v) if 32 <= v < 127 else ''
    return 'end' if v == 10 else 'eq' if c == '=' else 'digit' if c.isdigit() else 'letter' if c.isalpha() else 'sign'
def summ(pred, yt, vals):
    rc = {v: float((pred[yt == v] == v).mean()) for v in vals if (yt == v).any()}
    by = {}
    for v, r in rc.items(): by.setdefault(classes(v), []).append(r)
    m = lambda k: float(np.mean(by[k])) if k in by else float('nan')
    return dict(overall=float((pred == yt).mean()), end=m('end'), eq=m('eq'), digits=m('digit'), digits_above_half=int(sum(r > 0.5 for r in by.get('digit', []))),
                letters=m('letter'), never=int(sum(r == 0 for r in rc.values())), values=len(rc))
out = dict(label=label, args=kw, groups={})
print(f"== {label}   reader: overall | end mark, equals sign | digits (how many of 10 above 1/2) | letters | values never named")
for name, g in (('near output', near), ('within 1/8 of a driver', near8), ('all cells', ~isfb)):
    x = rec[:, g].astype(np.float64); sd = x[:h].std(0); keep = sd > 1e-9
    xs = np.clip((x[:, keep] - x[:h][:, keep].mean(0)) / sd[keep], -8, 8); xs = np.hstack([xs, np.ones((W, 1))])
    xf = np.where(rec[:, g] < 0.01, 0.0, rec[:, g]).astype(np.float64)
    G = xs[:h].T @ xs[:h]; lam = 0.01 * np.trace(G) / G.shape[0]; Gi = np.linalg.solve(G + lam * np.eye(G.shape[0]), xs[:h].T)
    res = {}
    for lag in (0, 1, 2, 3):
        y = b[lo - lag:hi - lag].astype(int); vals = sorted(set(y[:h].tolist())); yt = y[h:]
        Y = (y[:h, None] == np.array(vals)[None, :]).astype(np.float64)
        pred = np.array(vals)[(xs[h:] @ (Gi @ Y)).argmax(1)]
        C = np.stack([xf[:h][y[:h] == v].mean(0) for v in vals]); xt = xf[h:]
        dd = (xt ** 2).sum(1)[:, None] - 2 * xt @ C.T + (C ** 2).sum(1)[None, :]; predc = np.array(vals)[dd.argmin(1)]
        res[str(lag)] = dict(linear=summ(pred, yt, vals), centroid=summ(predc, yt, vals))
        for rn in ('linear', 'centroid'):
            r = res[str(lag)][rn]
            print(f"   {name:22s} ({int(g.sum()):4d} cells) byte {lag} back, {rn:8s}: {r['overall']:.3f} | {r['end']:.2f} {r['eq']:.2f} | {r['digits']:.2f} ({r['digits_above_half']}/10) | {r['letters']:.2f} | {r['never']}/{r['values']}")
    out['groups'][name] = dict(cells=int(g.sum()), lags=res)
os.makedirs('perbyte', exist_ok=True); json.dump(out, open(f'perbyte/{label}_linear.json', 'w')); sys.stdout.flush()
