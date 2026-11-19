# drive10.py, by E52's worker: one run of a tissue of the reviewer's tissue.c and tis.py at 97ba8349 (beside this file, unchanged; built
# with -ffp-contract=off by tis.py) gives both:
#   - perbyte3.py's readings, computed as in perbyte3.py (its code below, unchanged in substance: the last 8,192 bytes recorded, the
#     ridge reader fitted on the first half and read on the second, the nearest-centroid reader beside it; groups "near output",
#     "within 1/8 of a driver" (perbyte3's set: within ts.R2 of a driver), "all cells"; bytes 0 to 3 back), and
#   - drive9's figures: connections a cell (mean and largest, non-feedback cells), made in the whole run and in the last quarter, the
#     divergence of two copies over the 512 bytes before the recorded window (one cell's potential changed by 1e-9; an unchanged copy
#     as a check), and the number of non-feedback cells with at least one connection from an input line at the end.
# The run is cut at byte 7,680 (copies), 8,192 (the window) and 12,288 (the last quarter); cutting a run does not change it.
#   python3 -I drive10.py <label> <options as drive.py>
import sys, os, json, time, ctypes, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tis
ts = tis.ts
label = sys.argv[1]; kw = dict(a.split('=') for a in sys.argv[2:])
T = tis.make(kw); t, L, b, N, NB, A = T['t'], T['L'], T['b'], T['N'], T['NB'], T['A']
W = 8192; lo, hi = NB - W, NB
Q3 = 3 * NB // 4
assert lo - 512 > 0 and lo <= Q3 <= hi
SCAL = ('rng', 'made', 'given', 'given_by_use', 'k', 'active_sum')


def run(TT, a, z, r0=None, rec=None):
    if rec is None:
        rec = np.zeros((1, N), np.float32); r0 = z
    TT['L'].tis_run(ctypes.byref(TT['t']), ctypes.c_void_p(b.ctypes.data), ctypes.c_int64(a), ctypes.c_int64(z), ctypes.c_int64(r0), ctypes.c_int64(z), ctypes.c_void_p(rec.ctypes.data))
    return rec


def clone():
    C = tis.make(kw)
    for k, a in A.items():
        np.copyto(C['A'][k], a)
    for s in SCAL:
        setattr(C['t'], s, getattr(t, s))
    return C


t0 = time.process_time()
P0 = lo - 512
run(T, 0, P0)
T2 = clone(); T3 = clone()
j = int(np.argmax(T2['A']['xp'])); T2['A']['v'][j] += 1e-9
d1 = run(T, P0, lo, P0, np.zeros((512, N), np.float32)); d2 = run(T2, P0, lo, P0, np.zeros((512, N), np.float32)); d3 = run(T3, P0, lo, P0, np.zeros((512, N), np.float32))
diff = np.abs(d1.astype(np.float64) - d2).max(axis=1)
del T2, T3
rec = np.zeros((W, N), np.float32)
run(T, lo, Q3, lo, rec[:Q3 - lo]) if Q3 > lo else None
made_q3 = int(t.made)
run(T, Q3, hi, Q3, rec[Q3 - lo:])
cpu = time.process_time() - t0
B = T['B']; kind = np.asarray(B['kind']); isfb = kind == ts.FBK; P = np.asarray(B['P'])
deg = A['deg']; src = A['c_src']
from_line = int(sum(1 for i in range(N) if not isfb[i] and (src[i, :deg[i]] >= ts.LINE0).any()))
out = dict(label=label, args=kw, ms_per_byte=cpu / NB * 1e3, conns_mean=float(deg[~isfb].mean()), conns_max=int(deg[~isfb].max()), made=int(t.made),
           made_last_quarter=int(t.made) - made_q3, divergence={str(n): float(diff[n - 1]) for n in (1, 4, 16, 64, 256, 512)},
           copy_check_max_diff=float(np.abs(d1.astype(np.float64) - d3).max()), cells_from_input_line=from_line, groups={})
# ---- perbyte3.py's readings (its text, from "near = ..." to the end of its loop, with only the printing and the file name changed)
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
                letters=m('letter'), never=int(sum(r == 0 for r in rc.values())), values=len(rc),
                digits_above_09=int(sum(r > 0.9 for r in by.get('digit', []))), digit_count=len(by.get('digit', [])))
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
    out['groups'][name] = dict(cells=int(g.sum()), lags=res)
json.dump(out, open(os.path.join(HERE, 'out', label + '.json'), 'w'))
g = out['groups']['near output']['lags']
print(f"{label:28s} {out['ms_per_byte']:5.1f} ms/byte conns {out['conns_mean']:.1f}/{out['conns_max']} made {out['made']} lastq {out['made_last_quarter']} from-line {from_line} div {diff[-1]:.0e} copy {out['copy_check_max_diff']:.0e} | near linear 0-3 back: "
      + " ".join(f"{g[str(l)]['linear']['overall']:.3f}" for l in range(4)) + " | end/eq 0 back " + f"{g['0']['linear']['end']:.2f}/{g['0']['linear']['eq']:.2f}", flush=True)
