# perbyte.py by distance from the input lines: for the cells at 1, 2, 3, 4, 5 connections, for all cells, and for the cells near the
# output lines: how often a nearest-centroid reader names the present byte right, for the end mark, the equals sign, the digits, the
# letters and signs apart. Window: the last 8,192 bytes; centroids from its first half, read on its second half; floor 0.01.
# usage: perbyte2.py <label> rule=.. [options]
import sys, os, json, ctypes, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import tis
ts = tis.ts
label = sys.argv[1]; kw = dict(a.split('=') for a in sys.argv[2:])
T = tis.make(kw); t, L, b, N, NB = T['t'], T['L'], T['b'], T['N'], T['NB']
W = 8192; lo, hi = NB - W, NB
rec = np.zeros((W, N), np.float32)
L.tis_run(ctypes.byref(t), ctypes.c_void_p(b.ctypes.data), ctypes.c_int64(0), ctypes.c_int64(hi), ctypes.c_int64(lo), ctypes.c_int64(hi), ctypes.c_void_p(rec.ctypes.data))
A = T['A']; B = T['B']; kind = np.asarray(B['kind']); isfb = kind == ts.FBK
wb = b[lo:hi]; lines = set(ts.LINE0 + int(v) for v in set(wb.tolist()))
deg = A['deg']; src = A['c_src']
dist = np.zeros(N, np.int16)
fr = np.array([(not isfb[i]) and any(int(s) in lines for s in src[i, :deg[i]]) for i in range(N)]); dist[fr] = 1; d = 1
while fr.any():
    nxt = np.array([(dist[i] == 0) and (not isfb[i]) and bool(fr[src[i, :deg[i]][src[i, :deg[i]] < N]].any()) for i in range(N)])
    d += 1; dist[nxt] = d; fr = nxt
near = np.zeros(N, bool)
for i in np.where(kind == ts.OUTE)[0]:
    near[i] = True
    for m in B['members'][int(i)]:
        if m < N: near[m] = True
near &= ~isfb
x = np.where(rec < 0.01, 0.0, rec).astype(np.float64); h = W // 2
y = wb.astype(int); vals = sorted(set(y[:h].tolist()))
def classes(v):
    c = chr(v) if 32 <= v < 127 else ''
    return 'end' if v == 10 else 'eq' if c == '=' else 'digit' if c.isdigit() else 'letter' if c.isalpha() else 'sign'
out = dict(label=label, args=kw, groups={})
print(f"== {label}  group: cells | overall | end mark, equals sign | digits (mean recall; how many of 10 above 1/2) | letters (mean; of how many above 1/2) | signs | values never named")
for name, g in [('d=1', dist == 1), ('d=2', dist == 2), ('d=3', dist == 3), ('d=4', dist == 4), ('d=5', dist == 5), ('d>=6', dist >= 6), ('all', ~isfb), ('near output', near)]:
    if g.sum() < 2: continue
    xg = x[:, g]; C = np.stack([xg[:h][y[:h] == v].mean(0) for v in vals])
    xt = xg[h:]; yt = y[h:]
    dd = (xt ** 2).sum(1)[:, None] - 2 * xt @ C.T + (C ** 2).sum(1)[None, :]
    pred = np.array(vals)[dd.argmin(1)]
    rc = {v: float((pred[yt == v] == v).mean()) for v in vals if (yt == v).any()}
    by = {}
    for v, r in rc.items(): by.setdefault(classes(v), []).append(r)
    res = dict(cells=int(g.sum()), overall=float((pred == yt).mean()), recall={str(v): r for v, r in rc.items()}, never=int(sum(r == 0 for r in rc.values())), values=len(rc))
    out['groups'][name] = res
    f = lambda k: f"{np.mean(by.get(k, [np.nan])):.2f}"
    print(f"   {name:11s}: {res['cells']:4d} | {res['overall']:.3f} | {f('end')} {f('eq')} | {f('digit')} ({sum(r > 0.5 for r in by.get('digit', []))}/10) | {f('letter')} ({sum(r > 0.5 for r in by.get('letter', []))}/{len(by.get('letter', []))}) | {f('sign')} | {res['never']}/{res['values']}")
# the bytes one and two back among the cells near the output lines: overall share named right, and over the digits
xg = x[:, near]
for lag in (1, 2):
    yl = b[lo - lag:hi - lag].astype(int); vl = sorted(set(yl[:h].tolist()))
    C = np.stack([xg[:h][yl[:h] == v].mean(0) for v in vl]); xt = xg[h:]; yt = yl[h:]
    dd = (xt ** 2).sum(1)[:, None] - 2 * xt @ C.T + (C ** 2).sum(1)[None, :]; pred = np.array(vl)[dd.argmin(1)]
    dig = [float((pred[yt == v] == v).mean()) for v in vl if classes(v) == 'digit' and (yt == v).any()]
    out['groups'][f'near output, lag {lag}'] = dict(overall=float((pred == yt).mean()), digits=float(np.mean(dig)))
    print(f"   near output, the byte {lag} back: overall {float((pred == yt).mean()):.3f}, digits {np.mean(dig):.2f} ({sum(r > 0.5 for r in dig)}/10)")
os.makedirs('perbyte', exist_ok=True); json.dump(out, open(f'perbyte/{label}_by_distance.json', 'w')); sys.stdout.flush()
