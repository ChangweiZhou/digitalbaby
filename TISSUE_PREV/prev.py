# The previous byte at the reading step: measurement (item 1) and the candidates (item 2) of the reviewer's message of 19 November 2026
# (05:49 UTC). One tissue of tissue.c (the reviewer's program, copied; additions marked "TISSUE_PREV"), base setting rule=fixed gain=2 rho=1.
#
#   python3 -I prev.py <label> rule=fixed gain=2 rho=1 [cand=none|rho|tau|delay|input] [share=1/4] [rhohi=2] [tau=16] [delay=8] [insteps=8]
#
# Writes prev/<label>.json:
#   item1: by distance d = 1..11, the share of a cell's output variance explained by the previous byte and by the present byte (raw outputs,
#          diag.py's eta2), median and upper decile, read after steps 1, 2, 4, 8, 16 and 32 of the byte (window A2);
#   table: S(d, L) for d = 1, 2, 4 and L = 0, 1, 2 (readings.py, as drive.py), c0 at lags 1 and 2, the share neither silent nor constant at
#          d = 1, 2, 4, and the divergence of two copies (one cell's potential changed by 1e-9 at byte NB - 4096 - 512) after 1 ... 512 bytes.
# Cells or connections given a candidate are drawn once by numpy's default_rng(20261119) among the cells that are not feedback cells
# (connections: among the connections between cells present at birth).
import ctypes
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tis  # noqa: E402

ts, readings = tis.ts, tis.readings
label = sys.argv[1]; kw = dict(a.split('=') for a in sys.argv[2:])
cand = kw.pop('cand', 'none'); share = float(eval(kw.pop('share', '1/4')))
rhohi = float(eval(kw.pop('rhohi', '2'))); tau = float(eval(kw.pop('tau', '16'))); delay = int(kw.pop('delay', '8'))
insteps = int(kw.pop('insteps', '8'))
LEVEL = {'0': 0.1103, '1': 0.090, '2': 0.076}


def make():
    T = tis.make(kw)
    t, A, B, N = T['t'], T['A'], T['B'], T['N']
    rng = np.random.default_rng(20261119)
    nonfb = np.nonzero(B['kind'] != ts.FBK)[0]
    extra = {}
    if cand == 'rho':
        pick = rng.choice(nonfb, int(round(share * len(nonfb))), replace=False)
        r = np.full(N, t.rho); r[pick] = rhohi; extra['rhoc'] = r
        t.rhoc = r.ctypes.data
    elif cand == 'tau':
        pick = rng.choice(nonfb, int(round(share * len(nonfb))), replace=False)
        tv = np.full(N, 2.0); tv[pick] = tau; extra['tauv'] = tv
        t.tauv = tv.ctypes.data
    elif cand == 'delay':
        maxd = A['c_src'].shape[1]
        valid = np.arange(maxd)[None, :] < A['deg'][:, None]
        cc = np.argwhere(valid & (A['c_src'] < N))                  # connections between cells
        pick = cc[rng.choice(len(cc), int(round(share * len(cc))), replace=False)]
        dl = np.zeros((N, maxd), np.int32); dl[pick[:, 0], pick[:, 1]] = delay
        hl = delay + 2
        zh = np.zeros((hl, N)); extra['c_del'] = dl; extra['zh'] = zh
        t.c_del = dl.ctypes.data; t.zh = zh.ctypes.data; t.hlen = hl
    elif cand == 'input':
        t.in_steps = insteps
    else:
        assert cand == 'none'
    T['extra'] = extra                                              # kept alive with the tissue
    return T


def run(T, lo, hi, rec_lo=None):
    t, L, b = T['t'], T['L'], T['b']
    n = 0 if rec_lo is None else hi - rec_lo
    rec = np.zeros((max(n, 1), T['N']), np.float32)
    r0 = hi if rec_lo is None else rec_lo
    L.tis_run(ctypes.byref(t), ctypes.c_void_p(b.ctypes.data), ctypes.c_int64(lo), ctypes.c_int64(hi), ctypes.c_int64(r0), ctypes.c_int64(hi),
              ctypes.c_void_p(rec.ctypes.data))
    return rec


def eta2(x, lab):                                                   # diag.py's
    tot = x.var(axis=0); m = x.mean(axis=0); bet = np.zeros(x.shape[1])
    for v in np.unique(lab):
        s = lab == v; bet += s.mean() * (x[s].mean(axis=0) - m) ** 2
    return np.where(tot > 1e-30, bet / np.maximum(tot, 1e-30), np.nan)


c0t = time.process_time()
T1 = make(); T2 = make(); NB, N = T1['NB'], T1['N']
P0 = NB - 4096 - 512
run(T1, 0, P0); run(T2, 0, P0)
assert np.array_equal(T1['A']['v'], T2['A']['v'])
j = int(np.argmax(T2['A']['xp'])); T2['A']['v'][j] += 1e-9
d1 = run(T1, P0, P0 + 512, P0); d2 = run(T2, P0, P0 + 512, P0)
diff = np.abs(d1.astype(np.float64) - d2).max(axis=1)
divergence = {str(n): float(diff[n - 1]) for n in (1, 4, 16, 64, 256, 512)}
del T2
lo, hi = NB - 4096, NB
rec6 = np.zeros((6, hi - lo, N), np.float32)
T1['L'].tis_run_phases6(ctypes.byref(T1['t']), ctypes.c_void_p(T1['b'].ctypes.data), ctypes.c_int64(P0 + 512), ctypes.c_int64(hi),
                        ctypes.c_int64(lo), ctypes.c_int64(hi), ctypes.c_void_p(rec6.ctypes.data))
cpu = time.process_time() - c0t
b, A, B = T1['b'], T1['A'], T1['B']
lines = set(ts.LINE0 + int(v) for v in set(b[lo:hi].tolist()))
deg = A['deg']; src = A['c_src']; isfb = B['kind'] == ts.FBK
dist = np.zeros(N, np.int16)
fr = np.array([(not isfb[i]) and any(int(s) in lines for s in src[i, :deg[i]]) for i in range(N)]); dist[fr] = 1; d = 1
while fr.any():
    nxt = np.array([(dist[i] == 0) and (not isfb[i]) and bool(fr[src[i, :deg[i]][src[i, :deg[i]] < N]].any()) for i in range(N)])
    d += 1; dist[nxt] = d; fr = nxt
STEPS = (1, 2, 4, 8, 16, 32)
item1 = {}
for q, s in enumerate(STEPS):
    rec = rec6[q].astype(np.float64)
    e0 = eta2(rec[1:], b[lo + 1:hi]); e1 = eta2(rec[1:], b[lo:hi - 1])
    for dd in range(1, 12):
        c = dist == dd
        if not c.any():
            continue
        qf = lambda e, p: float(np.nanpercentile(e[c], p)) if np.isfinite(e[c]).any() else None
        item1.setdefault(str(dd), dict(cells=int(c.sum())))[f'step_{s}'] = dict(previous_median=qf(e1, 50), previous_p90=qf(e1, 90),
                                                                             present_median=qf(e0, 50), present_p90=qf(e0, 90),
                                                                             mean_output=float(rec[:, c].mean()))
X = rec6[5]
x = np.where(X < 0.01, 0.0, X)
r = readings.read_window(x, b[lo:hi], dist, T1['SEED'], stream=b, start=lo, lags=(0, 1, 2))
S = {str(dd): {str(Lg): (r['scores'][dd]['by_lag'][Lg] or {}).get('S') for Lg in (0, 1, 2)} for dd in (1, 2, 4)}
table = dict(S=S, above_level={dd: {Lg: (v is not None and v >= LEVEL[Lg]) for Lg, v in row.items()} for dd, row in S.items()},
             c0={str(k): (v.get('S') if isinstance(v, dict) else v) for k, v in r['c0'].items()},
             share={str(dd): r['shares'][dd]['share'] if r['shares'][dd]['n'] else None for dd in (1, 2, 4)},
             cells={str(dd): r['shares'][dd]['n'] for dd in (1, 2, 4)}, divergence=divergence)
out = dict(label=label, args=kw, cand=cand, share=share, rhohi=rhohi if cand == 'rho' else None, tau=tau if cand == 'tau' else None,
           delay=delay if cand == 'delay' else None, insteps=insteps if cand == 'input' else None, cpu_s=cpu,
           unreached=int(((dist == 0) & ~isfb).sum()), gain_median=float(np.median(A['gain'])), item1=item1, table=table)
os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prev'), exist_ok=True)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prev', f'{label}.json'), 'w'), indent=1, default=float)
f = lambda v: '  —  ' if v is None else f'{v:.3f}'
print(f"{label:24s} cpu {cpu:4.0f}s | S0 " + ' '.join(f(S[dd]['0']) for dd in ('1', '2', '4')) + ' | S1 ' + ' '.join(f(S[dd]['1']) for dd in ('1', '2', '4'))
      + ' | S2 ' + ' '.join(f(S[dd]['2']) for dd in ('1', '2', '4')) + f" | c0 L1 {f(table['c0'].get('1'))} L2 {f(table['c0'].get('2'))} | share "
      + ' '.join(f(table['share'][dd]) for dd in ('1', '2', '4')) + f" | div512 {divergence['512']:.1e}", flush=True)
