# Readings of one tissue of tissue.c at the end of one or more stretches of its life (the reviewer's task of 19 November, 06:16 UTC).
# Groups "near18" and "nearL" (added for the long-axon tissues): the drivers with every cell within 1/8 of a driver, and the drivers with
# every member of their (enlarged) neighbourhoods.
# usage: python3 -I ntd.py <label> ends=16384[,65536,262144] seed=.. rule=.. [options of tis.make]
# For each end E, window W = bytes E - 4096 .. E - 1 (the output of every cell after each byte), and:
#  - divergence: at byte E - 4608 a second tissue is made equal to the first in every state (memory copy), one cell's potential is
#    changed by 1e-9 (the most active cell, as diag.py does), both run 512 bytes; the number of cells whose output differs, and the
#    largest difference, after 1, 16, 64, 256 and 512 bytes;
#  - by distance d = 1 .. largest (from the lines of the byte values present in W, as drive.py) and for the pool of all ordinary cells:
#    cells; mean and largest output; share of (byte, cell) above 0.01; median share of a cell's output variance explained by the present
#    byte and by the previous byte (raw outputs, as diag.py); S(d, lag) for lags 0..4 (a) by the readings tool as it is (outputs below
#    0.01 set to 0) and (b) with no floor, each cell divided by its own standard deviation (as the helper's where.py, 4bda7aa1);
#  - C0(lag); connections a cell (mean, smallest, largest, over the 1,984 ordinary cells); connections holding more than half of their
#    cell's excitatory total; connections given up and made so far; cells sending a step; ms a byte.
import sys, os, json, ctypes, time
from random import Random
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import tis
import readings_tool as RD
label = sys.argv[1]; kw = dict(a.split('=') for a in sys.argv[2:])
ends = [int(e) for e in kw.pop('ends', '16384').split(',')]
kw['nbytes'] = str(max(ends))
LAGS = (0, 1, 2, 3, 4)
T1 = tis.make(kw); T2 = tis.make(kw); ts = tis.ts
N, SEED, b, A, B, L = T1['N'], T1['SEED'], T1['b'], T1['A'], T1['B'], T1['L']
isfb = B['kind'] == ts.FBK; ordinary = ~isfb


def run(T, lo, hi, rec_lo=None):
    n = 0 if rec_lo is None else hi - rec_lo
    rec = np.zeros((max(n, 1), N), np.float32)
    r0 = hi if rec_lo is None else rec_lo
    T['L'].tis_run(ctypes.byref(T['t']), ctypes.c_void_p(b.ctypes.data), ctypes.c_int64(lo), ctypes.c_int64(hi), ctypes.c_int64(r0), ctypes.c_int64(hi), ctypes.c_void_p(rec.ctypes.data))
    return rec


def copy_state(src, dst):
    for k, a in src['A'].items():
        dst['A'][k][...] = a
    ctypes.memmove(ctypes.byref(dst['t']), ctypes.byref(src['t']), ctypes.sizeof(src['t']))
    for k, a in dst['A'].items():
        setattr(dst['t'], k, a.ctypes.data)


def distances(lo, hi):
    lines = set(ts.LINE0 + int(v) for v in set(b[lo:hi].tolist()))
    deg = A['deg']; src = A['c_src']
    dist = np.zeros(N, np.int16)
    fr = np.array([(not isfb[i]) and any(int(s) in lines for s in src[i, :deg[i]]) for i in range(N)]); dist[fr] = 1; d = 1
    while fr.any():
        nxt = np.array([(dist[i] == 0) and (not isfb[i]) and bool(fr[src[i, :deg[i]][src[i, :deg[i]] < N]].any()) for i in range(N)])
        d += 1; dist[nxt] = d; fr = nxt
    return dist


def eta2(x, lab):
    tot = x.var(axis=0); m = x.mean(axis=0); bet = np.zeros(x.shape[1])
    for v in np.unique(lab):
        s = lab == v; bet += s.mean() * (x[s].mean(axis=0) - m) ** 2
    return np.where(tot > 1e-30, bet / np.maximum(tot, 1e-30), np.nan)


def strengths():
    deg = A['deg']; src = A['c_src'].astype(np.int64); w = A['c_w']; sig = A['sig']
    n = deg[ordinary]
    big = 0; tot_conn = 0
    for i in np.nonzero(ordinary)[0]:
        d = deg[i]
        if not d:
            continue
        s = src[i, :d]; ww = w[i, :d]; exc = sig[s] > 0
        et = ww[exc].sum()
        if et > 0:
            big += int((ww[exc] > 0.5 * et).sum())
        tot_conn += d
    return dict(conns_mean=float(n.mean()), conns_min=int(n.min()), conns_max=int(n.max()), conns_total=int(tot_conn), conns_over_half_of_exc_total=big)


out = dict(label=label, args=kw, seed=SEED, ends=ends, windows={})
pos = 0; cpu = 0.0
for E in ends:
    lo, P0 = E - 4096, E - 4096 - 512
    c = time.process_time(); run(T1, pos, P0); cpu += time.process_time() - c
    copy_state(T1, T2)
    j = int(np.argmax(T2['A']['xp'])); T2['A']['v'][j] += 1e-9
    c = time.process_time(); d1 = run(T1, P0, lo, P0); cpu += time.process_time() - c
    d2 = run(T2, P0, lo, P0)
    diff = np.abs(d1.astype(np.float64) - d2)
    div = {str(n): dict(cells=int((diff[n - 1] > 0).sum()), largest=float(diff[n - 1].max())) for n in (1, 16, 64, 256, 512)}
    c = time.process_time(); X = run(T1, lo, E, lo).astype(np.float64); cpu += time.process_time() - c
    pos = E
    t = T1['t']
    dist = distances(lo, E)
    wl = E - lo
    Xa = np.where(X < 0.01, 0.0, X); sd = X.std(axis=0); Xb = np.where(sd > 0, X / np.where(sd > 0, sd, 1.0), 0.0)
    e0 = eta2(X[1:], b[lo + 1:E]); e1 = eta2(X[1:], b[lo:E - 1])

    def score(F, cells, L_):
        if not cells:
            return None
        idx, lab = RD._samples(wl, L_, b[lo:E], b, lo)
        S, se, k = RD.classify(F[idx][:, cells], lab, idx, wl)
        return S

    def sample(cells):
        cells = sorted(int(c) for c in cells)
        return Random(SEED + 900000).sample(cells, 128) if len(cells) > 128 else cells

    rows = []
    groups = [(str(dd), np.nonzero(dist == dd)[0]) for dd in range(1, int(dist.max()) + 1)] + [("pool", np.nonzero(ordinary)[0])]
    # the cells near the output lines, two ways: the drivers with every cell within 1/8 of a driver (members at the definition's birth),
    # and the drivers with every member of their neighbourhoods as the tissue has them (enlarged by long axons, if any)
    drv = [int(x) for x in B['driver']]; B0 = ts.birth(SEED)
    n8 = set(drv); nL = set(drv)
    for i in drv:
        n8.update(int(m) for m in B0['members'][i] if m < N); nL.update(int(m) for m in B['members'][i] if m < N)
    groups += [("near18", np.array(sorted(c for c in n8 if not isfb[c]))), ("nearL", np.array(sorted(c for c in nL if not isfb[c])))]
    for name, cells in groups:
        if len(cells) == 0:
            continue
        samp = sample(cells); samp_b = [c_ for c_ in samp if sd[c_] > 0]
        med = lambda e: float(np.nanmedian(e[cells])) if np.isfinite(e[cells]).any() else None
        rows.append(dict(d=name, cells=int(len(cells)), mean_output=float(X[:, cells].mean()), max_output=float(X[:, cells].max()),
                         share_above_001=float((X[:, cells] > 0.01).mean()), cells_max_over_half=float((X[:, cells].max(axis=0) > 0.5).mean()), present=med(e0), previous=med(e1),
                         S_tool={str(L_): score(Xa, samp, L_) for L_ in LAGS}, S_nofloor={str(L_): score(Xb, samp_b, L_) for L_ in LAGS}))
    c0 = RD.c0(b[lo:E], b, lo, LAGS)
    out['windows'][str(E)] = dict(window=[lo, E - 1], divergence=div, rows=rows, c0={str(k): v["S"] for k, v in c0.items()}, max_distance=int(dist.max()),
                                  unreached=int(((dist == 0) & ordinary).sum()), given_up=int(t.given), made=int(t.made),
                                  cells_sending_a_step=float(t.active_sum / t.k), **strengths())
    w_ = out['windows'][str(E)]
    f = lambda v: '  -  ' if v is None else f'{v:.3f}'
    print(f"== {label} end {E}: conns {w_['conns_mean']:.2f} ({w_['conns_min']}-{w_['conns_max']}), >half {w_['conns_over_half_of_exc_total']}, given up {w_['given_up']}, made {w_['made']}, "
          f"cells a step {w_['cells_sending_a_step']:.0f}; divergence cells 1/16/64/256/512: " + ' '.join(str(div[k]['cells']) for k in ('1', '16', '64', '256', '512')), flush=True)
    for r in rows:
        print(f"   d={r['d']:>4s} n {r['cells']:4d} mean {r['mean_output']:.2e} max {r['max_output']:.3f} >.01 {r['share_above_001']:.3f} pres {f(r['present'])} prev {f(r['previous'])} | "
              f"tool S0-4 " + ' '.join(f(r['S_tool'][str(L_)]) for L_ in LAGS) + " | nofloor S0-4 " + ' '.join(f(r['S_nofloor'][str(L_)]) for L_ in LAGS), flush=True)
    print("   C0 0-4 " + ' '.join(f(out['windows'][str(E)]['c0'][str(L_)]) for L_ in LAGS), flush=True)
out['ms_per_byte'] = cpu / max(ends) * 1e3
os.makedirs(os.path.join(HERE, 'ntd'), exist_ok=True)
json.dump(out, open(os.path.join(HERE, 'ntd', label + '.json'), 'w'), default=float)
print(f"{label} ms/byte {out['ms_per_byte']:.2f}", flush=True)
