# drive7.py: drive6.py of PREP/tissue6 run on the reviewer's folder at 8ed2eacc (copied unchanged beside this file; option tau=), with
# these readings added: the cells near the output lines (far.py's set: the output-line drivers and the members of their neighbourhoods,
# feedback cells excluded) read by the tool as one distance; the largest distance reached; the seed's own levels; for the birth: input
# lines present in the window that some cell receives from, inhibitory cells near the output lines. drive6.py's text:
# drive6.py: a driver of the reviewer's tissue.c and tis.py at 419e93bf (copied unchanged beside this file), by E52's worker, for the
# task "the making rule on the new base". The readings are drive4.py's (PREP/tissue4): the helper's tool as it is, S(d, L) and the share
# at d = 1, 2, 4, 8, 10, connections a cell (mean and largest, non-feedback cells) at the end of every window, made and withdrawn. Added:
#   - cells at d with any output in the window (largest raw output > 0) and with a largest output above 1/2;
#   - the divergence of two copies (diag.py's method): 512 bytes before the last window the tissue is copied (every array and counter
#     of the C structure), one cell's potential is changed by 1e-9 in the copy, and the largest difference of outputs after 1, 4, 16,
#     64, 256, 512 bytes is read; an unchanged copy is run beside it and must give 0 (a check of the copying);
#   - per-cell data at the end, for the question of the cells with many connections: out/<label>_cells.npz (degree, distance in the
#     last window, number of connections holding more than 1/128 of the cell's excitatory or inhibitory total, M, abar, largest raw
#     output and the share of the output variance explained by the present byte in the last window).
# For a life of more than 16,384 bytes it reads four windows: the last 4,096 bytes of each quarter.
#   python3 -I drive6.py <label> <options as drive.py>
import sys, os, time, json, ctypes, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tis
from tis import ts, readings

DS = (1, 2, 4, 8, 10)
label = sys.argv[1]; kw = dict(a.split('=') for a in sys.argv[2:])
M1 = tis.make(kw)
t, A, B, L, b, N, NB, SEED = (M1[k] for k in ('t', 'A', 'B', 'L', 'b', 'N', 'NB', 'SEED'))
if NB > 16384:
    q = NB // 4
    WIN = {f'Q{j + 1}': ((j + 1) * q - 4096, (j + 1) * q) for j in range(4)}
else:
    WIN = {'A1': (2048, 6144), 'A2': (NB - 4096, NB)}
isfb = B['kind'] == ts.FBK
SCAL = ('rng', 'made', 'given', 'given_by_use', 'k', 'active_sum')


def run(T, lo, hi, r0=None):
    n = 0 if r0 is None else hi - r0
    rec = np.zeros((max(n, 1), N), np.float32)
    rr = hi if r0 is None else r0
    T['L'].tis_run(ctypes.byref(T['t']), ctypes.c_void_p(b.ctypes.data), ctypes.c_int64(lo), ctypes.c_int64(hi), ctypes.c_int64(rr), ctypes.c_int64(hi), ctypes.c_void_p(rec.ctypes.data))
    return rec


def clone():
    T = tis.make(kw)
    for k, a in A.items():
        np.copyto(T['A'][k], a)
    for s in SCAL:
        setattr(T['t'], s, getattr(t, s))
    return T


def distances(present):
    lines = set(ts.LINE0 + int(v) for v in present)
    deg = A['deg']; src = A['c_src']
    dist = np.zeros(N, np.int16)
    fr = np.array([(not isfb[i]) and any(int(s) in lines for s in src[i, :deg[i]]) for i in range(N)])
    dist[fr] = 1; d = 1
    while fr.any():
        nxt = np.array([(dist[i] == 0) and (not isfb[i]) and bool(fr[src[i, :deg[i]][src[i, :deg[i]] < N]].any()) for i in range(N)])
        d += 1; dist[nxt] = d; fr = nxt
    return dist


def conns():
    deg = A['deg'][~isfb]
    return dict(mean=float(deg.mean()), max=int(deg.max()), mean_all=float(A['deg'].sum()) / N)


out = dict(label=label, args=kw, windows=list(WIN)); pos = 0; t0 = time.process_time()
recs = {}; dists = {}; cw = {}
last = list(WIN)[-1]
for w, (lo, hi) in WIN.items():
    if w == last:
        P0 = lo - 512
        run(M1, pos, P0); pos = P0
        T2 = clone(); T3 = clone()
        j = int(np.argmax(T2['A']['xp'])); T2['A']['v'][j] += 1e-9
        d1 = run(M1, P0, lo, P0); d2 = run(T2, P0, lo, P0); d3 = run(T3, P0, lo, P0)
        diff = np.abs(d1.astype(np.float64) - d2).max(axis=1)
        out['divergence'] = {str(n): float(diff[n - 1]) for n in (1, 4, 16, 64, 256, 512)}
        out['copy_check_max_diff'] = float(np.abs(d1.astype(np.float64) - d3).max())
        del T2, T3
        pos = lo
    rec = run(M1, pos, hi, lo)
    recs[w] = rec; dists[w] = distances(sorted(set(b[lo:hi].tolist()))); pos = hi
    cw[w] = dict(conns(), made=int(t.made), given=int(t.given), given_by_use=int(t.given_by_use))
cpu = time.process_time() - t0
out.update(ms_per_byte=cpu / NB * 1e3, conns_per_cell=float(A['deg'].sum()) / N, conns_end=conns(), made=int(t.made), given=int(t.given),
           given_by_use=int(t.given_by_use), active_per_step=t.active_sum / t.k, gain_median=float(np.median(A['gain'])))
for w, (lo, hi) in WIN.items():
    raw = recs[w]; mx = raw.max(axis=0)
    x = np.where(raw < 0.01, 0.0, raw)
    r = readings.read_window(x, b[lo:hi], dists[w], SEED, stream=b, start=lo, distances=DS, lags=(0, 1, 2))
    out[w] = dict(at_end=cw[w], cells={str(d): int((dists[w] == d).sum()) for d in DS},
                  cells_any_output={str(d): int(((dists[w] == d) & (mx > 0)).sum()) for d in DS},
                  cells_above_half={str(d): int(((dists[w] == d) & (mx > 0.5)).sum()) for d in DS},
                  share={str(d): r['shares'][d]['share'] if r['shares'][d]['n'] else None for d in DS}, n={str(d): r['shares'][d]['n'] for d in DS},
                  S={str(d): {str(Lg): (r['scores'][d]['by_lag'][Lg] or {}).get('S') for Lg in (0, 1, 2)} for d in DS},
                  c0={str(k): (v.get('S') if isinstance(v, dict) else v) for k, v in r['c0'].items()})
# added in drive7: near the output lines, largest distance, levels, birth counts (last window)
lo7, hi7 = WIN[last]; x7 = np.where(recs[last] < 0.01, 0.0, recs[last]); kind = np.asarray(B['kind'])
near = np.zeros(N, bool)
for i in np.where(kind == ts.OUTE)[0]:
    near[i] = True
    for mm in B['members'][int(i)]:
        if mm < N: near[mm] = True
near &= ~isfb
r7 = readings.read_window(x7, b[lo7:hi7], np.where(near, 1, 0).astype(np.int16), SEED, stream=b, start=lo7, distances=(1,), lags=(0, 1, 2))
c07 = out[last]['c0']
lev = {'0': 0.1103, '1': (c07.get('1') or 0) + 0.02, '2': (c07.get('2') or 0) + 0.02}
mx7 = recs[last].max(axis=0); dl = dists[last]
present_lines = set(ts.LINE0 + int(v) for v in set(b[lo7:hi7].tolist()))
reached_lines = set(int(s) for i in range(N) if not isfb[i] for s in A['c_src'][i, :A['deg'][i]] if int(s) in present_lines)
out['levels'] = lev
out['near'] = dict(cells=int(near.sum()), answering=int((near & (mx7 > 0)).sum()), S={str(Lg): (r7['scores'][1]['by_lag'][Lg] or {}).get('S') for Lg in (0, 1, 2)},
                   share=r7['shares'][1]['share'] if r7['shares'][1]['n'] else None,
                   inhibitory=int((near & (np.asarray(B['sig'][:N]) < 0)).sum()), by_distance={str(k): int((near & (dl == k)).sum()) for k in range(0, int(dl.max()) + 1)})
out['max_distance'] = int(dl.max()); out['cells_d8_10'] = int(((dl >= 8) & (dl <= 10)).sum())
out['lines_present'] = len(present_lines); out['lines_reached'] = len(reached_lines)
out['cells_by_distance'] = {str(k): int((dl == k).sum()) for k in range(0, int(dl.max()) + 1)}
out['answering_by_distance'] = {str(k): int(((dl == k) & (mx7 > 0)).sum()) for k in range(0, int(dl.max()) + 1)}
# per-cell data at the end (last window)
lo, hi = WIN[last]; raw = recs[last].astype(np.float64)
lab = b[lo:hi]; tot = raw.var(axis=0); m = raw.mean(axis=0); bet = np.zeros(N)
for v in np.unique(lab):
    s = lab == v; bet += s.mean() * (raw[s].mean(axis=0) - m) ** 2
eta_present = np.where(tot > 1e-30, bet / np.maximum(tot, 1e-30), np.nan)
deg = A['deg']; src = A['c_src']; w_ = A['c_w']; sig = B['sig']
strong = np.zeros(N, np.int32); exc_n = np.zeros(N, np.int32)
for i in range(N):
    if isfb[i] or deg[i] == 0:
        continue
    ss = src[i, :deg[i]]; ww = w_[i, :deg[i]]; pos_ = sig[ss] > 0
    te = ww[pos_].sum(); ti = ww[~pos_].sum()
    strong[i] = int((ww[pos_] > te / 128).sum() + (ww[~pos_] > ti / 128).sum()); exc_n[i] = int(pos_.sum())
np.savez_compressed(os.path.join(HERE, 'out', label + '_cells.npz'), deg=deg.copy(), exc=exc_n, strong=strong, dist=dists[last], isfb=isfb,
                    M=A['M'][:N].copy(), abar=A['abar'].copy(), maxout=raw.max(axis=0), eta_present=eta_present, gain=A['gain'].copy())
os.makedirs(os.path.join(HERE, 'out'), exist_ok=True)
json.dump(out, open(os.path.join(HERE, 'out', label + '.json'), 'w'), default=float)
a = out[last]; f = lambda v: '  -  ' if v is None else f'{v:.3f}'
print(f"{label:34s} {out['ms_per_byte']:6.2f} ms/byte conns {out['conns_end']['mean']:5.1f} max {out['conns_end']['max']:3d} made {out['made']} given {out['given']} | {last} S0 d1 2 4 8 10 "
      + " ".join(f(a['S'][str(d)]['0']) for d in DS) + f" | any output d8 {a['cells_any_output']['8']}/{a['cells']['8']} d10 {a['cells_any_output']['10']}/{a['cells']['10']}"
      + f" | near S0 {f(out['near']['S']['0'])} maxd {out['max_distance']}" + f" | div512 {out['divergence']['512']:.1e} copy {out['copy_check_max_diff']:.1e}", flush=True)
