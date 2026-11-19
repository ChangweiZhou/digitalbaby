# The all-or-none test (claude/e63-allornone): one tissue, its readings by distance, its scores, its dependence on its own past.
# usage: aon.py <label> rule=.. [allornone=1] [astar=..] [eta_t=..] [tauv1=1] [noinh=1] [seed=..] [other options of drive.py]
# Prints and writes aon/<label>.json:
#  - by distance d from the input lines (all distances): cells; share of (position, cell) at which the cell is active (x > 0) at the
#    reading step; mean output; share of a cell's output variance explained by the present byte and by the previous byte (median and
#    upper decile over cells; raw outputs), in window A2 (the last 4,096 bytes);
#  - S(d, L) for L = 0, 1, 2 at d = 1, 2, 4 and the three largest distances, by the readings tool (outputs below 0.01 set to 0, as in
#    drive.py), in A2; C0(L);
#  - divergence: two copies of one tissue from byte NB - 4096 - 512; in copy 2 at that byte (a) one cell's potential is changed by 1e-9
#    (as diag.py does) and, in a third copy, (b) the output of one active cell is set to 0 for one step; the number of cells whose output
#    at the reading step differs after 1, 16, 64, 256, 512 bytes;
#  - cells sending a step (mean over the life), ms a byte (process time of copy 1 over the whole life).
import sys, os, json, time, ctypes, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import tis
label = sys.argv[1]; kw = dict(a.split('=') for a in sys.argv[2:])
def run(T, lo, hi, rec_lo=None):
    t, L, b = T['t'], T['L'], T['b']
    n = 0 if rec_lo is None else hi - rec_lo
    rec = np.zeros((max(n, 1), T['N']), np.float32)
    r0 = hi if rec_lo is None else rec_lo
    L.tis_run(ctypes.byref(t), ctypes.c_void_p(b.ctypes.data), ctypes.c_int64(lo), ctypes.c_int64(hi), ctypes.c_int64(r0), ctypes.c_int64(hi), ctypes.c_void_p(rec.ctypes.data))
    return rec
T1, T2, T3 = tis.make(kw), tis.make(kw), tis.make(kw); NB = T1['NB']; N = T1['N']; ts = tis.ts; readings = tis.readings
P0 = NB - 4096 - 512
c0 = time.process_time(); run(T1, 0, P0); cpu1 = time.process_time() - c0
run(T2, 0, P0); run(T3, 0, P0)
assert np.array_equal(T1['A']['v'], T2['A']['v']) and np.array_equal(T1['A']['xp'], T3['A']['xp'])
j = int(np.argmax(T2['A']['xp'])); T2['A']['v'][j] += 1e-9
act = np.nonzero(T3['A']['xp'] > 0)[0]; j3 = int(act[0]) if len(act) else j; T3['A']['xp'][j3] = 0.0
c0 = time.process_time(); d1 = run(T1, P0, P0 + 512, P0); cpu1 += time.process_time() - c0
d2 = run(T2, P0, P0 + 512, P0); d3 = run(T3, P0, P0 + 512, P0)
cnt = lambda a, b: [(np.abs(a.astype(np.float64) - b) > 0).sum(axis=1)[n - 1] for n in (1, 16, 64, 256, 512)]
out = dict(label=label, args=kw, divergence_v1e9={str(n): int(c) for n, c in zip((1, 16, 64, 256, 512), cnt(d1, d2))},
           divergence_flip={str(n): int(c) for n, c in zip((1, 16, 64, 256, 512), cnt(d1, d3))}, flipped_cell=j3, perturbed_cell=j)
lo, hi = NB - 4096, NB
c0 = time.process_time(); rec = run(T1, P0 + 512, hi, lo).astype(np.float64); cpu1 += time.process_time() - c0
t1 = T1['t']; out['ms_per_byte'] = cpu1 / NB * 1e3; out['cells_sending_a_step'] = t1.active_sum / t1.k
out['conns_per_cell'] = float(T1['A']['deg'].sum()) / N
b = T1['b']; A = T1['A']; B = T1['B']
lines = set(ts.LINE0 + int(v) for v in set(b[lo:hi].tolist()))
deg = A['deg']; src = A['c_src']; isfb = B['kind'] == ts.FBK
dist = np.zeros(N, np.int16)
fr = np.array([(not isfb[i]) and any(int(s) in lines for s in src[i, :deg[i]]) for i in range(N)]); dist[fr] = 1; d = 1
while fr.any():
    nxt = np.array([(dist[i] == 0) and (not isfb[i]) and bool(fr[src[i, :deg[i]][src[i, :deg[i]] < N]].any()) for i in range(N)])
    d += 1; dist[nxt] = d; fr = nxt
def eta2(x, lab):
    tot = x.var(axis=0); m = x.mean(axis=0); bet = np.zeros(x.shape[1])
    for v in np.unique(lab):
        s = lab == v; bet += s.mean() * (x[s].mean(axis=0) - m) ** 2
    return np.where(tot > 1e-30, bet / np.maximum(tot, 1e-30), np.nan)
e0 = eta2(rec[1:], b[lo + 1:hi]); e1 = eta2(rec[1:], b[lo:hi - 1])
rows = []
for dd in range(1, int(dist.max()) + 1):
    c = dist == dd
    if not c.any(): continue
    q = lambda e, p: float(np.nanpercentile(e[c], p)) if np.isfinite(e[c]).any() else None
    rows.append(dict(d=dd, cells=int(c.sum()), active_share=float((rec[:, c] > 0).mean()), mean_output=float(rec[:, c].mean()),
                     cells_never_active=int((rec[:, c].max(axis=0) == 0).sum()), cells_always_active=int((rec[:, c].min(axis=0) > 0).sum()),
                     present_median=q(e0, 50), present_p90=q(e0, 90), previous_median=q(e1, 50), previous_p90=q(e1, 90)))
out['by_distance'] = rows; out['unreached'] = int(((dist == 0) & ~isfb).sum()); out['max_distance'] = int(dist.max())
ds = sorted(set([1, 2, 4] + [r['d'] for r in rows][-3:]))
x = np.where(rec < 0.01, 0.0, rec)
r = readings.read_window(x, b[lo:hi], dist, T1['SEED'], stream=b, start=lo, distances=tuple(ds), lags=(0, 1, 2))
out['S'] = {str(dd): {str(Lg): (r['scores'][dd]['by_lag'][Lg] or {}).get('S') for Lg in (0, 1, 2)} for dd in ds}
out['c0'] = {str(k): (v.get('S') if isinstance(v, dict) else v) for k, v in r['c0'].items()}
os.makedirs(os.path.join(tis.HERE, 'aon'), exist_ok=True); json.dump(out, open(os.path.join(tis.HERE, 'aon', f'{label}.json'), 'w'), default=float)
f = lambda v: '  nan' if v is None else f'{v:.3f}'
print(f"== {label}  {out['ms_per_byte']:.2f} ms/byte, cells sending a step {out['cells_sending_a_step']:.0f}, conns {out['conns_per_cell']:.1f}, unreached {out['unreached']}")
print("   divergence (cells differing) after 1/16/64/256/512 bytes: v+1e-9 " + ' '.join(str(out['divergence_v1e9'][str(n)]) for n in (1, 16, 64, 256, 512))
      + " | one output flipped " + ' '.join(str(out['divergence_flip'][str(n)]) for n in (1, 16, 64, 256, 512)))
for rr in rows:
    print(f"   d={rr['d']:2d} cells {rr['cells']:4d} active {rr['active_share']:.3f} never {rr['cells_never_active']:4d} always {rr['cells_always_active']:4d} | present med {f(rr['present_median'])} p90 {f(rr['present_p90'])} | previous med {f(rr['previous_median'])} p90 {f(rr['previous_p90'])}")
print("   S(d,0/1/2): " + ' | '.join(f"d{dd} {f(out['S'][str(dd)]['0'])} {f(out['S'][str(dd)]['1'])} {f(out['S'][str(dd)]['2'])}" for dd in ds) + f" | C0 {f(out['c0'].get('0'))} {f(out['c0'].get('1'))} {f(out['c0'].get('2'))}")
sys.stdout.flush()
