# drive8.py, by E52's worker: a driver of tissue_t8.c through tis8.py (the reviewer's tissue.c and tis.py at 1ece60fc with the options
# test_age and zero_test added; the reviewer's files are beside this one, unchanged). The readings of drive7.py (PREP/tissue7): the
# helper's tool as it is at d = 1, 2, 4, 8, 10 and near the output lines (far.py's set), lags 0, 1, 2; cells answering; connections a
# cell; the divergence of two copies over the 512 bytes before the last window (with an unchanged copy as a check). Added:
#   - connections made in the last quarter of the life;
#   - loops at the end (loops.py's method: strongly connected sets of more than one cell among the connections between cells, all
#     senders and excitatory senders only);
#   - the cells that made a connection in the last quarter (klast in it), and the cells inside excitatory loops: their distance in the
#     last window, x coordinate, gain, number of sources, long mean M, whether any source had output in the last window (a cell with
#     output, or an input line present); for the connections inside excitatory loops, the share younger than test_age steps.
# For a life of more than 16,384 bytes it reads four windows, the last 4,096 bytes of each quarter.
#   python3 -I drive8.py <label> <options as drive.py, and test_age=.., zero_test=1>
import sys, os, time, json, ctypes, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tis8 as tis
from tis8 import ts, readings

DS = (1, 2, 4, 8, 10)
label = sys.argv[1]; kw = dict(a.split('=') for a in sys.argv[2:])
M1 = tis.make(kw)
t, A, B, L, b, N, NB, SEED = (M1[k] for k in ('t', 'A', 'B', 'L', 'b', 'N', 'NB', 'SEED'))
WIN = {f'Q{j + 1}': ((j + 1) * (NB // 4) - 4096, (j + 1) * (NB // 4)) for j in range(4)} if NB > 16384 else {'A1': (2048, 6144), 'A2': (NB - 4096, NB)}
last = list(WIN)[-1]
Q3 = 3 * NB // 4
isfb = B['kind'] == ts.FBK
kind = np.asarray(B['kind'])
SCAL = ('rng', 'made', 'given', 'given_by_use', 'k', 'active_sum')
TEST_AGE = float(eval(kw.get('test_age', '4096')))
X = np.array([p[0] for p in B['P']], float)
state = dict(pos=0, made_q3=None)


def run(T, lo, hi, r0=None):
    n = 0 if r0 is None else hi - r0
    rec = np.zeros((max(n, 1), N), np.float32)
    rr = hi if r0 is None else r0
    T['L'].tis_run(ctypes.byref(T['t']), ctypes.c_void_p(b.ctypes.data), ctypes.c_int64(lo), ctypes.c_int64(hi), ctypes.c_int64(rr), ctypes.c_int64(hi), ctypes.c_void_p(rec.ctypes.data))
    return rec


def mark():
    if state['pos'] == Q3 and state['made_q3'] is None:
        state['made_q3'] = int(t.made)


def adv(to):
    if state['pos'] < Q3 < to:
        run(M1, state['pos'], Q3); state['pos'] = Q3; mark()
    if state['pos'] < to:
        run(M1, state['pos'], to); state['pos'] = to
    mark()


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


def sccs(n, edges):                      # loops.py's function, unchanged
    adj = [[] for _ in range(n)]
    for a, b_ in edges: adj[a].append(b_)
    index = [-1] * n; low = [0] * n; on = [False] * n; st = []; out = []; c = 0
    for r in range(n):
        if index[r] != -1: continue
        work = [(r, 0)]
        while work:
            v, i = work.pop()
            if i == 0: index[v] = low[v] = c; c += 1; st.append(v); on[v] = True
            rec = False
            for j in range(i, len(adj[v])):
                w = adj[v][j]
                if index[w] == -1: work.append((v, j + 1)); work.append((w, 0)); rec = True; break
                elif on[w]: low[v] = min(low[v], index[w])
            if rec: continue
            if low[v] == index[v]:
                comp = []
                while True:
                    w = st.pop(); on[w] = False; comp.append(w)
                    if w == v: break
                out.append(comp)
            if work: u = work[-1][0]; low[u] = min(low[u], low[v])
    return out


def conns():
    deg = A['deg'][~isfb]
    return dict(mean=float(deg.mean()), max=int(deg.max()))


out = dict(label=label, args=kw, windows=list(WIN)); t0 = time.process_time()
recs = {}; dists = {}; cw = {}
for w, (lo, hi) in WIN.items():
    if w == last:
        P0 = lo - 512
        adv(P0)
        T2 = clone(); T3 = clone()
        j = int(np.argmax(T2['A']['xp'])); T2['A']['v'][j] += 1e-9
        d1 = run(M1, P0, lo, P0); d2 = run(T2, P0, lo, P0); d3 = run(T3, P0, lo, P0); state['pos'] = lo; mark()
        diff = np.abs(d1.astype(np.float64) - d2).max(axis=1)
        out['divergence'] = {str(n): float(diff[n - 1]) for n in (1, 4, 16, 64, 256, 512)}
        out['copy_check_max_diff'] = float(np.abs(d1.astype(np.float64) - d3).max())
        del T2, T3
    else:
        adv(lo)
    recs[w] = run(M1, lo, hi, lo); state['pos'] = hi; mark()
    dists[w] = distances(sorted(set(b[lo:hi].tolist())))
    cw[w] = dict(conns(), made=int(t.made), given=int(t.given))
cpu = time.process_time() - t0
out.update(ms_per_byte=cpu / NB * 1e3, conns_end=conns(), made=int(t.made), given=int(t.given), made_last_quarter=int(t.made) - state['made_q3'])
near = np.zeros(N, bool)
for i in np.where(kind == ts.OUTE)[0]:
    near[i] = True
    for mm in B['members'][int(i)]:
        if mm < N: near[mm] = True
near &= ~isfb
for w, (lo, hi) in WIN.items():
    raw = recs[w]; mx = raw.max(axis=0); x = np.where(raw < 0.01, 0.0, raw)
    r = readings.read_window(x, b[lo:hi], dists[w], SEED, stream=b, start=lo, distances=DS, lags=(0, 1, 2))
    rn = readings.read_window(x, b[lo:hi], np.where(near, 1, 0).astype(np.int16), SEED, stream=b, start=lo, distances=(1,), lags=(0, 1, 2))
    out[w] = dict(at_end=cw[w], cells={str(d): int((dists[w] == d).sum()) for d in DS},
                  cells_any_output={str(d): int(((dists[w] == d) & (mx > 0)).sum()) for d in DS},
                  S={str(d): {str(Lg): (r['scores'][d]['by_lag'][Lg] or {}).get('S') for Lg in (0, 1, 2)} for d in DS},
                  near=dict(cells=int(near.sum()), answering=int((near & (mx > 0)).sum()), S={str(Lg): (rn['scores'][1]['by_lag'][Lg] or {}).get('S') for Lg in (0, 1, 2)}),
                  c0={str(k): (v.get('S') if isinstance(v, dict) else v) for k, v in r['c0'].items()})
# loops at the end
deg = A['deg']; src = A['c_src']; sig = B['sig']; k_now = int(t.k)
alle = []; exc = []
for i in range(N):
    for d in range(deg[i]):
        s = int(src[i, d])
        if s < N:
            alle.append((s, i))
            if sig[s] > 0: exc.append((s, i))
loops = {}
for nm, e in (('all', alle), ('excitatory', exc)):
    cs = [c for c in sccs(N, e) if len(c) > 1]
    loops[nm] = dict(connections=len(e), loops=len(cs), cells_in_loops=int(sum(len(c) for c in cs)), sizes=sorted((len(c) for c in cs), reverse=True)[:10])
    if nm == 'excitatory':
        exc_sets = cs
out['loops_end'] = loops
# the cells of point 1
lo, hi = WIN[last]; mxl = recs[last].max(axis=0); dl = dists[last]
present = set(ts.LINE0 + int(v) for v in set(b[lo:hi].tolist()))


def has_active_source(i):
    for s in src[i, :deg[i]]:
        s = int(s)
        if (s < N and mxl[s] > 0) or s in present:
            return True
    return False


def describe(cells):
    cells = np.asarray(sorted(cells), int)
    if cells.size == 0:
        return dict(cells=0)
    q = lambda v: [float(np.percentile(v, p)) for p in (10, 50, 90)]
    return dict(cells=int(cells.size), dist_p10_50_90=q(dl[cells]), dist_unreached=int((dl[cells] == 0).sum()), x_p10_50_90=q(X[cells]),
                gain_p10_50_90=q(A['gain'][cells]), sources_p10_50_90=q(deg[cells]), M_p10_50_90=q(A['M'][cells]),
                share_with_active_source=float(np.mean([has_active_source(i) for i in cells])), share_with_output=float((mxl[cells] > 0).mean()))


klast = A['klast'].astype(np.int64)
made_late = [i for i in range(N) if not isfb[i] and klast[i] >= Q3 * ts.S and klast[i] > 0]
in_loops = sorted(set(i for c in exc_sets for i in c))
ages = []
for c in exc_sets:
    cs = set(c)
    for i in c:
        for d in range(deg[i]):
            s = int(src[i, d])
            if s in cs and sig[s] > 0:
                ages.append(k_now - int(A['c_k0'][i, d]))
ages = np.array(ages)
out['point1'] = dict(all_cells=describe([i for i in range(N) if not isfb[i]]), made_last_quarter=describe(made_late),
                     in_excitatory_loops=describe(in_loops),
                     loop_connections=dict(n=int(ages.size), younger_than_test_age=(float((ages < TEST_AGE).mean()) if ages.size else None),
                                           age_p10_50_90=([float(np.percentile(ages, p)) for p in (10, 50, 90)] if ages.size else None)))
np.savez_compressed(os.path.join(HERE, 'out', label + '_cells.npz'), deg=deg.copy(), dist=dl, isfb=isfb, x=X, gain=A['gain'].copy(), M=A['M'][:N].copy(),
                    klast=klast, maxout=mxl, in_loops=np.isin(np.arange(N), in_loops))
json.dump(out, open(os.path.join(HERE, 'out', label + '.json'), 'w'), default=float)
a = out[last]; f = lambda v: '  -  ' if v is None else f'{v:.3f}'
print(f"{label:30s} {out['ms_per_byte']:6.2f} ms/byte conns {out['conns_end']['mean']:4.1f}/{out['conns_end']['max']} made {out['made']} last quarter {out['made_last_quarter']} | {last} S0 d1 2 4 8 10 "
      + " ".join(f(a['S'][str(d)]['0']) for d in DS) + f" near {f(a['near']['S']['0'])} | exc loops {loops['excitatory']['loops']} ({loops['excitatory']['cells_in_loops']} cells)"
      + f" | div512 {out['divergence']['512']:.1e} copy {out['copy_check_max_diff']:.0e}", flush=True)
