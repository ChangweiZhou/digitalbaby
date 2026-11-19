# pbstand.py (the helper's, new file, 19 November after 12:35 UTC): stand.py's measures of one tissue, with perbyte3.py's readers (copied
# unchanged below, from the reviewer's folder at 96109386) on the last 8,192 bytes, and the count of cells that receive from an input line.
# The tool's window (far.py's readings, both levels) is the last 4,096 bytes of the same record. Below, stand.py's header.
# stand.py (the helper's, new file): does the long-axon tissue stand? One tissue of the reviewer's tis.py (folder from 10876732, build with
# -ffp-contract=off), read in one or more windows of 4,096 bytes, with:
#   - the readings of far.py (readings.read_window with the distances of far.py, lags 0..4; (a) the tool as it is, (b) no floor and each
#     cell divided by its own deviation), for every distance and for the cells near the output lines (far.py's definition: the drivers of
#     the output lines and the members of their neighbourhoods, feedback cells left out);
#   - the levels: lag 0 above 0.1103 under both; lag L >= 1 above c0(L) + 0.02 (level A) or C<L(L) + 0.02 (level B), of the same window;
#   - connections a cell (mean, largest), made and withdrawn (cumulative), made and withdrawn in the last quarter of the life, the share
#     of the connections (from cells) that come from long-axon cells, the largest output of the cells near the output lines in the window
#     and the share of them whose largest output in the window is above 1/2;
#   - the divergence of two copies: a second tissue, one cell's potential changed by 1e-9 at byte NB - 4608, largest output difference
#     after 1, 4, 16, 64, 256, 512 bytes;
#   - loops among the cells at the end of the life (loops.py's sccs and graph, copied unchanged below).
# Options beyond tis.py's: draw=own gives the helper's own draw of the long-axon cells and of the time constants (default: the reviewer's):
#   long-axon cells: Random(seed + 710000); in cell order every excitatory cell that is neither a feedback cell nor a driver of an output
#   line is a long-axon cell when random() < 1/K; then for every cell that is not a feedback cell, in cell order, the long-axon cells
#   within R not already members (and not the cell itself) are appended to its members and its starting connections are drawn again,
#   sample(members, 8) from the same generator. Time constants: Random(seed + 700000), one draw per cell in cell order from {2, 4, ..., top}.
#   windows=s1,s2,.. (default: the last 4,096 bytes).
# Added at 11:15 UTC for tissue LF-256 (the reviewer's longfast=1 in tis.py from e38c6514: the long-axon cells keep the time constant 2): in the
#   helper's own draw the long-axon cells' time constant is set to 2 after the draw of the time constants (the draw itself unchanged); and,
#   besides far.py's cells near the output lines (group (a), 'near'), near2.py's group (b): cells within the neighbourhood radius (1/8) of a
#   driver of an output line ('near_b'), and (c) the long-axon cells of (a) not in (b) ('near_c'), each read by the tool as it is, lags 0-4.
# usage: python3 -I pbstand.py <label> seed=.. <options of tis.py> [draw=own] [windows=..]
import sys, os, json, ctypes, time
from random import Random
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import tis
ts = tis.ts; readings = tis.readings
label = sys.argv[1]; kw = dict(a.split('=') for a in sys.argv[2:])
OWN = kw.get('draw') == 'own'
LAGS = (0, 1, 2, 3, 4)
t0 = time.process_time()
_birth = ts.birth


def own_birth(seed):
    B = _birth(seed); N = ts.N
    K, R = kw['long'].split('-'); R2L = float(eval(R)) ** 2; r = Random(seed + 710000)
    Pq = np.asarray(B['P'])
    longs = np.array([j for j in range(N) if B['sig'][j] > 0 and j not in B['fbset'] and j not in B['outset'] and r.random() < 1.0 / float(K)])
    for i in range(N):
        if i in B['fbset']:
            continue
        have = set(B['members'][i]); d2 = ((Pq[longs] - Pq[i]) ** 2).sum(1)
        B['members'][i] = list(B['members'][i]) + [int(j) for j, q in zip(longs, d2) if q <= R2L and j != i and int(j) not in have]
        B['src'][i] = [int(m) for m in r.sample(list(B['members'][i]), min(ts.START_N, len(B['members'][i])))]
    B['longs'] = longs
    return B


def own_tau(spec, seed):
    top = int(spec[3:].rstrip('e')); ch = [2.0 ** k for k in range(1, top.bit_length())]
    assert not spec.endswith('e')
    r = Random(seed + 700000); return np.array([ch[r.randrange(len(ch))] for _ in range(ts.N)])


def make():
    if OWN:
        k2 = {k: v for k, v in kw.items() if k != 'long'}
        ts.birth = own_birth
        try:
            TT = tis.make(k2)
        finally:
            ts.birth = _birth
        if kw.get('tau'):
            TT['A']['tauv'][:] = own_tau(kw['tau'], TT['SEED'])
        if kw.get('longfast', ''):
            TT['A']['tauv'][np.asarray(TT['B']['longs'], dtype=int)] = 2.0
    else:
        TT = tis.make(kw)
    return TT


# loops.py's sccs and graph, unchanged
def sccs(n, edges):
    adj = [[] for _ in range(n)]
    for a, b in edges: adj[a].append(b)
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
def graph(T):
    A = T['A']; N = T['N']; deg = A['deg']; src = A['c_src']; sig = A['sig']
    alle = []; exc = []
    for i in range(N):
        for d in range(deg[i]):
            s = int(src[i, d])
            if s < N:
                alle.append((s, i))
                if sig[s] > 0: exc.append((s, i))
    res = {}
    for name, e in (('all', alle), ('excitatory', exc)):
        cs = [c for c in sccs(N, e) if len(c) > 1]
        res[name] = dict(connections=len(e), loops=len(cs), cells_in_loops=int(sum(len(c) for c in cs)), largest=int(max([len(c) for c in cs], default=0)))
    return res


T = make(); N, NB, SEED, b, A, B = T['N'], T['NB'], T['SEED'], T['b'], T['A'], T['B']
longs = set(int(j) for j in B.get('longs', []))
isfb = np.asarray(B['kind']) == ts.FBK; kind = np.asarray(B['kind'])
near = np.zeros(N, bool)
for i in np.where(kind == ts.OUTE)[0]:
    near[i] = True
    for m in B['members'][int(i)]:
        if m < N: near[m] = True
near &= ~isfb
Pp = np.asarray(B['P']); near_b = np.zeros(N, bool)
for i in np.where(kind == ts.OUTE)[0]:
    near_b[((Pp - Pp[i]) ** 2).sum(1) <= ts.R2] = True
near_b &= ~isfb; near_c = near & ~near_b
WL = 4096
STARTS = sorted(int(x) for x in kw['windows'].split(',')) if 'windows' in kw else [NB - WL]
P0 = NB - WL - 512; Q3 = (3 * NB) // 4


def run(TT, a, c, r0=None):
    n = 0 if r0 is None else c - r0
    rr = np.zeros((max(n, 1), N), np.float32)
    TT['L'].tis_run(ctypes.byref(TT['t']), ctypes.c_void_p(TT['b'].ctypes.data), ctypes.c_int64(a), ctypes.c_int64(c), ctypes.c_int64(c if r0 is None else r0), ctypes.c_int64(c), ctypes.c_void_p(rr.ctypes.data))
    return rr


def distances(lo, hi):
    lines = set(ts.LINE0 + int(v) for v in set(b[lo:hi].tolist()))
    deg = A['deg']; src = A['c_src']
    dist = np.zeros(N, np.int16)
    fr = np.array([(not isfb[i]) and any(int(s) in lines for s in src[i, :deg[i]]) for i in range(N)]); dist[fr] = 1; d = 1
    while fr.any():
        nxt = np.array([(dist[i] == 0) and (not isfb[i]) and bool(fr[src[i, :deg[i]][src[i, :deg[i]] < N]].any()) for i in range(N)])
        d += 1; dist[nxt] = d; fr = nxt
    return dist


def conn_state():
    deg = A['deg']; src = A['c_src']
    tot = 0; fl = 0
    for i in range(N):
        for d in range(deg[i]):
            s = int(src[i, d])
            if s < N:
                tot += 1; fl += s in longs
    return dict(conns_per_cell=float(deg.sum()) / N, conns_max=int(deg.max()), from_cells=tot, from_long=fl, share_from_long=fl / tot if tot else None,
                made=int(T['t'].made), given=int(T['t'].given))


def analyse(rec, lo, hi):
    wb = b[lo:hi]; dist = distances(lo, hi); DS = tuple(range(1, int(dist.max()) + 1))
    xa = np.where(rec < 0.01, 0.0, rec).astype(np.float64)
    sd = rec.std(axis=0).astype(np.float64); xb = rec.astype(np.float64) / np.where(sd > 1e-30, sd, 1.0)
    def table(x, dd, ds):
        r = readings.read_window(x, wb, dd, SEED, stream=b, start=lo, distances=ds, lags=LAGS)
        return {str(d): {str(Lg): (r['scores'][d]['by_lag'][Lg] or {}).get('S') for Lg in LAGS} for d in ds}, r
    ta, r = table(xa, dist, DS); tb, _ = table(xb, dist, DS)
    c0 = {str(k): v['S'] for k, v in r['c0'].items()}; clt = {str(k): v['S'] for k, v in r['c_lt'].items()}
    dn = np.where(near, 1, 0).astype(np.int16)
    na, _ = table(xa, dn, (1,)); nb, _ = table(xb, dn, (1,))
    grp = {}
    for gname, g in (('near_b', near_b), ('near_c', near_c)):
        if g.any():
            ga_, _ = table(xa, np.where(g, 1, 0).astype(np.int16), (1,)); gm = rec[:, g].max(axis=0)
            gv = {str(L): ga_['1'][str(L)] for L in LAGS}
            grp[gname] = dict(cells=int(g.sum()), tool=gv, max_output=float(gm.max()), share_above_half=float((gm > 0.5).mean()))
        else:
            grp[gname] = dict(cells=0, tool=None)
    drv = np.where(kind == ts.OUTE)[0]
    grp['driver_connections'] = int(sum(int(A['deg'][i]) for i in drv)); grp['driver_connections_from_long'] = int(sum(int(s) in longs for i in drv for s in A['c_src'][i, :A['deg'][i]]))
    levA = {str(L): (c0[str(L)] + 0.02 if L else 0.1103) for L in LAGS}; levB = {str(L): (clt[str(L)] + 0.02 if L else 0.1103) for L in LAGS}
    nine = {f'{d},{L}': ta.get(str(d), {}).get(str(L)) for d in (1, 2, 4) for L in (0, 1, 2)}
    def below(vals, lev):
        return [k for k, v in vals.items() if v is None or v <= lev[k.split(',')[-1]]]
    nearv = {str(L): na['1'][str(L)] for L in LAGS}
    nmax = rec[:, near].max(axis=0)
    w = dict(window=[lo, hi - 1], c0=c0, c_lt=clt, level_A=levA, level_B=levB, tool=ta, nofloor=tb, cells={str(k): int((dist == k).sum()) for k in DS},
             max_output={str(k): float(rec[:, dist == k].max()) for k in DS}, near_cells=int(near.sum()), near_tool=nearv, near_nofloor=nb['1'],
             near_distances={str(k): int(((dist == k) & near).sum()) for k in DS}, near_max_output=float(nmax.max()), near_share_above_half=float((nmax > 0.5).mean()),
             nine=nine, nine_below_A=below(nine, levA), nine_below_B=below(nine, levB),
             near_below_A=[L for L, v in nearv.items() if v is None or v <= levA[L]], near_below_B=[L for L, v in nearv.items() if v is None or v <= levB[L]])
    w.update(conn_state()); w.update(grp)
    if grp['near_b'].get('tool'):
        w['near_b_below_A'] = [L for L, v in grp['near_b']['tool'].items() if v is None or v <= levA[L]]
        w['near_b_below_B'] = [L for L, v in grp['near_b']['tool'].items() if v is None or v <= levB[L]]
    return w



# the record of the last 8,192 bytes, in pieces at the quarter mark and at the divergence check (the record is the same either way)
W8 = 8192; lo8 = NB - W8
assert lo8 <= P0 and P0 + 512 <= NB and lo8 <= Q3 <= NB
run(T, 0, lo8); pos = lo8; pieces = []; q3 = None; divergence = None
for m in sorted(set([Q3, P0, NB])):
    if pos < m:
        pieces.append(run(T, pos, m, pos)); pos = m
    if m == Q3 and q3 is None:
        q3 = dict(made=int(T['t'].made), given=int(T['t'].given))
    if m == P0:
        T2 = make(); run(T2, 0, P0)
        assert np.array_equal(A['v'], T2['A']['v'])
        jj = int(np.argmax(T2['A']['xp'])); T2['A']['v'][jj] += 1e-9
        d1 = run(T, P0, P0 + 512, P0); d2 = run(T2, P0, P0 + 512, P0); pieces.append(d1); pos = P0 + 512
        divergence = {str(n): float(np.abs(d1[n - 1].astype(np.float64) - d2[n - 1]).max()) for n in (1, 4, 16, 64, 256, 512)}
        del T2
        if Q3 < pos and q3 is None:
            raise SystemExit('the quarter mark falls in the divergence check')
rec = np.concatenate(pieces)
assert rec.shape[0] == W8
wins = {str(NB - WL): analyse(rec[W8 - WL:], NB - WL, NB)}

# perbyte3.py's readers, unchanged except that rec, b, lo, hi come from above
lo, hi, W = lo8, NB, W8
P = np.asarray(B['P'])
near8 = np.zeros(N, bool)
for i in np.where(kind == ts.OUTE)[0]:
    near8[i] = True
    near8[((P - P[i]) ** 2).sum(1) <= ts.R2] = True
near8 &= ~isfb
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
pb = {}
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
    pb[name] = dict(cells=int(g.sum()), lags=res)

deg = A['deg']; src = A['c_src']
from_line = int(sum(1 for i in range(N) if not isfb[i] and any(ts.LINE0 <= int(s) < ts.LINE0 + ts.NIN for s in src[i, :deg[i]])))
end = conn_state()
out = dict(label=label, args=kw, draw='own: long-axon cells and starting connections Random(seed + 710000), time constants Random(seed + 700000)' if OWN else 'reviewer: long-axon cells and starting connections default_rng(seed + 20261120), time constants default_rng(seed + 20261119)',
           seed=SEED, nbytes=NB, long_cells=len(longs), divergence=divergence, windows=wins, end=end, cells_from_input_line=from_line, perbyte=pb,
           made_last_quarter=end['made'] - q3['made'], given_last_quarter=end['given'] - q3['given'], quarter_mark=Q3, loops_end=graph(T),
           gain=dict(median=float(np.median(A['gain'][~isfb])), at_bound=int((A['gain'][~isfb] >= T['t'].gmax).sum())),
           tau_counts={str(k): int((A['tauv'] == k).sum()) for k in sorted(set(A['tauv'].tolist()))}, total_cpu_s=time.process_time() - t0)
os.makedirs(os.path.join(HERE, 'pbstand'), exist_ok=True)
json.dump(out, open(os.path.join(HERE, 'pbstand', label + '.json'), 'w'), default=float)
r = pb['near output']['lags']
print(label, 'done', round(out['total_cpu_s'], 1), 's; near output, linear, end/eq/digits>1/2 by lag:', ' '.join(f"{r[l]['linear']['end']:.2f}/{r[l]['linear']['eq']:.2f}/{r[l]['linear']['digits_above_half']}" for l in '0123'), '; divergence', divergence['512'], flush=True)
