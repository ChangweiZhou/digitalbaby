# Is the tissue directed? After a life: the connections between cells as a graph (sender -> receiver); the sets of cells that can each
# reach the others (loops), by sign of the senders taken (all connections; excitatory senders only); the gains; and the divergence of
# two copies differing by 1e-9 over the last 512 bytes.
# usage: loops.py <label> rule=.. [options of drive.py]
import sys, os, json, ctypes, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import tis
ts = tis.ts
label = sys.argv[1]; kw = dict(a.split('=') for a in sys.argv[2:])
def run(T, lo, hi, rec_lo=None):
    t, L, b = T['t'], T['L'], T['b']
    n = 0 if rec_lo is None else hi - rec_lo
    rec = np.zeros((max(n, 1), T['N']), np.float32); r0 = hi if rec_lo is None else rec_lo
    L.tis_run(ctypes.byref(t), ctypes.c_void_p(b.ctypes.data), ctypes.c_int64(lo), ctypes.c_int64(hi), ctypes.c_int64(r0), ctypes.c_int64(hi), ctypes.c_void_p(rec.ctypes.data))
    return rec
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
T1 = tis.make(kw); T2 = tis.make(kw); NB = T1['NB']
P0 = NB - 512
out = dict(label=label, args=kw, at={})
for mark in (4096, NB // 2, P0):
    run(T1, T1['t'].k // 32, mark); out['at'][str(mark)] = graph(T1)
run(T2, 0, P0)
j = int(np.argmax(T2['A']['xp'])); T2['A']['v'][j] += 1e-9
d1 = run(T1, P0, NB, P0); d2 = run(T2, P0, NB, P0)
diff = np.abs(d1.astype(np.float64) - d2).max(axis=1)
g = T1['A']['gain']; isfb = np.asarray(T1['B']['kind']) == ts.FBK
out.update(divergence={str(n): float(diff[n - 1]) for n in (1, 16, 64, 256, 512)}, gain=dict(median=float(np.median(g[~isfb])), p90=float(np.percentile(g[~isfb], 90)), at_bound=int((g[~isfb] >= T1['t'].gmax).sum())),
           end=graph(T1), made=int(T1['t'].made), withdrawn=int(T1['t'].given))
os.makedirs('loops', exist_ok=True); json.dump(out, open(f'loops/{label}.json', 'w'), default=float)
print(f"== {label}  divergence after 1/16/64/256/512 bytes: " + ' '.join(f"{out['divergence'][str(n)]:.1e}" for n in (1, 16, 64, 256, 512)) + f" | gain median {out['gain']['median']:.2f} p90 {out['gain']['p90']:.2f} at the bound {out['gain']['at_bound']} | made {out['made']}")
for mark, r in list(out['at'].items()) + [('end', out['end'])]:
    print(f"   at byte {mark:>6}: all connections {r['all']['connections']:5d}: {r['all']['loops']:3d} loops, {r['all']['cells_in_loops']:4d} cells in loops, largest {r['all']['largest']:4d} | from excitatory cells only: {r['excitatory']['loops']:3d} loops, {r['excitatory']['cells_in_loops']:4d} cells, largest {r['excitatory']['largest']:4d}")
sys.stdout.flush()
