# What the made connections are (step 2 of the task "a rule of making and withdrawing"). Runs one tissue of the reviewer's tissue.c
# (tis.make, unchanged) in pieces of P bytes and compares the connection lists after each piece: a connection is (sender, receiver,
# step made); one present now and not before was made in the piece, one present before and not now was withdrawn in it. Times are
# therefore known to within P bytes (P = 64 by default; a connection made and withdrawn within one piece is not seen and is counted
# by the core's own counters only).
#   python3 -I made_conns.py <label> <options as drive.py> [piece=64]
# Distance of a cell: by the birth's connections, from all input lines (distance 0 for a line): a made connection comes from a nearer
# sender (dist[s] < dist[i]), one at the same distance, or a farther one. Share of the cell's total: |w| over the sum of |w| of the
# cell's connections of the same sign (the excitatory or inhibitory total), at the end of the piece in which it was seen.
import sys, os, json, ctypes, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tis
from tis import ts

label = sys.argv[1]; kw = dict(a.split('=') for a in sys.argv[2:])
P = int(kw.pop('piece', 64))
M = tis.make(kw)
t, A, B, L, b, N, NB = (M[k] for k in ('t', 'A', 'B', 'L', 'b', 'N', 'NB'))
isfb = B['kind'] == ts.FBK
sig = B['sig']


def birth_dist():
    deg = A['deg']; src = A['c_src']
    dist = np.full(ts.NZ, -1, np.int32); dist[ts.LINE0:] = 0
    fr = np.array([(not isfb[i]) and any(int(s) >= ts.LINE0 for s in src[i, :deg[i]]) for i in range(N)])
    dist[:N][fr] = 1; d = 1
    while fr.any():
        nxt = np.array([(dist[i] < 0) and (not isfb[i]) and bool(fr[src[i, :deg[i]][src[i, :deg[i]] < N]].any()) for i in range(N)])
        d += 1; dist[:N][nxt] = d; fr = nxt
    return dist


dist = birth_dist()


def snapshot():
    deg = A['deg']; src = A['c_src']; k0 = A['c_k0']; w = A['c_w']
    s = {}
    for i in range(N):
        if isfb[i]:
            continue
        for d in range(deg[i]):
            s[(int(src[i, d]), i, int(k0[i, d]))] = float(w[i, d])
    return s


def totals(snap):
    te = np.zeros(N); ti = np.zeros(N)
    for (s, i, _), w in snap.items():
        if sig[s] > 0:
            te[i] += abs(w)
        else:
            ti[i] += abs(w)
    return te, ti


prev = snapshot(); born0 = set(prev)
made = {}       # key -> dict(seen, gone, share_first, share_max, wmax)
pos = 0
dummy = np.zeros((1, N), np.float32)
while pos < NB:
    hi = min(NB, pos + P)
    L.tis_run(ctypes.byref(t), ctypes.c_void_p(b.ctypes.data), ctypes.c_int64(pos), ctypes.c_int64(hi), ctypes.c_int64(0), ctypes.c_int64(0), ctypes.c_void_p(dummy.ctypes.data))
    pos = hi
    cur = snapshot(); te, ti = totals(cur)
    for key, w in cur.items():
        if key in born0:
            continue
        s, i, _ = key
        tot = te[i] if sig[s] > 0 else ti[i]
        sh = abs(w) / tot if tot > 0 else 0.0
        if key not in made:
            made[key] = dict(seen=pos, gone=None, share_first=sh, share_max=sh, share_last=sh)
        else:
            m = made[key]; m['share_max'] = max(m['share_max'], sh); m['share_last'] = sh
    for key in prev:
        if key not in cur and key in made and made[key]['gone'] is None:
            made[key]['gone'] = pos
    prev = cur

rows = []
for (s, i, k0), m in made.items():
    rel = 'nearer' if dist[s] < dist[i] else ('same' if dist[s] == dist[i] else 'farther')
    life = (m['gone'] if m['gone'] is not None else NB) - m['seen']
    rows.append(dict(rel=rel, exc=bool(sig[s] > 0), alive=m['gone'] is None, life=life, share_first=m['share_first'], share_max=m['share_max'], share_last=m['share_last']))
n = len(rows)


def summ(sel):
    sel = list(sel)
    if not sel:
        return dict(n=0)
    lf = np.array([r['life'] for r in sel]); smx = np.array([r['share_max'] for r in sel]); sl = np.array([r['share_last'] for r in sel])
    return dict(n=len(sel), alive_at_end=int(sum(r['alive'] for r in sel)), life_median_bytes=float(np.median(lf)), life_p90_bytes=float(np.percentile(lf, 90)),
                share_max_median=float(np.median(smx)), share_max_p90=float(np.percentile(smx, 90)), share_last_median=float(np.median(sl)))


out = dict(label=label, args=kw, piece=P, made_seen=n, made_core=int(t.made), given_core=int(t.given), given_by_use_core=int(t.given_by_use),
           birth_connections=len(born0), birth_alive_at_end=len(born0 & set(prev)),
           by_rel={rel: dict(share=(sum(r['rel'] == rel for r in rows) / n if n else None), **summ(r for r in rows if r['rel'] == rel)) for rel in ('nearer', 'same', 'farther')},
           excitatory=summ(r for r in rows if r['exc']), inhibitory=summ(r for r in rows if not r['exc']), all=summ(rows),
           conns_end_mean=float(A['deg'][~isfb].mean()), conns_end_max=int(A['deg'][~isfb].max()))
os.makedirs(os.path.join(HERE, 'made'), exist_ok=True)
json.dump(out, open(os.path.join(HERE, 'made', label + '.json'), 'w'), indent=1, default=float)
print(json.dumps({k: out[k] for k in ('label', 'made_seen', 'made_core', 'given_core', 'birth_connections', 'birth_alive_at_end', 'conns_end_mean', 'conns_end_max')}))
for rel in ('nearer', 'same', 'farther'):
    print(rel, json.dumps(out['by_rel'][rel]))
