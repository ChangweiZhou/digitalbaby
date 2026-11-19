# drive12.py, by E52's worker: part B of the reviewer's message of 13:40. A tissue of the reviewer's tissue.c and tis.py at 97ba8349 (beside
# this file, unchanged; -ffp-contract=off) lives NB bytes (65,536 by default); at the reading step of every byte, online readers at the
# output end name the NEXT byte.
#   Units: 256, one a byte value v. Unit v sees the cells (index < N) among the members of the neighbourhood of the driver of output line v
#   (B['driver'][v]; neighbourhoods as enlarged by long=).
#   Inputs of a cell: x (its output at the reading step), and for the input set "x,s2,s4" also s_k(t) = s_k(t-1) + (x(t) - s_k(t-1))/k,
#   k = 2, 4, once a byte. Each input component: minus its running mean, over its running deviation (rate 1/1024; both updated after
#   use; deviation from the running mean of squares less the square of the running mean, at least 1e-6), clipped to [-8, 8]; a constant
#   1 appended; n = |U|^2.
#   D: every unit W[v] += (1/64)(target_v - a_v) U / n, target_v = 1 for the next byte's value and 0 for the others.
#   Q: emission e = the largest a_v; with chance 1/16 a value seen so far at random instead; ok = 1 if e is the next byte; W[e] +=
#      (1/16)(ok - a_e) U / n; on yes every other unit W[v] += (1/16)(0 - a_v) U / n; on no the others do nothing.
#   Read: the share of bytes in the last quarter whose next byte is the largest a_v (before the update); for Q also the share emitted
#   right (with the random choices). Beside them, from the stream alone: the next byte named by the most frequent successor of the last
#   1, 2, 3 bytes, the counts taken online from the start of the life (a context never seen names nothing).
#   python3 -I drive12.py <label> <options as drive.py> [nbytes=65536]
import sys, os, json, time, ctypes, numpy as np
from collections import defaultdict, Counter
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tis
ts = tis.ts
label = sys.argv[1]; kw = dict(a.split('=') for a in sys.argv[2:])
kw.setdefault('nbytes', '65536')
T = tis.make(kw); t, L, b, N, NB, B = T['t'], T['L'], T['b'], T['N'], T['NB'], T['B']
rng = np.random.default_rng(int(kw.get('seed', 73129)) + 777)
drv = [int(d) for d in B['driver']]
mem = [[int(m) for m in B['members'][d] if m < N] for d in drv]
M = max(len(m) for m in mem)
idx = np.zeros((256, M), np.int64); msk = np.zeros((256, M), bool)
for v, m in enumerate(mem):
    idx[v, :len(m)] = m; msk[v, :len(m)] = True
RATE = 1 / 1024
SETS = ('x', 'x,s2,s4')
ncomp = {'x': 1, 'x,s2,s4': 3}
st = {s: dict(mu=np.zeros((ncomp[s], N)), sq=np.zeros((ncomp[s], N)), WD=np.zeros((256, ncomp[s] * M + 1)), WQ=np.zeros((256, ncomp[s] * M + 1))) for s in SETS}
s2 = np.zeros(N); s4 = np.zeros(N)
Q4 = 3 * NB // 4
cnt = {s: dict(D=0, Q=0, Qemit=0) for s in SETS}; nread = 0
seen = []; seen_set = set()
ctx = {k: defaultdict(Counter) for k in (1, 2, 3)}; cctx = {k: 0 for k in (1, 2, 3)}
CH = 1024; pos = 0; t0 = time.process_time()
while pos < NB - 1:
    end = min(NB, pos + CH)
    rec = np.zeros((end - pos, N), np.float32)
    L.tis_run(ctypes.byref(t), ctypes.c_void_p(b.ctypes.data), ctypes.c_int64(pos), ctypes.c_int64(end), ctypes.c_int64(pos), ctypes.c_int64(end), ctypes.c_void_p(rec.ctypes.data))
    for r in range(end - pos):
        tb = pos + r
        x = rec[r].astype(np.float64)
        s2 += (x - s2) / 2; s4 += (x - s4) / 4
        cur = int(b[tb])
        if cur not in seen_set: seen_set.add(cur); seen.append(cur)
        if tb + 1 >= NB: break
        nxt = int(b[tb + 1])
        late = tb >= Q4
        if late:
            nread += 1
            for k in (1, 2, 3):
                if tb - k + 1 >= 0:
                    c = tuple(b[tb - k + 1:tb + 1].tolist())
                    if c in ctx[k] and ctx[k][c].most_common(1)[0][0] == nxt: cctx[k] += 1
        for k in (1, 2, 3):
            if tb - k + 1 >= 0: ctx[k][tuple(b[tb - k + 1:tb + 1].tolist())][nxt] += 1
        for s in SETS:
            S = st[s]
            comps = (x,) if s == 'x' else (x, s2, s4)
            Z = np.stack(comps)                                   # ncomp x N
            sd = np.sqrt(np.maximum(S['sq'] - S['mu'] ** 2, 1e-6))
            Zn = np.clip((Z - S['mu']) / sd, -8, 8)
            S['mu'] += RATE * (Z - S['mu']); S['sq'] += RATE * (Z * Z - S['sq'])       # updated after use
            U = np.concatenate([np.where(msk, Zn[c][idx], 0.0) for c in range(Zn.shape[0])] + [np.ones((256, 1))], axis=1)
            n = (U * U).sum(1)
            aD = (S['WD'] * U).sum(1)
            if late and int(np.argmax(aD)) == nxt: cnt[s]['D'] += 1
            tgt = np.zeros(256); tgt[nxt] = 1.0
            S['WD'] += ((1 / 64) * (tgt - aD) / n)[:, None] * U
            aQ = (S['WQ'] * U).sum(1)
            e = int(np.argmax(aQ))
            if late and e == nxt: cnt[s]['Q'] += 1
            if rng.random() < 1 / 16: e = int(seen[rng.integers(len(seen))])
            ok = 1.0 if e == nxt else 0.0
            if late and ok: cnt[s]['Qemit'] += 1
            if ok:
                dv = (1 / 16) * (0.0 - aQ) / n; dv[e] = (1 / 16) * (ok - aQ[e]) / n[e]
                S['WQ'] += dv[:, None] * U
            else:
                S['WQ'][e] += (1 / 16) * (ok - aQ[e]) / n[e] * U[e]
    pos = end
cpu = time.process_time() - t0
out = dict(label=label, args=kw, bytes=NB, read=nread, cpu_s=cpu,
           D={s: cnt[s]['D'] / nread for s in SETS}, Q={s: cnt[s]['Q'] / nread for s in SETS}, Q_emitted={s: cnt[s]['Qemit'] / nread for s in SETS},
           context={str(k): cctx[k] / nread for k in (1, 2, 3)}, units_cells_min_med_max=[int(min(map(len, mem))), int(np.median(list(map(len, mem)))), int(max(map(len, mem)))])
os.makedirs(os.path.join(HERE, 'outB'), exist_ok=True)
json.dump(out, open(os.path.join(HERE, 'outB', label + '.json'), 'w'))
print(f"{label:20s} cpu {cpu:.0f}s | last quarter, next byte right: D x {out['D']['x']:.3f} x,s2,s4 {out['D']['x,s2,s4']:.3f} | Q x {out['Q']['x']:.3f} x,s2,s4 {out['Q']['x,s2,s4']:.3f}"
      f" | stream: 1 byte {out['context']['1']:.3f} 2 bytes {out['context']['2']:.3f} 3 bytes {out['context']['3']:.3f}", flush=True)
