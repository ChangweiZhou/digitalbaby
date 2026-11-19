"""Section 16 (reviewer's task of 14:07 UTC): rules D (eta 1/64) and Q (eta 1/16) on [x, s_2, s_4] (own_trace.py's readings: copies of
the raw output, each component standardised with its own running mean and deviation and clipped at +-8, plus a constant 1), with the
cells a unit sees chosen by `cells`:
  a  every unit sees the cells holding a connection from an input line at the end of the life (distance 1 from the input lines)
  b  every unit sees the cells at distance 1 or 2
  c  each unit sees its own driver's cells (set (a) cells of the driver's enlarged neighbourhood, the driver included), as before
  d  every unit sees every cell that is not a feedback cell
Distances by breadth-first search along the final connections (the life's .conn file), from the input lines of every byte value of
the stream, feedback cells not passed (my_tissue.distances).
digits=1 adds to every unit's inputs a one-hot of the present byte's digit (10 inputs, 0/1, not standardised; all 0 if not a digit).
Scores: last quarter (from 3(NB-1)/4 + 1); share per block of 32,768 bytes and the largest weight-row norm at each block's end; the
recall of each byte value in the last quarter (values D or Q never names there). Q's generator default_rng([seed, 3]) as own_trace.py
form (iii) (so cells=c, digits=0 repeats own_trace.py's D and Q (iii)).
Also, for the cells seen (cells a, b, d) or the union of the drivers' cells (c): the linear reader of perbyte_own.py (last 8,192 bytes,
outputs x) for the present byte: overall share and mean over the digits.

    python3 own_trace2.py <rec> <conn file> <seed> <nbytes> <tag> <A4|L8> cells=a|b|c|d [digits=0|1] [reader=1]
"""
import json, sys, time
import numpy as np
sys.path.insert(0, "/home/user/tissue_deps"); sys.path.insert(0, "/home/user/tissue_check"); sys.path.insert(0, "/home/user/prep_deps")
import tissue_step1 as ts, my_tissue as mt, perbyte_own as PB
import life_stream as LS, score_emissions as SE

recf, connf, seed, NB, tag, kind = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6]
kw = dict(a.split("=") for a in sys.argv[7:]); mode = kw["cells"]; digits = kw.get("digits", "0") == "1"; reader = kw.get("reader", "1") == "1"
t0 = time.time()
FR, RE = (1.0, 0.25) if kind == "A4" else (1 / 8, 0.5)
Bt = ts.birth(seed); _, members, _ = mt.long_axons(Bt, seed, FR, RE)
b, _ = ts.stream(seed, NB); b = b.astype(np.int64)
L = LS.life(seed); assert np.array_equal(L["bytes"][:NB], b)
isfb = np.asarray(Bt["kind"]) == ts.FBK; drivers = list(Bt["driver"])
conns = [(int(r[0]), int(r[1])) for r in (l.split() for l in open(connf)) if len(r) >= 2]
dist = mt.distances(Bt, conns, b)
if mode == "c":
    near = set(drivers)
    for i in drivers: near.update(int(m) for m in members[i] if m < ts.N)
    cells = np.array(sorted(i for i in near if not isfb[i])); col = {c: k for k, c in enumerate(cells)}
    own = [sorted({col[c] for c in [i] + [int(m) for m in members[i] if m < ts.N] if c in col}) for i in drivers]
else:
    sel = {"a": (dist == 1), "b": (dist >= 1) & (dist <= 2), "d": np.ones(ts.N, bool)}[mode] & ~isfb
    cells = np.flatnonzero(sel); own = None
K = len(cells)
R = np.memmap(recf, dtype=np.float32, mode='r').reshape(-1, ts.N)            # read row by row (a 262,144-byte record is 2 GB)
M = NB - 1; y = b[1:]; q0 = (3 * M) // 4 + 1
DIG = np.zeros((256, 10)); DIG[48:58] = np.eye(10)
if own is not None:
    G = max(map(len, own)); IDX = np.full((256, G), K)
    for v in range(256): IDX[v, :len(own[v])] = own[v]
    F = 3 * G + 1 + (10 if digits else 0)
else:
    F = 3 * K + 1 + (10 if digits else 0)
W = [np.zeros((256, F)) for _ in range(2)]; gen = np.random.default_rng([seed, 3])
pred = np.full((2, NB), -1, np.int64); seen = np.zeros(256, bool)
S = np.zeros((3, K)); m = np.zeros((3, K)); s2 = np.zeros((3, K))
blocks = {0: [], 1: []}
for t in range(M):
    x = np.asarray(R[t, cells], dtype=np.float64); seen[b[t]] = True; r = 1.0 / min(t + 1, 1024)
    for j, k in enumerate((1, 2, 4)): S[j] += (x - S[j]) / k
    m += r * (S - m); s2 += r * ((S - m) ** 2 - s2); sd = np.sqrt(s2)
    Z = np.clip(np.where(sd > 0, (S - m) / np.where(sd > 0, sd, 1), 0.0), -8, 8)
    extra = [np.ones(1)] + ([DIG[b[t]]] if digits else [])
    if own is not None:
        Zp = np.concatenate([Z, np.zeros((3, 1))], axis=1)
        U = np.ones((256, F)); U[:, :3 * G] = Zp[:, IDX].transpose(1, 0, 2).reshape(256, 3 * G)
        if digits: U[:, 3 * G + 1:] = DIG[b[t]]
    else:
        u = np.concatenate([Z.reshape(-1)] + extra); U = np.broadcast_to(u, (256, F))
    n2 = (U * U).sum(1) if own is not None else np.full(256, float(u @ u))
    for c in range(2):
        a = (W[c] * U).sum(1) if own is not None else W[c] @ u
        if c == 0:                                             # D
            pred[0, t + 1] = int(a.argmax()); tg = np.zeros(256); tg[y[t]] = 1.0
            W[0] += ((1 / 64) * (tg - a) / n2)[:, None] * U
        else:                                                  # Q
            e = int(a.argmax())
            if gen.random() < 1 / 16:
                sv = np.flatnonzero(seen); e = int(sv[gen.integers(0, len(sv))])
            pred[1, t + 1] = e; ok = e == y[t]
            if ok:
                dq = (1 / 16) * (0.0 - a) / n2; dq[e] = (1 / 16) * (1.0 - a[e]) / n2[e]; W[1] += dq[:, None] * U
            else:
                W[1][e] += (1 / 16) * (0.0 - a[e]) / n2[e] * U[e]
    if (t + 2) % 32768 == 0:
        for c in range(2):
            lo = max(1, t + 2 - 32768); blocks[c].append(dict(end=t + 2, share=float((pred[c, lo:t + 2] == b[lo:t + 2]).mean()),
                                                               max_row_norm=float(np.linalg.norm(W[c], axis=1).max())))
refs = SE.references(L["bytes"][:NB])
lq = {k: float((refs[k] == b)[q0:].mean()) for k in ("ref_1", "ref_2", "ref_3")}; lq["end mark"] = float((b[q0:] == 10).mean())
tq = b[q0:]; vals = sorted(set(tq.tolist()))
res = {}
for c, nm in enumerate(("D 1/64", "Q 1/16")):
    p = pred[c, q0:]
    rec = {int(v): float((p[tq == v] == v).mean()) for v in vals}
    res[nm] = dict(last_quarter=float((p == tq).mean()), blocks=blocks[c], recall=rec,
                   never_named=[int(v) for v in vals if rec[v] == 0.0], values=len(vals))
rd = None
if reader:
    rc = cells if own is None else cells                       # c: the union of the drivers' cells
    lo = NB - 8192; Xr = np.asarray(R[lo:NB][:, rc], dtype=np.float64)
    rr = PB.read(Xr, b, lo)["0"]; rd = dict(overall=rr["overall"], digits=rr["digits_mean"], end=rr["end"], eq=rr["eq"])
out = dict(tag=tag, seed=seed, bytes=NB, kind=kind, cells=mode, digits=digits, cells_seen=int(K),
           distance_counts={int(d): int(((dist == d) & ~isfb).sum()) for d in sorted(set(dist.tolist()))},
           rules=res, levels_last_quarter=lq, reader_present_byte=rd, seconds=time.time() - t0)
json.dump(out, open(f"/home/user/tissue_check/out/tr2_{tag}.json", "w"), indent=0)
print(f"== {tag} cells={mode} digits={int(digits)} ({K} cells): " + " ".join(f"{k} {100*v['last_quarter']:.2f}" for k, v in res.items()) +
      (f" | reader present byte {rd['overall']:.3f}, digits {rd['digits']:.3f}" if rd else "") + f" | {time.time() - t0:.0f} s", flush=True)
