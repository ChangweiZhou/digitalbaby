"""Reviewer's questions of 14:21 UTC on the A4-0 lives (65,536 bytes), no new run: (1) input lines with no receiver at the end: how
many had none at birth (starting connections of the birth file), how many were made to during the life; (2) cells at distance 1 with no
receiver of their own at the end, and the linear reader of the present byte on the distance-1 cells with at least one receiver against
all distance-1 cells (perbyte_own.read, last 8,192 bytes)."""
import sys, json, numpy as np
sys.path.insert(0, "/home/user/tissue_deps"); sys.path.insert(0, "/home/user/tissue_check")
import tissue_step1 as ts, my_tissue as mt, perbyte_own as PB
out = {}
for s in (73129, 93683, 116089):
    B = ts.birth(s); _, members, src = mt.long_axons(B, s, 1.0, 0.25); isfb = np.asarray(B["kind"]) == ts.FBK
    b, _ = ts.stream(s, 65536); b = b.astype(np.int64); vals = sorted(set(b.tolist()))
    birth_rcv = {v: sum(1 for i in range(ts.N) if not isfb[i] and ts.LINE0 + v in src[i]) for v in vals}
    conns = [(int(r[0]), int(r[1]), int(r[3])) for r in (l.split() for l in open(f"out/OE_A40_{s}.conn"))]
    end_rcv = {v: sum(1 for i, j, mk in conns if j == ts.LINE0 + v) for v in vals}
    made_from_line = {v: sum(1 for i, j, mk in conns if j == ts.LINE0 + v and mk > 0) for v in vals}
    none_end = [v for v in vals if end_rcv[v] == 0]
    dist = mt.distances(B, [(i, j) for i, j, _ in conns], b)
    d1 = np.flatnonzero((dist == 1) & ~isfb)
    nrecv = np.zeros(ts.N + 300, int)
    for i, j, _ in conns: nrecv[j] += 1
    d1r = d1[nrecv[d1] > 0]
    R = np.memmap(f"out/OE_A40_{s}.rec", dtype=np.float32, mode="r").reshape(-1, ts.N); lo = 65536 - 8192
    rd = {}
    for nm, cs in (("all distance-1", d1), ("with a receiver", d1r), ("without a receiver", d1[nrecv[d1] == 0])):
        r0 = PB.read(np.asarray(R[lo:65536][:, cs], dtype=np.float64), b, lo)["0"] if len(cs) else None
        rd[nm] = dict(cells=int(len(cs)), overall=r0["overall"] if r0 else None, digits=r0["digits_mean"] if r0 else None)
    ch = lambda v: "END" if v == 10 else chr(v)
    out[s] = dict(lines_none_at_end=[ch(v) for v in none_end], of_these_none_at_birth=sum(birth_rcv[v] == 0 for v in none_end),
                  lines_none_at_birth=sum(birth_rcv[v] == 0 for v in vals), lines_with_made_connection_at_end=sum(made_from_line[v] > 0 for v in vals),
                  receivers_at_birth_median=float(np.median([birth_rcv[v] for v in vals])), receivers_at_end_median=float(np.median([end_rcv[v] for v in vals])),
                  distance1=rd)
    print(s, json.dumps(out[s]))
json.dump(out, open("out/lines_q.json", "w"), indent=1)
