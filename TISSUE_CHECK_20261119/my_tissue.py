"""Front end of the independent tissue program (my_tissue.c). It uses only E63's tissue_step1.birth and stream and the helper's
readings.read_window, as THE_TISSUE_AS_RUN_20261119.md section 5 allows.

    python3 my_tissue.py <N0|N64|N256> <seed> <nbytes> [--u1 0|1] [--trace STEPS] [--out DIR]

Writes DIR/<tag>.{rec,conn,stats,trace} and DIR/<tag>.json: the connections, made and withdrawn, and S(d, L) by the tool over the last
4,096 bytes (or the whole run if shorter), with d the distance by breadth-first search along the final connections from the lines of the
byte values present in that window (as the reviewer's far.py computes it).
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, "/home/user/tissue_deps")
import tissue_step1 as ts        # noqa: E402
import readings                  # noqa: E402

TAUS = {"N0": (2,), "N64": (2, 4, 8, 16, 32, 64), "N128": (2, 4, 8, 16, 32, 64, 128), "N256": (2, 4, 8, 16, 32, 64, 128, 256)}


def long_axons(B, seed, frac=1 / 8, reach=0.5):
    """Section 3a: one excitatory cell in 8 that is neither a driver nor a feedback cell is long; it is a member of every non-feedback
    cell within `reach`; the 8 starting connections are drawn again from the enlarged lists."""
    rng = np.random.default_rng(seed + 20261120)
    P = B["P"]
    long = [i for i in range(ts.N) if B["sig"][i] > 0 and i not in B["fbset"] and i not in B["outset"] and rng.random() < frac]
    members, src = {}, {}
    for i in range(ts.N):
        if i in B["fbset"]:
            members[i], src[i] = [], []
            continue
        xi, yi = P[i]; have = set(B["members"][i])
        add = [j for j in long if j != i and j not in have and (P[j][0] - xi) ** 2 + (P[j][1] - yi) ** 2 <= reach ** 2]
        members[i] = list(B["members"][i]) + add
        src[i] = [int(v) for v in rng.choice(members[i], size=min(8, len(members[i])), replace=False)]
    return long, members, src


def prepare(kind, seed, nbytes, out, long=False):
    B = ts.birth(seed)
    if long:
        B = dict(B); B["long"], B["members"], B["src"] = long_axons(B, seed)
    tau = np.full(ts.N, 2.0) if kind == "N0" else np.random.default_rng(seed + 20261119).choice(TAUS[kind], size=ts.N).astype(float)
    lines = [str(seed)]
    for i in range(ts.N):
        src = B["src"][i]
        lines.append(f"{i} {int(i in B['fbset'])} {B['sig'][i]:.1f} {tau[i]:.1f} {len(src)} " + " ".join(map(str, src)))
        mem = B["members"][i]
        lines.append(f"{len(mem)} " + " ".join(map(str, mem)))
    bp = out / f"birth_{seed}_{kind}{'_L8' if long else ''}.txt"; tmp = Path(f"{bp}.{os.getpid()}"); tmp.write_text("\n".join(lines) + "\n"); os.replace(tmp, bp)
    b, _ = ts.stream(seed, nbytes)
    sp = out / f"stream_{seed}_{nbytes}.bin"; tmp = Path(f"{sp}.{os.getpid()}"); tmp.write_bytes(b.tobytes()); os.replace(tmp, sp)
    return B, b, bp, sp


def distances(B, conns, wb):
    lines = {ts.LINE0 + int(v) for v in set(wb.tolist())}
    src = {i: [] for i in range(ts.N)}
    for i, s in conns:
        src[i].append(s)
    isfb = np.asarray(B["kind"]) == ts.FBK
    dist = np.zeros(ts.N, np.int16)
    fr = np.array([(not isfb[i]) and any(s in lines for s in src[i]) for i in range(ts.N)])
    dist[fr] = 1; d = 1
    while fr.any():
        nxt = np.array([(dist[i] == 0) and (not isfb[i]) and any(s < ts.N and fr[s] for s in src[i]) for i in range(ts.N)])
        d += 1; dist[nxt] = d; fr = nxt
    return dist


def main(a):
    kind, seed, nbytes = a[0], int(a[1]), int(a[2])
    opt = lambda k, d: a[a.index(k) + 1] if k in a else d
    u1 = int(opt("--u1", "1")); trace = int(opt("--trace", "0")); out = Path(opt("--out", str(HERE / "out"))); out.mkdir(parents=True, exist_ok=True)
    tag = f"{kind}_{seed}_{nbytes}_u{u1}"
    B, b, bp, sp = prepare(kind, seed, nbytes, out)
    lo, hi = max(0, nbytes - 4096), nbytes
    subprocess.run([str(HERE / "my_tissue"), str(bp), str(sp), str(nbytes), str(lo), str(hi), str(out / tag), str(trace), str(u1)], check=True)
    st = json.loads((out / f"{tag}.stats").read_text())
    conns = [(int(r.split()[0]), int(r.split()[1])) for r in (out / f"{tag}.conn").read_text().split("\n") if r]
    rec = np.fromfile(out / f"{tag}.rec", dtype=np.float32).reshape(hi - lo, ts.N)
    wb = b[lo:hi]
    dist = distances(B, conns, wb)
    DS = tuple(range(1, int(dist.max()) + 1))
    xa = np.where(rec < 0.01, 0.0, rec).astype(np.float64)
    r = readings.read_window(xa, wb, dist, seed, stream=b, start=lo, distances=DS, lags=(0, 1, 2))
    S = {str(d): {str(L): (r["scores"][d]["by_lag"][L] or {}).get("S") for L in (0, 1, 2)} for d in DS}
    res = dict(kind=kind, seed=seed, nbytes=nbytes, u1=u1, window=[lo, hi], stats=st, cells_at={str(d): int((dist == d).sum()) for d in DS}, S=S,
               connections_a_cell_all_2048=st["connections"] / ts.N)
    (out / f"{tag}.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(dict(stats=st, cells_at=res["cells_at"], S={d: S[d] for d in S if int(d) in (1, 2, 4, 8, 10)}), default=float))


if __name__ == "__main__":
    main(sys.argv[1:])
