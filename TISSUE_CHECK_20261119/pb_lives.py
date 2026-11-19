"""Part 2/3 of the A4 task: the linear per-byte reading near the output lines (perbyte_own.read) on every A4 life:
the reference lives (OE_A4_<seed>, every byte recorded) at 16,384 and 65,536, and the last-bit lives (cell p, 1e-12 at byte 0):
A4s_<seed>_p<p> (16,384 bytes, recorded 8,192-16,384) and A4l_<seed>_p<p> (65,536 bytes, recorded 57,344-65,536).
Writes out/pb_A4_lives.json; skips a life whose record is not complete.      python3 pb_lives.py"""
import json, os, sys
import numpy as np
sys.path.insert(0, "/home/user/tissue_deps"); sys.path.insert(0, "/home/user/tissue_check")
import tissue_step1 as ts, perbyte_own as P
O = "/home/user/tissue_check/out/"
res = json.load(open(O + "pb_A4_lives.json")) if os.path.exists(O + "pb_A4_lives.json") else {}
for seed in (93683, 73129, 116089):
    near, _ = P.near_cells(seed, "A4"); b, _ = ts.stream(seed, 65536)
    lives = [("ref", "OE_A4_%d" % seed, 0, e) for e in (16384, 65536)]
    if seed != 116089:
        ps = (100, 197, 294, 391, 488) if seed == 93683 else (100, 197, 1024, 391, 488)   # 73129: cell 294 carries nothing on
        lives += [("p%d" % p, "A4s_%d_p%d" % (seed, p), 8192, 16384) for p in ps]                # (life bit-identical to the reference)
        lives += [("p%d" % p, "A4l_%d_p%d" % (seed, p), 57344, 65536) for p in ps]
    for name, f, rlo, end in lives:
        key = f"{seed} {name} {end}"
        if key in res or not os.path.exists(O + f + ".stats"): continue
        rec = np.memmap(O + f + ".rec", dtype=np.float32, mode="r").reshape(-1, ts.N)
        X = np.asarray(rec[end - 8192 - rlo: end - rlo][:, near], dtype=np.float64)
        res[key] = P.read(X, b, end - 8192)
        json.dump(res, open(O + "pb_A4_lives.json", "w"), indent=0)
        print(key, {L: [round(v[k], 3) for k in ("overall", "end", "eq", "digits_mean")] for L, v in res[key].items()}, flush=True)
