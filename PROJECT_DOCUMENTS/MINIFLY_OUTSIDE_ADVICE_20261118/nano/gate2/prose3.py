#!/usr/bin/env python3
"""PLAN_8.md: gate3.py's five answering cells on nano/merge's prose system without classes, through a life with three books.
    python3 prose3.py [seed=1]
The books are read from the repository's tree (git show) into build/, which is not committed."""
import json, os, subprocess, sys, time
import numpy as np
import gate3 as H
G = H.G; sys.path.insert(0, os.path.join(H.HERE, "..", "merge"))
import score
BOOKS = [("minifly_v155_keys_dev_v1_20261026", b) for b in ("chesterton-ball", "austen-emma", "melville-moby_dick")]

def book(folder, name):
    p = os.path.join(H.BUILD, name + ".txt")
    if not os.path.exists(p): open(p, "wb").write(subprocess.check_output(["git", "show", f"HEAD:{folder}/data/{name}.txt"], cwd=H.HERE))
    return np.frombuffer(open(p, "rb").read(), np.uint8)

def run(seed=1):
    t0 = time.process_time(); xa, cut, _ = score.load(); vals = score.byte_values(); tab = np.full(256, 1, np.int64); tab[vals] = np.arange(65); parts = [("A", xa)]; replaced = {}
    for folder, b in BOOKS:
        raw = book(folder, b); raw = raw[raw != 13]; parts.append((b, tab[raw])); replaced[b] = round(100 * float((~np.isin(raw, vals)).mean()), 2)
    x = np.concatenate([p[1] for p in parts]); segs = {}; off = 0
    for name, p in parts:
        c = cut if name == "A" else int(0.9 * len(p)); segs[name] = (off, off + c, off + len(p)); off += len(p)
    K = 6; s = H.Sys2(65, K, [tuple(range(1, k + 1)) for k in range(K + 1)], 0.00005, 0.02, 0.25, False, text=x, P=0, mode="none", dseed=seed)
    def sc(lo, hi):
        Ps = s.say_all(G.ctx_of(x, lo, hi, s.D)); r = {}
        for c, P in zip(H.CELLS, Ps): q = score.score(P, x[lo:hi]); r[c] = dict(nats=q["nats"], right=q["right"])
        return r
    out = dict(task="prose_life", seed=seed, segments={k: list(map(int, v)) for k, v in segs.items()}, replaced_percent=replaced, scores={})
    for name, (a, c, b) in segs.items():
        lo = max(a, s.D)
        for m in range(lo, c, 125000): hi = min(c, m + 125000); s.learn(G.ctx_of(x, m, hi, s.D), x[m:hi])
        out["scores"]["A_after_its_stream" if name == "A" else name] = sc(c, b); print(name, {k: round(v["nats"], 4) for k, v in out["scores"]["A_after_its_stream" if name == "A" else name].items()}, flush=True)
    a, c, b = segs["A"]; out["scores"]["A_at_the_end"] = sc(c, b); print("A at the end", {k: round(v["nats"], 4) for k, v in out["scores"]["A_at_the_end"].items()}, flush=True)
    out.update(sizes=s.sizes(), parts=int(s.Gt[0].size), parts_used=[int((s.Gu[g] > 0).sum()) for g in range(s.nG)], largest_part=[float(np.abs(s.Gt[g]).max()) for g in range(s.nG)], cpu=round(time.process_time() - t0, 1))
    json.dump(out, open(os.path.join(H.HERE, "res", f"prose_life_new_seed{seed}.json"), "w")); print("cells", out["sizes"]["cells"], "parts used", out["parts_used"], "of", out["parts"], "cpu", out["cpu"]); return out

if __name__ == "__main__":
    kw = dict(a.split("=") for a in sys.argv[1:]); run(int(kw.get("seed", 1)))
