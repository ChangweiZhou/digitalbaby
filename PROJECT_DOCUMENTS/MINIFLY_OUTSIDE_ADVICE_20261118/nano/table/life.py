#!/usr/bin/env python3
"""PLAN_5.md: what the system keeps and what it gains when the text changes.
    python3 life.py T2|T0|M0
A = tiny Shakespeare (stream: its first nine tenths); B = three other texts read one after another (first nine tenths of each).
A byte of B that A does not have becomes a space; carriage returns are removed. Every scoring is done with nothing changing."""
import json, os, sys, time
import numpy as np
import score, table, words, trust
HERE = os.path.dirname(os.path.abspath(__file__)); S = 65
DATA = os.path.join(HERE, "..", "..", "..", "..", "minifly_v148_pretrain_probe_dev_20261025", "data")
BT = ("bryant-stories.txt", "burgess-busterbrown.txt", "carroll-alice.txt")

def load_all():
    xa, cut, _ = score.load(); vals = score.byte_values(); tab = np.full(256, 1, np.int64); tab[vals] = np.arange(S); parts = [("A", xa)]; replaced = {}
    for f in BT:
        raw = np.frombuffer(open(os.path.join(DATA, f), "rb").read(), np.uint8); raw = raw[raw != 13]; parts.append((f.split(".")[0], tab[raw])); replaced[f.split(".")[0]] = round(100 * float((~np.isin(raw, vals)).mean()), 2)
    x = np.concatenate([p[1] for p in parts]); segs = {}; off = 0
    for name, p in parts:
        c = cut if name == "A" else int(0.9 * len(p)); segs[name] = (off, off + c, off + len(p)); off += len(p)
    return x, segs, replaced

class Summing:
    """The summing form of PLAN_1.md (strengths changed by the mismatch), over the same ids."""
    def __init__(s, x, K):
        s.x, s.K = x, K; s.ids, s.ncell, _, _ = words.make_ids(x, [], K, len(x), len(x), K, 1); s.W = np.zeros((s.ncell, S)); s.uses = np.zeros(s.ncell); s.L = table.kern()
    def go(s, lo, hi, learn, want_p=False):
        ids = np.ascontiguousarray(s.ids[lo - s.K:hi - s.K]); y = np.ascontiguousarray(s.x[lo:hi].astype(np.int32)); P = np.zeros((hi - lo, S)) if want_p else None
        words.step(s.L, ids, y, s.W, s.uses, 0.5, int(learn), P); return P
    def cells(s): return int((s.uses > 0).sum())

class Trusting:
    def __init__(s, x, K, keys): s.s = trust.Sys(x, keys, K)
    def go(s, lo, hi, learn, want_p=False): return s.s.go(lo, hi, 0.00005, 0.02, learn, want_p)
    def cells(s): return int((s.s.N > 0).sum())

def make(name, x):
    if name == "M0": return Summing(x, 6)
    return Trusting(x, 6, words.word_keys(x) if name == "T2" else [])

def scores(sy, x, segs):
    out = {}
    for name, (a, c, b) in segs.items():
        r = score.score(sy.go(c, b, False, True), x[c:b]); out[name] = dict(nats=r["nats"], right=r["right"], n=int(b - c))
    bn = [k for k in segs if k != "A"]; out["B"] = dict(nats=float(sum(out[k]["nats"] * out[k]["n"] for k in bn) / sum(out[k]["n"] for k in bn)), right=float(sum(out[k]["right"] * out[k]["n"] for k in bn) / sum(out[k]["n"] for k in bn)))
    return out

def run(name):
    x, segs, replaced = load_all(); t0 = time.process_time(); K = 6; r = dict(system=name, segments={k: list(map(int, v)) for k, v in segs.items()}, replaced_percent=replaced)
    sy = make(name, x); a, c, b = segs["A"]; sy.go(K, c, True); r["cells_after_A"] = sy.cells(); r["after_A"] = scores(sy, x, segs)
    for k in segs:
        if k != "A": sy.go(segs[k][0], segs[k][1], True)
    r["cells_after_A_and_B"] = sy.cells(); r["after_A_and_B"] = scores(sy, x, segs)
    s2 = make(name, x)
    for k in segs:
        if k != "A": s2.go(segs[k][0], segs[k][1], True)
    r["cells_B_only"] = s2.cells(); r["B_only"] = scores(s2, x, segs); r["cpu"] = round(time.process_time() - t0, 1); return r

if __name__ == "__main__":
    name = sys.argv[1]; r = run(name); json.dump(r, open(os.path.join(HERE, "res", f"life_{name}.json"), "w"))
    f = lambda d: " ".join(f"{k} {d[k]['nats']:.4f}/{d[k]['right']:.1f}" for k in d)
    print(name, "| after A:", f(r["after_A"]), "| after A and B:", f(r["after_A_and_B"]), "| B only:", f(r["B_only"]), "| cells", r["cells_after_A"], r["cells_after_A_and_B"], r["cells_B_only"], "| replaced %", r["replaced_percent"], "| cpu", r["cpu"], flush=True)
