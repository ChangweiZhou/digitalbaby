#!/usr/bin/env python3
"""PLAN_2.md: cells that step once a word, beside the table's cells (mismatch form of table.py, kernel unchanged).
    python3 words.py arm=0|1|2|3|4 [K=6] [eta=...] [readings=...] [first=1|2]     settings chosen on the last twentieth of the stream unless all three are given
    python3 words.py show                                                       a few positions with their contexts, to check the keys by eye
Arms: 0 the table's cells; 1 with A (sep, cur); 2 with A and B (prev, sep, cur); 3 with A, B and C (prev2, prev, sep, cur);
4 as 3 with a clock that is blind to the words (four bytes "letters", the fifth "other", whatever they are)."""
import ctypes, json, os, sys, time
import numpy as np
import score, table
HERE = os.path.dirname(os.path.abspath(__file__)); S = 65; MASK = (1 << 62) - 1
PD = ctypes.POINTER(ctypes.c_double); PI = ctypes.POINTER(ctypes.c_int64); PY = ctypes.POINTER(ctypes.c_int32)

def word_keys(x, clock=0, ends=None):
    """For every position t (after byte t has been read) the contexts of kinds A, B, C as numbers below 2**62.
    A word is a run of letters (byte indices 13..64); with clock=c the 'letters' are the positions with t % c < c - 1;
    with ends given (a boolean array) a position where ends[t] is true is 'another byte' and all others are 'letters'."""
    n = len(x); isl = (x >= 13) if not clock else (np.arange(n) % clock < clock - 1)
    if ends is not None: isl = ~ends
    A = np.zeros(n, np.int64); B = np.zeros(n, np.int64); C = np.zeros(n, np.int64); xs = x.tolist(); il = isl.tolist()
    cur = sep = prev = prev2 = 0; inword = False
    for t in range(n):
        if il[t]:
            inword = True; cur = (cur * 1000003 + xs[t] + 1) & MASK
        else:
            if inword: prev2 = prev; prev = cur; cur = 0; sep = 0; inword = False
            sep = (sep * 1000003 + xs[t] + 1) & MASK
        a = (sep * 6364136223846793005 + cur * 1442695040888963407 + 1) & MASK
        b = (a * 2862933555777941757 + prev * 3202034522624059733 + 2) & MASK
        A[t] = a; B[t] = b; C[t] = (b * 3935559000370003845 + prev2 * 2685821657736338717 + 3) & MASK
    return [A, B, C]

def make_ids(x, keys, lo, hi, build_hi, K, first):
    """Cells active before each byte lo..hi-1: the table's K + 1 columns, then one column for each added kind; cells made from x[:build_hi].
    first=2: a cell of an added kind is made at the second occurrence of its context (contexts that occur once have no cell)."""
    ids, off = table.cell_ids(x, lo, hi, K, build_hi); tb = np.arange(K - 1, build_hi - 1); t = np.arange(lo - 1, hi - 1); cols = [ids]; kinds = []; firsts = []
    for kk in keys:
        uc, fi, cnt = np.unique(kk[tb], return_index=True, return_counts=True)
        if first == 2: keep = cnt >= 2; uc = uc[keep]; fi = fi[keep]
        j = np.searchsorted(uc, kk[t]); j[j == len(uc)] = 0; cols.append(np.where(uc[j] == kk[t], j + off, -1)[:, None]); off += len(uc); kinds.append(int(len(uc))); firsts.append(tb[fi])
    return np.ascontiguousarray(np.hstack(cols)), off, kinds, firsts

def step(L, ids, y, W, uses, eta, learn, P=None):
    L.table_steps(len(y), ids.shape[1], ids.ctypes.data_as(PI), y.ctypes.data_as(PY), W.ctypes.data_as(PD), uses.ctypes.data_as(PD), eta, learn, P.ctypes.data_as(PD) if P is not None else None)

def teach(x, keys, build_hi, lo_s, hi_s, K, eta, readings, first):
    """Cells read x[:build_hi] `readings` times; after each reading x[lo_s:hi_s] is scored with learning off. Returns the scores, the last P, cell numbers."""
    L = table.kern(); ids, ncell, kinds, firsts = make_ids(x, keys, K, build_hi, build_hi, K, first); y = np.ascontiguousarray(x[K:build_hi].astype(np.int32)); ids1 = ids
    if first == 2 and keys:
        ids1 = ids.copy()
        for c, f in enumerate(firsts): ids1[f - (K - 1), K + 1 + c] = -1                 # at the first occurrence of its context the cell is not there yet
    ids2 = make_ids(x, keys, lo_s, hi_s, build_hi, K, first)[0]; y2 = np.ascontiguousarray(x[lo_s:hi_s].astype(np.int32)); W = np.zeros((ncell, S)); uses = np.zeros(ncell); out = []; P = np.zeros((len(y2), S))
    for r in range(readings):
        step(L, ids1 if r == 0 else ids, y, W, uses, eta, 1); step(L, ids2, y2, W, uses, eta, 0, P); out.append(score.score(P, x[lo_s:hi_s]))
    return out, P, ncell, kinds

def arm_keys(x, arm, ends=None):
    if arm == 0: return []
    kk = word_keys(x, clock=5 if arm == 4 else 0, ends=ends); return kk[:min(arm, 3)]

def run(kw, ends=None, tag=None):
    x, cut, _ = score.load(); arm = int(kw.get("arm", 3)); K = int(kw.get("K", 6)); dev = cut - cut // 20; t0 = time.process_time(); keys = arm_keys(x, arm, ends); grid = {}
    if all(k in kw for k in ("eta", "readings", "first")): eta, rd, first = float(kw["eta"]), int(kw["readings"]), int(kw["first"])
    else:
        for f in ((1, 2) if keys else (1,)):
            for e in (0.125, 0.25, 0.5, 1.0):
                for r, sc in enumerate(teach(x, keys, dev, dev, cut, K, e, 4, f)[0]): grid[f"{e}/{r + 1}/{f}"] = sc["nats"]
        eta, rd, first = min(grid, key=grid.get).split("/"); eta, rd, first = float(eta), int(rd), int(first)
    out, P, ncell, kinds = teach(x, keys, cut, cut, len(x), K, eta, rd, first); r = dict(out[-1])
    r.update(arm=arm, K=K, eta=eta, readings=rd, first=first, grid=grid, cells=ncell, cells_added=kinds, test_after_each_reading=[round(o["nats"], 4) for o in out], cpu=round(time.process_time() - t0, 1)); return r

def show():
    x, cut, _ = score.load(); vals = score.byte_values(); kk = word_keys(x); txt = lambda a, b: bytes(vals[x[a:b]]).decode().replace("\n", "/")
    for t in (120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130): print(t, repr(txt(t - 30, t + 1)), [int(k[t]) % 100000 for k in kk])
    for name, k in zip("ABC", kk): print(name, "contexts in the stream:", len(np.unique(k[:cut - 1])), "of them once:", int((np.unique(k[:cut - 1], return_counts=True)[1] == 1).sum()))
    i = np.flatnonzero(x[:cut] < 13); print("words in the stream:", int((np.diff(i) > 1).sum()), "mean bytes from the end of one word to the end of the next:", round(float(np.diff(i[np.append(True, np.diff(i) > 1)]).mean()), 2))

if __name__ == "__main__":
    if sys.argv[1] == "show": show()
    else:
        kw = dict(a.split("=") for a in sys.argv[1:]); r = run(kw); tag = "words_" + "_".join(f"{k}{v}" for k, v in sorted(kw.items()))
        json.dump(r, open(os.path.join(HERE, "res", tag + ".json"), "w")); print(tag, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k not in ("n", "grid")}, "best of grid:", sorted((round(v, 4), k) for k, v in r["grid"].items())[:3], flush=True)
