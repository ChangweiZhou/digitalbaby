#!/usr/bin/env python3
"""PLAN_21.md: a cell that recalls. One more sender for gate_all.py's system: at each byte, the byte that followed the last earlier place where the same M bytes stood.
    python3 recall.py [control=1] [M=8]
The system: the bare system of PLAN_20.md (no word cells, no companions, no relay system), second cell with parts by kind and byte."""
import json, os, sys, time
import numpy as np
import score, gate_all as GA
HERE = os.path.dirname(os.path.abspath(__file__)); S = 65; E0 = 0.02
EDGES = np.array([8, 10, 12, 16, 20, 24, 32, 48, 64])                      # the levels of how long the two runs agree: none | 8-9 | 10-11 | 12-15 | 16-19 | 20-23 | 24-31 | 32-47 | 48-63 | 64 or more

def last_places(x, M, limit=None):
    """prev[t]: the last q < t (and q < limit, if given) with x[q-M:q] == x[t-M:t]; -1 if none. Only bytes before t are used."""
    n = len(x); key = np.zeros(n, np.uint64)
    for i in range(1, M + 1): key[M:] |= x[M - i:n - i].astype(np.uint64) << np.uint64(7 * (i - 1))
    order = np.argsort(key[M:], kind="stable") + M; same = key[order][1:] == key[order][:-1]; prev = np.full(n, -1, np.int64); prev[order[1:][same]] = order[:-1][same]
    if limit is not None:
        for _ in range(64):
            m = prev >= limit
            if not m.any(): break
            prev[m] = prev[prev[m]]
        assert not (prev >= limit).any()
    return prev

def agree(x, prev, M, cap=64):
    """How many bytes before t and before prev[t] agree, counted back from the byte just before each (at least M where prev >= 0), up to cap."""
    n = len(x); L = np.where(prev >= 0, M, 0); t = np.flatnonzero(prev >= 0); q = prev[t]; alive = np.ones(len(t), bool)
    for j in range(M + 1, cap + 1):
        ok = alive & (q - j >= 0); ok[ok] = x[t[ok] - j] == x[q[ok] - j]; alive = ok
        if not alive.any(): break
        L[t[alive]] = j
    return L

def sender(x, prev, M, rng=None):
    n = len(x); L = agree(x, prev, M); has = prev >= 0; X = np.full((n, S), -np.log(S), np.float32); src = prev.copy()
    if rng is not None: src[has] = rng.integers(0, np.maximum(np.flatnonzero(has), 1))          # the control: the byte that followed a place drawn by chance before t
    hi = np.float32(np.log((1 + E0) / (1 + S * E0))); lo = np.float32(np.log(E0 / (1 + S * E0))); t = np.flatnonzero(has); X[t] = lo; X[t, x[src[t]]] = hi
    lev = np.where(has, 1 + np.searchsorted(EDGES, L, side="right") - 1, 0).astype(np.int32); lev = np.clip(lev, 0, 9).astype(np.int32)
    return X, lev, L, src

def run(kw):
    x, cut, _ = score.load(); t0 = time.process_time(); M = int(kw.get("M", 8)); control = int(kw.get("control", 0)); rng = np.random.default_rng(7) if control else None
    go_prev = last_places(x, M); fz_prev = last_places(x, M, cut); assert (go_prev[:cut] == fz_prev[:cut]).all()
    Xg, lg, Lg, sg = sender(x, go_prev, M, rng); rng = np.random.default_rng(7) if control else None; Xf, lf, Lf, sf = sender(x, fz_prev, M, rng)
    s = GA.SysGate(x, ("words", "companions", "relay"), 0, 1); s.T[s.K1] = 0.0; s.X, s.xlev = Xg, lg; s.go(6, cut, True)
    # with nothing changing: only places in the stream may be recalled
    s.X, s.xlev = Xf, lf; r = dict(task="recall", args=kw, M=M, kinds=int(s.K1), frozen=GA.both(s, cut, len(x)), trust_of_the_recalling_cell_by_level=np.round(s.T[s.K1, :10], 3).tolist())
    y = x[cut:]; has = fz_prev[cut:] >= 0; rec = x[np.maximum(sf[cut:], 0)]
    r["test_bytes_with_a_place"] = float(has.mean()); r["recalled_is_true_by_level"] = {int(l): (int((lf[cut:] == l).sum()), float((rec[lf[cut:] == l] == y[lf[cut:] == l]).mean()) if (lf[cut:] == l).any() else None) for l in range(1, 10)}
    long_ = Lf[cut:] >= 16; r["recalled_is_true_where_16_or_longer"] = (int(long_.sum()), float((rec[long_] == y[long_]).mean()) if long_.any() else None)
    s.X, s.xlev = Xg, lg; r["going_on"] = GA.both(s, cut, len(x), True); r["cpu"] = round(time.process_time() - t0, 1); return r

def check():
    """last_places and agree against plain Python on a short piece; the sender uses nothing at or after t."""
    x, cut, _ = score.load(); x = x[:6000].copy(); M = 4; prev = last_places(x, M); L = agree(x, prev, M); ok = True; seen = {}
    for t in range(len(x)):
        k = tuple(x[t - M:t]) if t >= M else None; q = seen.get(k, -1) if k is not None else -1
        if q != prev[t]: ok = False
        if q >= 0:
            j = 0
            while j < 64 and q - j - 1 >= 0 and x[t - j - 1] == x[q - j - 1]: j += 1
            if j != L[t]: ok = False
        if k is not None: seen[k] = t
    x2 = x.copy(); x2[3000:] = (x2[3000:] + 1) % S; p2 = last_places(x2, M); ok &= bool((p2[:3001] == prev[:3001]).all()); lim = last_places(x, M, 3000); ok &= bool((lim < 3000).all()) and bool((lim[:3000] == prev[:3000]).all())
    print(f"recalled places and lengths of agreement against plain Python on {len(x)} bytes (M = {M}); nothing at or after a byte changes its place; places under a limit:", "PASSED" if ok else "FAILED"); return ok

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "check": sys.exit(0 if check() else 1)
    kw = dict(a.split("=") for a in sys.argv[1:]); r = run(kw); tag = "recall" + "".join(f"_{k}{v}" for k, v in sorted(kw.items())); json.dump(r, open(os.path.join(HERE, "res", tag + ".json"), "w"))
    f = lambda d: (round(d["cell1"]["nats"], 4), round(d["cell2"]["nats"], 4))
    print(tag, "| last tenth, nothing changing (first cell, second):", f(r["frozen"]), "| going on:", f(r["going_on"]), "| trust by level:", r["trust_of_the_recalling_cell_by_level"], "| test bytes with a place:", round(r["test_bytes_with_a_place"], 3), "| recalled byte is the true one, by level (bytes, share):", {k: (v[0], None if v[1] is None else round(v[1], 3)) for k, v in r["recalled_is_true_by_level"].items()}, "| 16 or longer:", r["recalled_is_true_where_16_or_longer"], "| cpu", r["cpu"], flush=True)
