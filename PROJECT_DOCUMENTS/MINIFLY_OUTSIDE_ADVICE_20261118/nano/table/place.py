#!/usr/bin/env python3
"""PLAN_8.md: cells that hear how long ago a byte value last came, beside T2 (trust form).
    python3 place.py arm=0|A|A2|B|C|P|P2|Pc [birth=0] [eta=0.0005]      birth: the trust at which an added kind is born (PLAN_8.md, addition 1)
For a byte value v the cells are keyed on (bytes since v last came, capped at 63; the present byte)."""
import ctypes, json, os, subprocess, sys, time
import numpy as np
import score, table, words, trust
HERE = os.path.dirname(os.path.abspath(__file__)); S = 65; NL = 40; ETA, E0 = 0.00005, 0.02
PD = ctypes.POINTER(ctypes.c_double); PF = ctypes.POINTER(ctypes.c_float); PI = ctypes.POINTER(ctypes.c_int64); PY = ctypes.POINTER(ctypes.c_int32)
SRC = trust.SRC.replace("void mix_steps(", "void mix_steps_w(").replace("s[32][S]; int lev[32];", "s[96][S]; int lev[96];"); assert "s[96][S]" in SRC          # room for 96 kinds; nothing else changed

def kern():
    so = os.path.join(HERE, "kern_w.so"); src = os.path.join(HERE, "kern_w.c")
    if not os.path.exists(src) or open(src).read() != SRC: open(src, "w").write(SRC)
    if not os.path.exists(so) or os.path.getmtime(so) < os.path.getmtime(src): subprocess.check_call(["gcc", "-O3", "-march=native", "-shared", "-fPIC", "-o", so, src, "-lm"])
    L = ctypes.CDLL(so); L.mix_steps_w.argtypes = [ctypes.c_int64, ctypes.c_int64, PI, PY, PF, PF, PF, PD, ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_int, ctypes.c_int, PD, PD, ctypes.c_int]; L.mix_steps_w.restype = None; return L

def since(x, v):
    """bytes since v last came (0 where the present byte is v), capped at 63; 63 before v has come at all."""
    t = np.arange(len(x)); last = np.maximum.accumulate(np.where(x == v, t, -1)); return np.where(last >= 0, np.minimum(t - last, 63), 63)

def pool(x, shift=0, sizes=(2, 3)):
    """PLAN_9.md: a kind for every set of two or of three places among the last eight bytes (place 0 = the present byte), less the two sets the table's cells
    already are; shift moves every place that many bytes further back (the control)."""
    import itertools
    sh = {d: np.concatenate([np.zeros(d + shift, np.int64), x[:len(x) - d - shift]]) for d in range(8)}; keys = []; names = []
    for m in sizes:
        for c in itertools.combinations(range(8), m):
            if c in ((0, 1), (0, 1, 2)): continue
            k = np.zeros(len(x), np.int64)
            for d in c: k = k * S + sh[d]
            keys.append(k); names.append(tuple(d + shift for d in c))
    return keys, names

NAMES = []
def arm_keys(x, arm):
    keys = words.word_keys(x); xp = np.concatenate([[0], x[:-1]])
    if arm in ("A", "A2"): keys.append((since(x, 0) * S + x).astype(np.int64))
    if arm == "A2": keys.append(((since(x, 0) // 4 * S + x) * S + xp).astype(np.int64))
    if arm == "B": keys += [(since(x, v) * S + x).astype(np.int64) for v in range(S)]
    if arm == "C": keys.append(((np.arange(len(x)) % 45) * S + x).astype(np.int64))
    if arm in ("P", "P2", "Pc"):
        k, n = pool(x, 8 if arm == "Pc" else 0, (2,) if arm == "P2" else (2, 3)); keys += k; NAMES.extend(n)
    return keys

def run(arm, birth=0.25, eta=ETA):
    x, cut, _ = score.load(); t0 = time.process_time(); K = 6; keys = arm_keys(x, arm); L = kern(); ids, ncell, kinds, _ = words.make_ids(x, keys, K, len(x), len(x), K, 1); K1 = ids.shape[1]; assert K1 <= 96
    C = np.zeros((ncell, S), np.float32); N = np.zeros(ncell, np.float32); U = np.zeros(ncell, np.float32); T = np.full((K1, NL), 0.25); Tu = np.zeros((K1, NL)); T[10:] = birth          # PLAN_8.md, addition 1: the trust in the added kinds may be born at nothing
    def go(lo, hi, learn, want_p=False):
        i = np.ascontiguousarray(ids[lo - K:hi - K]); y = np.ascontiguousarray(x[lo:hi].astype(np.int32)); P = np.zeros((hi - lo, S)) if want_p else None
        L.mix_steps_w(hi - lo, K1, i.ctypes.data_as(PI), y.ctypes.data_as(PY), C.ctypes.data_as(PF), N.ctypes.data_as(PF), U.ctypes.data_as(PF), T.ctypes.data_as(PD), eta, E0, int(learn), int(learn), 0, P.ctypes.data_as(PD) if want_p else None, Tu.ctypes.data_as(PD), 0); return P
    go(K, cut, True); P = go(cut, len(x), False, True); r = dict(score.score(P)); y = x[cut:]; prev = x[cut - 1:-1]; m = (prev >= 13) & (y == 0)
    r.update(arm=arm, birth=birth, eta=eta, kinds=K1, cells=int((N > 0).sum()), cells_added=kinds[3:], nats_at_a_line_end_after_a_letter=float(-np.log(P[m, 0]).mean()), cpu=round(time.process_time() - t0, 1))
    if arm in ("P", "P2", "Pc"):                          # which kinds the answering cell came to trust (places counted back from the present byte, 0)
        used = Tu[10:] > 0; mt = np.where(used.any(1), (T[10:] * used).sum(1) / np.maximum(used.sum(1), 1), 0.0); o = np.argsort(-mt)[:12]; r["most_trusted_kinds"] = [(list(NAMES[i]), round(float(mt[i]), 3)) for i in o]; r["trust_by_kind"] = [(list(NAMES[i]), round(float(mt[i]), 4)) for i in range(len(NAMES))]
    if arm == "B":                                        # which byte values the answering cell came to trust: the mean trust over the levels that were used
        vals = score.byte_values(); used = Tu[10:] > 0; mt = np.where(used.any(1), (T[10:] * used).sum(1) / np.maximum(used.sum(1), 1), 0.0); o = np.argsort(-mt)[:8]; r["most_trusted_values"] = [(repr(chr(vals[i])), round(float(mt[i]), 3)) for i in o]; r["trust_by_value"] = np.round(mt, 3).tolist()
    return r

if __name__ == "__main__":
    kw = dict(a.split("=") for a in sys.argv[1:]); arm = kw["arm"]; r = run(arm, float(kw.get("birth", 0.25)), float(kw.get("eta", ETA))); tag = "_".join(f"{k}{v}" for k, v in sorted(kw.items())); json.dump(r, open(os.path.join(HERE, "res", f"place_{tag}.json"), "w")); print("arm", arm, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k not in ("n", "trust_by_value", "trust_by_kind", "cells_added")}, flush=True)
