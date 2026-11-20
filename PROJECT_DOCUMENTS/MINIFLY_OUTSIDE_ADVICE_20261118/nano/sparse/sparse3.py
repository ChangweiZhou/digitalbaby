#!/usr/bin/env python3
"""PLAN_6.md: the relay system's answer read as one more sender, with its own trust, beside the counting cells of sparse.py.
    python3 sparse3.py <label> seed=1 [plan=add:400000[,sub:200000]] [m=3] [eta=...] [e=...] [T0=...] [Tx0=0] [block=50] [every=50000] [relay=1]
    python3 sparse3.py check
The relay system is loop_small_case/cells2.py's plain system (256 cells in between, its settings unchanged), run by its own compiled kernel; the files
cells2.py, fastk.py, fastk3.py, kern.c and kern3.c are copied unchanged into relay_copy/. It reads the same stream at the same time. Its answers for a
block of lines are computed from its strengths at the start of the block; then it learns the block (the simplification of nano/table/relay.py).
Its message at a byte: the logs of its branches' outputs, cut off below at 0.01 and scaled to sum to one. The answering cell of the whole holds a
trust for it by ten levels of its confidence (its largest output, cut to 0..1), born at Tx0 and changed by the mismatch as every other trust.
Nothing of the counting cells reaches the relay system. The first byte of a line, which the relay system is never taught, gets an even message.
relay=0: no sender (must reproduce sparse.py)."""
import ctypes, json, os, subprocess, sys, time
import numpy as np
import sparse as sp
from sparse import S, END, NL, HP, text, level_np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "relay_copy"))

SRC = sp.SRC
def rep(a, b):
    global SRC
    assert SRC.count(a) == 1, a; SRC = SRC.replace(a, b)
rep("static void look(sys_t *q, const int32_t *h, int make, double *p) {", "static void look(sys_t *q, const int32_t *h, int make, double *p, const float *sx, int xl) {")
rep("    for (int v = 0; v < S; v++) if (a[v] > mx) mx = a[v];\n    for (int v = 0; v < S; v++) { p[v] = exp(a[v] - mx); z += p[v]; }\n    for (int v = 0; v < S; v++) p[v] /= z;\n}",
    "    if (sx) { const double tk = q->T[q->K * q->nplace * NL + xl]; for (int v = 0; v < S; v++) a[v] += tk * sx[v]; }      /* one more sender: its message is given */\n"
    "    for (int v = 0; v < S; v++) if (a[v] > mx) mx = a[v];\n    for (int v = 0; v < S; v++) { p[v] = exp(a[v] - mx); z += p[v]; }\n    for (int v = 0; v < S; v++) p[v] /= z;\n}")
rep("int64_t steps(sys_t *q, int64_t n, const int32_t *x, int64_t learn_c, int64_t learn_t, double *P) {", "int64_t steps(sys_t *q, int64_t n, const int32_t *x, int64_t learn_c, int64_t learn_t, double *P, const float *X, const int32_t *xlev) {")
rep("        look(q, x + t, (int)learn_c, p); const int y = x[t];", "        look(q, x + t, (int)learn_c, p, X ? X + t * S : 0, X ? xlev[t] : 0); const int y = x[t];")
rep("        if (learn_c) for (int64_t k = 0; k < q->K; k++) { float *c",
    "        if (learn_t && X) { const float *sx = X + t * S; double g = sx[y]; for (int v = 0; v < S; v++) g -= p[v] * sx[v]; const int64_t i = q->K * q->nplace * NL + xlev[t]; q->T[i] += q->eta * g; q->Tu[i] += 1.0; }\n"
    "        if (learn_c) for (int64_t k = 0; k < q->K; k++) { float *c")
rep("void answers(sys_t *q, int64_t n, const int32_t *lines, int64_t L, int64_t from, int64_t cnt, int64_t own, int32_t *out, double *acc) {", "void answers(sys_t *q, int64_t n, const int32_t *lines, int64_t L, int64_t from, int64_t cnt, int64_t own, int32_t *out, double *acc, const float *X, const int32_t *xlev) {")
rep("            look(q, buf + HP + from + c, 0, p); int b = 0;", "            look(q, buf + HP + from + c, 0, p, X ? X + i * q->S : 0, X ? xlev[i] : 0); int b = 0;")

def kern():
    so = os.path.join(HERE, "kern_s3.so"); src = os.path.join(HERE, "kern_s3.c")
    if not os.path.exists(src) or open(src).read() != SRC: open(src, "w").write(SRC)
    if not os.path.exists(so) or os.path.getmtime(so) < os.path.getmtime(src):
        tmp = so + f".{os.getpid()}"; subprocess.check_call(["gcc", "-O3", "-march=native", "-shared", "-fPIC", "-o", tmp, src, "-lm"]); os.replace(tmp, so)
    L = ctypes.CDLL(so); L.steps.restype = ctypes.c_int64; L.steps.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    L.answers.restype = None; L.answers.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p] + [ctypes.c_int64] * 4 + [ctypes.c_void_p] * 4; return L

class Sys3(sp.Sys):
    """sparse.Sys with one more sender; its trusts are the last row of T (ten levels used)."""
    def __init__(s, Tx0=0.0, **kw):
        sp.Sys.__init__(s, **kw); s.L = kern(); s.T = np.vstack([s.T, np.full((1, NL), float(Tx0))]); s.Tu = np.zeros_like(s.T)
    def read(s, x, X=None, xlev=None, learn=True, want_p=False):
        x = np.asarray(x, np.int32).ravel(); buf = np.ascontiguousarray(np.concatenate([s.tail, x])); n = len(x); P = np.zeros((n, s.S)) if want_p else None; done = 0
        if X is not None: X = np.ascontiguousarray(X, np.float32); xlev = np.ascontiguousarray(xlev, np.int32); assert X.shape == (n, s.S) and len(xlev) == n
        while done < n:
            q = s.cs(); done += s.L.steps(ctypes.byref(q), n - done, buf.ctypes.data + 4 * (HP + done), int(learn), int(learn), P.ctypes.data + 8 * s.S * done if want_p else None,
                                           X.ctypes.data + 4 * s.S * done if X is not None else None, xlev.ctypes.data + 4 * done if X is not None else None); s.ncell = q.ncell
            if done < n: s.grow()
        if learn: s.tail = buf[-HP:].copy()
        return P
    def answer_at(s, lines, pos, X=None, xlev=None):
        """The byte at place pos of each line, named from the bytes before it in the line as given (and the sender's message for it)."""
        lines = np.ascontiguousarray(lines, np.int32); out = np.zeros(len(lines), np.int32); q = s.cs()
        if X is not None: X = np.ascontiguousarray(X, np.float32); xlev = np.ascontiguousarray(xlev, np.int32)
        s.L.answers(ctypes.byref(q), len(lines), lines.ctypes.data, lines.shape[1], pos, 1, 0, out.ctypes.data, None, X.ctypes.data if X is not None else None, xlev.ctypes.data if X is not None else None); return out
    def size(s):
        d = sp.Sys.size(s); return d

def ref_read(r, x, X, xlev):
    """numpy reference: sparse.Ref's arithmetic with the given sender added (r.T has the extra row)."""
    P = []; hist = list(r.hist); row = r.K * r.nplace
    for t, y in enumerate(np.asarray(x).ravel().tolist()):
        _, act, ids = r.look(hist, True); a = np.zeros(r.S)
        for k, ti, sv in act: a += r.T[ti] * sv
        sx = X[t].astype(np.float64); a += r.T[row, xlev[t]] * sx; p = np.exp(a - a.max()); p /= p.sum(); P.append(p)
        for k, ti, sv in act: r.T[ti] += r.eta * (sv[y] - p @ sv); r.Tu[ti] += 1.0
        r.T[row, xlev[t]] += r.eta * (sx[y] - p @ sx)
        for i in ids:
            if r.C[i][y] == 0: r.U[i] += 1
            r.C[i][y] += 1; r.N[i] += 1
        hist.append(y); hist = hist[-HP:]
    r.hist = hist; return np.array(P)

class Relay:
    """cells2.py's plain system on its compiled kernel, with its answers taken for bytes it has not yet learned."""
    D = 10
    def __init__(s, seed):
        import cells2; s.c2 = cells2; assert cells2.SYMS == sp.SYMS; s.s = cells2.Sys(seed); s.s.fast = 3; s.tail = np.full(s.D, END, np.int64)
    def message(s, ctx):
        """ctx: (n, 10) the ten bytes before each byte, the latest first. Returns the message, the level of confidence and the outputs."""
        s.s.flush(); a = s.s.out(s.s.W1.T[s.c2.OFF + ctx].sum(1) + s.s.b1); q = np.maximum(a, 0.01); X = np.log(q / q.sum(1, keepdims=True)).astype(np.float32)
        return X, np.minimum((np.clip(a.max(1), 0, 1) * 10).astype(np.int32), 9), a
    def block(s, Xl):
        """The messages for every byte of a block of lines (from the strengths before the block), then the block is learned."""
        x = Xl.ravel().astype(np.int64); full = np.concatenate([s.tail, x]); n = len(x); idx = np.arange(n)[:, None] + s.D - 1 - np.arange(s.D)[None, :]; M, lev, _ = s.message(full[idx])
        first = np.arange(n) % Xl.shape[1] == 0; M[first] = -np.log(S); lev[first] = 0
        s.s.bx = Xl.ravel().astype(np.int32); s.s.by = np.concatenate([Xl[:, 1:], np.full((len(Xl), 1), -1)], 1).ravel().astype(np.int32); s.s.flush(); s.tail = full[-s.D:]; return M, lev
    def at(s, lines, pos):
        """For scoring: before the line stand line ends only, as in cells2.py."""
        buf = np.concatenate([np.full((len(lines), s.D), END, np.int64), lines], 1); ctx = buf[:, s.D + pos - 1 - np.arange(s.D)]; return s.message(ctx)

def score(s, rl, op, pairs):
    """As sparse.score. With the relay system: at each answer byte its message is computed from the bytes that stand before (own: the whole system's own bytes).
    relay_alone: the relay system's own answers, whole answer right, the line's bytes before each (cells2.py's score)."""
    Ls = np.array([text(op, a, b) for a, b in pairs], np.int32); res = {}
    for own in (1, 0):
        buf = Ls.copy(); ok = np.zeros((len(Ls), 3), bool); ra = np.zeros((len(Ls), 3), bool)
        for c in range(3):
            if rl is not None: M, lev, a = rl.at(buf, 6 + c); ra[:, c] = a.argmax(1) == Ls[:, 6 + c]; out = s.answer_at(buf, 6 + c, M, lev)
            else: out = s.answer_at(buf, 6 + c)
            ok[:, c] = out == Ls[:, 6 + c]
            if own: buf[:, 6 + c] = out
        if own: res.update(whole=float(100 * ok.all(1).mean()), place=(100 * ok.mean(0)).round(1).tolist())
        else:
            res.update(forced=float(100 * ok.all(1).mean()))
            if rl is not None: res.update(relay_alone=float(100 * ra.all(1).mean()), relay_alone_place=(100 * ra.mean(0)).round(1).tolist())
    return res

def run(kw):
    t0 = time.process_time(); seed = int(kw.get("seed", 1)); rng, held, shown = sp.split(seed); every = int(kw.get("every", 50000)); block = int(kw.get("block", 50)); use = int(kw.get("relay", 1))
    s = Sys3(Tx0=float(kw.get("Tx0", 0)), D=int(kw.get("D", 9)), m=int(kw.get("m", 3)), eta=float(kw.get("eta", 0.008)), e=float(kw.get("e", 0.1)), T0=float(kw.get("T0", 0.02)), shrink=int(kw.get("shrink", 0)))
    rl = Relay(seed) if use else None; TAB = {op: np.array([text(op, a, b) for a, b in shown[op]], np.int32) for op in "+-"}; out = dict(label=kw.get("label"), args=kw, kinds=len(s.kinds), rows=[]); first = {5000, 10000, 20000, 40000}
    for pi, ph in enumerate(kw.get("plan", "add:400000").split(",")):
        name, n = ph.split(":"); n = int(n); op0 = {"add": "+", "sub": "-"}[name]; it = 0
        while it < n:
            nxt = min([n] + [j for j in list(first) + [(it // every + 1) * every] if j > it]); Xl = TAB[op0][rng.integers(8000, size=nxt - it)]
            for b in range(0, len(Xl), block):
                B = Xl[b:b + block]
                if rl is not None: M, lev = rl.block(B); s.read(B, M, lev)
                else: s.read(B)
            it = nxt; a = score(s, rl, "+", held["+"]); d = score(s, rl, "-", held["-"]); row = dict(phase=pi, name=name, lines=it, add=a["whole"], sub=d["whole"], add_place=a["place"], sub_place=d["place"], add_forced=a["forced"], sub_forced=d["forced"])
            if rl is not None: row.update(relay_add=a["relay_alone"], relay_sub=d["relay_alone"], relay_add_place=a["relay_alone_place"], relay_trust=np.round(s.T[-1, :10], 3).tolist())
            out["rows"].append(row)
    out.update(s.size()); out["cpu"] = round(time.process_time() - t0, 1); out["top_trust_add"] = {}
    for ph in sorted({r["phase"] for r in out["rows"]}):
        rr = [r for r in out["rows"] if r["phase"] == ph]; out[f"last3_add_phase{ph}"] = float(np.mean([r["add"] for r in rr[-3:]])); out[f"last3_sub_phase{ph}"] = float(np.mean([r["sub"] for r in rr[-3:]]))
        if rl is not None: out[f"last3_relay_add_phase{ph}"] = float(np.mean([r["relay_add"] for r in rr[-3:]]))
    os.makedirs(os.path.join(HERE, "res"), exist_ok=True); json.dump(out, open(os.path.join(HERE, "res", f"s_{kw['label']}.json"), "w")); return out

def show(o):
    print(o["label"], f"kinds {o['kinds']} cells {o['cells']} cpu {o['cpu']}s")
    for r in o["rows"]:
        print(f"  {r['phase']}:{r['name']} {r['lines']:>7}  add {r['add']:5.1f} (h,t,u {r['add_place']}; forced {r['add_forced']:5.1f})  sub {r['sub']:5.1f}" + (f"  relay alone: add {r['relay_add']:5.1f} {r['relay_add_place']} sub {r['relay_sub']:5.1f}  trust of the relay by confidence {r['relay_trust']}" if "relay_add" in r else ""), flush=True)

def check():
    g = np.random.default_rng(0); rng, held, shown = sp.split(1); X = np.array([text("+-"[i % 2], *shown["+-"[i % 2]][i]) for i in range(120)], np.int32); n = X.size
    M = g.normal(0, 1, (n, S)); M = (M - np.log(np.exp(M).sum(1, keepdims=True))).astype(np.float32); lev = g.integers(0, 10, n).astype(np.int32)
    a = Sys3(Tx0=0.1, m=3, eta=0.008, e=0.1, T0=0.02); Pa = a.read(X, M, lev, True, True); b = sp.Ref(m=3, eta=0.008, e=0.1, T0=0.02); b.T = np.vstack([b.T, np.full((1, NL), 0.1)]); b.Tu = np.zeros_like(b.T); Pb = ref_read(b, X, M, lev)
    d = max(np.abs(Pa - Pb).max(), np.abs(a.T - b.T).max()); print(f"with one given sender, {n} bytes, 129 kinds: compiled against numpy, largest difference {d:.1e}", "PASSED" if d < 1e-9 else "FAILED")
    a = Sys3(m=3, eta=0.008, e=0.1, T0=0.02); c = sp.Sys(m=3, eta=0.008, e=0.1, T0=0.02); d0 = np.abs(a.read(X, None, None, True, True) - c.read(X, True, True)).max(); Ls = np.array([text("+", *p) for p in held["+"][:60]], np.int32)
    same = all((a.answer_at(Ls, 6 + i) == c.answers(Ls, 0)[:, i]).all() for i in range(3)); print(f"with no sender against sparse.py's kernel: largest difference {d0:.1e}; answers the same: {same}", "PASSED" if d0 == 0 and same else "FAILED")
    import cells2                                                       # the relay wrapper against cells2.py's own run and score
    o = cells2.run(dict(seed="1", plan="add:20000", every="10000", fast="3", nowrite="1", label="x")); rl = Relay(1); rng, held, shown = sp.split(1); TAB = np.array([text("+", a, b) for a, b in shown["+"]], np.int32); got = []
    for part in range(2):
        Xl = TAB[rng.integers(8000, size=10000)]
        for bl in range(0, len(Xl), 50): rl.block(Xl[bl:bl + 50])
        Ls = np.array([text("+", *p) for p in held["+"]], np.int32); got.append(float(100 * np.all([rl.at(Ls, 6 + c)[2].argmax(1) == Ls[:, 6 + c] for c in range(3)], 0).mean()))
    want = [r["add"] for r in o["rows"]]; print(f"the relay system through this program's blocks against cells2.py's own run (never-shown additions at 10,000 and 20,000 lines): {got} against {want}", "PASSED" if np.allclose(got, want) else "FAILED")

if __name__ == "__main__":
    if sys.argv[1] == "check": check()
    else:
        kw = dict(a.split("=") for a in sys.argv[2:]); kw["label"] = sys.argv[1]; show(run(kw))
