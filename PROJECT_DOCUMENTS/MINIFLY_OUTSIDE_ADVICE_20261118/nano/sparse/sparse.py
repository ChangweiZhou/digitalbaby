#!/usr/bin/env python3
"""PLAN_1.md: cells that count what follows them, each keyed on the byte values at a few chosen delays; an answering cell that learns a trust
for each kind and level of experience of cell. The text, the split and the symbols are those of loop_small_case/cells2.py (text() and the
split in run() are copied from it); the level of experience and the trust rule are those of nano/table/trust.py (copied, with the number
of byte values made a parameter).
    python3 sparse.py <label> seed=1 [plan=add:40000[,sub:20000]] [D=9] [m=3] [eta=0.002] [e=0.02] [T0=0.25] [every=5000] [place=0] [rev=0] [shrink=0] [ref=1]
    python3 sparse.py check
A kind of cell is a set of at most m delays among 1..D (delay d: the byte d places before the byte to be named). A cell of a kind exists for
each combination of byte values that has occurred at those delays; it counts the bytes that follow. Before each byte, every cell whose
combination is present and which has counted says s(v) = log((c(v) + e) / (n + S e)); the answering cell's branches compete on the sum of
trust times s; when the byte arrives the trust that was used changes by eta times (s(arrived) - sum p s), and the cells count.
place=P (PLAN_3): the trust also depends on how many bytes have passed since the last line end (0..P-1). rev=1: the answer is written units first. shrink=1 (PLAN_5): a trust's step is eta over the square root of one plus its uses.
ref=1: the numpy reference in place of the compiled kernel (slow)."""
import ctypes, itertools, json, os, subprocess, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
SYMS = list(b"0123456789+=\n-"); S = len(SYMS); SI = {c: i for i, c in enumerate(SYMS)}; END = SI[10]; NL = 40; HP = 16; LINE = 10

def text(op, a, b, rev=0):
    if op == "-" and a < b: a, b = b, a
    ans = f"{(a + b if op == '+' else a - b):03d}"
    return [SI[c] for c in f"{a:02d}{op}{b:02d}={ans[::-1] if rev else ans}\n".encode()]

def kinds_of(D, m): return [c for j in range(1, m + 1) for c in itertools.combinations(range(1, D + 1), j)]

SRC = r'''
#include <math.h>
#include <stdint.h>
#define NL 40
#define MAXS 64
#define MAXK 2048
#define HP 16
typedef struct { int64_t S, K, nplace, END, ncell, cap; const int32_t *nd, *del; const int64_t *off; int32_t *map; float *C, *N, *U; double *T, *Tu; double eta, e; int64_t flat, shrink; } sys_t;
static int64_t ids[MAXK]; static int tix[MAXK]; static double sv[MAXK][MAXS]; static double lt[4096]; static double lte = -1;
static inline int level(float n, float u, int flat) {
    if (flat) return 0;
    int bn = 0; int64_t m = (int64_t)n; while (m > 1 && bn < 9) { m >>= 1; bn++; }
    return bn * 4 + (u <= 1 ? 0 : u <= 2 ? 1 : u <= 4 ? 2 : 3);
}
/* h[-d] is the byte d places back. The active cells that have counted speak; p receives the answering cell's probabilities. */
static void look(sys_t *q, const int32_t *h, int make, double *p) {
    const int64_t S = q->S; double a[MAXS], mx = -1e300, z = 0; int pl = 0;
    if (lte != q->e) { for (int c = 0; c < 4096; c++) lt[c] = log(c + q->e); lte = q->e; }
    if (q->nplace > 1) { while (pl < q->nplace - 1 && h[-1 - pl] != q->END) pl++; }
    for (int v = 0; v < S; v++) a[v] = 0;
    for (int64_t k = 0; k < q->K; k++) {
        int64_t key = 0; for (int j = 0; j < q->nd[k]; j++) key = key * S + h[-q->del[k * 4 + j]];
        int64_t id = q->map[q->off[k] + key];
        if (id < 0 && make) { id = q->ncell++; q->map[q->off[k] + key] = (int32_t)id; }
        ids[k] = id; tix[k] = -1; if (id < 0 || q->N[id] < 1) continue;
        const float *c = q->C + id * S; tix[k] = (int)((k * q->nplace + pl) * NL + level(q->N[id], q->U[id], (int)q->flat));
        const double tk = q->T[tix[k]], den = log(q->N[id] + S * q->e);
        for (int v = 0; v < S; v++) { const double s = (c[v] < 4096 ? lt[(int)c[v]] : log(c[v] + q->e)) - den; sv[k][v] = s; a[v] += tk * s; }
    }
    for (int v = 0; v < S; v++) if (a[v] > mx) mx = a[v];
    for (int v = 0; v < S; v++) { p[v] = exp(a[v] - mx); z += p[v]; }
    for (int v = 0; v < S; v++) p[v] /= z;
}
/* x[t] is named from the bytes before it (HP bytes before x[0] are valid). Returns how many bytes were done (fewer than n if cells ran out). */
int64_t steps(sys_t *q, int64_t n, const int32_t *x, int64_t learn_c, int64_t learn_t, double *P) {
    const int64_t S = q->S; double p[MAXS];
    for (int64_t t = 0; t < n; t++) {
        if (learn_c && q->ncell + q->K > q->cap) return t;
        look(q, x + t, (int)learn_c, p); const int y = x[t];
        if (P) for (int v = 0; v < S; v++) P[t * S + v] = p[v];
        if (learn_t) for (int64_t k = 0; k < q->K; k++) if (tix[k] >= 0) {
            double g = sv[k][y]; for (int v = 0; v < S; v++) g -= p[v] * sv[k][v];
            q->T[tix[k]] += (q->shrink ? q->eta / sqrt(1.0 + q->Tu[tix[k]]) : q->eta) * g; q->Tu[tix[k]] += 1.0;
        }
        if (learn_c) for (int64_t k = 0; k < q->K; k++) { float *c = q->C + ids[k] * S; if (c[y] == 0) q->U[ids[k]] += 1; c[y] += 1; q->N[ids[k]] += 1; }
    }
    return n;
}
/* For each line (L bytes; before it only line ends): the bytes at places from..from+cnt-1 are named; own=1: the system's own byte is written in
   place of the line's before the next is named. acc (cnt x K x 2, may be null) receives the sum of the trust used and the number of times active. */
void answers(sys_t *q, int64_t n, const int32_t *lines, int64_t L, int64_t from, int64_t cnt, int64_t own, int32_t *out, double *acc) {
    int32_t buf[HP + 64]; double p[MAXS];
    for (int64_t i = 0; i < n; i++) {
        for (int j = 0; j < HP; j++) buf[j] = (int32_t)q->END;
        for (int j = 0; j < L; j++) buf[HP + j] = lines[i * L + j];
        for (int64_t c = 0; c < cnt; c++) {
            look(q, buf + HP + from + c, 0, p); int b = 0; for (int v = 1; v < q->S; v++) if (p[v] > p[b]) b = v;
            out[i * cnt + c] = b; if (own) buf[HP + from + c] = b;
            if (acc) for (int64_t k = 0; k < q->K; k++) if (tix[k] >= 0) { acc[(c * q->K + k) * 2] += q->T[tix[k]]; acc[(c * q->K + k) * 2 + 1] += 1; }
        }
    }
}
'''
class CSys(ctypes.Structure):
    _fields_ = [(n, ctypes.c_int64) for n in ("S", "K", "nplace", "END", "ncell", "cap")] + [("nd", ctypes.c_void_p), ("del_", ctypes.c_void_p), ("off", ctypes.c_void_p), ("map", ctypes.c_void_p),
                ("C", ctypes.c_void_p), ("N", ctypes.c_void_p), ("U", ctypes.c_void_p), ("T", ctypes.c_void_p), ("Tu", ctypes.c_void_p), ("eta", ctypes.c_double), ("e", ctypes.c_double), ("flat", ctypes.c_int64), ("shrink", ctypes.c_int64)]

def kern():
    so = os.path.join(HERE, "kern_s.so"); src = os.path.join(HERE, "kern_s.c")
    if not os.path.exists(src) or open(src).read() != SRC: open(src, "w").write(SRC)
    if not os.path.exists(so) or os.path.getmtime(so) < os.path.getmtime(src):
        tmp = so + f".{os.getpid()}"; subprocess.check_call(["gcc", "-O3", "-march=native", "-shared", "-fPIC", "-o", tmp, src, "-lm"]); os.replace(tmp, so)
    L = ctypes.CDLL(so); L.steps.restype = ctypes.c_int64; L.steps.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
    L.answers.restype = None; L.answers.argtypes = [ctypes.c_void_p] + [ctypes.c_int64] * 1 + [ctypes.c_void_p] + [ctypes.c_int64] * 4 + [ctypes.c_void_p, ctypes.c_void_p]; return L

def level_np(n, u):
    bn = 0; m = int(n)
    while m > 1 and bn < 9: m >>= 1; bn += 1
    return bn * 4 + (0 if u <= 1 else 1 if u <= 2 else 2 if u <= 4 else 3)

class Sys:
    """Compiled. The table from context to cell is `map` (one entry for each possible combination of a kind; -1 where no cell has been made)."""
    def __init__(s, D=9, m=3, eta=0.002, e=0.02, T0=0.25, nplace=1, kinds=None, nsym=S, cap=1 << 16, shrink=0):
        s.shrink = int(shrink); s.kinds = kinds if kinds is not None else kinds_of(D, m); s.K = K = len(s.kinds); s.S = nsym; s.nplace = max(1, nplace); s.eta, s.e = eta, e
        s.nd = np.array([len(k) for k in s.kinds], np.int32); s.dl = np.zeros((K, 4), np.int32)
        for i, k in enumerate(s.kinds): s.dl[i, :len(k)] = k
        sz = np.array([nsym ** len(k) for k in s.kinds], np.int64); s.off = np.concatenate([[0], np.cumsum(sz)[:-1]]).astype(np.int64); s.map = np.full(int(sz.sum()), -1, np.int32)
        s.cap = cap; s.C = np.zeros((cap, nsym), np.float32); s.N = np.zeros(cap, np.float32); s.U = np.zeros(cap, np.float32); s.T = np.full((K * s.nplace, NL), float(T0)); s.Tu = np.zeros_like(s.T)
        s.ncell = 0; s.tail = np.full(HP, END, np.int32); s.L = kern()
    def cs(s):
        q = CSys(s.S, s.K, s.nplace, END, s.ncell, s.cap, s.nd.ctypes.data, s.dl.ctypes.data, s.off.ctypes.data, s.map.ctypes.data, s.C.ctypes.data, s.N.ctypes.data, s.U.ctypes.data, s.T.ctypes.data, s.Tu.ctypes.data, s.eta, s.e, 0, s.shrink); return q
    def grow(s):
        s.cap *= 2; C = np.zeros((s.cap, s.S), np.float32); C[:len(s.C)] = s.C; s.C = C
        for nm in "NU": a = np.zeros(s.cap, np.float32); a[:len(getattr(s, nm))] = getattr(s, nm); setattr(s, nm, a)
    def read(s, x, learn=True, want_p=False):
        """The stream goes on with the bytes x: each is named, then (with learn) the trust and the counts change."""
        x = np.asarray(x, np.int32).ravel(); buf = np.ascontiguousarray(np.concatenate([s.tail, x])); n = len(x); P = np.zeros((n, s.S)) if want_p else None; done = 0
        while done < n:
            q = s.cs(); done += s.L.steps(ctypes.byref(q), n - done, buf.ctypes.data + 4 * (HP + done), int(learn), int(learn), P.ctypes.data + 8 * s.S * done if want_p else None); s.ncell = q.ncell
            if done < n: s.grow()
        if learn: s.tail = buf[-HP:].copy()
        return P
    def answers(s, lines, own=1, frm=6, cnt=3, want_acc=False):
        lines = np.ascontiguousarray(lines, np.int32); out = np.zeros((len(lines), cnt), np.int32); acc = np.zeros((cnt, s.K, 2)) if want_acc else None; q = s.cs()
        s.L.answers(ctypes.byref(q), len(lines), lines.ctypes.data, lines.shape[1], frm, cnt, own, out.ctypes.data, acc.ctypes.data if want_acc else None)
        return (out, acc) if want_acc else out
    def size(s): return dict(cells=int((s.N[:s.ncell] > 0).sum()), counts_above_zero=int((s.C[:s.ncell] > 0).sum()), trusts_used=int((s.Tu > 0).sum()), trusts=int(s.T.size))

class Ref:
    """numpy reference: the same arithmetic, a dictionary from (kind, values) to cell, one byte at a time."""
    def __init__(s, D=9, m=3, eta=0.002, e=0.02, T0=0.25, nplace=1, kinds=None, nsym=S, shrink=0):
        s.shrink = int(shrink); s.kinds = kinds if kinds is not None else kinds_of(D, m); s.K = len(s.kinds); s.S = nsym; s.nplace = max(1, nplace); s.eta, s.e = eta, e; s.cell = {}; s.C = []; s.N = []; s.U = []
        s.T = np.full((s.K * s.nplace, NL), float(T0)); s.Tu = np.zeros_like(s.T); s.hist = [END] * HP
    def look(s, hist, make):
        """hist[-d] is the byte d back. Returns p and the active cells as (kind, cell, trust index, s)."""
        pl = 0
        if s.nplace > 1:
            while pl < s.nplace - 1 and hist[-1 - pl] != END: pl += 1
        a = np.zeros(s.S); act = []; ids = []
        for k, dl in enumerate(s.kinds):
            key = (k,) + tuple(hist[-d] for d in dl); i = s.cell.get(key, -1)
            if i < 0 and make: i = len(s.C); s.cell[key] = i; s.C.append(np.zeros(s.S, np.float32)); s.N.append(0.0); s.U.append(0.0)
            ids.append(i)
            if i < 0 or s.N[i] < 1: continue
            ti = (k * s.nplace + pl, level_np(s.N[i], s.U[i])); sv = np.log(s.C[i].astype(np.float64) + s.e) - np.log(s.N[i] + s.S * s.e); act.append((k, ti, sv)); a += s.T[ti] * sv
        p = np.exp(a - a.max()); p /= p.sum(); return p, act, ids
    def read(s, x, learn=True, want_p=False):
        P = []; hist = list(s.hist)
        for y in np.asarray(x).ravel().tolist():
            p, act, ids = s.look(hist, learn); P.append(p)
            if learn:
                for k, ti, sv in act: s.T[ti] += (s.eta / np.sqrt(1.0 + s.Tu[ti]) if s.shrink else s.eta) * (sv[y] - p @ sv); s.Tu[ti] += 1.0
                for i in ids:
                    if s.C[i][y] == 0: s.U[i] += 1
                    s.C[i][y] += 1; s.N[i] += 1
            hist.append(y); hist = hist[-HP:]
        if learn: s.hist = hist
        return np.array(P) if want_p else None
    def answers(s, lines, own=1, frm=6, cnt=3, want_acc=False):
        out = np.zeros((len(lines), cnt), np.int32); acc = np.zeros((cnt, s.K, 2))
        for i, L in enumerate(np.asarray(lines).tolist()):
            buf = [END] * HP + L
            for c in range(cnt):
                p, act, _ = s.look(buf[:HP + frm + c], False); b = int(p.argmax()); out[i, c] = b
                if own: buf[HP + frm + c] = b
                for k, ti, sv in act: acc[c, k, 0] += s.T[ti]; acc[c, k, 1] += 1
        return (out, acc) if want_acc else out
    def size(s): return dict(cells=int(sum(n > 0 for n in s.N)), counts_above_zero=int(sum((c > 0).sum() for c in s.C)), trusts_used=int((s.Tu > 0).sum()), trusts=int(s.T.size))

def split(seed):
    """The split of cells2.py: for each operator 500 pairs scored that are never shown, 8,000 pairs shown; the generator goes on to draw the lines."""
    rng = np.random.default_rng([seed, 99]); held, shown = {}, {}
    for op in "+-":
        o = rng.permutation(10000); P = [(int(q) // 100, int(q) % 100) for q in o]; held[op] = P[:500]; shown[op] = P[2000:]
    return rng, held, shown

def score(s, op, pairs, rev=0, want_acc=False):
    """Never-shown pairs: the system writes its own three answer bytes after '='. whole: all three right (%); place: each answer byte right as written
    (hundreds, tens, units whatever the order of writing); forced: whole answer right when the line's own bytes stand before each (the scoring of cells2.py)."""
    Ls = np.array([text(op, a, b, rev) for a, b in pairs], np.int32); r = s.answers(Ls, 1, 6, 3, want_acc); own = r[0] if want_acc else r; ok = own == Ls[:, 6:9]; fo = s.answers(Ls, 0, 6, 3) == Ls[:, 6:9]
    pl = (100 * ok.mean(0)).round(1).tolist(); pf = (100 * fo.mean(0)).round(1).tolist()
    d = dict(whole=float(100 * ok.all(1).mean()), place=pl[::-1] if rev else pl, forced=float(100 * fo.all(1).mean()), forced_place=pf[::-1] if rev else pf)
    return (d, r[1]) if want_acc else d

def make(kw, cls=None):
    cls = cls or (Ref if int(kw.get("ref", 0)) else Sys)
    drop = {tuple(int(c) for c in d) for d in kw["drop"].split("/")} if kw.get("drop") else set()      # drop=3467/2356/4578: these sets of delays have no cells (PLAN_5, addition 1)
    return cls(kinds=[k for k in kinds_of(int(kw.get("D", 9)), int(kw.get("m", 3))) if k not in drop], D=int(kw.get("D", 9)), m=int(kw.get("m", 3)), eta=float(kw.get("eta", 0.002)), e=float(kw.get("e", 0.02)), T0=float(kw.get("T0", 0.25)), nplace=int(kw.get("place", 0)), shrink=int(kw.get("shrink", 0)))

def run(kw, s=None):
    t0 = time.process_time(); seed = int(kw.get("seed", 1)); rng, held, shown = split(seed); rev = int(kw.get("rev", 0)); every = int(kw.get("every", 5000)); s = s or make(kw)
    TAB = {op: np.array([text(op, a, b, rev) for a, b in shown[op]], np.int32) for op in "+-"}; out = dict(label=kw.get("label"), args=kw, kinds=len(s.kinds), rows=[]); first = {5000, 10000, 20000, 40000} if every > 5000 else set()
    for pi, ph in enumerate(kw.get("plan", "add:40000").split(",")):
        name, n = ph.split(":"); n = int(n); op0 = {"add": "+", "sub": "-", "mix": None}[name]; it = 0
        while it < n:
            nxt = min([n] + [j for j in list(first) + [(it // every + 1) * every] if j > it])
            if op0 is None: ops = rng.integers(2, size=nxt - it); idx = rng.integers(8000, size=nxt - it); X = np.where(ops[:, None] == 0, TAB["+"][idx], TAB["-"][idx])
            else: X = TAB[op0][rng.integers(8000, size=nxt - it)]
            s.read(X); it = nxt; a = score(s, "+", held["+"], rev); b = score(s, "-", held["-"], rev)
            out["rows"].append(dict(phase=pi, name=name, lines=it, add=a["whole"], sub=b["whole"], add_place=a["place"], sub_place=b["place"], add_forced=a["forced"], sub_forced=b["forced"], cells=s.size()["cells"]))
    a, acc = score(s, "+", held["+"], rev, True); top = []
    for c in range(3):
        mean = np.where(acc[c, :, 1] > 0, acc[c, :, 0] / np.maximum(acc[c, :, 1], 1), -np.inf); o = np.argsort(-mean)[:8]
        top.append([dict(delays=list(map(int, s.kinds[k])), mean_trust=round(float(mean[k]), 3), active_on=int(acc[c, k, 1])) for k in o])
    out["top_trust_add"] = dict(zip(("first_answer_byte", "second_answer_byte", "third_answer_byte"), top)); out.update(s.size()); out["cpu"] = round(time.process_time() - t0, 1)
    for ph in sorted({r["phase"] for r in out["rows"]}):
        rr = [r for r in out["rows"] if r["phase"] == ph]; out[f"last3_add_phase{ph}"] = float(np.mean([r["add"] for r in rr[-3:]])); out[f"last3_sub_phase{ph}"] = float(np.mean([r["sub"] for r in rr[-3:]]))
    run.last = s
    if not kw.get("nowrite"): os.makedirs(os.path.join(HERE, "res"), exist_ok=True); json.dump(out, open(os.path.join(HERE, "res", f"s_{kw['label']}.json"), "w"))
    return out

def show(o):
    print(o["label"], f"kinds {o['kinds']} cells {o['cells']} counts>0 {o['counts_above_zero']} trusts used {o['trusts_used']} cpu {o['cpu']}s")
    for r in o["rows"]: print(f"  {r['phase']}:{r['name']} {r['lines']:>7} lines  add {r['add']:5.1f} (h,t,u {r['add_place']}; forced {r['add_forced']:5.1f})  sub {r['sub']:5.1f} (h,t,u {r['sub_place']}; forced {r['sub_forced']:5.1f})  cells {r['cells']}")
    print("  last3", {k: round(v, 1) for k, v in o.items() if k.startswith("last3")}, flush=True)

def check():
    """The compiled kernel against the numpy reference: probabilities, trust, counts, cells, and the answers written."""
    for kw in (dict(m=3), dict(m=4, D=6), dict(m=3, place=10), dict(m=2, rev=1), dict(m=3, D=6, shrink=1, eta=0.1)):
        rng, held, shown = split(1); rev = kw.get("rev", 0); X = np.array([text("+-"[i % 2], *shown["+-"[i % 2]][i], rev) for i in range(120)], np.int32)
        a = make(kw, Sys); b = make(kw, Ref); Pa = a.read(X, True, True); Pb = b.read(X, True, True); n = b.size()["cells"]
        Ls = np.array([text("+", *p, rev) for p in held["+"][:40]] + [text("-", *p, rev) for p in shown["-"][:40]], np.int32); same = all((a.answers(Ls, own) == b.answers(Ls, own)).all() for own in (0, 1))
        Cb = np.zeros((a.ncell, S));
        for key, i in b.cell.items():
            k = key[0]; j = 0
            for v in key[1:]: j = j * S + v
            Cb[a.map[a.off[k] + j]] = b.C[i]
        d = max(np.abs(Pa - Pb).max(), np.abs(a.T - b.T).max(), np.abs(a.C[:a.ncell] - Cb).max()); ok = d < 1e-9 and same and a.size() == b.size() and a.ncell == len(b.C)
        print(f"{kw}: {X.size} bytes, {a.K} kinds, {n} cells that have counted: compiled against numpy, largest difference {d:.1e}; answers written the same: {same}; sizes the same: {a.size() == b.size()}", "PASSED" if ok else "FAILED")

if __name__ == "__main__":
    if sys.argv[1] == "check": check()
    else:
        kw = dict(a.split("=") for a in sys.argv[2:]); kw["label"] = sys.argv[1]; show(run(kw))
