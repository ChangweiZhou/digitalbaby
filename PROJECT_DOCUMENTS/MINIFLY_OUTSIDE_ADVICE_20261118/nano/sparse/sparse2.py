#!/usr/bin/env python3
"""PLAN_3.md: the system of sparse.py with a second layer of counting cells that are keyed on what a first cell names.
    python3 sparse2.py <label> seed=1 [plan=add:40000] [m=3] [mz=3] [eta=...] [e=...] [T0=...] [T02=...] [ref=1] [every=5000]
    python3 sparse2.py check
First layer: as sparse.py (a kind is a set of at most m delays; a cell for each combination of byte values that has occurred; it counts the bytes
that follow). Second layer: for each first-layer kind X and each kind Z of at most mz delays, a cell for each combination of byte values at Z's
delays that has occurred while the active cell of kind X was undecided (two, three or four different bytes have followed it). The second-layer
cell does not count bytes. It counts which of the X cell's followers arrived: the one the X cell has counted most, the second, the third, the
fourth (ties: the smaller byte value first), or none of them. Before a byte it says, for each byte value, the log of its smoothed share for
that byte's rank in the X cell (the share of 'none of them' divided among the byte values that have not followed the X cell). The answering
cell adds these to the first layer's with a trust for each pair of kinds and level of experience, learned by the same rule.
The text, split and score are sparse.py's (imported); the kernel is a copy of sparse.py's with the second layer added."""
import ctypes, json, os, subprocess, sys, time
import numpy as np
import sparse as sp
from sparse import S, END, NL, HP, text, kinds_of, level_np
HERE = os.path.dirname(os.path.abspath(__file__)); R = 4

SRC = r'''
#include <math.h>
#include <stdint.h>
#define NL 40
#define MAXS 64
#define MAXK 512
#define MAXP 70000
#define HP 16
#define R 4
typedef struct { int64_t S, K, nplace, END, ncell, cap; const int32_t *nd, *del; const int64_t *off; int32_t *map; float *C, *N, *U; double *T, *Tu; double eta, e; int64_t flat, shrink; double ncap;
                 int64_t KZ, ZTOT, ncell2, cap2; int32_t *map2; float *C2, *N2, *U2; double *T2, *Tu2; } sys_t;
static int64_t ids[MAXK], keyv[MAXK]; static int tix[MAXK]; static double sv[MAXK][MAXS]; static double lt[4096]; static double lte = -1;
static int8_t rk[MAXK][MAXS]; static int n2, m2; static struct { int kx, t; double sr[R + 1]; } act2[MAXP]; static struct { int kx; int64_t id; } all2[MAXP];
static inline int level(float n, float u, int flat) {
    if (flat) return 0;
    int bn = 0; int64_t m = (int64_t)n; while (m > 1 && bn < 9) { m >>= 1; bn++; }
    return bn * 4 + (u <= 1 ? 0 : u <= 2 ? 1 : u <= 4 ? 2 : 3);
}
static inline double lg(float c, double e) { return c < 4096 ? lt[(int)c] : log(c + e); }
static void look(sys_t *q, const int32_t *h, int make, double *p) {
    const int64_t S = q->S; double a[MAXS], mx = -1e300, z = 0; int pl = 0;
    if (lte != q->e) { for (int c = 0; c < 4096; c++) lt[c] = log(c + q->e); lte = q->e; }
    if (q->nplace > 1) { while (pl < q->nplace - 1 && h[-1 - pl] != q->END) pl++; }
    for (int v = 0; v < S; v++) a[v] = 0;
    for (int64_t k = 0; k < q->K; k++) {
        int64_t key = 0; for (int j = 0; j < q->nd[k]; j++) key = key * S + h[-q->del[k * 4 + j]];
        keyv[k] = key; int64_t id = q->map[q->off[k] + key];
        if (id < 0 && make) { id = q->ncell++; q->map[q->off[k] + key] = (int32_t)id; }
        ids[k] = id; tix[k] = -1; if (id < 0 || q->N[id] < 1) continue;
        const float *c = q->C + id * S; tix[k] = (int)((k * q->nplace + pl) * NL + level(q->N[id], q->U[id], (int)q->flat));
        const double tk = q->T[tix[k]], den = log(q->N[id] + S * q->e);
        for (int v = 0; v < S; v++) { const double s = lg(c[v], q->e) - den; sv[k][v] = s; a[v] += tk * s; }
    }
    n2 = 0; m2 = 0;
    if (q->KZ) for (int64_t kx = 0; kx < q->K; kx++) {
        const int64_t id = ids[kx]; if (id < 0 || q->N[id] < 1 || q->U[id] < 2 || q->U[id] > R) continue;
        const float *c = q->C + id * S; int8_t *r = rk[kx]; int f[R], nf = 0;
        for (int v = 0; v < S; v++) { r[v] = R; if (c[v] > 0) f[nf++] = v; }
        for (int i = 1; i < nf; i++) { int x = f[i], j = i; while (j > 0 && c[f[j - 1]] < c[x]) { f[j] = f[j - 1]; j--; } f[j] = x; }
        for (int j = 0; j < nf; j++) r[f[j]] = (int8_t)j;
        const double lognf = log((double)(S - nf));
        for (int64_t kz = 0; kz < q->KZ; kz++) {
            const int64_t slot = kx * q->ZTOT + q->off[kz] + keyv[kz]; int64_t id2 = q->map2[slot];
            if (id2 < 0 && make) { id2 = q->ncell2++; q->map2[slot] = (int32_t)id2; }
            all2[m2].kx = (int)kx; all2[m2].id = id2; m2++;
            if (id2 < 0 || q->N2[id2] < 1) continue;
            const float *c2 = q->C2 + id2 * (R + 1); const int t = (int)((kx * q->KZ + kz) * NL + level(q->N2[id2], q->U2[id2], 0));
            const double tk = q->T2[t], den = log(q->N2[id2] + (R + 1) * q->e); double *sr = act2[n2].sr;
            for (int j = 0; j < R; j++) sr[j] = lg(c2[j], q->e) - den;
            sr[R] = lg(c2[R], q->e) - den - lognf;
            for (int v = 0; v < S; v++) a[v] += tk * sr[r[v]];
            act2[n2].kx = (int)kx; act2[n2].t = t; n2++;
        }
    }
    for (int v = 0; v < S; v++) if (a[v] > mx) mx = a[v];
    for (int v = 0; v < S; v++) { p[v] = exp(a[v] - mx); z += p[v]; }
    for (int v = 0; v < S; v++) p[v] /= z;
}
int64_t steps(sys_t *q, int64_t n, const int32_t *x, int64_t learn_c, int64_t learn_t, double *P) {
    const int64_t S = q->S; double p[MAXS];
    for (int64_t t = 0; t < n; t++) {
        if (learn_c && (q->ncell + q->K > q->cap || q->ncell2 + q->K * q->KZ > q->cap2)) return t;
        look(q, x + t, (int)learn_c, p); const int y = x[t];
        if (P) for (int v = 0; v < S; v++) P[t * S + v] = p[v];
        if (learn_t) {
            for (int64_t k = 0; k < q->K; k++) if (tix[k] >= 0) {
                double g = sv[k][y]; for (int v = 0; v < S; v++) g -= p[v] * sv[k][v];
                q->T[tix[k]] += q->eta * g; q->Tu[tix[k]] += 1.0;
            }
            for (int i = 0; i < n2; i++) {
                const int8_t *r = rk[act2[i].kx]; const double *sr = act2[i].sr; double g = sr[r[y]]; for (int v = 0; v < S; v++) g -= p[v] * sr[r[v]];
                q->T2[act2[i].t] += q->eta * g; q->Tu2[act2[i].t] += 1.0;
            }
        }
        if (learn_c) {
            for (int i = 0; i < m2; i++) { float *c2 = q->C2 + all2[i].id * (R + 1); const int j = rk[all2[i].kx][y]; if (c2[j] == 0) q->U2[all2[i].id] += 1; c2[j] += 1; q->N2[all2[i].id] += 1; }
            for (int64_t k = 0; k < q->K; k++) { float *c = q->C + ids[k] * S; if (c[y] == 0) q->U[ids[k]] += 1; c[y] += 1; q->N[ids[k]] += 1; }
        }
    }
    return n;
}
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
/* the second-layer cells active before the byte at h, for inspection: pair index and trust index, up to cap of them; returns how many */
int64_t active2(sys_t *q, const int32_t *h, int64_t cap, int32_t *t_out, double *sr_out) {
    double p[MAXS]; look(q, h, 0, p); int64_t n = n2 < cap ? n2 : cap;
    for (int64_t i = 0; i < n; i++) { t_out[i] = act2[i].t; for (int j = 0; j <= R; j++) sr_out[i * (R + 1) + j] = act2[i].sr[j]; }
    return n;
}
'''
class CSys2(ctypes.Structure):
    _fields_ = sp.CSys._fields_ + [(n, ctypes.c_int64) for n in ("KZ", "ZTOT", "ncell2", "cap2")] + [(n, ctypes.c_void_p) for n in ("map2", "C2", "N2", "U2", "T2", "Tu2")]

def kern():
    so = os.path.join(HERE, "kern_s2.so"); src = os.path.join(HERE, "kern_s2.c")
    if not os.path.exists(src) or open(src).read() != SRC: open(src, "w").write(SRC)
    if not os.path.exists(so) or os.path.getmtime(so) < os.path.getmtime(src):
        tmp = so + f".{os.getpid()}"; subprocess.check_call(["gcc", "-O3", "-march=native", "-shared", "-fPIC", "-o", tmp, src, "-lm"]); os.replace(tmp, so)
    L = ctypes.CDLL(so); L.steps.restype = ctypes.c_int64; L.steps.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
    L.answers.restype = None; L.answers.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p] + [ctypes.c_int64] * 4 + [ctypes.c_void_p, ctypes.c_void_p]
    L.active2.restype = ctypes.c_int64; L.active2.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p]; return L

class Sys2(sp.Sys):
    def __init__(s, D=9, m=3, mz=3, eta=0.002, e=0.02, T0=0.25, T02=None, cap2=1 << 18, **kw):
        sp.Sys.__init__(s, D=D, m=m, eta=eta, e=e, T0=T0, **kw); s.L = kern(); s.KZ = len(kinds_of(D, mz)) if mz else 0; assert mz <= m and s.K * max(s.KZ, 1) < 70000 and s.K <= 512
        s.ZTOT = int(sum(s.S ** len(k) for k in s.kinds[:s.KZ])); s.map2 = np.full(max(1, s.K * s.ZTOT), -1, np.int32); s.cap2 = cap2; s.C2 = np.zeros((cap2, R + 1), np.float32); s.N2 = np.zeros(cap2, np.float32); s.U2 = np.zeros(cap2, np.float32)
        s.T2 = np.full((max(1, s.K * s.KZ), NL), float(T0 if T02 is None else T02)); s.Tu2 = np.zeros_like(s.T2); s.ncell2 = 0
    def cs(s):
        return CSys2(s.S, s.K, s.nplace, END, s.ncell, s.cap, s.nd.ctypes.data, s.dl.ctypes.data, s.off.ctypes.data, s.map.ctypes.data, s.C.ctypes.data, s.N.ctypes.data, s.U.ctypes.data, s.T.ctypes.data, s.Tu.ctypes.data, s.eta, s.e, 0, 0, 0.0,
                     s.KZ, s.ZTOT, s.ncell2, s.cap2, s.map2.ctypes.data, s.C2.ctypes.data, s.N2.ctypes.data, s.U2.ctypes.data, s.T2.ctypes.data, s.Tu2.ctypes.data)
    def grow2(s):
        s.cap2 *= 2; C = np.zeros((s.cap2, R + 1), np.float32); C[:len(s.C2)] = s.C2; s.C2 = C
        for nm in ("N2", "U2"): a = np.zeros(s.cap2, np.float32); a[:len(getattr(s, nm))] = getattr(s, nm); setattr(s, nm, a)
    def read(s, x, learn=True, want_p=False):
        x = np.asarray(x, np.int32).ravel(); buf = np.ascontiguousarray(np.concatenate([s.tail, x])); n = len(x); P = np.zeros((n, s.S)) if want_p else None; done = 0
        while done < n:
            q = s.cs(); done += s.L.steps(ctypes.byref(q), n - done, buf.ctypes.data + 4 * (HP + done), int(learn), int(learn), P.ctypes.data + 8 * s.S * done if want_p else None); s.ncell = q.ncell; s.ncell2 = q.ncell2
            if done < n:
                if s.ncell + s.K > s.cap: s.grow()
                if s.ncell2 + s.K * s.KZ > s.cap2: s.grow2()
        if learn: s.tail = buf[-HP:].copy()
        return P
    def size(s):
        d = sp.Sys.size(s); d.update(cells2=int((s.N2[:s.ncell2] > 0).sum()), counts2_above_zero=int((s.C2[:s.ncell2] > 0).sum()), trusts2_used=int((s.Tu2 > 0).sum()), trusts2=int(s.T2.size)); return d
    def active2(s, hist):
        """hist: the bytes before the byte to be named (at least HP of them). Returns the trust indices (pair of kinds, level) and what each active second-layer cell says by rank."""
        buf = np.ascontiguousarray(np.asarray(hist, np.int32)); t = np.zeros(70000, np.int32); sr = np.zeros((70000, R + 1)); q = s.cs(); n = s.L.active2(ctypes.byref(q), buf.ctypes.data + 4 * len(buf), 70000, t.ctypes.data, sr.ctypes.data); return t[:n], sr[:n]

class Ref2(sp.Ref):
    def __init__(s, D=9, m=3, mz=3, eta=0.002, e=0.02, T0=0.25, T02=None, **kw):
        sp.Ref.__init__(s, D=D, m=m, eta=eta, e=e, T0=T0, **kw); s.KZ = len(kinds_of(D, mz)) if mz else 0; s.cell2 = {}; s.C2 = []; s.N2 = []; s.U2 = []
        s.T2 = np.full((max(1, s.K * s.KZ), NL), float(T0 if T02 is None else T02)); s.Tu2 = np.zeros_like(s.T2)
    def look(s, hist, make):
        a = np.zeros(s.S); act = []; ids = []; keys = []
        for k, dl in enumerate(s.kinds):
            key = (k,) + tuple(hist[-d] for d in dl); keys.append(key); i = s.cell.get(key, -1)
            if i < 0 and make: i = len(s.C); s.cell[key] = i; s.C.append(np.zeros(s.S, np.float32)); s.N.append(0.0); s.U.append(0.0)
            ids.append(i)
            if i < 0 or s.N[i] < 1: continue
            ti = (k, level_np(s.N[i], s.U[i])); sv = np.log(s.C[i].astype(np.float64) + s.e) - np.log(s.N[i] + s.S * s.e); act.append((k, ti, sv)); a += s.T[ti] * sv
        act2 = []; all2 = []
        for kx in range(s.K if s.KZ else 0):
            i = ids[kx]
            if i < 0 or s.N[i] < 1 or s.U[i] < 2 or s.U[i] > R: continue
            c = s.C[i]; f = sorted([v for v in range(s.S) if c[v] > 0], key=lambda v: (-c[v], v)); r = np.full(s.S, R); r[f] = np.arange(len(f)); lognf = np.log(float(s.S - len(f)))
            for kz in range(s.KZ):
                key = (kx,) + keys[kz]; j = s.cell2.get(key, -1)
                if j < 0 and make: j = len(s.C2); s.cell2[key] = j; s.C2.append(np.zeros(R + 1, np.float32)); s.N2.append(0.0); s.U2.append(0.0)
                all2.append((j, r))
                if j < 0 or s.N2[j] < 1: continue
                ti = (kx * s.KZ + kz, level_np(s.N2[j], s.U2[j])); sr = np.log(s.C2[j].astype(np.float64) + s.e) - np.log(s.N2[j] + (R + 1) * s.e); sr[R] -= lognf; sv = sr[r]; act2.append((ti, sv)); a += s.T2[ti] * sv
        p = np.exp(a - a.max()); p /= p.sum(); s.act2 = act2; s.all2 = all2; return p, act, ids
    def read(s, x, learn=True, want_p=False):
        P = []; hist = list(s.hist)
        for y in np.asarray(x).ravel().tolist():
            p, act, ids = s.look(hist, learn); P.append(p)
            if learn:
                for k, ti, sv in act: s.T[ti] += s.eta * (sv[y] - p @ sv); s.Tu[ti] += 1.0
                for ti, sv in s.act2: s.T2[ti] += s.eta * (sv[y] - p @ sv); s.Tu2[ti] += 1.0
                for j, r in s.all2:
                    if s.C2[j][r[y]] == 0: s.U2[j] += 1
                    s.C2[j][r[y]] += 1; s.N2[j] += 1
                for i in ids:
                    if s.C[i][y] == 0: s.U[i] += 1
                    s.C[i][y] += 1; s.N[i] += 1
            hist.append(y); hist = hist[-HP:]
        if learn: s.hist = hist
        return np.array(P) if want_p else None
    def size(s):
        d = sp.Ref.size(s); d.update(cells2=int(sum(n > 0 for n in s.N2)), counts2_above_zero=int(sum((c > 0).sum() for c in s.C2)), trusts2_used=int((s.Tu2 > 0).sum()), trusts2=int(s.T2.size)); return d

def make(kw, cls=None):
    cls = cls or (Ref2 if int(kw.get("ref", 0)) else Sys2)
    return cls(D=int(kw.get("D", 9)), m=int(kw.get("m", 3)), mz=int(kw.get("mz", 3)), eta=float(kw.get("eta", 0.002)), e=float(kw.get("e", 0.02)), T0=float(kw.get("T0", 0.25)), T02=float(kw["T02"]) if "T02" in kw else None)

def run(kw):
    t0 = time.process_time(); s = make(kw); kw2 = dict(kw, nowrite=1); o = sp.run(kw2, s); o["args"] = kw; o.update(s.size()); o["cpu"] = round(time.process_time() - t0, 1)
    if isinstance(s, Sys2) and s.KZ:                                   # which pairs of kinds have the most trust where the tens digit of never-shown additions is named
        rng, held, shown = sp.split(int(kw.get("seed", 1))); rev = int(kw.get("rev", 0)); o["top_trust2_add"] = {}
        for c, nm in enumerate(("first_answer_byte", "second_answer_byte", "third_answer_byte")):
            tot = {}
            for a, b in held["+"][:200]:
                L = text("+", a, b, rev); t, sr = s.active2([END] * HP + L[:6 + c])
                for ti in t.tolist(): pr = ti // NL; tot.setdefault(pr, [0.0, 0]); tot[pr][0] += s.T2.ravel()[ti]; tot[pr][1] += 1
            top = sorted(tot.items(), key=lambda kv: -kv[1][0] / kv[1][1])[:8]
            o["top_trust2_add"][nm] = [dict(first_cell_delays=list(map(int, s.kinds[pr // s.KZ])), second_cell_delays=list(map(int, s.kinds[pr % s.KZ])), mean_trust=round(v[0] / v[1], 3), active_on=v[1]) for pr, v in top]
    os.makedirs(os.path.join(HERE, "res"), exist_ok=True); json.dump(o, open(os.path.join(HERE, "res", f"s_{kw['label']}.json"), "w")); return o

def check():
    for kw in (dict(m=2, mz=2, D=7), dict(m=3, mz=1, D=6), dict(m=2, mz=0)):
        rng, held, shown = sp.split(1); X = np.array([text("+-"[i % 2], *shown["+-"[i % 2]][i % 40]) for i in range(300)], np.int32)
        a = make(kw, Sys2); b = make(kw, Ref2); Pa = a.read(X, True, True); Pb = b.read(X, True, True)
        Ls = np.array([text("+", *p) for p in held["+"][:40]] + [text("-", *p) for p in shown["-"][:40]], np.int32); same = all((a.answers(Ls, own) == b.answers(Ls, own)).all() for own in (0, 1))
        d = max(np.abs(Pa - Pb).max(), np.abs(a.T - b.T).max(), np.abs(a.T2 - b.T2).max()); ok = d < 1e-9 and same and a.size() == b.size()
        if kw.get("mz") == 0:                                           # with no second layer the kernel is sparse.py's
            c = sp.make(kw, sp.Sys); Pc = c.read(X, True, True); d0 = np.abs(Pa - Pc).max(); ok = ok and d0 < 1e-12; print(f"   against sparse.py's kernel: largest difference {d0:.1e}")
        print(f"{kw}: {X.size} bytes, {a.K} first-layer kinds, {a.K * a.KZ} pairs of kinds, {a.size()['cells']} first-layer and {a.size()['cells2']} second-layer cells that have counted, {a.size()['trusts2_used']} second-layer trusts used: compiled against numpy, largest difference {d:.1e}; answers written the same: {same}; sizes the same: {a.size() == b.size()}", "PASSED" if ok else "FAILED")

if __name__ == "__main__":
    if sys.argv[1] == "check": check()
    else:
        kw = dict(a.split("=") for a in sys.argv[2:]); kw["label"] = sys.argv[1]; o = run(kw); sp.show(o); print("  second layer:", {k: o[k] for k in ("cells2", "counts2_above_zero", "trusts2_used")})
