#!/usr/bin/env python3
"""PLAN_2.md of nano/ahead2: the system of ahead.py with the depth of the look-ahead made a number H, on sums of nd-digit numbers.
    python3 ahead3.py check                      the compiled kernel against the plain Python reference; and H=3, nd=2 against ahead.py's figures
    python3 ahead3.py run label seed=1 H=3 [nd=3] [lines=80000] [every=10000]
Layers of cells: layer 0 plain, H sets of counts; layer q (1..H-1) keyed on the guess g_(H-q+1), H-q sets of counts.
The guess g_d (d = H..2) and the answer (d = 1) are formed from the d-th counts of the layers 0..H-d."""
import ctypes, json, math, os, subprocess, sys, time
import numpy as np
from ahead import SYMS, S, SI, END, NL, MAXK, kinds, level
HERE = os.path.dirname(os.path.abspath(__file__)); MAXH = 4

def text(nd, a, b): return [SI[c] for c in f"{a:0{nd}d}+{b:0{nd}d}={a + b:0{nd + 1}d}\n".encode()]

def split(seed, nd):
    """As cells2.py: a permutation of all ordered pairs; the first 500 scored, the first fifth never shown, the rest shown."""
    rng = np.random.default_rng([seed, 99]); n = 10 ** nd; o = rng.permutation(n * n); nh = n * n // 5
    return rng, [(int(q) // n, int(q) % n) for q in o[:500]], o[nh:]

def birth(H, tp=0.02, tg=0.0):
    T = np.zeros((MAXH, MAXH, MAXK, NL)); T[0] = tp; T[1:] = tg; return T

class Ref:
    def __init__(s, D=9, m=4, mg=3, H=3, eta=0.1, e=2.0):
        s.D, s.H, s.eta, s.e = D, H, eta, e; s.ks = [kinds(D, m)] + [kinds(D, mg, True)] * (H - 1); s.T = birth(H); s.U = np.zeros_like(s.T); s.cells = {}; s.h = [END] * (D + 1); s.past = []
    def form(s, h):
        H = s.H; r = dict(c=[None] * H, used=[[] for _ in range(H + 1)], p=[None] * (H + 1), g=[-1] * (H + 1))
        r["c"][0] = [s.cells.setdefault((0, k, tuple(h[d] for d in ds)), np.zeros((H, S))) for k, ds in enumerate(s.ks[0])]
        for d in range(H, 0, -1):
            a = np.zeros(S)
            for q in range(H - d + 1):
                for k, c in enumerate(r["c"][q]):
                    c = c[d - 1]; n = c.sum()
                    if n < 1: continue
                    lv = level(n, (c > 0).sum()); sv = np.array([math.log(v + s.e) for v in c]) - math.log(n + S * s.e); a += s.T[q, d - 1, k, lv] * sv; r["used"][d].append((q, k, lv, sv))
            r["g"][d] = int(a.argmax()); p = np.exp(a - a.max()); r["p"][d] = p / p.sum()
            if d >= 2:
                q = H - d + 1; r["c"][q] = [s.cells.setdefault((q, k, tuple(h[x] for x in ds) + (r["g"][d],)), np.zeros((H - q, S))) for k, ds in enumerate(s.ks[q])]
        return r
    def step(s, y):
        r = s.form(s.h); recs = [r] + s.past[::-1]              # recs[d-1]: formed d-1 bytes ago; its guess d is of the byte arriving now
        for d in range(1, min(s.H, len(recs)) + 1):
            for q, k, lv, sv in recs[d - 1]["used"][d]:
                s.T[q, d - 1, k, lv] += s.eta / np.sqrt(1.0 + s.U[q, d - 1, k, lv]) * (sv[y] - recs[d - 1]["p"][d] @ sv); s.U[q, d - 1, k, lv] += 1
        for d in range(1, min(s.H, len(recs)) + 1):
            for q in range(s.H - d + 1):
                for c in recs[d - 1]["c"][q]: c[d - 1, y] += 1
        s.past = (s.past + [r])[-(s.H - 1):] if s.H > 1 else []; s.h = [0, y] + s.h[1:s.D]; return r
    def sizes(s):
        return sum(1 for c in s.cells.values() if c[0].sum() > 0), int(sum((c > 0).sum() for c in s.cells.values())), int((s.U > 0).sum())

SRC = r'''
#include <math.h>
#include <stdint.h>
#include <string.h>
#define S 14
#define NL 40
#define MAXK 256
#define MAXH 4
typedef struct { int32_t D, H, nk[MAXH]; double eta, e; int32_t *C[MAXH]; double *T, *U; const int32_t *kd[MAXH], *kn[MAXH]; const int64_t *base[MAXH]; } St;
typedef struct { int64_t id[MAXH][MAXK]; int lev[MAXH][MAXH][MAXK]; double s[MAXH][MAXH][MAXK][S]; double p[MAXH + 1][S]; int g[MAXH + 1]; } Rec;
static Rec ring[MAXH], scratch; static int hist[16]; static int64_t tt;
static double lt[8192], ln_[65536], cur_e = -1;
static void tables(double e) { if (e == cur_e) return; cur_e = e; for (int c = 0; c < 8192; c++) lt[c] = log(c + e); for (int n = 0; n < 65536; n++) ln_[n] = log(n + S * e); }
static inline int level(int64_t n, int u) { int bn = 0; while (n > 1 && bn < 9) { n >>= 1; bn++; } return bn * 4 + (u <= 1 ? 0 : u <= 2 ? 1 : u <= 4 ? 2 : 3); }
#define TI(q, d, k, lv) (((((int64_t)(q) * MAXH + (d)) * MAXK + (k)) * NL) + (lv))
static void form(const St *st, const int *h, Rec *r) {
    const int H = st->H; double a[S]; int64_t key[MAXH][MAXK];
    for (int q = 0; q < H; q++) for (int k = 0; k < st->nk[q]; k++) {
        int64_t idx = 0, mul = 1; for (int i = 0; i < st->kn[q][k]; i++) { idx += h[st->kd[q][k * 4 + i]] * mul; mul *= S; }
        key[q][k] = st->base[q][k] + (q ? idx * S : idx); }
    for (int k = 0; k < st->nk[0]; k++) r->id[0][k] = key[0][k];
    for (int d = H; d >= 1; d--) {
        memset(a, 0, sizeof a);
        for (int q = 0; q <= H - d; q++) { const int ns = H - q;
            for (int k = 0; k < st->nk[q]; k++) {
                const int32_t *c = st->C[q] + (r->id[q][k] * ns + (d - 1)) * S; int64_t n = 0; int u = 0;
                for (int v = 0; v < S; v++) { n += c[v]; u += c[v] > 0; }
                if (n < 1) { r->lev[q][d - 1][k] = -1; continue; }
                const int lv = level(n, u); r->lev[q][d - 1][k] = lv; const double tk = st->T[TI(q, d - 1, k, lv)], den = n < 65536 ? ln_[n] : log(n + S * st->e); double *s = r->s[q][d - 1][k];
                for (int v = 0; v < S; v++) { const double sv = (c[v] < 8192 ? lt[c[v]] : log(c[v] + st->e)) - den; s[v] = sv; a[v] += tk * sv; }
            } }
        int g = 0; double z = 0, *p = r->p[d]; for (int v = 1; v < S; v++) if (a[v] > a[g]) g = v;
        for (int v = 0; v < S; v++) { p[v] = exp(a[v] - a[g]); z += p[v]; }
        for (int v = 0; v < S; v++) p[v] /= z;
        r->g[d] = g;
        if (d >= 2) { const int q = H - d + 1; for (int k = 0; k < st->nk[q]; k++) r->id[q][k] = key[q][k] + g; }
    }
}
static void arrive(const St *st, int y, Rec **recs, int nrec) {
    const int H = st->H;
    for (int d = 1; d <= H && d <= nrec; d++) { const Rec *r = recs[d - 1]; const double *p = r->p[d];
        for (int q = 0; q <= H - d; q++) for (int k = 0; k < st->nk[q]; k++) { const int lv = r->lev[q][d - 1][k]; if (lv < 0) continue;
            const double *s = r->s[q][d - 1][k]; double g = s[y]; for (int v = 0; v < S; v++) g -= p[v] * s[v];
            const int64_t i = TI(q, d - 1, k, lv); st->T[i] += st->eta / sqrt(1.0 + st->U[i]) * g; st->U[i] += 1.0; } }
    for (int d = 1; d <= H && d <= nrec; d++) { const Rec *r = recs[d - 1];
        for (int q = 0; q <= H - d; q++) { const int ns = H - q; for (int k = 0; k < st->nk[q]; k++) st->C[q][(r->id[q][k] * ns + (d - 1)) * S + y]++; } }
}
void reset(int end) { tt = 0; for (int d = 0; d < 16; d++) hist[d] = end; }
void run(const St *st, const int32_t *x, int64_t n, double *outp, int32_t *outg) {
    const int H = st->H; tables(st->e);
    for (int64_t i = 0; i < n; i++, tt++) {
        Rec *recs[MAXH]; int nrec = 0; for (int d = 0; d < H && d <= tt; d++) recs[nrec++] = &ring[(tt - d) % H];
        form(st, hist, recs[0]);
        if (outp) { memcpy(outp + i * (MAXH + 1) * S, recs[0]->p, sizeof recs[0]->p); memcpy(outg + i * (MAXH + 1), recs[0]->g, sizeof recs[0]->g); }
        arrive(st, x[i], recs, nrec);
        for (int d = st->D; d > 1; d--) hist[d] = hist[d - 1];
        hist[1] = x[i];
    }
}
void answer(const St *st, const int32_t *h, int32_t *out) {
    int hh[16]; tables(st->e); for (int d = 0; d <= st->D; d++) hh[d] = h[d];
    form(st, hh, &scratch); for (int d = 0; d <= MAXH; d++) out[d] = scratch.g[d];
}
void sizes(const St *st, const int64_t *ncell, int64_t *out) {
    int64_t cells = 0, above = 0;
    for (int q = 0; q < st->H; q++) { const int ns = st->H - q; for (int64_t i = 0; i < ncell[q]; i++) { const int32_t *c = st->C[q] + i * ns * S; int64_t n = 0;
        for (int v = 0; v < S; v++) n += c[v]; cells += n > 0; for (int v = 0; v < ns * S; v++) above += c[v] > 0; } }
    out[0] = cells; out[1] = above;
}
'''
PI32 = ctypes.POINTER(ctypes.c_int32); PI64 = ctypes.POINTER(ctypes.c_int64); PD = ctypes.POINTER(ctypes.c_double)
class St(ctypes.Structure):
    _fields_ = [("D", ctypes.c_int32), ("H", ctypes.c_int32), ("nk", ctypes.c_int32 * MAXH), ("eta", ctypes.c_double), ("e", ctypes.c_double), ("C", PI32 * MAXH), ("T", PD), ("U", PD),
                ("kd", PI32 * MAXH), ("kn", PI32 * MAXH), ("base", PI64 * MAXH)]

def kern():
    so = os.path.join(HERE, "kern_h.so"); src = os.path.join(HERE, "kern_h.c")
    if not os.path.exists(src) or open(src).read() != SRC: open(src, "w").write(SRC)
    if not os.path.exists(so) or os.path.getmtime(so) < os.path.getmtime(src):
        tmp = so + f".{os.getpid()}"; subprocess.check_call(["gcc", "-O3", "-march=native", "-ffp-contract=off", "-shared", "-fPIC", "-o", tmp, src, "-lm"]); os.replace(tmp, so)
    L = ctypes.CDLL(so); L.run.argtypes = [ctypes.POINTER(St), PI32, ctypes.c_int64, PD, PI32]; L.answer.argtypes = [ctypes.POINTER(St), PI32, PI32]; L.sizes.argtypes = [ctypes.POINTER(St), PI64, PI64]; L.reset.argtypes = [ctypes.c_int]
    for f in (L.run, L.answer, L.sizes, L.reset): f.restype = None
    return L

class Fast:
    def __init__(s, D=9, m=4, mg=3, H=3, eta=0.1, e=2.0):
        s.D, s.H = D, H; s.ks = [kinds(D, m)] + [kinds(D, mg, True)] * (H - 1); s.L = kern(); s.T = birth(H); s.U = np.zeros_like(s.T); s.keep = []; s.st = St(); s.st.D = D; s.st.H = H; s.st.eta = eta; s.st.e = e
        s.ncell = np.zeros(MAXH, np.int64); s.C = []
        for q, ks in enumerate(s.ks):
            assert len(ks) <= MAXK; d = np.zeros((len(ks), 4), np.int32); n = np.zeros(len(ks), np.int32); base = np.zeros(len(ks), np.int64); tot = 0
            for k, ds in enumerate(ks): d[k, :len(ds)] = ds; n[k] = len(ds); base[k] = tot; tot += S ** len(ds) * (S if q else 1)
            C = np.zeros(tot * (H - q) * S, np.int32); s.C.append(C); s.ncell[q] = tot; s.keep += [d, n, base]
            s.st.nk[q] = len(ks); s.st.C[q] = C.ctypes.data_as(PI32); s.st.kd[q] = d.ctypes.data_as(PI32); s.st.kn[q] = n.ctypes.data_as(PI32); s.st.base[q] = base.ctypes.data_as(PI64)
        s.st.T = s.T.ctypes.data_as(PD); s.st.U = s.U.ctypes.data_as(PD); s.L.reset(END)
    def run(s, x, want=False):
        x = np.ascontiguousarray(x, np.int32); P = np.zeros((len(x), MAXH + 1, S)) if want else None; G = np.zeros((len(x), MAXH + 1), np.int32) if want else None
        s.L.run(ctypes.byref(s.st), x.ctypes.data_as(PI32), len(x), P.ctypes.data_as(PD) if want else None, G.ctypes.data_as(PI32) if want else None); return P, G
    def answer(s, h):
        hh = np.zeros(16, np.int32); hh[:len(h)] = h; out = np.zeros(MAXH + 1, np.int32); s.L.answer(ctypes.byref(s.st), hh.ctypes.data_as(PI32), out.ctypes.data_as(PI32)); return out.tolist()
    def sizes(s):
        out = np.zeros(2, np.int64); s.L.sizes(ctypes.byref(s.st), s.ncell.ctypes.data_as(PI64), out.ctypes.data_as(PI64)); return int(out[0]), int(out[1]), int((s.U > 0).sum())

def score(sys_, nd, pairs):
    """Own bytes; line ends before the line. Whole %, by place %, and guess[i][d]: at answer byte i, the share of guesses d ahead (d = 2..H) that equal the line's byte there."""
    D, H = sys_.D, sys_.H; na = nd + 1; pre = 2 * nd + 2; whole = 0; place = np.zeros(na); gr = np.zeros((na, MAXH + 1)); n = len(pairs)
    for a, b in pairs:
        L = text(nd, a, b); h = [0] + [END] * D; ok = []
        for c in L[:pre]: h = [0, c] + h[1:D]
        for i in range(na):
            g = sys_.answer(h); ok.append(g[1] == L[pre + i]); h = [0, g[1]] + h[1:D]
            for d in range(2, H + 1):
                if i + d - 1 < na: gr[i, d] += g[d] == L[pre + i + d - 1]
        whole += all(ok); place += ok
    return 100 * whole / n, (100 * place / n).round(1).tolist(), (100 * gr / n).round(1).tolist()

def run(kw):
    t0 = time.process_time(); seed = int(kw.get("seed", 1)); nd = int(kw.get("nd", 3)); H = int(kw.get("H", 3)); lines = int(kw.get("lines", 80000)); every = int(kw.get("every", 10000)); n = 10 ** nd
    rng, held, shown = split(seed, nd); s = Fast(H=H); out = dict(label=kw["label"], args=kw, rows=[]); seen = {}; read = None
    for b in range(lines // every):
        idx = shown[rng.integers(len(shown), size=every)]; first = 0
        for q in idx.tolist(): seen[q] = seen.get(q, 0) + 1
        if read is None: read = [(q // n, q % n) for q in list(dict.fromkeys(idx.tolist()))[:500]]         # 500 different pairs read in the first block
        s.run(np.array([text(nd, int(q) // n, int(q) % n) for q in idx], np.int32).ravel())
        a, ap, ag = score(s, nd, held); r, rp, _ = score(s, nd, read); out["rows"].append(dict(lines=(b + 1) * every, add=a, add_place=ap, add_guess=ag, read=r, read_place=rp))
    cnt = np.array(list(seen.values())); out["first_readings_share"] = float(len(cnt) / lines); out["pairs_read_three_times_or_more"] = int((cnt >= 3).sum())
    c, ab, tu = s.sizes(); out["sizes"] = dict(cells=c, counts_above_zero=ab, trusts_used=tu); out["last3"] = float(np.mean([r["add"] for r in out["rows"][-3:]])); out["last3_read"] = float(np.mean([r["read"] for r in out["rows"][-3:]]))
    out["cpu"] = round(time.process_time() - t0, 1)
    if not kw.get("nowrite"): os.makedirs(os.path.join(HERE, "res"), exist_ok=True); json.dump(out, open(os.path.join(HERE, "res", f"{kw['label']}.json"), "w"))
    return out

def check():
    ok_all = True
    for kw, nb in ((dict(D=5, m=2, mg=1, H=4), 1500), (dict(D=6, m=3, mg=2, H=3), 1500), (dict(D=5, m=2, mg=2, H=1), 1500), (dict(D=6, m=2, mg=1, H=2), 1500), (dict(D=9, m=4, mg=3, H=4), 300)):
        rng, held, shown = split(7, 3); x = np.concatenate([text(3, int(q) // 1000, int(q) % 1000) for q in shown[rng.integers(len(shown), size=nb // 13 + 1)]]).astype(np.int32)[:nb]
        f = Fast(**kw); P, G = f.run(x[:nb // 2], True); P2, G2 = f.run(x[nb // 2:], True); P = np.concatenate([P, P2]); G = np.concatenate([G, G2]); r = Ref(**kw); dp = 0.0; same = True; H = kw["H"]
        for t in range(nb):
            q = r.step(int(x[t]))
            for d in range(1, H + 1): dp = max(dp, float(np.abs(q["p"][d] - P[t, d]).max())); same &= q["g"][d] == G[t, d]
        dT = float(np.abs(r.T - f.T).max()); sz = r.sizes() == f.sizes(); hs = [[0] + [int(v) for v in rng.integers(S, size=kw["D"])] for _ in range(50)]; ans = all(f.answer(h)[1:H + 1] == r.form(h)["g"][1:] for h in hs)
        ok = dp < 1e-9 and dT < 1e-9 and same and sz and ans; ok_all &= bool(ok)
        print(f"{kw}: {nb} bytes, sizes (cells, counts above zero, trusts used) {r.sizes()}: compiled against the reference: shares {dp:.1e}, trusts {dT:.1e}; answers and guesses the same: {bool(same)}; "
              f"sizes the same: {sz}; answers without learning the same: {ans}", "PASSED" if ok else "FAILED", flush=True)
    # H = 3 on the two-digit text must give ahead.py's figures (res/la_73129.json, addition phase), the stream being drawn in the same way
    import ahead
    rng, held, shown = ahead.split(73129); TAB = np.array([ahead.text("+", a, b) for a, b in shown["+"]], np.int32); s = Fast(H=3); got = []
    for b in range(8): s.run(TAB[rng.integers(8000, size=5000)].ravel()); got.append(score(s, 2, held["+"])[0])
    want = [r["add"] for r in json.load(open(os.path.join(HERE, "res", "la_73129.json")))["rows"] if r["phase"] == 0]; ok = got == want; ok_all &= ok
    print("H = 3, two-digit text, seed 73129, eight readings to 40,000 lines:", got, "ahead.py:", want, "PASSED" if ok else "FAILED"); return ok_all

if __name__ == "__main__":
    if sys.argv[1] == "check": sys.exit(0 if check() else 1)
    kw = dict(a.split("=") for a in sys.argv[3:]); kw["label"] = sys.argv[2]; o = run(kw); L = o["rows"][-1]
    print(o["label"], "last3", round(o["last3"], 1), "read", round(o["last3_read"], 1), "|", " ".join(f"{r['lines'] // 1000}k {r['add']:.1f}" for r in o["rows"]), "| place", L["add_place"], "| guess at first byte", L["add_guess"][0][2:],
          "| first readings", round(o["first_readings_share"], 3), o["sizes"], o["cpu"], "s", flush=True)
