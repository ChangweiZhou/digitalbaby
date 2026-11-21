#!/usr/bin/env python3
"""nano/search (PLAN_1.md): cells that count; the KINDS of cells found by a search judged by the learned trust. One program for every text.
    python3 search.py check
    python3 search.py sums  tag=... seed=1 plan=add:40000 [nd=2] [K=128 R=8192 prot=8 nrep=8 eta=.. e=.. ncap=.. step=0|1]
    python3 search.py prose tag=... mode=dev|life [same settings] [kinds=runs6]
The compiled kernel is kern_q.c; the plain Python reference is class Ref below (same store, same order of operations)."""
import ctypes, json, os, subprocess, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); D = 16; MAXL = 6; NLEV = 40; M64 = (1 << 64) - 1
c_p = ctypes.c_void_p

class St(ctypes.Structure):
    _fields_ = [(n, ctypes.c_int64) for n in "S H K NLAY NB stepmode pos evict ins".split()] + [(n, ctypes.c_double) for n in "eta e ncap".split()] + \
               [(n, c_p) for n in "KEY KU NN UU CC dead nl ln lay uid T A Wa Wg rkey rslot rlev rs rp rvalid".split()]

def kern():
    so = os.path.join(HERE, "kern_q.so"); src = os.path.join(HERE, "kern_q.c")
    if not os.path.exists(so) or os.path.getmtime(so) < os.path.getmtime(src): subprocess.check_call(["gcc", "-O3", "-march=native", "-shared", "-fPIC", "-o", so, src, "-lm"])
    L = ctypes.CDLL(so); P = ctypes.POINTER(St)
    L.run.argtypes = [P, c_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int, ctypes.c_int, c_p, c_p]; L.backfill.argtypes = [P, c_p, c_p, ctypes.c_int64, ctypes.c_int64, c_p, ctypes.c_int64]; L.backfill.restype = None; L.run.restype = None
    L.gen.argtypes = [P, c_p, ctypes.c_int64, ctypes.c_int64, c_p]; L.gen.restype = None; L.sizeof_st.restype = ctypes.c_int64
    assert L.sizeof_st() == ctypes.sizeof(St); return L

def line_name(l, H=4): return f"d{l + 1}" if l < D else f"g{(l - D) // H}.{(l - D) % H + 1}"

class Sys:
    """State in numpy arrays (shared with the kernel); the search itself is done here, between calls of the kernel."""
    def __init__(s, S, K=128, seed=1, H=4, NLAY=2, logslots=22, eta=0.1, e=0.5, ncap=16, step=0, R=8192, prot=8, nrep=8, kinds=None, ref=False, maxuid=1 << 20, pmut=0.5, wg=1.0, B=0):
        s.S, s.K, s.H, s.NLAY, s.R, s.prot, s.nrep, s.pmut, s.wg = S, K, H, NLAY, R, prot, nrep, pmut, wg; s.nlines = D + NLAY * H; s.rng = np.random.default_rng([seed, 7])
        ns = 1 << logslots; s.NB = ns // 4; Rr = NLAY + 1
        s.KEY = np.zeros(ns, np.uint64); s.KU = np.zeros(ns, np.uint32); s.NN = np.zeros((ns, H), np.uint32); s.UU = np.zeros((ns, H), np.uint8); s.CC = np.zeros((ns, H, S), np.uint16); s.dead = np.zeros(maxuid, np.uint8)
        s.nl = np.zeros(K, np.int32); s.ln = np.zeros((K, MAXL), np.int32); s.lay = np.zeros(K, np.int32); s.uid = np.zeros(K, np.uint32)
        s.T = np.zeros((Rr, H, K, NLEV)); s.A = np.zeros((Rr, H, K, NLEV)); s.Wa = np.zeros(K); s.Wg = np.zeros(K)
        s.rkey = np.zeros((H + 1, K), np.uint64); s.rslot = np.full((H + 1, K), -1, np.int64); s.rlev = np.full((H + 1, K, H), -1, np.int8); s.rs = np.zeros((H + 1, K, H, S)); s.rp = np.zeros((H + 1, Rr, H, S)); s.rvalid = np.zeros(H + 1, np.int8)
        s.st = St(S=S, H=H, K=K, NLAY=NLAY, NB=s.NB, stepmode=step, pos=0, evict=0, ins=0, eta=eta, e=e, ncap=ncap if ncap else 1e30)
        for n in "KEY KU NN UU CC dead nl ln lay uid T A Wa Wg rkey rslot rlev rs rp rvalid".split(): setattr(s.st, n, getattr(s, n).ctypes.data)
        s.L = None if ref else kern(); s.ref = Ref(s) if ref else None
        s.next_uid = 0; s.age = np.zeros(K, np.int64); s.kinds = [None] * K; s.earn = np.zeros(K); s.earn_g = np.zeros(K); s.tried = 0; s.rounds = 0; s.since = 0; s.cont = 0; s.bytes = 0; s.tail = None; s.last_W = np.zeros(K); s.B = B; s.hx = np.zeros(0, np.int32); s.hg = np.zeros((0, NLAY * H), np.int8)
        for k in range(K): s.set_kind(k, kinds[k] if kinds is not None else s.random_kind())
    # --- kinds
    def random_kind(s):
        while True:
            m = int(s.rng.integers(1, 5)); kd = tuple(sorted(int(v) for v in s.rng.choice(s.nlines, m, replace=False)))
            if kd not in s.kinds: return kd
    def changed_kind(s, w):
        for _ in range(50):
            kd = list(s.kinds[int(s.rng.choice(s.K, p=w))]); op = int(s.rng.integers(3)); free = [l for l in range(s.nlines) if l not in kd]
            if op == 0 and len(kd) < MAXL: kd.append(int(s.rng.choice(free)))
            elif op == 1 and len(kd) > 1: kd.pop(int(s.rng.integers(len(kd))))
            elif op == 2: kd[int(s.rng.integers(len(kd)))] = int(s.rng.choice(free))
            else: continue
            kd = tuple(sorted(kd))
            if kd not in s.kinds: return kd
        return s.random_kind()
    def set_kind(s, k, kd):
        if s.kinds[k] is not None: s.dead[s.uid[k]] = 1
        s.kinds[k] = tuple(kd); s.uid[k] = s.next_uid; s.next_uid += 1; s.tried += 1; s.nl[k] = len(kd); s.ln[k, :len(kd)] = kd; g = [(l - D) // s.H for l in kd if l >= D]; s.lay[k] = 1 + max(g) if g else 0
        s.T[:, :, k] = 0; s.A[:, :, k] = 0; s.Wa[k] = 0; s.Wg[k] = 0; s.age[k] = 0; s.earn[k] = 0; s.earn_g[k] = 0; s.rkey[:, k] = 0; s.rslot[:, k] = -1; s.rlev[:, k] = -1
    def search(s):
        W = s.Wa + s.wg * s.Wg; s.earn += s.Wa; s.earn_g += s.Wg; s.age += 1; s.rounds += 1; s.last_W = W.copy()
        if s.nrep:
            el = [k for k in range(s.K) if s.age[k] > s.prot]; out = sorted(el, key=lambda k: W[k])[:s.nrep]; w = np.maximum(W, 0); w[out] = 0
            for k in out:
                s.set_kind(k, s.changed_kind(w / w.sum()) if w.sum() > 0 and s.rng.random() < s.pmut else s.random_kind())
            if s.B and out and len(s.hg):
                ks = np.array(out, np.int32); hx = np.ascontiguousarray(s.hx); hg = np.ascontiguousarray(s.hg)
                if s.ref: s.ref.backfill(hx, hg, D, len(hx), ks)
                else: s.L.backfill(ctypes.byref(s.st), hx.ctypes.data, hg.ctypes.data, D, len(hx), ks.ctypes.data, len(ks))
        s.Wa[:] = 0; s.Wg[:] = 0
    # --- reading
    def _run(s, buf, lo, hi, learn, cont, P):
        buf = np.ascontiguousarray(buf, np.int32); G = np.zeros((hi - lo, s.NLAY * s.H), np.int8) if learn and s.B else None
        if s.ref: s.ref.run(buf, lo, hi, learn, cont, P, G)
        else: s.L.run(ctypes.byref(s.st), buf.ctypes.data, lo, hi, learn, cont, P.ctypes.data if P is not None else None, G.ctypes.data if G is not None else None)
        if G is not None:                                           # the bytes kept: the last B read, with the D before them and the guesses as they were made
            s.hx = np.concatenate([s.hx if len(s.hx) else buf[lo - D:lo], buf[lo:hi]])[-(s.B + D):]; s.hg = np.concatenate([s.hg, G])[-s.B:]
    def start(s, tail): s.tail = np.asarray(tail, np.int32)[-D:].copy(); s.cont = 0
    def feed(s, xb):
        buf = np.concatenate([s.tail, np.asarray(xb, np.int32)]); n = len(xb); i = 0
        while i < n:
            m = min(n - i, s.R - s.since); s._run(buf, D + i, D + i + m, 1, s.cont, None); s.cont = 1; i += m; s.since += m; s.bytes += m
            if s.since >= s.R: s.search(); s.since = 0
        s.tail = buf[-D:].copy()
    def probs(s, tail, xb):
        buf = np.concatenate([np.asarray(tail, np.int32)[-D:], np.asarray(xb, np.int32)]); P = np.zeros((len(xb), s.S)); s._run(buf, D, len(buf), 0, 0, P); return P
    def write(s, ctx, ngen):
        ctx = np.ascontiguousarray(ctx, np.int32); out = np.zeros((len(ctx), ngen), np.int32)
        if s.ref: s.ref.gen(ctx, ngen, out)
        else: s.L.gen(ctypes.byref(s.st), ctx.ctypes.data, len(ctx), ngen, out.ctypes.data)
        return out
    # --- size
    def size(s):
        cells = 0; above = 0; step = 1 << 18
        for i in range(0, len(s.KEY), step):
            live = (s.KEY[i:i + step] != 0) & (s.dead[s.KU[i:i + step]] == 0); cells += int(live.sum()); above += int((s.CC[i:i + step][live] > 0).sum())
        return dict(cells=cells, counts_above_zero=above, store=len(s.KEY), displaced=int(s.st.evict), kinds_alive=s.K, kinds_tried=s.tried, rounds=s.rounds, trusts_used=int((s.A > 0).sum()))
    def kinds_report(s, top=None):
        tm = np.abs(s.T[s.NLAY, 0]).max(1); rows = [dict(lines=" ".join(line_name(l, s.H) for l in s.kinds[k]), layer=int(s.lay[k]), age=int(s.age[k]), earned_answer=round(float(s.earn[k] + s.Wa[k]), 2), earned_guesses=round(float(s.earn_g[k] + s.Wg[k]), 2),
                                                      last_period=round(float(s.last_W[k]), 3), largest_trust=round(float(tm[k]), 3)) for k in range(s.K)]
        rows.sort(key=lambda r: -r["earned_answer"]); return rows[:top] if top else rows

class Ref:
    """Plain Python, the same store and the same order of operations as kern_q.c, written from PLAN_1.md section 2."""
    def __init__(s, sy): s.y = sy
    @staticmethod
    def key(uid, vals):
        h = ((uid + 1) * 0x9E3779B97F4A7C15) & M64
        for v in vals: h = ((h ^ (v + 1)) * 0x100000001B3) & M64
        h ^= h >> 31; h = (h * 0xBF58476D1CE4E5B9) & M64; h ^= h >> 29; return h or 1
    @staticmethod
    def level(n, u):
        bn = 0; m = int(n)
        while m > 1 and bn < 9: m >>= 1; bn += 1
        return bn * 4 + (0 if u <= 1 else 1 if u <= 2 else 2 if u <= 4 else 3)
    def lookup(s, key):
        y = s.y; b = (key % y.NB) * 4
        for i in range(4):
            if int(y.KEY[b + i]) == key: return b + i
        return -1
    def insert(s, key, uid):
        y = s.y; b = (key % y.NB) * 4; w = -1
        for i in range(4):
            if int(y.KEY[b + i]) == 0 or y.dead[y.KU[b + i]]: w = b + i; break
        if w < 0:
            w = b
            for i in range(1, 4):
                if y.NN[b + i, 0] < y.NN[w, 0]: w = b + i
            y.st.evict += 1
        y.KEY[w] = key; y.KU[w] = uid; y.NN[w] = 0; y.UU[w] = 0; y.CC[w] = 0; y.st.ins += 1; return w
    def fwd(s, hist, a):
        """hist[d-1] = the byte d places back."""
        y = s.y; S, H, K, NLAY = y.S, y.H, y.K, y.NLAY; e, ncap = y.st.e, y.st.ncap; G = np.zeros((NLAY, H), np.int64)
        for l in range(NLAY + 1):
            for k in range(K):
                if y.lay[k] != l: continue
                vals = [int(hist[ln]) if ln < D else int(G[(ln - D) // H, (ln - D) % H]) for ln in y.ln[k, :y.nl[k]]]; key = s.key(int(y.uid[k]), vals); sl = s.lookup(key); y.rkey[a, k] = key; y.rslot[a, k] = sl
                for h in range(H):
                    y.rlev[a, k, h] = -1
                    if sl < 0 or y.NN[sl, h] < 1: continue
                    n = float(y.NN[sl, h]); c = y.CC[sl, h].astype(np.float64); y.rlev[a, k, h] = s.level(n, int(y.UU[sl, h]))
                    if n > ncap: c = c * (ncap / n); n = ncap
                    y.rs[a, k, h] = np.log(c + e) - np.log(n + S * e)
            for h in range(H if l < NLAY else 1):
                br = np.zeros(S)
                for k in range(K):
                    if y.lay[k] <= l and y.rlev[a, k, h] >= 0: br = br + y.T[l, h, k, y.rlev[a, k, h]] * y.rs[a, k, h]
                p = np.exp(br - br.max()); y.rp[a, l, h] = p / p.sum()
                if l < NLAY: G[l, h] = int(np.argmax(br))
        return G
    def learn(s, yb):
        y = s.y; H, K, NLAY = y.H, y.K, y.NLAY; pos = int(y.st.pos); eta = y.st.eta
        for h in range(H):
            if pos - h < 0 or not y.rvalid[(pos - h) % H]: continue
            a = (pos - h) % H
            for r in range(NLAY + 1):
                if r == NLAY and h > 0: continue
                p = y.rp[a, r, h]
                for k in range(K):
                    lv = y.rlev[a, k, h]
                    if y.lay[k] > r or lv < 0: continue
                    sv = y.rs[a, k, h]; g = sv[yb] - float(p @ sv)
                    if r == NLAY: y.Wa[k] += y.T[r, h, k, lv] * g
                    else: y.Wg[k] += y.T[r, h, k, lv] * g
                    if y.st.stepmode == 0: y.T[r, h, k, lv] += eta / np.sqrt(1.0 + y.A[r, h, k, lv]) * g; y.A[r, h, k, lv] += 1.0
                    else:
                        y.A[r, h, k, lv] += g * g
                        if y.A[r, h, k, lv] > 0: y.T[r, h, k, lv] += eta * g / np.sqrt(y.A[r, h, k, lv])
        for h in range(H):
            if pos - h < 0 or not y.rvalid[(pos - h) % H]: continue
            a = (pos - h) % H
            for k in range(K):
                if y.lay[k] == NLAY and h > 0: continue
                key = int(y.rkey[a, k]); sl = int(y.rslot[a, k])
                if not key: continue
                if sl < 0 or int(y.KEY[sl]) != key:
                    sl = s.lookup(key) if h == 0 else -1
                    if sl < 0 and h == 0: sl = s.insert(key, int(y.uid[k]))
                    y.rslot[a, k] = sl
                if sl < 0: continue
                if y.CC[sl, h, yb] == 65535: y.CC[sl, h] >>= 1; y.NN[sl, h] = y.CC[sl, h].sum(); y.UU[sl, h] = (y.CC[sl, h] > 0).sum()
                if y.CC[sl, h, yb] == 0: y.UU[sl, h] += 1
                y.CC[sl, h, yb] += 1; y.NN[sl, h] += 1
    def backfill(s, x, g, lo, hi, ks):
        y = s.y; H = y.H
        for t in range(lo, hi):
            for k in ks:
                vals = [int(x[t - (ln + 1)]) if ln < D else int(g[t - lo, ln - D]) for ln in y.ln[k, :y.nl[k]]]; key = s.key(int(y.uid[k]), vals); sl = s.lookup(key)
                if sl < 0: sl = s.insert(key, int(y.uid[k]))
                for h in range(1 if y.lay[k] == y.NLAY else H):
                    if t + h >= hi: break
                    yb = int(x[t + h])
                    if y.CC[sl, h, yb] == 65535: y.CC[sl, h] >>= 1; y.NN[sl, h] = y.CC[sl, h].sum(); y.UU[sl, h] = (y.CC[sl, h] > 0).sum()
                    if y.CC[sl, h, yb] == 0: y.UU[sl, h] += 1
                    y.CC[sl, h, yb] += 1; y.NN[sl, h] += 1
    def run(s, x, lo, hi, learn, cont, P, Gout=None):
        y = s.y; H = y.H
        if learn and not cont: y.st.pos = 0; y.rvalid[:] = 0
        for t in range(lo, hi):
            a = int(y.st.pos) % H if learn else H; G = s.fwd(x[t - 1::-1][:D] if t > 0 else [], a)
            if Gout is not None: Gout[t - lo] = G.ravel()
            if P is not None: P[t - lo] = y.rp[a, y.NLAY, 0]
            if learn: y.rvalid[a] = 1; s.learn(int(x[t])); y.st.pos += 1
    def gen(s, ctx, ngen, out):
        y = s.y
        for i in range(len(ctx)):
            buf = list(ctx[i])
            for j in range(ngen):
                s.fwd(buf[::-1][:D], y.H); g = int(np.argmax(y.rp[y.H, y.NLAY, 0])); buf.append(g); out[i, j] = g

# ---------------------------------------------------------------- sums (text, split and score of loop_small_case/cells2.py; nd = 3 is this line's)
SYMS = list(b"0123456789+=\n-"); SI = {c: i for i, c in enumerate(SYMS)}; END = SI[10]
def text(op, a, b, nd=2):
    if op == "-" and a < b: a, b = b, a
    return [SI[c] for c in f"{a:0{nd}d}{op}{b:0{nd}d}={(a + b if op == '+' else a - b):0{nd + 1}d}\n".encode()]

class Sums:
    def __init__(s, seed, nd=2):
        s.nd = nd; s.seed = seed; s.rng = np.random.default_rng([seed, 99]); s.held, s.shown, s.TAB = {}, {}, {}; N = 10 ** nd
        for op in "+-":
            if nd == 2: o = s.rng.permutation(N * N); P = [(int(q) // N, int(q) % N) for q in o]; s.held[op] = P[:500]; s.shown[op] = P[2000:]; s.TAB[op] = np.array([text(op, a, b) for a, b in s.shown[op]], np.int32)
            else: q = s.rng.choice(N * N, 2000, replace=False); s.held[op] = [(int(v) // N, int(v) % N) for v in q[:500]]; s.heldset = set(int(v) for v in q)       # every other pair may be shown
        if nd == 2:
            w = set(tuple(text("-", a, b)) for a, b in s.shown["-"]); s.clean = [p for p in s.held["-"] if tuple(text("-", *p)) not in w]
    def block(s, op, n):
        if s.nd == 2: return s.TAB[op][s.rng.integers(8000, size=n)].ravel()
        N = 10 ** s.nd; out = []
        while len(out) < n: out += [int(v) for v in s.rng.integers(N * N, size=n) if int(v) not in s.heldset]
        return np.array([text(op, v // N, v % N, s.nd) for v in out[:n]], np.int32).ravel()
    def score(s, sy, op, pairs, past=None):
        """past=None: the empty past of cells2.py (line ends before the line). past=op2 (PLAN_4.md): before each held-out line stand two lines drawn by chance from the lines of op2 that may be read; mean of five draws."""
        Ls = np.array([text(op, a, b, s.nd) for a, b in pairs]); pre = 2 * s.nd + 2; ctx = np.full((len(pairs), D), END, np.int32); ctx[:, D - pre:] = Ls[:, :pre]
        if past is None: out = sy.write(ctx, s.nd + 1); ok = out == Ls[:, pre:pre + s.nd + 1]; return dict(whole=round(100 * float(ok.all(1).mean()), 2), place=[round(100 * float(v), 1) for v in ok.mean(0)])
        if not hasattr(s, "prng"): s.prng = np.random.default_rng([s.seed, 77]); s.keep = s.rng; N = 10 ** s.nd
        w = []
        for _ in range(5):
            s.rng = s.prng; prev = s.block(past, 2 * len(pairs)).reshape(len(pairs), -1); s.rng = s.keep          # the stream's own generator is not touched
            ctx[:, :D - pre] = prev[:, -(D - pre):]; out = sy.write(ctx, s.nd + 1); w.append(100 * float((out == Ls[:, pre:pre + s.nd + 1]).all(1).mean()))
        return round(float(np.mean(w)), 2)

SETTINGS = ("K", "R", "prot", "nrep", "logslots", "step", "NLAY", "B"); FSET = ("eta", "e", "ncap", "pmut", "wg")
def make(S, kw, kinds=None):
    a = {k: int(kw[k]) for k in SETTINGS if k in kw}; a.update({k: float(kw[k]) for k in FSET if k in kw}); return Sys(S, seed=int(kw.get("seed", 1)), kinds=kinds, **a)

def run_sums(kw):
    t0 = time.process_time(); seed = int(kw.get("seed", 1)); nd = int(kw.get("nd", 2)); sm = Sums(seed, nd); sy = make(len(SYMS), kw); sy.start([END] * D); rows = []; every = int(kw.get("every", 5000))
    for pi, ph in enumerate(kw.get("plan", "add:40000").split(",")):
        name, n = ph.split(":"); op = {"add": "+", "sub": "-"}[name]
        for it in range(0, int(n), every):
            sy.feed(sm.block(op, every)); r = dict(phase=pi, name=name, lines=it + every, add=sm.score(sy, "+", sm.held["+"]))
            if nd == 2 and "sub" in kw.get("plan", ""): r["sub_all"] = sm.score(sy, "-", sm.held["-"])["whole"]; r["sub_clean"] = sm.score(sy, "-", sm.clean)["whole"]; r["n_clean"] = len(sm.clean)
            if int(kw.get("past", 0)):
                r["add_past"] = sm.score(sy, "+", sm.held["+"], past=op)
                if "sub_clean" in r: r["sub_clean_past"] = sm.score(sy, "-", sm.clean, past=op)
            r["tried"] = sy.tried; rows.append(r)
            if int(kw.get("verbose", 0)): print(r, flush=True)
    a = [r["add"]["whole"] for r in rows if r["phase"] == 0]
    ap = [r["add_past"] for r in rows if r["phase"] == 0 and "add_past" in r]
    return dict(task="sums", args=kw, rows=rows, add_last=a[-1], add_last3=round(float(np.mean(a[-3:])), 2), past_last=ap[-1] if ap else None, past_last3=round(float(np.mean(ap[-3:])), 2) if ap else None, size=sy.size(), kinds=sy.kinds_report(), cpu=round(time.process_time() - t0, 1))

# ---------------------------------------------------------------- prose (text, split and life of nano/table/score.py and life.py)
TEXT = os.path.join(HERE, "..", "data", "tinyshakespeare.txt"); BT = ("bryant-stories", "burgess-busterbrown", "carroll-alice")
def book(name):
    d = os.path.join(HERE, "data_local"); f = os.path.join(d, name + ".txt")
    if not os.path.exists(f):
        os.makedirs(d, exist_ok=True); open(f, "wb").write(subprocess.check_output(["git", "show", f"5d00eae:minifly_v148_pretrain_probe_dev_20261025/data/{name}.txt"], cwd=HERE))
    return np.frombuffer(open(f, "rb").read(), np.uint8)
def load_prose(books=True):
    raw = np.frombuffer(open(TEXT, "rb").read(), np.uint8); vals = np.unique(raw); assert len(raw) == 1115394 and len(vals) == 65
    tab = np.full(256, 1, np.int64); tab[vals] = np.arange(65); cut = int(0.9 * len(raw)); parts = [("A", tab[raw], cut)]
    if books:
        for b in BT: r = book(b); r = r[r != 13]; parts.append((b, tab[r], int(0.9 * len(r))))
    x = np.concatenate([p[1] for p in parts]).astype(np.int32); segs = {}; off = 0
    for name, p, c in parts: segs[name] = (off, off + c, off + len(p)); off += len(p)
    return x, segs
def nats(P, y): return float(-np.log(P[np.arange(len(y)), y]).mean()), float(100 * (P.argmax(1) == y).mean())
def hand_kinds(name, K):
    if name.startswith("runs"): n = int(name[4:]); kd = [tuple(range(m)) for m in range(1, n + 1)]
    else: raise ValueError(name)
    return kd

def run_prose(kw):
    t0 = time.process_time(); mode = kw.get("mode", "dev"); x, segs = load_prose(mode == "life"); kinds = None
    if "kinds" in kw: kinds = hand_kinds(kw["kinds"], 0); kw = dict(kw, K=len(kinds), nrep=0)
    sy = make(65, kw, kinds); out = dict(task="prose", args=kw, mode=mode); a, c, b = segs["A"]
    if mode == "dev":
        dev = int(kw.get("devn", c - c // 20)); c = min(c, dev + 50000) if "devn" in kw else c       # devn: read that many bytes of the stream and score the next 50,000 (development only)
        sy.start(x[:D]); sy.feed(x[D:dev]); n, r = nats(sy.probs(x[dev - D:dev], x[dev:c]), x[dev:c]); out.update(dev_nats=round(n, 4), dev_right=round(r, 2))
    else:
        sy.start(x[:D]); sy.feed(x[D:c]); out["size_after_A"] = sy.size(); out["kinds_after_A"] = sy.kinds_report(40); out["books"] = {}
        for name in BT:
            a2, c2, b2 = segs[name]; n0, _ = nats(sy.probs(x[c2 - D:c2], x[c2:b2]), x[c2:b2]); sy.start(x[a2 - D:a2]); sy.feed(x[a2:c2]); n, r = nats(sy.probs(x[c2 - D:c2], x[c2:b2]), x[c2:b2])
            out["books"][name] = dict(nats=round(n, 4), right=round(r, 2), before_reading_it=round(n0, 4), n=int(b2 - c2), tried=sy.tried)
        n, r = nats(sy.probs(x[c - D:c], x[c:b]), x[c:b]); out["A_last_tenth_at_end"] = dict(nats=round(n, 4), right=round(r, 2))
    out.update(size=sy.size(), kinds=sy.kinds_report(), cpu=round(time.process_time() - t0, 1)); return out

# ---------------------------------------------------------------- check
def check():
    ok = True
    for name, step, logslots, K, B in (("shrinking step, store large", 0, 16, 10, 0), ("step by summed squares, store large", 1, 16, 10, 0), ("shrinking step, store of 256 places (cells displaced)", 0, 8, 10, 0), ("shrinking step, new kinds count over the 500 bytes kept", 0, 16, 10, 500), ("summed squares, 300 bytes kept, store of 256 places", 1, 8, 10, 300)):
        res = []
        for ref in (False, True):
            sm = Sums(5); sy = Sys(len(SYMS), K=K, seed=3, logslots=logslots, eta=0.1, e=0.5, ncap=4, step=step, R=400, prot=1, nrep=3, ref=ref, maxuid=4096, B=B)
            if not ref: kinds0 = list(sy.kinds)
            sy.start([END] * D); sy.feed(sm.block("+", 120)); P = sy.probs([END] * D, sm.block("+", 30)); w = sm.score(sy, "+", sm.held["+"][:40]); Ls = np.array([text("+", a, b) for a, b in sm.held["+"][:40]]); ctx = np.full((40, D), END, np.int32); ctx[:, D - 6:] = Ls[:, :6]
            res.append((P, sy.T.copy(), sy.KEY.copy(), sy.CC.copy(), sy.NN.copy(), sy.write(ctx, 3), sy.Wa.copy(), list(sy.kinds), int(sy.st.evict), sy.tried, int((sy.lay > 0).sum())))
        a, b = res; d = max(np.abs(a[0] - b[0]).max(), np.abs(a[1] - b[1]).max(), np.abs(a[6] - b[6]).max()); same = all(np.array_equal(a[i], b[i]) for i in (2, 3, 4, 5)) and a[7] == b[7] and a[8] == b[8]
        good = d < 1e-9 and same; ok &= good
        print(f"{name}: 1,200 bytes read with {a[9]} kinds tried ({a[10]} of the {K} living keyed on a guess at the end), {a[8]} cells displaced; compiled against plain Python: probabilities, trusts, earnings differ by at most {d:.1e}; store, counts, written bytes and kinds equal: {same}", "PASSED" if good else "FAILED", flush=True)
    return ok

if __name__ == "__main__":
    if sys.argv[1] == "check": sys.exit(0 if check() else 1)
    kw = dict(a.split("=") for a in sys.argv[2:]); tag = kw.pop("tag"); r = (run_sums if sys.argv[1] == "sums" else run_prose)(kw); json.dump(r, open(os.path.join(HERE, "res", tag + ".json"), "w"), indent=0)
    if r["task"] == "sums": print(tag, "add last", r["add_last"], "last3", r["add_last3"], "after a read line: last", r["past_last"], "last3", r["past_last3"], "place", r["rows"][-1]["add"]["place"], {k: r["size"][k] for k in ("cells", "counts_above_zero", "displaced", "kinds_tried")}, "cpu", r["cpu"], flush=True)
    else: print(tag, r.get("dev_nats"), {k: v["nats"] for k, v in r.get("books", {}).items()}, r.get("A_last_tenth_at_end"), {k: r["size"][k] for k in ("cells", "counts_above_zero", "displaced", "kinds_tried")}, "cpu", r["cpu"], flush=True)
    print("   top kinds:", "; ".join(f"[{k['lines']}] {k['earned_answer']}" for k in r["kinds"][:12]), flush=True)
