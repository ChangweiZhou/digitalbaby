#!/usr/bin/env python3
"""nano/lives, PLAN_14.md: the relay network of PLAN_12.md with one more local change, in the life of PLAN_13.md (three returns).
    python3 relay14.py check
    python3 relay14.py life seed=1 form=Rgtqr|Rgtqrs [kappa=10] [theta=4] [lines=400000] tag=..
The letter s: every answering strength (and every branch's constant) keeps the sum of its own squared drivers, G, and its step is divided by
the root of (1 + G / kappa). A strength that has been changed much changes little; one that has never been used changes as before. The return
strength of the same connection takes the same change, as it always did. A driver is the branch's mismatch times the sending cell's activity
(for a constant, the mismatch). With s off the kernel does the arithmetic of relay12.py's (checked). The life and the marks are relay13.py's."""
import copy, ctypes, hashlib, json, os, subprocess, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "loop"))
import aba as AB, loop as LP, relay12 as R12
I64, F64, PD, PI, P32 = LP.I64, LP.F64, LP.PD, LP.PI, LP.P32; rep = R12.rep

SRC = R12.SRC
SRC = rep(SRC, "int64_t gate, hold, rect, holdb; } PNet;", "int64_t gate, hold, rect, holdb, stiff; double kappa; double *G2, *Gb; } PNet;      /* PLAN_14: a sum of squared drivers for every answering strength and constant */")
SRC = rep(SRC, """            for (int64_t j = 0; j < H; j++) { const double q = f * h[j] / cs[j]; r[j] += mv * t[j] * cs[j]; w[j] += q; t[j] += q; }
            if (!n->holdb) n->b2[v] += g * mv;""", """            if (n->stiff) {                                 /* a branch with no mismatch changes nothing and sends nothing back */
                if (mv != 0.0) {
                    double *restrict G = n->G2 + v * H;
                    for (int64_t j = 0; j < H; j++) { const double dr = mv * h[j]; G[j] += dr * dr; const double q = f * h[j] / cs[j] / sqrt(1.0 + G[j] / n->kappa); r[j] += mv * t[j] * cs[j]; w[j] += q; t[j] += q; }
                    n->Gb[v] += mv * mv; if (!n->holdb) n->b2[v] += g * mv / sqrt(1.0 + n->Gb[v] / n->kappa);
                }
            } else {
            for (int64_t j = 0; j < H; j++) { const double q = f * h[j] / cs[j]; r[j] += mv * t[j] * cs[j]; w[j] += q; t[j] += q; }
            if (!n->holdb) n->b2[v] += g * mv;
            }""")

class Net(ctypes.Structure):
    _fields_ = [(k, I64) for k in "H NL S D soft right".split()] + [(k, F64) for k in "eta1 eta2 lam loss".split()] + [(k, PD) for k in "W1t b1 W2 b2 Rt".split()] + [("hist", PI)] + [(k, I64) for k in "gate hold rect holdb stiff".split()] + [("kappa", F64), ("G2", PD), ("Gb", PD)]
_lib = None
def lib():
    global _lib
    if _lib is None:
        h = hashlib.md5(SRC.encode()).hexdigest()[:8]; so = os.path.join(HERE, f"kern_relay14_{h}.so"); src = os.path.join(HERE, "kern_relay14.c")
        if not os.path.exists(so):
            open(src, "w").write(SRC); tmp = f"{so}.{os.getpid()}.tmp"; subprocess.check_call(["gcc", *LP.FLAGS, "-o", tmp, src, "-lm"]); os.replace(tmp, so)
        _lib = ctypes.CDLL(so); _lib.p_steps.restype = I64; _lib.p_steps.argtypes = [ctypes.POINTER(Net), I64, P32, P32]
    return _lib

class Sys(R12.Sys):
    def __init__(s, seed, H, D, S, pad, stiff=0, kappa=10.0, **kw):
        super().__init__(seed, H, D, S, pad, **kw); s.stiff, s.kappa = stiff, kappa; s.G2 = np.zeros((S, H)); s.Gb = np.zeros(S)
    def step(s, x, y):
        """The numpy reference (least-squares arm), with every switch of relay12.py and the stiffening."""
        assert not s.soft; s.hist[1:] = s.hist[:-1].copy(); s.hist[0] = x; act = s.OFF + s.hist
        h = LP.sg(s.W1t[act].sum(0) + s.b1); a = s.W2 @ h + s.b2; s.right += int(a.argmax() == y)
        q = np.maximum(a, 1e-3); s.loss -= np.log(q[y] / q.sum()); m = -(np.maximum(a, 0) if s.rect else a)
        m[y] += 1; g = s.eta2 / (h @ h + 1); d = s.eta1 * (m @ s.Rt) * h * (1 - h); dr = np.outer(m, h)
        if s.stiff: s.G2 += dr * dr; s.Gb += m * m; d2 = g * dr / np.sqrt(1 + s.G2 / s.kappa); db = g * m / np.sqrt(1 + s.Gb / s.kappa)
        else: d2 = g * dr; db = g * m
        s.Rt += d2; s.W2 += d2; s.W1t[act] += d
        if not s.holdb: s.b2 += db
        if not s.hold: s.b1 += d
        f = (1 - s.lam * h)[None, :] if s.gate else 1 - s.lam; s.W2 *= f; s.Rt *= f
    def read(s, xs, ys, fast=True):
        if not fast:
            for x, y in zip(xs.tolist(), ys.tolist()): s.step(x, y)
            return
        xs = np.ascontiguousarray(xs, np.int32); ys = np.ascontiguousarray(ys, np.int32); n = Net(); n.H, n.NL, n.S, n.D, n.soft, n.right = s.H, s.NL, s.S, s.D, s.soft, 0; n.eta1, n.eta2, n.lam, n.loss = s.eta1, s.eta2, s.lam, 0.0
        n.gate, n.hold, n.rect, n.holdb, n.stiff, n.kappa = s.gate, s.hold, s.rect, s.holdb, s.stiff, s.kappa
        for k in ("W1t", "b1", "W2", "b2", "Rt", "G2", "Gb"): a = getattr(s, k); assert a.dtype == np.float64 and a.flags.c_contiguous; setattr(n, k, a.ctypes.data_as(PD))
        n.hist = s.hist.ctypes.data_as(PI); lib().p_steps(ctypes.byref(n), len(xs), xs.ctypes.data_as(P32), ys.ctypes.data_as(P32)); s.loss += n.loss; s.right += n.right

def switches(form):
    return dict(R12.switches(form.replace("s", "")), stiff=int("s" in form))

def run(kw):
    seed = int(kw.get("seed", 1)); form = kw["form"]; lines = int(kw.get("lines", 400000)); theta = float(kw.get("theta", 4.0)); kappa = float(kw.get("kappa", 10.0)); tag = kw["tag"]; D = 10; t0 = time.time()
    x, sp = AB.build(seed, lines); eng = AB.LV.raw("shakespeare")[0]; held_eng = eng[400000:420000]; vals = np.unique(x); idx = np.zeros(256, np.int64); idx[vals] = np.arange(len(vals)); S = len(vals)
    rg = np.random.default_rng([seed, 99]); o = rg.permutation(10000); pairs = [(int(q) // 100, int(q) % 100) for q in o]; TAB = np.array([AB.text(a, b) for a, b in pairs[2000:]], np.int64)
    B2 = eng[420000:760000]; left_out = int((~np.isin(B2, vals)).sum()); B2 = B2[np.isin(B2, vals)]; B3 = eng[760000:1100000]; assert np.isin(B3, vals).all() and np.isin(held_eng, vals).all()
    more = [B2, TAB[np.random.default_rng([seed, 56]).integers(8000, size=5000)].ravel(), B3, TAB[np.random.default_rng([seed, 57]).integers(8000, size=5000)].ravel()]
    head = x[:sp["A2"][1]]; xm = idx[np.concatenate([head] + more)].astype(np.int32); cuts = np.cumsum([len(head)] + [len(m) for m in more])            # the stream of relay13.py
    spans = {"sums": sp["A"], "English 1": sp["B"], "return 1": sp["A2"], "English 2": (int(cuts[0]), int(cuts[1])), "return 2": (int(cuts[1]), int(cuts[2])), "English 3": (int(cuts[2]), int(cuts[3])), "return 3": (int(cuts[3]), int(cuts[4]))}
    s = Sys(seed, 256, D, S, int(idx[10]), eta1=1.0, eta2=1 / 16, lam=1e-5, rscale=1.0, soft=0, theta=theta, kappa=kappa, **switches(form)); ev = idx[x[sp["ev"]:]].reshape(5, 500, 20); rows = []; digits = np.array([int(idx[v]) for v in range(48, 58)]); he = idx[held_eng].astype(np.int64)
    HS = []; SH = []; Lsh = idx[TAB[:500]]
    for d in range(5):
        prev = ev[d, :, :10]; Ls = ev[d, :, 10:]
        for t in (5, 6, 7):
            for L, out in ((Ls, HS), (Lsh, SH)): hist = np.zeros((500, D), np.int64); hist[:, :t + 1] = L[:, t::-1]; hist[:, t + 1:] = prev[:, ::-1][:, :D - t - 1]; out.append((hist, L[:, t + 1]))
    HE = np.stack([he[D - 1 - j: len(he) - 1 - j] for j in range(D)], 1); YE = he[D:]; HSall = np.concatenate([h for h, _ in HS]); YS = np.concatenate([y for _, y in HS]); others = np.setdiff1d(np.arange(S), digits)
    def whole(sy, among=None, states=None):
        ok = np.ones((5, 500), bool)
        for i, (hist, y) in enumerate(states or HS):
            a = sy.branches(hist)
            if among is not None: b = np.full_like(a, -np.inf); b[:, among] = a[:, among]; a = b
            ok[i // 3] &= a.argmax(1) == y
        return float(100 * ok.mean())
    def english(sy): return float(100 * (sy.branches(HE).argmax(1) == YE).mean())
    def read(sy, lo, hi): sy.read(xm[lo - 1:hi - 1], xm[lo:hi])
    def mark(label, sy=None): sy = sy or s; r = dict(at=label, whole=whole(sy), among_digits=whole(sy, digits), english=english(sy), shown_whole=whole(sy, None, SH)); rows.append(r); print(tag, label, {k: round(v, 1) for k, v in r.items() if k != "at"}, flush=True)
    def voices(sy):
        a = sy.branches(HSall); right = a[np.arange(len(YS)), YS]; o_ = a[:, others]; top = o_.argmax(1); loud = o_[np.arange(len(YS)), top]
        return dict(right_digit_output=float(right.mean()), loudest_other_branch_output=float(loud.mean()), share_of_states_where_another_branch_is_loudest=float((loud > a[:, digits].max(1)).mean()))
    def steps(sy):
        """How far the answering strengths have stiffened: the divisor of the step, at the median over the strengths of the digit branches and of the other branches that have been used at all."""
        if not sy.stiff: return None
        dv = np.sqrt(1 + sy.G2 / sy.kappa); dg = dv[digits]; ot = dv[others]
        return dict(divisor_digit_strengths_median=float(np.median(dg)), divisor_other_strengths_used_median=float(np.median(ot[ot > 1])) if (ot > 1).any() else 1.0, share_of_other_strengths_never_used=float((ot == 1).mean()))
    read(s, *spans["sums"]); mark("end of the sums"); notes = dict(end_of_the_sums=dict(voices=voices(s), steps=steps(s))); right = {}
    for k in (1, 2, 3):
        kept = dict(W2=s.W2.copy(), b2=s.b2.copy()); lo, hi = spans[f"English {k}"]; r0 = s.right; read(s, lo, hi); right[f"English {k}"] = 100 * (s.right - r0) / (hi - lo); mark(f"end of English {k}")
        c = copy.deepcopy(s); c.W2[...] = kept["W2"]; c.b2[...] = kept["b2"]; mark(f"end of English {k}, the answering strengths of before it put back", c); notes[f"English {k}"] = dict(voices=voices(s), steps=steps(s), digit_strengths_change_over_size_before=float(np.linalg.norm(s.W2[digits] - kept["W2"][digits]) / np.linalg.norm(kept["W2"][digits])))
        lo, hi = spans[f"return {k}"]; done = lo
        for n in (10, 20, 50, 100, 200, 500, 1000, 2000, 5000): read(s, done, lo + n * 10); done = lo + n * 10; mark(f"return {k}, {n} lines")
    out = dict(task="relay14", tag=tag, seed=seed, form=form, theta=theta, kappa=kappa, lines=lines, values=S, bytes_left_out_of_english_2=left_out, rows=rows, notes=notes, english_bytes_named_right_while_reading=right, wall=round(time.time() - t0, 1)); json.dump(out, open(os.path.join(HERE, "res", f"relay14__{tag}.json"), "w")); return out

def check():
    bad = 0; x, sp = AB.build(1, 5000, back=500, english=3000); vals = np.unique(x); idx = np.zeros(256, np.int64); idx[vals] = np.arange(len(vals)); xm = idx[x].astype(np.int32); S = len(vals)
    lo = sp["B"][0] - 1500; n = 3000; xs = xm[lo:lo + n]; ys = xm[lo + 1:lo + n + 1]
    for form in ("Rs", "Rgtqrs", "Rgtqrbs", "Rrs", "Rgtqr", "R"):
        for kw in (dict(H=64, lam=1e-5, kappa=10.0), dict(H=48, lam=0.02, kappa=0.5)):
            c = dict(kw); H = c.pop("H"); A = Sys(3, H, 10, S, int(idx[10]), eta1=1.0, eta2=1 / 16, rscale=1.0, soft=0, theta=3.0, **c, **switches(form)); B = copy.deepcopy(A)
            A.read(xs, ys, fast=False); B.read(xs[:n // 2], ys[:n // 2]); B.read(xs[n // 2:], ys[n // 2:])
            w = max(np.abs(getattr(A, k) - getattr(B, k)).max() / (np.abs(getattr(A, k)).max() + 1e-300) for k in ("W1t", "b1", "W2", "b2", "Rt", "G2", "Gb") if np.abs(getattr(A, k)).max() > 0); used = float(A.G2.max())
            ok = w < 1e-6 and A.right == B.right and abs(A.loss - B.loss) / n < 1e-6 and (used > 0) == ("s" in form); bad += not ok
            print(f"form {form:8s} H={H} lam={kw['lam']:g} kappa={kw['kappa']:g}: compiled against the numpy reference, largest difference over largest value {w:.1e}; named right {A.right} and {B.right}; largest sum of squared drivers {used:.3g} {'PASSED' if ok else 'FAILED'}", flush=True)
    for form in ("R", "Rgtqr", "Rgtqrb"):
        A = Sys(3, 64, 10, S, int(idx[10]), eta1=1.0, eta2=1 / 16, lam=1e-5, rscale=1.0, soft=0, theta=3.0, **switches(form)); B = R12.Sys(3, 64, 10, S, int(idx[10]), eta1=1.0, eta2=1 / 16, lam=1e-5, rscale=1.0, soft=0, theta=3.0, **R12.switches(form)); A.read(xs, ys); B.read(xs, ys)
        w = max(np.abs(getattr(A, k) - getattr(B, k)).max() / (np.abs(getattr(B, k)).max() + 1e-300) for k in ("W1t", "b1", "W2", "b2", "Rt")); ok = w < 1e-9 and A.right == B.right; bad += not ok
        print(f"form {form:8s} with the stiffening off against relay12.py's kernel (compiled separately): largest difference over largest strength {w:.1e}; named right {A.right} and {B.right} {'PASSED' if ok else 'FAILED'}")
    # a strength that was never driven is as free as at birth; one that was driven much moves less for the same driver
    A = Sys(3, 8, 10, S, int(idx[10]), eta1=0.0, eta2=1 / 16, lam=0.0, rscale=1.0, soft=0, stiff=1, kappa=10.0, hold=1); A.G2[0, :] = 990.0; g0 = A.G2.copy(); w0 = A.W2.copy(); A.read(xs[:1], ys[:1]); y = int(ys[0]); d = A.W2 - w0
    ok = bool(np.all(A.G2 >= g0)) and (y == 0 or abs(d[y]).max() > 0); bad += not ok
    print(f"after one byte: sums of squared drivers never fall, and the branch of the byte that arrived has moved: {ok} {'PASSED' if ok else 'FAILED'}")
    print("CHECK", "FAILED" if bad else "PASSED"); return not bad

if __name__ == "__main__":
    os.makedirs(os.path.join(HERE, "res"), exist_ok=True)
    if sys.argv[1] == "check": sys.exit(0 if check() else 1)
    kw = dict(a.split("=", 1) for a in sys.argv[2:]); out = run(kw); g = {r["at"]: r for r in out["rows"]}
    print(out["tag"], "| never-shown sums, whole answer (among digit branches; held-out English): " + " | ".join(f"{k}: {g[k]['whole']:.1f} ({g[k]['among_digits']:.1f}; {g[k]['english']:.1f})" for k in ["end of the sums"] + [f"{a}{b}" for n in (1, 2, 3) for a, b in ((f"end of English {n}", ""), (f"return {n}, ", "100 lines"), (f"return {n}, ", "5000 lines"))]), "| wall", out["wall"], flush=True)
