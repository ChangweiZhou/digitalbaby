#!/usr/bin/env python3
"""nano/lives, PLAN_12.md: the relay network of PLAN_11.md with four local changes, each of which can be switched on alone.
    python3 relay12.py check
    python3 relay12.py life seed=1 form=R|Rg|Rt|Rq|Rr|...|Rgtqr [theta=3] [lines=400000] tag=..
The letters after R:
  g  a strength of the answering cell (and its return strength) fades only as its sending cell is active: by lam x the cell's activity at each byte,
     where the network as it is fades every such strength by lam at every byte;
  t  the cells' thresholds are held where they were born (as it is, a threshold moves at every byte);
  q  the cells are born quiet: every threshold is born at -theta (as it is, at nought, where a cell is half active for anything);
  r  a branch of the answering cell whose output is below nought counts as silent: its output is taken as nought in the mismatch, so a branch that is
     silent and whose byte did not arrive is not taught (as it is, every branch is drawn to nought from both sides at every byte);
  b  (PLAN_12.md, addition 1) the branches' own constants are held at nought (as it is, a branch's constant moves whenever the branch is taught,
     whichever cells were active).
With no letter the kernel does the arithmetic of nano/loop/kernp.c (checked). The life, the alphabet, the score and the diagnostics are relay_aba.py's."""
import copy, ctypes, hashlib, json, os, subprocess, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "loop"))
import aba as AB, loop as LP
I64, F64, PD, PI, P32 = LP.I64, LP.F64, LP.PD, LP.PI, LP.P32

def rep(t, a, b):
    assert t.count(a) == 1, (t.count(a), a[:70]); return t.replace(a, b)

SRC = open(os.path.join(HERE, "..", "loop", "kernp.c")).read()
SRC = rep(SRC, "double *W1t, *b1, *W2, *b2, *Rt; int64_t *hist; } PNet;", "double *W1t, *b1, *W2, *b2, *Rt; int64_t *hist; int64_t gate, hold, rect, holdb; } PNet;      /* PLAN_12: four switches */")
SRC = rep(SRC, "double m[256], a[256];", "double m[256], a[256]; double *restrict cs = malloc(H * sizeof(double)); for (int64_t j = 0; j < H; j++) cs[j] = 1.0;      /* gate: the true strengths from cell j are cs[j] times the stored ones */")
SRC = rep(SRC, "double q = 0; for (int64_t j = 0; j < H; j++) q += w[j] * h[j]; a[v] = sc * q + n->b2[v];", "double q = 0; for (int64_t j = 0; j < H; j++) q += w[j] * (cs[j] * h[j]); a[v] = sc * q + n->b2[v];")
SRC = rep(SRC, "for (int64_t v = 0; v < S; v++) m[v] = -a[v];", "for (int64_t v = 0; v < S; v++) m[v] = (n->rect && a[v] < 0) ? 0.0 : -a[v];      /* rect: a branch below nought is silent */")
SRC = rep(SRC, "for (int64_t j = 0; j < H; j++) { const double q = f * h[j]; r[j] += mv * t[j]; w[j] += q; t[j] += q; }", "for (int64_t j = 0; j < H; j++) { const double q = f * h[j] / cs[j]; r[j] += mv * t[j] * cs[j]; w[j] += q; t[j] += q; }")
SRC = rep(SRC, "for (int64_t j = 0; j < H; j++) b1[j] += dv[j];", "if (!n->hold) for (int64_t j = 0; j < H; j++) b1[j] += dv[j];      /* hold: the thresholds stay where they were born */")
SRC = rep(SRC, "n->b2[v] += g * mv;", "if (!n->holdb) n->b2[v] += g * mv;      /* holdb: the branches' constants stay at nought */")
SRC = rep(SRC, "sc *= 1.0 - n->lam;", """if (n->gate) { for (int64_t j = 0; j < H; j++) { cs[j] *= 1.0 - n->lam * h[j]; if (cs[j] < 1e-100) { for (int64_t v = 0; v < S; v++) { W2[v * H + j] *= cs[j]; Rt[v * H + j] *= cs[j]; } cs[j] = 1.0; } } }
        else sc *= 1.0 - n->lam;""")
SRC = rep(SRC, "    n->loss += loss; n->right += right;", "    for (int64_t v = 0; v < S; v++) for (int64_t j = 0; j < H; j++) { W2[v * H + j] *= cs[j]; Rt[v * H + j] *= cs[j]; }\n    free(cs);\n    n->loss += loss; n->right += right;")

class Net(ctypes.Structure):
    _fields_ = [(k, I64) for k in "H NL S D soft right".split()] + [(k, F64) for k in "eta1 eta2 lam loss".split()] + [(k, PD) for k in "W1t b1 W2 b2 Rt".split()] + [("hist", PI)] + [(k, I64) for k in "gate hold rect holdb".split()]
_lib = None
def lib():
    global _lib
    if _lib is None:
        h = hashlib.md5(SRC.encode()).hexdigest()[:8]; so = os.path.join(HERE, f"kern_relay12_{h}.so"); src = os.path.join(HERE, "kern_relay12.c")
        if not os.path.exists(so):
            open(src, "w").write(SRC); tmp = f"{so}.{os.getpid()}.tmp"; subprocess.check_call(["gcc", *LP.FLAGS, "-o", tmp, src, "-lm"]); os.replace(tmp, so)
        _lib = ctypes.CDLL(so); _lib.p_steps.restype = I64; _lib.p_steps.argtypes = [ctypes.POINTER(Net), I64, P32, P32]
    return _lib

class Sys(LP.Sys):
    """loop.Sys with the switches gate, hold, rect, and with thresholds born at -theta when quiet is set."""
    def __init__(s, seed, H, D, S, pad, gate=0, hold=0, rect=0, quiet=0, holdb=0, theta=3.0, **kw):
        super().__init__(seed, H, D, S, pad, **kw); s.gate, s.hold, s.rect, s.quiet, s.holdb, s.theta = gate, hold, rect, quiet, holdb, theta
        if quiet: s.b1[:] = -theta
    def step(s, x, y):
        """The numpy reference with the switches (least-squares arm only)."""
        assert not s.soft; s.hist[1:] = s.hist[:-1].copy(); s.hist[0] = x; act = s.OFF + s.hist
        h = LP.sg(s.W1t[act].sum(0) + s.b1); a = s.W2 @ h + s.b2; s.right += int(a.argmax() == y)
        q = np.maximum(a, 1e-3); s.loss -= np.log(q[y] / q.sum()); m = -(np.maximum(a, 0) if s.rect else a)
        m[y] += 1; g = s.eta2 / (h @ h + 1)
        d = s.eta1 * (m @ s.Rt) * h * (1 - h); d2 = np.outer(g * m, h); s.Rt += d2; s.W2 += d2; s.W1t[act] += d
        if not s.holdb: s.b2 += g * m
        if not s.hold: s.b1 += d
        f = (1 - s.lam * h)[None, :] if s.gate else 1 - s.lam; s.W2 *= f; s.Rt *= f
    def read(s, xs, ys, fast=True):
        if not fast:
            for x, y in zip(xs.tolist(), ys.tolist()): s.step(x, y)
            return
        xs = np.ascontiguousarray(xs, np.int32); ys = np.ascontiguousarray(ys, np.int32); n = Net(); n.H, n.NL, n.S, n.D, n.soft, n.right = s.H, s.NL, s.S, s.D, s.soft, 0; n.eta1, n.eta2, n.lam, n.loss = s.eta1, s.eta2, s.lam, 0.0
        n.gate, n.hold, n.rect, n.holdb = s.gate, s.hold, s.rect, s.holdb
        for k in ("W1t", "b1", "W2", "b2", "Rt"): a = getattr(s, k); assert a.dtype == np.float64 and a.flags.c_contiguous; setattr(n, k, a.ctypes.data_as(PD))
        n.hist = s.hist.ctypes.data_as(PI); lib().p_steps(ctypes.byref(n), len(xs), xs.ctypes.data_as(P32), ys.ctypes.data_as(P32)); s.loss += n.loss; s.right += n.right

LETTERS = dict(g="gate", t="hold", q="quiet", r="rect", b="holdb")
def switches(form):
    assert form[0] == "R" and all(c in LETTERS for c in form[1:]) and len(set(form[1:])) == len(form) - 1, form; return {LETTERS[c]: 1 for c in form[1:]}
FORMS = ["R" + "".join(c for c, on in zip("gtqrb", (g, t, q, r, b)) if on) for b in (0, 1) for g in (0, 1) for t in (0, 1) for q in (0, 1) for r in (0, 1)]      # the sixteen of the plan, then the same with b

def run(kw):
    seed = int(kw.get("seed", 1)); form = kw["form"]; lines = int(kw.get("lines", 400000)); theta = float(kw.get("theta", 3.0)); tag = kw["tag"]; D = 10; t0 = time.time()
    x, sp = AB.build(seed, lines); eng = AB.LV.raw("shakespeare")[0]; held_eng = eng[400000:420000]           # 20,000 bytes of the English that are never read: its score says what the English keeps
    vals = np.unique(x); idx = np.zeros(256, np.int64); idx[vals] = np.arange(len(vals)); xm = idx[x].astype(np.int32); S = len(vals)          # the alphabet of relay_aba.py: the byte values of the life
    known = np.isin(held_eng, vals); he = np.where(known, idx[held_eng], idx[10]).astype(np.int64)                                                  # a held-out byte whose value the life never has is left out of the score, with the ten places after it
    s = Sys(seed, 256, D, S, int(idx[10]), eta1=1.0, eta2=1 / 16, lam=1e-5, rscale=1.0, soft=0, theta=theta, **switches(form)); ev = xm[sp["ev"]:].reshape(5, 500, 20); rows = []; digits = np.array([int(idx[v]) for v in range(48, 58)])
    HS = []
    for d in range(5):
        prev = ev[d, :, :10]; Ls = ev[d, :, 10:]
        for t in (5, 6, 7): hist = np.zeros((500, D), np.int64); hist[:, :t + 1] = Ls[:, t::-1]; hist[:, t + 1:] = prev[:, ::-1][:, :D - t - 1]; HS.append((hist, Ls[:, t + 1]))
    rg = np.random.default_rng([seed, 99]); o = rg.permutation(10000); shown = [(int(q) // 100, int(q) % 100) for q in o][2000:2500]            # addition 2: 500 of the shown pairs (aba.build's split), each after the same read lines as the never-shown ones
    SH = []; Lsh = idx[np.array([AB.text(a, b) for a, b in shown], np.int64)]
    for d in range(5):
        prev = ev[d, :, :10]
        for t in (5, 6, 7): hist = np.zeros((500, D), np.int64); hist[:, :t + 1] = Lsh[:, t::-1]; hist[:, t + 1:] = prev[:, ::-1][:, :D - t - 1]; SH.append((hist, Lsh[:, t + 1]))
    use = np.array([known[t - D:t + 1].all() for t in range(D, len(he))]); HE = np.stack([he[D - 1 - j: len(he) - 1 - j] for j in range(D)], 1)[use]; YE = he[D:][use]      # the states of the held-out English and the bytes that follow
    def never_shown(sy, among=None, states=None):
        """Whole answers of the 500 never-shown sums (five draws), the true bytes before each place. among: only these branches may answer. states: SH for the shown sums."""
        ok = np.ones((5, 500), bool)
        for i, (hist, y) in enumerate(states or HS):
            a = sy.branches(hist)
            if among is not None: b = np.full_like(a, -np.inf); b[:, among] = a[:, among]; a = b
            ok[i // 3] &= a.argmax(1) == y
        return float(100 * ok.mean())
    def english(sy): return float(100 * (sy.branches(HE).argmax(1) == YE).mean())
    def read(sy, lo, hi): sy.read(xm[lo - 1:hi - 1], xm[lo:hi])
    def mark(label, sy=None): sy = sy or s; r = dict(at=label, whole=never_shown(sy), among_digits=never_shown(sy, digits), english=english(sy), shown_whole=never_shown(sy, None, SH), shown_among_digits=never_shown(sy, digits, SH)); rows.append(r); print(tag, label, {k: round(v, 1) for k, v in r.items() if k != "at"}, flush=True)
    a0, b0 = sp["A"]; done = a0
    for n in sorted({5000, 40000, lines}):
        if n <= lines: read(s, done, a0 + n * 10); done = a0 + n * 10; mark(f"sums {n} lines")
    rows[-1]["at"] = "end of the sums"
    HSall = np.concatenate([h for h, _ in HS]); q = np.linspace(sp["B"][0] + D, sp["B"][1] - 1, 5000).astype(np.int64); HB = np.stack([xm[q - 1 - j] for j in range(D)], 1).astype(np.int64)
    def acts(sy, hist): return LP.sg(sy.W1t[hist + sy.OFF].sum(1) + sy.b1)
    def code(sy):
        hs = acts(sy, HSall); hb = acts(sy, HB); quiet = hb.mean(0) < 0.05
        return dict(mean_activity_sums=float(hs.mean()), mean_activity_english=float(hb.mean()), cells_above_half_in_a_state_sums=float((hs > 0.5).sum(1).mean()), cells_above_half_in_a_state_english=float((hb > 0.5).sum(1).mean()),
                    cells_quiet_through_the_english=int(quiet.sum()), share_of_the_sums_activity_on_cells_quiet_through_the_english=float(hs[:, quiet].sum() / hs.sum()))
    YS = np.concatenate([y for _, y in HS]); others = np.setdiff1d(np.arange(S), digits)
    def voices(sy):
        """In the states where the never-shown answers are named: the output of the right digit's branch, the largest output among the branches that are not digits, that branch's constant, and how often such a branch is the loudest of all."""
        a = sy.branches(HSall); right = a[np.arange(len(YS)), YS]; o = a[:, others]; top = o.argmax(1); loud = o[np.arange(len(YS)), top]
        return dict(right_digit_output=float(right.mean()), loudest_other_branch_output=float(loud.mean()), its_constant=float(sy.b2[others[top]].mean()), share_of_states_where_another_branch_is_loudest=float((loud > a[:, digits].max(1)).mean()), largest_digit_output=float(a[:, digits].max(1).mean()))
    kept = dict(W1t=s.W1t.copy(), b1=s.b1.copy(), W2=s.W2.copy(), b2=s.b2.copy()); hs_a = acts(s, HSall); codes = dict(end_of_the_sums=code(s), voices_end_of_the_sums=voices(s))
    r0 = s.right; read(s, sp["B"][0], sp["B"][1]); english_right = 100 * (s.right - r0) / (sp["B"][1] - sp["B"][0]); mark("end of the English")
    def with_parts(names):
        c = copy.deepcopy(s)
        for k in names: getattr(c, k)[...] = kept[k]
        return c
    mark("end of the English, the answering strengths of the end of the sums put back", with_parts(("W2", "b2")))
    mark("end of the English, the answering strengths and the cells' thresholds of the end of the sums put back", with_parts(("W2", "b2", "b1")))
    codes["end_of_the_english"] = code(s); codes["voices_end_of_the_english"] = voices(s); codes["mean_change_of_a_cell_activity_in_the_sums_states_through_the_english"] = float(np.abs(acts(s, HSall) - hs_a).mean())
    codes["answering_strengths_of_the_digit_branches_change_over_size_before"] = float(np.linalg.norm(s.W2[digits] - kept["W2"][digits]) / np.linalg.norm(kept["W2"][digits]))
    done = sp["A2"][0]
    for n in (10, 20, 50, 100, 200, 500, 1000, 2000, 5000): hi = sp["A2"][0] + n * 10; read(s, done, hi); done = hi; mark(f"return {n} lines")
    out = dict(task="relay12", tag=tag, seed=seed, form=form, theta=theta, lines=lines, values=S, rows=rows, code=codes, english_bytes_named_right_while_reading=english_right, wall=round(time.time() - t0, 1)); json.dump(out, open(os.path.join(HERE, "res", f"relay12__{tag}.json"), "w")); return out

def check():
    bad = 0; x, sp = AB.build(1, 5000, back=500, english=3000); vals = np.unique(x); idx = np.zeros(256, np.int64); idx[vals] = np.arange(len(vals)); xm = idx[x].astype(np.int32); S = len(vals)
    lo = sp["B"][0] - 1500; n = 3000; xs = xm[lo:lo + n]; ys = xm[lo + 1:lo + n + 1]                         # 1,500 bytes of sums, then 1,500 of English
    for form in FORMS:
        for kw in (dict(H=64, D=10, eta1=1.0, eta2=1 / 16, lam=1e-5), dict(H=48, D=10, eta1=1.0, eta2=1 / 16, lam=0.02)):
            c = dict(kw); H = c.pop("H"); D = c.pop("D"); A = Sys(3, H, D, S, int(idx[10]), rscale=1.0, soft=0, theta=3.0, **c, **switches(form)); B = copy.deepcopy(A)
            A.read(xs, ys, fast=False); B.read(xs[:n // 2], ys[:n // 2]); B.read(xs[n // 2:], ys[n // 2:])
            w = max(np.abs(getattr(A, k) - getattr(B, k)).max() / (np.abs(getattr(A, k)).max() + 1e-300) for k in ("W1t", "b1", "W2", "b2", "Rt")); moved = float(np.abs(A.b1 - (-3.0 if "q" in form else 0.0)).max()); movedb = float(np.abs(A.b2).max())
            ok = w < 1e-6 and A.right == B.right and abs(A.loss - B.loss) / n < 1e-6 and (moved == 0) == ("t" in form) and (movedb == 0) == ("b" in form); bad += not ok
            print(f"form {form:6s} H={H} lam={kw['lam']:g}: compiled against the numpy reference, largest difference over largest strength {w:.1e}; named right {A.right} and {B.right}; the thresholds moved by at most {moved:.3f}, the branches' constants by at most {movedb:.3f} {'PASSED' if ok else 'FAILED'}", flush=True)
    A = Sys(3, 64, 10, S, int(idx[10]), eta1=1.0, eta2=1 / 16, lam=1e-5, rscale=1.0, soft=0); B = LP.Sys(3, 64, 10, S, int(idx[10]), eta1=1.0, eta2=1 / 16, lam=1e-5, rscale=1.0, soft=0); A.read(xs, ys); B.read(xs, ys)
    w = max(np.abs(getattr(A, k) - getattr(B, k)).max() / (np.abs(getattr(B, k)).max() + 1e-300) for k in ("W1t", "b1", "W2", "b2", "Rt")); ok = w < 1e-9 and A.right == B.right; bad += not ok
    print(f"with no switch the kernel against nano/loop/kernp.c (compiled separately): largest difference over largest strength {w:.1e}; named right {A.right} and {B.right} {'PASSED' if ok else 'FAILED'}")
    # the gate: a cell that is silent keeps its outgoing strengths exactly (a network whose cells are all far below threshold, read with the gate)
    A = Sys(3, 32, 10, S, int(idx[10]), eta1=0.0, eta2=1 / 16, lam=0.01, rscale=1.0, soft=0, gate=1, hold=1); A.b1[:] = -800.0; A.W2[:] = 1.0; w0 = A.W2.copy(); A.read(xs[:500], ys[:500]); ok = bool((A.W2 == w0).all()); bad += not ok
    print(f"with the gate, cells that are wholly silent keep their answering strengths bit for bit through 500 bytes of fading at 0.01: {ok} {'PASSED' if ok else 'FAILED'}")
    print("CHECK", "FAILED" if bad else "PASSED"); return not bad

if __name__ == "__main__":
    os.makedirs(os.path.join(HERE, "res"), exist_ok=True)
    if sys.argv[1] == "check": sys.exit(0 if check() else 1)
    kw = dict(a.split("=", 1) for a in sys.argv[2:]); out = run(kw)
    print(out["tag"], "| never-shown sums, whole answer (among digit branches; held-out English named right):", " | ".join(f"{r['at']}: {r['whole']:.1f} ({r['among_digits']:.1f}; {r['english']:.1f})" for r in out["rows"]), "| wall", out["wall"], flush=True)
