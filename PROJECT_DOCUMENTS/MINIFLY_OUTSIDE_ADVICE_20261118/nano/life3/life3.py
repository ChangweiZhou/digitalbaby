#!/usr/bin/env python3
"""nano/life3, PLAN_1.md: a life of three (or more) tasks on lines of ten bytes, with the plain system and three routes of keeping side by side.
    python3 life3.py <label> seed=1 arm=P|O|R|K|S [plan=add:400000,sub:200000,max:200000] [every=50000] [fast=1] [H=..] [r=1] [K=8] [syms=..] [flat=1]
    arm=P: the plain system (the relay system of loop_small_case/cells2.py: 256 logistic cells in between, one answering cell).
    arm=O (the upper reference): the plain system; after each real line the program shows one shown line of an earlier task (an earlier task
           chosen with equal chance; in the first phase, where there is none, a line of the first task, as the control of nano/replay did).
    arm=R (route 2, nano/replay): the plain system and cells that count runs of up to K=8 bytes of the real lines; after each real line r=1
           replayed line made from the counting cells, from which the relay system learns as from a real line. Nothing is counted in replay.
    arm=K (route 1, nano/keep's final candidate): H=384 cells of which H0=1 is present at birth; cells join while the answering cell hears
           fewer than 128 (theta=20); a strength opens if its line was active on at least meet=30 of the cell's first young=4500 learned bytes;
           a cell with a closed strength on an active line is silent, unchanged and not fading; the answering cell's constants become still
           (exp(-n/1e4)).
    arm=N (PLAN_6.md): as R, but after each real line cand=4 lines are drawn from the counting cells and the relay system learns from the one
           on which its own mismatch is largest. No compiled code is added: the choice is made in numpy.
    arm=S (route 3, nano/sparse): cells that count, keyed on at most m=4 of the delays 1..9, read with a learned trust whose step shrinks with
           use (eta 0.1, e 2, trusts born at 0.02, shrink=1). The program of that line (sparse.py) is used unchanged, with one more byte value.
           PLAN_7.md: ncap=16 (a cell that has counted more than 16 times speaks as if it had counted 16 times, in the same proportions) and
           la=1 (route 3b: that line's look-ahead system, sparse4.py: every cell also counts what arrives two and three bytes after it, and
           cells may be keyed on the answering cell's guesses of those bytes).
    fast=1: the compiled kernels (kernr.c for P, O, R; kage.c for K; kern_s.c for S). fast=0: the numpy references (check_life3.py compares).
Tasks: add aa+bb=ccc; sub aa-bb=ccc (larger number first, as cells2.py); max aa^bb=ccc (the larger of the two); sat aa&bb=ccc (the sum plus
the first number's tens digit); srv aa#bb=ccc (the first number plus the second with its digits exchanged). The symbol set is that of
cells2.py with one further byte for each further operator in the plan (or in syms=). Added for PLAN_2.md: min aa_bb=ccc (the smaller of the two).
Added for PLAN_3.md: xam aa=bb+ccc (the larger of the two, written in the byte values of addition; no further byte).
The first fourteen byte values, the split of additions and subtractions, the stream that draws the lines and the birth values of the relay
system on the first fourteen byte values are those of cells2.py, so that the first two phases are the lives the other lines lived.
Copied (see PLAN_1.md for the commit): the relay system and the counting cells from nano/replay/replay.py; the rules of route 1 from
nano/keep/kage.py (the options of the final candidate only); kernr.c, fastr.py, kage.c, sparse.py unchanged; fastage.py with one import line changed."""
import json, os, random, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
BASE = list(b"0123456789+=\n-"); D = 10
TASKS = {"add": ("+", lambda a, b: (a, b, a + b)), "sub": ("-", lambda a, b: (max(a, b), min(a, b), abs(a - b))), "max": ("^", lambda a, b: (a, b, max(a, b))),
         "sat": ("&", lambda a, b: (a, b, a + b + a // 10)), "srv": ("#", lambda a, b: (a, b, a + 10 * (b % 10) + b // 10)),
         "min": ("_", lambda a, b: (a, b, min(a, b))), "xam": ("+", lambda a, b: (a, b, max(a, b)))}
TID = {"add": 0, "sub": 1, "max": 2, "sat": 3, "srv": 4, "xam": 5, "min": 6}
FMT = {"xam": lambda a, b, c: f"{a:02d}={b:02d}+{c:03d}\n"}      # a task written otherwise than aa?bb=ccc (PLAN_3.md): `max` in the byte values of addition, aa=bb+ccc

def sg(u): return 1 / (1 + np.exp(-u))

class World:
    """The byte values of a life and its lines."""
    def __init__(w, names):
        w.syms = list(BASE)
        for n in names:
            if ord(TASKS[n][0]) not in w.syms: w.syms.append(ord(TASKS[n][0]))
        w.S = len(w.syms); w.SI = {c: i for i, c in enumerate(w.syms)}; w.END = w.SI[10]; w.EQ = w.SI[ord("=")]; w.NL = D * w.S; w.OFF = np.arange(D) * w.S
    def text(w, name, a, b):
        op, f = TASKS[name]; a, b, c = f(a, b)
        return [w.SI[ch] for ch in (FMT[name](a, b, c) if name in FMT else f"{a:02d}{op}{b:02d}={c:03d}\n").encode()]

def split(seed, names):
    """cells2.py's split for additions and subtractions (one generator, which goes on to draw the lines); a generator of its own for each further task."""
    rng = np.random.default_rng([seed, 99]); P = {}
    for n in ("add", "sub"): o = rng.permutation(10000); P[n] = [(int(q) // 100, int(q) % 100) for q in o]
    for n in names:
        if n not in P: o = np.random.default_rng([seed, 99, TID[n]]).permutation(10000); P[n] = [(int(q) // 100, int(q) % 100) for q in o]
    return rng, P

class Relay:
    """The plain system of cells2.py (as in nano/replay/replay.py), with the number of byte values a parameter."""
    def __init__(s, seed, w, H=256, eta1=.25, eta2=1 / 16, lam=1e-5):
        wr = np.random.default_rng([seed, 20261119]); S = w.S; s.w = w; s.H, s.eta1, s.eta2, s.lam = H, eta1, eta2, lam
        W = [wr.uniform(-.5, .5, (H, D, 14))]; R = [wr.uniform(-1, 1, (H, 14))]       # the draws of cells2.py, in its order
        for k in range(S - 14):                       # each further byte value has a generator of its own, so that a life with more byte values begins as a life with fewer
            xr = np.random.default_rng([seed, 20261120, 3] + ([k] if k else [])); W.append(xr.uniform(-.5, .5, (H, D, 1))); R.append(xr.uniform(-1, 1, (H, 1)))
        s.W1 = np.ascontiguousarray(np.concatenate(W, 2).reshape(H, D * S)); s.R = np.ascontiguousarray(np.concatenate(R, 1))
        s.b1 = np.zeros(H); s.W2 = np.zeros((S, H)); s.b2 = np.zeros(S); s.hist = [w.END] * D; s.fast = 0
    def learn(s, y):
        act = s.w.OFF + np.asarray(s.hist); h = sg(s.W1[:, act].sum(1) + s.b1); a = s.W2 @ h + s.b2; m = -a; m[y] += 1; g = s.eta2 / (h @ h + 1)
        sig = (s.R @ m) * h * (1 - h); d2 = np.outer(g * m, h); d = s.eta1 * sig
        s.R += d2.T; s.W1[:, act] += np.repeat(d[:, None], D, 1); s.b1 += d; s.W2 += d2; s.b2 += g * m; s.W2 *= 1 - s.lam; s.R *= 1 - s.lam
    def steps(s, lines):
        lines = np.asarray(lines, np.int32); xs = lines.ravel(); ys = np.concatenate([lines[:, 1:], np.full((len(lines), 1), -1, np.int32)], 1).ravel()
        if s.fast: s.go_fast(xs, ys); return
        for x, y in zip(xs.tolist(), ys.tolist()):
            s.hist.insert(0, x); s.hist.pop()
            if y >= 0: s.learn(y)
    def go_fast(s, xs, ys): import fastr; fastr.run(s, xs, ys)
    def cells_out(s, hist): return sg(s.W1.T[s.w.OFF + hist].sum(1) + s.b1)
    def answers(s, Ls):
        """The three answer bytes of each line, each named with the line's own bytes before it (the scoring of cells2.py)."""
        out = []
        for t in (5, 6, 7):
            hist = np.full((len(Ls), D), s.w.END); hist[:, :t + 1] = Ls[:, t::-1]; out.append((s.cells_out(hist) @ s.W2.T + s.b2).argmax(1))
        return np.stack(out, 1)
    def feed(s, X, pi): s.steps(X)
    def census(s): return {}

class Keep(Relay):
    """Route 1: the final candidate of nano/keep (kage.py with H0=1 join=hear K=128 theta=20 young=4500 meet=30 blaw=exp btau=1e4), those options only."""
    def __init__(s, seed, w, H=384, H0=1, K=128, theta=20.0, young=4500, meet=30, btau=1e4, rhoA=1e-2, **k):
        super().__init__(seed, w, H=H, **k); s.law = 0; s.tau = 1e5; s.young = young; s.usemode = 0; s.sigjoin = 2; s.K = float(K); s.rhoA = rhoA; s.A = 0.0; s.meet = meet; s.openall = 0
        s.rhoE = 1e-3; s.rhoS = 1e-5; s.delta = .25; s.blaw = 1; s.btau = btau; s.G = 0.0; s.E = 0.0; s.C = 0.0; s.n = 0; s.nl = 0; s.U = np.zeros(H)
        s.born = np.zeros(H, np.int64); s.born[H0:] = np.iinfo(np.int64).max; s.thr = np.zeros(H); s.thr[H0:] = np.arange(1, H - H0 + 1) * float(theta); s.seen = np.zeros((H, w.NL), np.uint8)
    def learn(s, y):
        while s.nl < s.H and s.thr[s.nl] <= s.C: s.born[s.nl] = s.n; s.nl += 1
        live = np.arange(s.H) < s.nl; act = s.w.OFF + np.asarray(s.hist); h = sg(s.W1[:, act].sum(1) + s.b1) * live
        ygm = live & (s.n - np.where(live, s.born, 0) < s.young); yg = np.nonzero(ygm)[0]; blk = s.seen[np.ix_(yg, act)]; s.seen[np.ix_(yg, act)] = np.where(blk < 255, blk + 1, 255)
        op = ygm | (s.seen[:, act] >= s.meet).all(1); h = h * op; rate = (live & op).astype(float); heard = int((live & op).sum())      # silent at this byte: unchanged, the fading included
        m = -(s.W2 @ h + s.b2); m[y] += 1; g = s.eta2 / (h @ h + 1)
        sig = (s.R @ m) * h * (1 - h) * rate; d2 = np.outer(g * m, h * rate)
        s.W1[:, act] += s.eta1 * sig[:, None]; s.b1 += s.eta1 * sig
        s.W2 += d2; s.W2 *= 1 - s.lam * rate; s.R += d2.T; s.R *= 1 - s.lam * rate[:, None]
        s.b2 += g * m * np.exp(-s.n / s.btau); s.E += s.rhoE * (m @ m - s.E); s.G += s.rhoS * (m @ m - s.G)
        s.A += s.rhoA * (heard - s.A); s.C += max(0.0, 1 - s.A / s.K); s.n += 1
    def go_fast(s, xs, ys): import fastage; s.bx = xs; s.by = ys; fastage.run(s)
    def may_speak(s, hist):
        live = np.arange(s.H) < s.nl; return (s.seen.T[s.w.OFF + hist] >= s.meet).all(1) | (live & (s.n - np.where(live, s.born, 0) < s.young))
    def cells_out(s, hist): return sg(s.W1.T[s.w.OFF + hist].sum(1) + s.b1) * (np.arange(s.H) < s.nl) * s.may_speak(hist)
    def census(s): return dict(joined=int(s.nl), waiting=int(s.H - s.nl), born=[int(x) for x in s.born[:s.nl]], heard_mean=float(s.A))
    def speaking(s, Ls):
        """A measure, not used by the system: for each joined cell, the share of the answer bytes of these lines at which it may speak."""
        m = []
        for t in (5, 6, 7):
            hist = np.full((len(Ls), D), s.w.END); hist[:, :t + 1] = Ls[:, t::-1]; m.append(s.may_speak(hist) & (np.arange(s.H) < s.nl))
        return np.concatenate(m).mean(0)

class Counting:
    """Cells that count (nano/replay/replay.py): a cell is a run of 1 to K bytes that has occurred within a line (the newline before it included);
    it holds, for each byte value, how often that byte followed the run."""
    def __init__(s, K, w, flat=0): s.K = K; s.cells = {}; s.w = w; s.flat = flat
    def count(s, line):
        seq = [s.w.END] + line
        for p in range(len(line)):
            key = bytes(seq[max(0, p + 1 - s.K):p + 1]); c = s.cells.get(key)
            if c is None: c = s.cells[key] = [0] * s.w.S
            c[line[p]] += 1
    def replay(s, rnd):
        seq = [s.w.END]
        while len(seq) < 16:
            c = s.cells.get(bytes(seq[-s.K:]))
            if c is None: break
            if s.flat:                                # PLAN_5.md: every byte that has followed the run is equally likely, however often it followed
                f = [v for v in range(s.w.S) if c[v]]; v = f[int(rnd.random() * len(f))]
            else:
                u = rnd.random() * sum(c); v = 0
                while u >= c[v]: u -= c[v]; v += 1
            seq.append(v)
            if v == s.w.END: break
        return seq[1:]
    def size(s): return len(s.cells), sum(1 for c in s.cells.values() for x in c if x)

class Mixed(Relay):
    """Arms O and R: the plain system with one further line after each real line (old lines drawn by the program, or replayed lines)."""
    def __init__(s, seed, w, who, tabs, order, r=1.0, K=8, flat=0, cand=4, blk=500, **k):
        super().__init__(seed, w, **k); s.who = who; s.r = r; s.cand = cand; s.blk = blk; s.cells = Counting(K, w, flat); s.rnd = random.Random(seed * 1000 + 7); s.tabs = tabs; s.order = order; s.t_extra = 0.0
        s.lineset = {n: set(map(tuple, tabs[n])) for n in order}; s.ops = {w.SI[ord(TASKS[n][0])]: n for n in order if n not in FMT}; s.st = None; s.newst()
    def newst(s): s.st = dict(extra=0, well_formed=0, of_a_shown_line=0, **{"x_" + n: 0 for n in s.order})
    def kind(s, E):
        """Of a further line: is it well formed, which task's operator it carries, and is it one of the lines shown (of any task of the life)."""
        w = s.w; wf = len(E) == 10 and E[9] == w.END and E[5] == w.EQ and E[2] in s.ops and all(0 <= E[i] <= 9 for i in (0, 1, 3, 4, 6, 7, 8))
        if not wf:                                    # a task written otherwise (PLAN_3.md) is recognised by being one of its shown lines
            for n in s.order:
                if tuple(E) in s.lineset[n]: return len(E) == 10, n, True
            return False, None, False
        n = s.ops[E[2]]; return True, n, tuple(E) in s.lineset[n]
    def mismatch(s, C):
        """The relay system's squared mismatch summed over the nine named bytes of each line, the line standing alone (line ends before it)."""
        n = len(C); tot = np.zeros(n); ar = np.arange(n)
        for t in range(9):
            hist = np.full((n, D), s.w.END); hist[:, :t + 1] = C[:, t::-1]; m = -(s.cells_out(hist) @ s.W2.T + s.b2); m[ar, C[:, t + 1]] += 1; tot += (m * m).sum(1)
        return tot
    def feed_need(s, X):
        """PLAN_6.md: after each real line, `cand` lines are drawn from the counting cells as in route 2, and the relay system learns from the one on
        which its own mismatch is largest (measured with its strengths as they stood at the start of the present block of `blk` real lines)."""
        for i in range(0, len(X), s.blk):
            t1 = time.process_time(); Xb = X[i:i + s.blk].tolist(); cands = []
            for L in Xb:
                s.cells.count(L)
                for _ in range(s.cand):
                    E = s.cells.replay(s.rnd); assert len(E) == 10; cands.append(E)
            C = np.array(cands, np.int32); pick = s.mismatch(C).reshape(len(Xb), s.cand).argmax(1); lines = []
            for j, L in enumerate(Xb):
                E = cands[j * s.cand + int(pick[j])]; wf, n, ok = s.kind(E); s.st["extra"] += 1; s.st["well_formed"] += wf; s.st["of_a_shown_line"] += ok
                if n: s.st["x_" + n] += 1
                lines.append(L); lines.append(E)
            s.t_extra += time.process_time() - t1; s.steps(np.array(lines, np.int32))
    def feed(s, X, pi):
        if s.who == "N": return s.feed_need(X)
        t1 = time.process_time(); lines = []; earlier = s.order[:pi] or s.order[:1]
        for L in X.tolist():
            lines.append(L)
            if s.who == "R": s.cells.count(L)
            for _ in range(int(s.r) + (s.rnd.random() < s.r - int(s.r))):
                if s.who == "O":
                    n = earlier[s.rnd.randrange(len(earlier))] if len(earlier) > 1 else earlier[0]; E = s.tabs[n][s.rnd.randrange(8000)]
                else: E = s.cells.replay(s.rnd)
                wf, n, ok = s.kind(E); s.st["extra"] += 1; s.st["well_formed"] += wf; s.st["of_a_shown_line"] += ok
                if n: s.st["x_" + n] += 1
                if len(E) == 10: lines.append(E)      # a line of another length would be left out; none arises while K covers the line
        s.t_extra += time.process_time() - t1; s.steps(np.array(lines, np.int32))
    def census(s):
        st = s.st; s.newst(); nc, nz = s.cells.size(); return dict(extra_since_last_reading=st, counting_cells=nc, counts_above_zero=nz)

class Sparse:
    """Route 3: nano/sparse/sparse.py's system (compiled) or its numpy reference, with the byte values of this life."""
    def __init__(s, seed, w, m=4, eta=0.1, e=2.0, T0=0.02, shrink=1, ref=0, ncap=0, la=0):
        import sparse; assert sparse.END == w.END; s.w = w
        if la:                                        # PLAN_7.md: nano/sparse's look-ahead system (sparse4.py, unchanged; its number of byte values is a module constant, set here)
            import sparse4; sparse4.S = w.S; assert w.S <= 16; s.s = (sparse4.Ref4 if ref else sparse4.Sys4)(m=m, eta=eta, e=e, T0=T0, Tg0=0.0, shrink=shrink, ncap=ncap)
        elif ref: assert not ncap; s.s = sparse.Ref(kinds=sparse.kinds_of(9, m), eta=eta, e=e, T0=T0, nsym=w.S, shrink=shrink)
        else: s.s = sparse.Sys(kinds=sparse.kinds_of(9, m), eta=eta, e=e, T0=T0, nsym=w.S, shrink=shrink, ncap=ncap)
    def feed(s, X, pi): s.s.read(np.asarray(X, np.int32))
    def answers(s, Ls, own=0): return s.s.answers(np.ascontiguousarray(Ls, np.int32), own, 6, 3)
    def census(s): return s.s.size()

def run(kw):
    t0 = time.process_time(); seed = int(kw.get("seed", 1)); plan = [(p.split(":")[0], int(p.split(":")[1])) for p in kw.get("plan", "add:400000,sub:200000,max:200000").split(",")]
    order = list(dict.fromkeys(n for n, _ in plan)); names = order + [n for n in kw.get("syms", "").split(",") if n and n not in order]; w = World(names); rng, P = split(seed, names)
    arm = kw.get("arm", "P"); every = int(kw.get("every", 50000)); fast = int(kw.get("fast", 1)); score_on = names if kw.get("score_all") else order
    TAB = {n: np.array([w.text(n, a, b) for a, b in P[n][2000:]], np.int32) for n in order}; TL = {n: TAB[n].tolist() for n in order}
    HELD, CLEAN = {}, {}
    for n in score_on:                                 # the 500 never-shown pairs of cells2.py; and, where a never-shown pair can write a shown line, the never-shown pairs (of 2,000) that write no shown line
        shown = set(tuple(w.text(n, a, b)) for a, b in P[n][2000:]); h = [q for q in P[n][:500]]; c = [q for q in h if tuple(w.text(n, *q)) not in shown]
        if len(c) < len(h): c = [q for q in P[n][:2000] if tuple(w.text(n, *q)) not in shown]
        HELD[n] = np.array([w.text(n, *q) for q in h], np.int32); CLEAN[n] = np.array([w.text(n, *q) for q in c], np.int32)
    Hk = dict(H=int(kw["H"])) if "H" in kw else {}
    if arm == "P": s = Relay(seed, w, **Hk)
    elif arm in "ORN": s = Mixed(seed, w, arm, TL, order, r=float(eval(kw.get("r", "1"))), K=int(kw.get("K", 8)), flat=int(kw.get("flat", 0)), cand=int(kw.get("cand", 4)), blk=int(kw.get("blk", 500)), **Hk)
    elif arm == "K": s = Keep(seed, w, **Hk, **{k: float(kw[k]) for k in ("theta", "btau") if k in kw}, **{k: int(kw[k]) for k in ("H0", "K", "young", "meet") if k in kw})
    elif arm == "S": s = Sparse(seed, w, ref=1 - fast, **{k: float(kw[k]) for k in ("eta", "e", "T0") if k in kw}, **{k: int(kw[k]) for k in ("m", "shrink", "la") if k in kw}, **{k: float(kw[k]) for k in ("ncap",) if k in kw})
    else: raise SystemExit("arm?")
    if arm != "S": s.fast = fast
    whole = lambda Ls, own=None: float(100 * ((s.answers(Ls) if own is None else s.answers(Ls, own)) == Ls[:, 6:9]).all(1).mean()) if len(Ls) else None
    out = dict(label=kw.get("label"), args=kw, arm=arm, plan=plan, syms=bytes(w.syms).decode(), n_held={n: len(HELD[n]) for n in score_on}, n_clean={n: len(CLEAN[n]) for n in score_on}, rows=[])
    for pi, (name, n) in enumerate(plan):
        it = 0
        while it < n:
            nxt = min((it // every + 1) * every, n) - 1; X = TAB[name][rng.integers(8000, size=nxt - it + 1)]; it = nxt + 1; s.feed(X, pi)
            row = dict(phase=pi, name=name, lines=it)
            for t in score_on:
                row[t] = whole(HELD[t]); row[t + "_clean"] = whole(CLEAN[t])
                if arm == "S": row[t + "_own"] = whole(HELD[t], 1); row[t + "_clean_own"] = whole(CLEAN[t], 1)
                if arm == "K": sp = s.speaking(HELD[t]); row[t + "_cells_may_speak"] = float(sp.sum()); row[t + "_cells_speak_mostly"] = int((sp > .5).sum())
            row.update(s.census()); out["rows"].append(row)
    ends = []
    for pi, (name, n) in enumerate(plan):
        rr = [r for r in out["rows"] if r["phase"] == pi]; e = dict(phase=pi, name=name)
        for t in score_on:
            for k in (t, t + "_clean") + ((t + "_own", t + "_clean_own") if arm == "S" else ()): e[k] = rr[-1][k]; e[k + "_last3"] = float(np.mean([r[k] for r in rr[-3:]]))
        ends.append(e)
    out["ends"] = ends; out["cpu"] = round(time.process_time() - t0, 1); run.last = s
    if arm in "ORN": out["cpu_further_lines_python"] = round(s.t_extra, 1)
    if not kw.get("nowrite"):
        os.makedirs(os.path.join(HERE, "res"), exist_ok=True); json.dump(out, open(os.path.join(HERE, "res", f"{kw['label']}.json"), "w"))
    return out

def show(o):
    ts = list(o["n_held"]); print(o["label"], "arm", o["arm"], "cpu", o["cpu"], "s | clean pairs", o["n_clean"])
    for r in o["rows"]:
        x = "  ".join(f"{t} {r[t]:5.1f}" + (f" (clean {r[t + '_clean']:5.1f})" if o["n_clean"][t] != o["n_held"][t] else "") for t in ts)
        y = f" | joined {r['joined']} speak " + "/".join(str(r[t + "_cells_speak_mostly"]) for t in ts) if "joined" in r else ""
        z = " | further lines " + str({k: v for k, v in r["extra_since_last_reading"].items()}) if "extra_since_last_reading" in r else ""
        print(f"  {r['phase']}:{r['name']} {r['lines']:>7}  {x}{y}{z}", flush=True)

if __name__ == "__main__":
    kw = dict(a.split("=") for a in sys.argv[2:]); kw["label"] = sys.argv[1]; show(run(kw))
