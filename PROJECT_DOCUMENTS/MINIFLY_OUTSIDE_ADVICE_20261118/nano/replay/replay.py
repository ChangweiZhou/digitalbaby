#!/usr/bin/env python3
"""nano/replay, PLAN_1.md: keeping by replay from the system's own counting cells. Two kinds of cells on the sums text: the relay system (the plain
system of loop_small_case/cells2.py: 256 logistic cells in between, one answering cell) and cells that count (a cell for a run of up to K bytes,
counting what follows it). Between real lines the system replays lines made from the counting cells; the relay system learns from a replayed line
as from a real one; nothing is counted during replay.
    python3 replay.py <label> seed=1 [who=P|R|O|Z] [r=1] [K=8] [plan=add:400000,sub:200000] [every=50000] [fast=1]
    who=P: the plain system, no replay (must reproduce cells2.py).
    who=R: after each real line, r replayed lines (the whole part of r, and one more with a chance of its fractional part). A replayed line begins
           at a line start; each next byte is drawn, in proportion to its counts, from the longest active counting cell; it stops at the newline.
    who=O (a control): in place of a replayed line, a real line of the first task drawn by the program from the lines it shows.
    who=Z (a control): as R, with the counting cells emptied by the program when the task changes.
    fast=1: the relay system's inner loop runs compiled (kernr.c through fastr.py); its numpy loop here is the reference (check_replay.py).
Copied from loop_small_case/cells2.py at commit fe397ca: the text, the split, the streams of random numbers, the plain system of class Sys, the
drawing of lines in blocks up to the next reading, and the scoring of all pairs at once."""
import json, os, random, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
SYMS = list(b"0123456789+=\n-"); S = len(SYMS); SI = {c: i for i, c in enumerate(SYMS)}; END = SI[10]; D = 10; NL = D * S; OFF = np.arange(D) * S; PLUS, EQ, MINUS = SI[ord("+")], SI[ord("=")], SI[ord("-")]

def text(op, a, b):
    if op == "-" and a < b: a, b = b, a
    return [SI[c] for c in f"{a:02d}{op}{b:02d}={(a + b if op == '+' else a - b):03d}\n".encode()]

def sg(u): return 1 / (1 + np.exp(-u))

class Sys:
    """The plain system of cells2.py."""
    def __init__(s, seed, H=256, eta1=.25, eta2=1 / 16, lam=1e-5):
        wr = np.random.default_rng([seed, 20261119]); s.H, s.eta1, s.eta2, s.lam = H, eta1, eta2, lam
        s.W1 = wr.uniform(-.5, .5, (H, NL)); s.b1 = np.zeros(H); s.R = wr.uniform(-1, 1, (H, S)); s.W2 = np.zeros((S, H)); s.b2 = np.zeros(S); s.hist = [END] * D
    def learn(s, y):
        act = OFF + np.asarray(s.hist); h = sg(s.W1[:, act].sum(1) + s.b1); a = s.W2 @ h + s.b2; m = -a; m[y] += 1; g = s.eta2 / (h @ h + 1)
        sig = (s.R @ m) * h * (1 - h); d2 = np.outer(g * m, h); d = s.eta1 * sig
        s.R += d2.T; s.W1[:, act] += np.repeat(d[:, None], D, 1); s.b1 += d; s.W2 += d2; s.b2 += g * m; s.W2 *= 1 - s.lam; s.R *= 1 - s.lam
    def steps(s, xs, ys):
        for x, y in zip(xs.tolist(), ys.tolist()):
            s.hist.insert(0, x); s.hist.pop()
            if y >= 0: s.learn(y)
    def score(s, op, pairs):
        Ls = np.array([text(op, a, b) for a, b in pairs]); ok = np.ones(len(pairs), bool); W1T = s.W1.T
        for t in (5, 6, 7):
            hist = np.full((len(pairs), D), END); hist[:, :t + 1] = Ls[:, t::-1]
            ok &= (sg(W1T[OFF + hist].sum(1) + s.b1) @ s.W2.T + s.b2).argmax(1) == Ls[:, t + 1]
        return float(100 * ok.sum() / len(pairs))

class Counting:
    """Cells that count. A cell is a run of 1 to K bytes that has occurred; it holds, for each byte value, how often that byte followed the run.
    Held here: for each byte of a line, the longest cell whose run does not reach back beyond the newline before the line (that newline included).
    Shorter cells, and cells whose run crosses from one line into the next, are never the longest cell that has counted during a replay, which
    begins at a newline with nothing before it, and are not held."""
    def __init__(s, K): s.K = K; s.cells = {}
    def count(s, line):
        seq = [END] + line
        for p in range(len(line)):
            key = bytes(seq[max(0, p + 1 - s.K):p + 1]); c = s.cells.get(key)
            if c is None: c = s.cells[key] = [0] * S
            c[line[p]] += 1
    def replay(s, rnd):
        seq = [END]
        while len(seq) < 16:
            c = s.cells.get(bytes(seq[-s.K:]))
            if c is None: break                        # no cell has counted after this run (does not happen while K covers what was counted)
            u = rnd.random() * sum(c); v = 0
            while u >= c[v]: u -= c[v]; v += 1
            seq.append(v)
            if v == END: break
        return seq[1:]
    def empty(s): s.cells = {}
    def size(s): return len(s.cells), sum(1 for c in s.cells.values() for x in c if x)

def kind(line):
    """(well formed, correct, operator) of a replayed line."""
    if len(line) != 10 or line[9] != END or line[5] != EQ or line[2] not in (PLUS, MINUS) or any(not 0 <= line[i] <= 9 for i in (0, 1, 3, 4, 6, 7, 8)): return False, False, None
    a, b, c = line[0] * 10 + line[1], line[3] * 10 + line[4], line[6] * 100 + line[7] * 10 + line[8]; op = "+" if line[2] == PLUS else "-"
    return True, (a + b == c if op == "+" else (a >= b and a - b == c)), op

def run(kw):
    t0 = time.process_time(); seed = int(kw.get("seed", 1)); rng = np.random.default_rng([seed, 99]); held, shown = {}, {}
    for op in "+-":
        o = rng.permutation(10000); P = [(int(q) // 100, int(q) % 100) for q in o]; held[op] = P[:500]; shown[op] = P[2000:]
        if op == "-":                                 # PLAN_1.md section 12: a subtraction line is written with the larger number first, so a never-shown pair (a, b) writes the line of (b, a), which may be shown
            wr = set(tuple(text("-", a, b)) for a, b in shown["-"]); clean = [q for q in P[:500] if tuple(text("-", *q)) not in wr]; clean_wide = [q for q in P[:2000] if tuple(text("-", *q)) not in wr]
    s = Sys(seed, H=int(kw.get("H", 256))); who = kw.get("who", "P"); r = float(eval(kw.get("r", "1"))); K = int(kw.get("K", 8)); cells = Counting(K); rnd = random.Random(seed * 1000 + 7)
    every = int(kw.get("every", 50000)); fast = int(kw.get("fast", 0)); out = dict(label=kw.get("label"), args=kw, rows=[])
    TAB = {op: np.array([text(op, a, b) for a, b in shown[op]], np.int32) for op in "+-"}; OLD = TAB["+"].tolist(); t_rep = 0.0
    def go(xs, ys):
        if fast: import fastr; fastr.run(s, xs, ys)
        else: s.steps(xs, ys)
    for pi, ph in enumerate(kw.get("plan", "add:400000,sub:200000").split(",")):
        name, n = ph.split(":"); n = int(n); op0 = {"add": "+", "sub": "-"}[name]; it = 0
        if who == "Z" and pi > 0: cells.empty()
        while it < n:
            nxt = min((it // every + 1) * every, n) - 1; X = TAB[op0][rng.integers(8000, size=nxt - it + 1)]; it = nxt + 1; st = dict(extra=0, x_add=0, x_sub=0, well_formed=0, correct=0)
            if who == "P": lines = X
            else:
                t1 = time.process_time(); lines = []
                for L in X.tolist():
                    lines.append(L)
                    if who != "O": cells.count(L)
                    for _ in range(int(r) + (rnd.random() < r - int(r))):
                        if who == "O": E = OLD[rnd.randrange(8000)]
                        else: E = cells.replay(rnd)
                        wf, ok, op = kind(E); st["extra"] += 1; st["well_formed"] += wf; st["correct"] += ok; st["x_add"] += op == "+"; st["x_sub"] += op == "-"
                        if wf: lines.append(E)        # a line that is not well formed has no fixed length; none arises while K covers what was counted, and one that did would be left out
                lines = np.array(lines, np.int32); t_rep += time.process_time() - t1
            xs = lines.ravel(); ys = np.concatenate([lines[:, 1:], np.full((len(lines), 1), -1, np.int32)], 1).ravel(); go(xs, ys)
            if (nxt + 1) % every == 0 or nxt + 1 == n: out["rows"].append(dict(phase=pi, name=name, lines=nxt + 1, add=s.score("+", held["+"]), sub=s.score("-", held["-"]), sub_clean=s.score("-", clean), sub_clean_wide=s.score("-", clean_wide), **st))
    ph = lambda i: [x for x in out["rows"] if x["phase"] == i]; nc, nz = cells.size()
    out.update(add_after_first=float(np.mean([x["add"] for x in ph(0)[-3:]])), cells=nc, counts_above_zero=nz, cpu=round(time.process_time() - t0, 1), cpu_counting_and_replay=round(t_rep, 1))
    out.update(n_clean=len(clean), n_clean_wide=len(clean_wide))
    if len(ph(1)): out.update(sub_clean_after_second=ph(1)[-1]["sub_clean"], sub_clean_wide_after_second=ph(1)[-1]["sub_clean_wide"], add_after_second=ph(1)[-1]["add"], sub_after_second=ph(1)[-1]["sub"], add_last3_second=float(np.mean([x["add"] for x in ph(1)[-3:]])), sub_last3_second=float(np.mean([x["sub"] for x in ph(1)[-3:]])))
    run.last = s; run.cells = cells
    if not kw.get("nowrite"):
        os.makedirs(os.path.join(HERE, "res"), exist_ok=True); json.dump(out, open(os.path.join(HERE, "res", f"p_{kw['label']}.json"), "w"))
    return out

if __name__ == "__main__":
    kw = dict(a.split("=") for a in sys.argv[2:]); kw["label"] = sys.argv[1]; o = run(kw)
    print(o["label"], f"addition after its phase {o['add_after_first']:.1f}; after the second phase: addition {o.get('add_after_second')}, subtraction {o.get('sub_after_second')} |",
          " ".join(f"{x['phase']}:{x['lines']//1000}k {x['add']:.0f}/{x['sub']:.0f}" for x in o["rows"]), "| second phase, extra lines by reading (addition/subtraction/correct of all):",
          " ".join(f"{x['x_add']}/{x['x_sub']}/{x['correct']}of{x['extra']}" for x in o["rows"] if x["phase"] == 1), f"| cells {o['cells']} counts above zero {o['counts_above_zero']} | cpu {o['cpu']} s, of which counting and replay {o['cpu_counting_and_replay']} s")
