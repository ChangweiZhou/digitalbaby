#!/usr/bin/env python3
"""nano/replay, PLAN_4.md ("dreams"): arm R of replay.py (one replayed line for each real line, K = 8) with one change: a share q of the extra lines
are not replays but dreams. A dream's two numbers and signs are drawn from short counting cells, so that pairs never shown occur; its answer is
written by the relay system itself; the dreamed line is then counted by the counting cells like a real line, at weight w; the relay system does
not learn from it at that moment. Later replays draw from everything counted, dreamed lines included.
    python3 replay4.py <label> seed=1 [q=0.1] [w=1] [start=0] [ans=relay|chance] [plan=add:400000,sub:200000] [every=50000] [chunk=500] [fast=1]
    q=0: arm R.   start=n: no dream before n real lines have been read (an addition of mine).   ans=chance (a control): the three digits of a
    dream's answer are drawn by chance in place of the relay system's.
How a dream is made.  First number and sign: byte by byte from the cells that begin at the newline (runs of 1 to 3 bytes), as a replay would.  Second
number and the next sign: from cells that begin at the sign (runs of 1 to 3 bytes: the sign; the sign and a digit; the sign and two digits), which
do not see the first number.  Answer: the relay system reads the six bytes and names the next byte, reads its own byte and names the next, four
times; the dream is kept if the three answer bytes are digits and the fourth is the newline.
The life is read in chunks of `chunk` real lines.  The dreams of a chunk are made at its start, from the cells and the relay system as they then
are (so a dream's answer is the relay system's of at most `chunk` real lines earlier), and counted at their places in the chunk; the relay system
then learns the chunk's real and replayed lines in order.  The relay system, the counting of real lines and the replay are replay.py's."""
import json, os, random, sys, time
import numpy as np
import replay
from replay import HERE, Sys, text, sg, S, END, D, OFF, PLUS, EQ, MINUS, kind

class Cells(replay.Counting):
    """replay.py's counting cells with weights, and the short cells that begin at a sign."""
    def __init__(s, K): super().__init__(K); s.short = {}; s.n_real = 0; s.n_dream = 0
    def count(s, line, w=1.0):
        seq = [END] + line
        for p in range(len(line)):
            key = bytes(seq[max(0, p + 1 - s.K):p + 1]); c = s.cells.get(key)
            if c is None: c = s.cells[key] = [0.0] * S
            c[line[p]] += w
        for n in (1, 2, 3):
            key = bytes(line[2:2 + n]); c = s.short.get(key)
            if c is None: c = s.short[key] = [0.0] * S
            c[line[2 + n]] += w
    @staticmethod
    def draw(c, rnd):
        u = rnd.random() * sum(c); v = 0
        while v < S - 1 and u >= c[v]: u -= c[v]; v += 1
        return v
    def prefix(s, rnd):
        """The first six bytes of a dream, or None if a cell is missing."""
        seq = [END]
        for _ in range(3):
            c = s.cells.get(bytes(seq));
            if c is None: return None
            seq.append(s.draw(c, rnd))
        out = seq[1:]
        for n in (1, 2, 3):
            c = s.short.get(bytes(out[2:2 + n]))
            if c is None: return None
            out.append(s.draw(c, rnd))
        return out
    def answer(s, six):
        """The store's own answer to six bytes: the byte with the largest count in the longest cell, four times; None if a cell is missing."""
        seq = [END] + list(six)
        for _ in range(4):
            c = s.cells.get(bytes(seq[-s.K:]))
            if c is None or sum(c) == 0: return None
            seq.append(int(np.argmax(c)))
        return seq[1:]
    def size(s): return len(s.cells) + len(s.short), sum(1 for d in (s.cells, s.short) for c in d.values() for x in c if x)

def write(s, P):
    """The relay system writes bytes 6 to 9 of the lines that begin with the rows of P (n by 6), reading its own bytes."""
    L = np.full((len(P), 10), END, np.int64); L[:, :6] = P; W1T = s.W1.T
    for t in (5, 6, 7, 8):
        hist = np.full((len(P), D), END); hist[:, :t + 1] = L[:, t::-1]; L[:, t + 1] = (sg(W1T[OFF + hist].sum(1) + s.b1) @ s.W2.T + s.b2).argmax(1)
    return L
def well(line): return line[9] == END and all(0 <= line[i] <= 9 for i in (6, 7, 8))

def run(kw):
    t0 = time.process_time(); seed = int(kw.get("seed", 1)); rng = np.random.default_rng([seed, 99]); PP = {}
    for op in "+-": o = rng.permutation(10000); PP[op] = [(int(x) // 100, int(x) % 100) for x in o]
    TAB = {op: np.array([text(op, a, b) for a, b in PP[op][2000:]], np.int32) for op in "+-"}; shown = {op: set(map(tuple, TAB[op].tolist())) for op in "+-"}
    heldA = PP["+"][:500]; wideA = PP["+"][:2000]; cleanS = [p for p in PP["-"][:2000] if tuple(text("-", *p)) not in shown["-"]]; mixedS = PP["-"][:500]
    s = Sys(seed); q = float(eval(kw.get("q", "0.1"))); w = float(eval(kw.get("w", "1"))); start = int(kw.get("start", 0)); ans = kw.get("ans", "relay"); cells = Cells(int(kw.get("K", 8)))
    rnd = random.Random(seed * 1000 + 7); drnd = random.Random(seed * 1000 + 11); every = int(kw.get("every", 50000)); chunk = int(kw.get("chunk", 500)); fast = int(kw.get("fast", 0))
    out = dict(label=kw.get("label"), args=kw, rows=[]); nreal = 0; t_rep = 0.0
    def go(lines):
        lines = np.array(lines, np.int32); xs = lines.ravel(); ys = np.concatenate([lines[:, 1:], np.full((len(lines), 1), -1, np.int32)], 1).ravel()
        if fast: import fastr; fastr.run(s, xs, ys)
        else: s.steps(xs, ys)
    def store(pairs, op):
        """Of these pairs: how many have an answer in the store, and how many of those are right; and how many the relay system has right."""
        Ls = [text(op, a, b) for a, b in pairs]; A = [cells.answer(L[:6]) for L in Ls]; has = sum(a is not None for a in A); right = sum(a is not None and a[6:] == L[6:] for a, L in zip(A, Ls))
        return dict(n=len(pairs), in_store=has, store_right=right, relay=s.score(op, pairs))
    for pi, ph in enumerate(kw.get("plan", "add:400000,sub:200000").split(",")):
        name, n = ph.split(":"); n = int(n); op0 = {"add": "+", "sub": "-"}[name]; it = 0
        while it < n:
            nxt = min((it // every + 1) * every, n) - 1; X = TAB[op0][rng.integers(8000, size=nxt - it + 1)].tolist(); it = nxt + 1
            st = dict(replays=0, replays_wrong=0, replays_add=0, dreams=0, dreams_kept=0, dreams_add=0, dreams_add_right=0, dreams_add_never_shown=0, dreams_add_never_shown_right=0, dreams_sub=0, dreams_sub_right=0, dreams_sub_small_first=0)
            for c0 in range(0, len(X), chunk):
                t1 = time.process_time(); part = X[c0:c0 + chunk]; isd = [q > 0 and nreal + i >= start and drnd.random() < q for i in range(len(part))]; nd = sum(isd)
                pre = [cells.prefix(drnd) for _ in range(nd)]; ok = [p for p in pre if p is not None]; dl = []
                if ok:
                    L = write(s, np.array(ok))
                    if ans == "chance": L[:, 6:9] = np.array([[drnd.randrange(10) for _ in range(3)] for _ in ok]); L[:, 9] = END
                    dl = L.tolist()
                dl = iter(dl); pre = iter(pre); lines = []
                for Lr, d in zip(part, isd):
                    lines.append(Lr); cells.count(Lr); cells.n_real += 1
                    if d:
                        st["dreams"] += 1
                        if next(pre) is None: continue
                        E = next(dl)
                        if not well(E): continue
                        cells.count(E, w); cells.n_dream += 1; st["dreams_kept"] += 1; wf, right, op = kind(E)
                        if op == "+":
                            ns = tuple(text("+", E[0] * 10 + E[1], E[3] * 10 + E[4])) not in shown["+"]
                            st["dreams_add"] += 1; st["dreams_add_right"] += right; st["dreams_add_never_shown"] += ns; st["dreams_add_never_shown_right"] += ns and right
                        elif op == "-": st["dreams_sub"] += 1; st["dreams_sub_right"] += right; st["dreams_sub_small_first"] += E[0] * 10 + E[1] < E[3] * 10 + E[4]
                    else:
                        rnd.random()                  # replay.py draws once here for the fractional part of r; kept so that q = 0 is arm R to the last draw
                        E = cells.replay(rnd); wf, right, op = kind(E); st["replays"] += 1; st["replays_wrong"] += not right; st["replays_add"] += op == "+"
                        if wf: lines.append(E)
                nreal += len(part); t_rep += time.process_time() - t1; go(lines)
            out["rows"].append(dict(phase=pi, name=name, lines=nxt + 1, add=s.score("+", heldA), sub=s.score("-", mixedS), sub_clean_wide=s.score("-", cleanS), **st))
        out[f"store_after_phase_{pi}"] = dict(held_add=store(heldA, "+"), never_shown_add_wide=store(wideA, "+"), clean_sub_wide=store(cleanS, "-"))
    ph = lambda i: [x for x in out["rows"] if x["phase"] == i]; nc, nz = cells.size(); e = ph(1)[-1] if ph(1) else {}
    out.update(add_after_first=float(np.mean([x["add"] for x in ph(0)[-3:]])), add_after_second=e.get("add"), sub_after_second=e.get("sub"), sub_clean_wide_after_second=e.get("sub_clean_wide"), cells=nc, counts_above_zero=nz,
               lines_counted_real=cells.n_real, lines_counted_dreamed=cells.n_dream, cpu=round(time.process_time() - t0, 1), cpu_counting_replay_dreams=round(t_rep, 1)); run.last = s; run.cells = cells
    if not kw.get("nowrite"):
        os.makedirs(os.path.join(HERE, "res"), exist_ok=True); json.dump(out, open(os.path.join(HERE, "res", f"d_{kw['label']}.json"), "w"))
    return out

if __name__ == "__main__":
    kw = dict(a.split("=") for a in sys.argv[2:]); kw["label"] = sys.argv[1]; o = run(kw); f = o.get("store_after_phase_1", o["store_after_phase_0"])["held_add"]
    print(o["label"], f"addition after its phase {o['add_after_first']:.1f}; at the end: addition {o['add_after_second']}, subtraction clean wide {o['sub_clean_wide_after_second']} | store alone on the 500 held additions: {f['in_store']} in the store, {f['store_right']} right |",
          " ".join(f"{x['phase']}:{x['lines']//1000}k {x['add']:.0f}" for x in o["rows"]), "| dreams kept / addition right of addition / wrong replays of replays, by reading:",
          " ".join(f"{x['dreams_kept']}/{x['dreams_add_right']}of{x['dreams_add']}/{x['replays_wrong']}of{x['replays']}" for x in o["rows"]), f"| cells {o['cells']} counts {o['counts_above_zero']} | cpu {o['cpu']} s")
