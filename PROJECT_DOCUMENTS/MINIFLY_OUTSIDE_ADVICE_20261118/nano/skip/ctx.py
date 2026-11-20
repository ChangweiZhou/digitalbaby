#!/usr/bin/env python3
"""PLAN_9.md: the bytes to pass over are not a set of byte values. Text h2: aa+<0 to 4 fillers>bb=ccc, a filler being a letter followed by a digit
drawn by chance, so a digit matters in one place and not in another. The system and kernel are those of skip.py; this file has the text, the
clocks and the scoring (numpy for everything but the kernel's inner loop).

    python3 ctx.py <label> seed=1 lines=2400000 every=200000 clock=none|oracle|trial key=value|pair [alpha=2 p0=0.2 P=256 eps=0.01 win=3000]

clock=oracle   the letters and the digit after each letter are passed over (told by the program: a check).
clock=trial    the trial clock of PLAN_2.md, second form, with one difference from skip.py: an entry's evidence changes only in a period in which
               its byte came. key=value: one entry for each byte value (18). key=pair: one entry for each byte value after each byte value (324):
               the entry of the byte v that arrived after the byte p decides whether that v enters the chain.
"""
import json, os, sys, time
import numpy as np
import skip
from skip import sg

def fills(a, b, k): return "".join(chr(skip.LET[(7 * a + 3 * b + 5 * i * i + i) % 4]) + str((a + 2 * b + 3 * i) % 10) for i in range(k))      # the fillers used when scoring: fixed by the pair
def text(a, b, k): return [skip.SI[c] for c in f"{a:02d}+{fills(a, b, k)}{b:02d}={a + b:03d}\n".encode()]
def stream(shown, n, rng):
    P = np.asarray(shown)[rng.integers(len(shown), size=n)]; a, b = P[:, 0], P[:, 1]; k = rng.integers(5, size=n); c = a + b
    F = np.empty((n, 8), np.int64); F[:, 0::2] = np.asarray(skip.LETS)[rng.integers(4, size=(n, 4))]; F[:, 1::2] = rng.integers(10, size=(n, 4)); F = np.where(np.arange(8)[None, :] < 2 * k[:, None], F, -1)
    col = lambda v: np.asarray(v)[:, None]; const = lambda ch: np.full((n, 1), skip.SI[ord(ch)])
    M = np.concatenate([col(a // 10), col(a % 10), const("+"), F, col(b // 10), col(b % 10), const("="), col(c // 100), col(c // 10 % 10), col(c % 10), const("\n")], 1)
    xs = M[M >= 0].astype(np.int32); ys = np.empty_like(xs); ys[:-1] = xs[1:]; ys[-1] = -1; ys[xs == skip.END] = -1; return xs, ys

def run(kw):
    t0 = time.process_time(); S = skip.S; seed = int(kw.get("seed", 1)); lines = int(kw.get("lines", 2400000)); every = int(kw.get("every", 200000)); ck = kw.get("clock", "none"); key = kw.get("key", "value")
    rng = np.random.default_rng([seed, 99]); o = rng.permutation(10000); Pp = [(int(q) // 100, int(q) % 100) for q in o]; held, shown = Pp[:500], Pp[2000:]
    s = skip.Sk(seed); s.fast = int(kw.get("fast", 1)); crng = np.random.default_rng([seed, 20261125]); isl = np.zeros(S, bool); isl[skip.LETS] = True
    P = int(kw.get("P", 256)); alpha = float(kw.get("alpha", 2)); eps = float(kw.get("eps", 0.01)); beta = 0.02; p0 = float(kw.get("p0", 0.2)); gam = 1 - 1 / float(kw.get("win", 3000)); u0 = np.log(p0 / (1 - p0))
    closedM = np.zeros((S, S), bool)                                       # closedM[p, v]: the byte v after the byte p is passed over (the clock as it stands; used for scoring)
    if ck == "oracle": closedM[:, skip.LETS] = True; closedM[skip.LETS, :10] = True
    shape = (S, S) if key == "pair" else (S,); u = np.full(shape, u0); ev = np.zeros((4,) + shape); tr = dict(b=None, w=None); seen = np.zeros((S, S))
    def full(c): return c if key == "pair" else np.broadcast_to(c[None, :], (S, S))
    def score(gapk):
        hists = []; want = []; own = []
        for k, (a, b) in enumerate(held):
            L = text(a, b, gapk); eq = L.index(skip.EQ); hist = list(s.hist); pend = bool(s.pend)
            for t in range(len(L) - 1):
                p = hist[0]
                if pend: hist = [L[t], p] + hist[1:-1]
                else: hist[0] = L[t]
                pend = not closedM[p, L[t]]
                if t >= eq and L[t + 1] != skip.END: hists.append(list(hist)); want.append(L[t + 1]); own.append((k, t - eq))
        hists = np.array(hists); out = sg(s.W1.T[skip.OFF + hists].sum(1) + s.b1) @ s.W2.T + s.b2; r = out.argmax(1) == np.array(want); ok = np.ones(len(held), bool); pl = np.zeros(3)
        for (k, j), rr in zip(own, r): ok[k] &= bool(rr); pl[j] += rr
        return float(100 * ok.mean()), [int(round(100 * x / len(held))) for x in pl]
    out = dict(label=kw["label"], args=kw, rows=[]); done = 0
    while done < lines:
        n = min(every, lines - done); xs, ys = stream(shown, n, rng)
        if ck == "trial":
            for i in range(0, len(xs), P):
                c = np.clip(sg(u), eps, 1 - eps); closed = crng.random(shape) < c; x = xs[i:i + P]; xp = np.concatenate([[s.hist[0]], x[:-1]]); s.esum = 0.0; s.ecnt = 0
                s.run(x, ys[i:i + P], ~full(closed)[xp, x]); occ2 = np.zeros((S, S)); np.add.at(occ2, (xp, x), 1); seen += occ2
                if s.ecnt < P // 4: continue
                r = s.esum / s.ecnt
                if tr["b"] is None: tr["b"] = r; tr["w"] = 0.05 * r; continue
                z = float(np.clip((tr["b"] - r) / max(tr["w"], 1e-12), -3, 3)); occ = (occ2 > 0) if key == "pair" else (occ2.sum(0) > 0)        # an entry whose byte did not come in the period learns nothing from it
                ev *= np.where(occ, gam, 1.0); ev[0] += z * (closed & occ); ev[1] += closed & occ; ev[2] += z * (~closed & occ); ev[3] += ~closed & occ
                u = u0 + alpha * (ev[0] / (ev[1] + 1) - ev[2] / (ev[3] + 1)) / np.sqrt(1 / (ev[1] + 1) + 1 / (ev[3] + 1))
                tr["w"] += beta * (abs(r - tr["b"]) - tr["w"]); tr["b"] += beta * (r - tr["b"])
            closedM = np.array(full(u > 0))
        else:
            xp = np.concatenate([[s.hist[0]], xs[:-1]]); s.run(xs, ys, ~closedM[xp, xs] if ck == "oracle" else None); np.add.at(seen, (xp, xs), 1)
        done += n; row = dict(lines=done, never_by_fillers={}, place={})
        for g in (0, 2, 4): row["never_by_fillers"][g], row["place"][g] = score(g)
        oc = seen > 0; dig = np.zeros(S, bool); dig[:10] = True; fd = oc & isl[:, None] & dig[None, :]; mk = oc & isl[None, :]; ot = oc & ~fd & ~mk; nm = lambda v: "nl" if skip.SYMS[v] == 10 else chr(skip.SYMS[v])
        row["filler_digits_closed"] = [int((closedM & fd).sum()), int(fd.sum())]; row["letters_closed"] = [int((closedM & mk).sum()), int(mk.sum())]
        row["others_closed"] = [nm(p) + nm(v) for p, v in zip(*np.nonzero(closedM & ot))] if key == "pair" or ck == "oracle" else sorted({nm(v) for p, v in zip(*np.nonzero(closedM & ot))}); row["others_that_occur"] = int(ot.sum())
        out["rows"].append(row)
    out["last3_by_fillers"] = {g: float(np.mean([r["never_by_fillers"][g] for r in out["rows"][-3:]])) for g in (0, 2, 4)}; out["cpu"] = round(time.process_time() - t0, 1); run.last = s
    if not kw.get("nowrite"): json.dump(out, open(os.path.join(skip.HERE, "res", f"{kw['label']}.json"), "w"))
    return out

if __name__ == "__main__":
    kw = dict(a.split("=") for a in sys.argv[2:]); kw["label"] = sys.argv[1]; o = run(kw); r = o["rows"][-1]
    print(o["label"], "| last3 by fillers", {g: round(v, 1) for g, v in o["last3_by_fillers"].items()}, "|", " ".join(f"{q['lines']//1000}k " + "/".join(f"{v:.0f}" for v in q["never_by_fillers"].values()) for q in o["rows"]),
          "| place", r["place"], "| closed (filler digits, letters; of those that occur):", [(q["filler_digits_closed"][0], q["letters_closed"][0], len(q["others_closed"])) for q in o["rows"]], "of", r["filler_digits_closed"][1], r["letters_closed"][1], r["others_that_occur"], "| others closed at the end", r["others_closed"][:30], "|", o["cpu"], "s")
