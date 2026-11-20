#!/usr/bin/env python3
"""PLAN_9.md: the bytes to pass over are not a set of byte values. Text h2: aa+<0 to 4 fillers>bb=ccc, a filler being a letter followed by a digit
drawn by chance, so a digit matters in one place and not in another. The system and kernel are those of skip.py; this file has the text, the
clocks and the scoring (numpy for everything but the kernel's inner loop).

    python3 ctx.py <label> seed=1 lines=2400000 every=200000 clock=none|oracle|trial key=value|pair [alpha=2 p0=0.2 P=256 eps=0.01 win=3000] [listen=T] [text=h2|g2]

clock=oracle   the letters and the digit after each letter are passed over (told by the program: a check).
clock=trial    the trial clock of PLAN_2.md, second form, with one difference from skip.py: an entry's evidence changes only in a period in which
               its byte came. key=value: one entry for each byte value (18). key=pair: one entry for each byte value after each byte value (324):
               the entry of the byte v that arrived after the byte p decides whether that v enters the chain.
listen=T       (addition 1 to PLAN_9.md) an entry draws its state anew at each of its bytes, and listens to the number sent out by the answering cell
               only over the T bytes that follow that byte: the mean over those bytes is what it enters into its closed or its open mean. The
               tendencies are brought up to date every 2,048 bytes; a byte less than T/2 named bytes from the end of such a stretch is not counted.
bit=1          (addition 2 to PLAN_9.md; with key=pair and listen) an entry has two tendencies and two sets of evidence: one for when the byte
               before its own entered the chain, one for when that byte was passed over.
text=g2        the text of skip.py (letters only), for a check of the listening form where the plain form is known to work.
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
    tx = kw.get("text", "h2"); T = int(kw.get("listen", 0)); mk_text = text if tx == "h2" else (lambda a, b, g: skip.text("g2", a, b, skip.fill(a, b, 2 * g))); mk_stream = stream if tx == "h2" else (lambda sh, n, r: skip.stream("g2", sh, 8, n, r))
    rng = np.random.default_rng([seed, 99]); o = rng.permutation(10000); Pp = [(int(q) // 100, int(q) % 100) for q in o]; held, shown = Pp[:500], Pp[2000:]
    s = skip.Sk(seed); s.fast = int(kw.get("fast", 1)); crng = np.random.default_rng([seed, 20261125]); isl = np.zeros(S, bool); isl[skip.LETS] = True
    P = int(kw.get("P", 256)); alpha = float(kw.get("alpha", 2)); eps = float(kw.get("eps", 0.01)); beta = 0.02; p0 = float(kw.get("p0", 0.2)); gam = 1 - 1 / float(kw.get("win", 3000)); u0 = np.log(p0 / (1 - p0))
    closedM = np.zeros((S, S), bool)                                       # closedM[p, v]: the byte v after the byte p is passed over (the clock as it stands; used for scoring)
    if ck == "oracle": closedM[:, skip.LETS] = True; closedM[skip.LETS, :10] = True
    bit = int(kw.get("bit", 0)); assert not bit or (key == "pair" and T)
    closedB = np.zeros((2, S, S), bool); prevclosed = False               # with bit=1: closedB[1 if the byte before was passed over else 0, p, v]
    shape = (2, S, S) if bit else (S, S) if key == "pair" else (S,); u = np.full(shape, u0); ev = np.zeros((4,) + shape); tr = dict(b=None, w=None); seen = np.zeros((S, S))
    def full(c): return c if key == "pair" else np.broadcast_to(c[None, :], (S, S))
    def score(gapk):
        hists = []; want = []; own = []
        for k, (a, b) in enumerate(held):
            L = mk_text(a, b, gapk); eq = L.index(skip.EQ); hist = list(s.hist); pend = bool(s.pend)
            for t in range(len(L) - 1):
                p = hist[0]
                if pend: hist = [L[t], p] + hist[1:-1]
                else: hist[0] = L[t]
                pend = not (closedB[0 if pend else 1, p, L[t]] if bit else closedM[p, L[t]])
                if t >= eq and L[t + 1] != skip.END: hists.append(list(hist)); want.append(L[t + 1]); own.append((k, t - eq))
        hists = np.array(hists); out = sg(s.W1.T[skip.OFF + hists].sum(1) + s.b1) @ s.W2.T + s.b2; r = out.argmax(1) == np.array(want); ok = np.ones(len(held), bool); pl = np.zeros(3)
        for (k, j), rr in zip(own, r): ok[k] &= bool(rr); pl[j] += rr
        return float(100 * ok.mean()), [int(round(100 * x / len(held))) for x in pl]
    out = dict(label=kw["label"], args=kw, rows=[]); done = 0
    while done < lines:
        n = min(every, lines - done); xs, ys = mk_stream(shown, n, rng)
        if ck == "trial" and T:
            CH = 2048; gc = gam ** (CH / P)
            for i in range(0, len(xs), CH):
                c = np.clip(sg(u), eps, 1 - eps); x = xs[i:i + CH]; xp = np.concatenate([[s.hist[0]], x[:-1]]); kidx = (xp, x) if key == "pair" else (x,)
                if bit:                                   # each byte's state depends on whether the byte before it was passed over: a chain of two-state steps, worked out for the whole stretch at once
                    rr = crng.random(len(x)); c0 = rr < c[0][kidx]; c1 = rr < c[1][kidx]; const = c0 == c1; j0 = np.arange(len(x))
                    last = np.maximum.accumulate(np.where(const, j0, -1)); base = np.where(last >= 0, c0[np.maximum(last, 0)], prevclosed)
                    # between resets the state is copied (c1 and not c0) or flipped (c0 and not c1) from the byte before
                    flip = (~const & c0).astype(np.int64); cf = np.cumsum(flip); par = (cf - np.where(last >= 0, cf[np.maximum(last, 0)], 0)) % 2
                    closed = base ^ (par == 1); pb = np.concatenate([[prevclosed], closed[:-1]]).astype(np.int64); prevclosed = bool(closed[-1]); kidx = (pb, xp, x)
                else: closed = crng.random(len(x)) < c[kidx]
                s.eout = []; s.run(x, ys[i:i + CH], ~closed); e = np.array(s.eout); s.eout = None; named = (ys[i:i + CH] >= 0).astype(float); np.add.at(seen, (xp, x), 1)
                ce = np.concatenate([[0.0], np.cumsum(e)]); cn = np.concatenate([[0.0], np.cumsum(named)]); j = np.arange(len(x)); hi = np.minimum(j + 1 + T, len(x)); cnt = cn[hi] - cn[j + 1]; ok = cnt >= T / 2
                if not ok.any(): continue
                W = (ce[hi] - ce[j + 1]) / np.maximum(cnt, 1)                  # the mean mismatch over the T bytes after each byte
                if tr["b"] is None: tr["b"] = float(W[ok].mean()); tr["w"] = float(np.abs(W[ok] - tr["b"]).mean()) + 1e-12
                z = np.clip((tr["b"] - W) / tr["w"], -6, 6); kk = tuple(q[ok] for q in kidx); cl = closed[ok]; zz = z[ok]; occ = np.zeros(shape, bool); occ[kk] = True
                ev *= np.where(occ, gc, 1.0); np.add.at(ev[0], kk, zz * cl); np.add.at(ev[1], kk, cl * 1.0); np.add.at(ev[2], kk, zz * ~cl); np.add.at(ev[3], kk, ~cl * 1.0)
                u = u0 + alpha * (ev[0] / (ev[1] + 1) - ev[2] / (ev[3] + 1)) / np.sqrt(1 / (ev[1] + 1) + 1 / (ev[3] + 1))
                tr["w"] += beta * (float(np.abs(W[ok] - tr["b"]).mean()) - tr["w"]); tr["b"] += beta * (float(W[ok].mean()) - tr["b"])
            if bit: closedB = u > 0; closedM = closedB[0]
            else: closedM = np.array(full(u > 0))
        elif ck == "trial":
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
        if ck == "trial":                                                  # the evidence (u less its start, over alpha), mean over the entries of each kind that occur
            tt = np.array(full(((u[0] if bit else u) - u0) / max(alpha, 1e-12))); row["evidence_mean"] = dict(filler_digits=float(tt[fd].mean()) if fd.any() else None, letters=float(tt[mk].mean()), others=float(tt[ot].mean()), others_max=float(tt[ot].max()))
        if bit:
            row["after_a_byte_passed_over"] = dict(filler_digits_closed=int((closedB[1] & fd).sum()), letters_closed=int((closedB[1] & mk).sum()), others_closed=[nm(p) + nm(v) for p, v in zip(*np.nonzero(closedB[1] & ot))])
        out["rows"].append(row)
    out["last3_by_fillers"] = {g: float(np.mean([r["never_by_fillers"][g] for r in out["rows"][-3:]])) for g in (0, 2, 4)}; out["cpu"] = round(time.process_time() - t0, 1); run.last = s
    if not kw.get("nowrite"): json.dump(out, open(os.path.join(skip.HERE, "res", f"{kw['label']}.json"), "w"))
    return out

def crit(seed=1):
    """Addition 1 to PLAN_9.md: what the expectation rule of PLAN_7.md (next two bytes, table counting across the end mark) would pass over on h2, by kind of byte."""
    import expect
    rng = np.random.default_rng([seed, 99]); o = rng.permutation(10000); Pp = [(int(q) // 100, int(q) % 100) for q in o]; shown = Pp[2000:]; out = {}
    names = ["a digit 1", "a digit 2", "+", "filler letter", "filler digit", "b digit 1", "b digit 2", "=", "c digit 1", "c digit 2", "c digit 3", "end mark"]
    def kinds(xs):
        k = np.zeros(len(xs), np.int64); st = 0
        for i, x in enumerate(xs):
            if x == skip.END: k[i] = 11; st = 0
            elif x in skip.LETS: k[i] = 3
            elif i and xs[i - 1] in skip.LETS: k[i] = 4
            elif st < 2: k[i] = st; st += 1
            elif st == 2: k[i] = 2; st += 1
            else: k[i] = st + 2; st += 1
        return k
    for theta in (0.2, 0.3, 0.5):
        T = expect.Table(theta=theta, across=1); T.two = 1; pre = np.full(expect.K, skip.END); r = np.random.default_rng([seed, 99, 1])
        for _ in range(200): xs, ys = stream(shown, 2000, r); T.count(xs, ys, pre); pre = xs[-expect.K:]
        xs, ys = stream(shown, 3000, np.random.default_rng([seed, 557])); ent, tv = T.enters(xs, np.full(expect.K, skip.END)); kd = kinds(xs)
        out[str(theta)] = [float((~ent)[kd == k].mean()) for k in range(12)]; print(f"theta {theta}: share passed over after 400,000 lines: " + "  ".join(f"{n}: {v:.2f}" for n, v in zip(names, out[str(theta)])))
        if theta == 0.2: print("   the total variation itself, median by kind: " + "  ".join(f"{n}: {np.nanmedian(tv[kd == k]):.2f}" for k, n in enumerate(names)))
    json.dump(dict(kinds=names, share_passed_over=out), open(os.path.join(skip.HERE, "res", f"p9_crit_h2_s{seed}.json"), "w"))

if __name__ == "__main__":
    if sys.argv[1] == "crit": crit(); sys.exit()
    kw = dict(a.split("=") for a in sys.argv[2:]); kw["label"] = sys.argv[1]; o = run(kw); r = o["rows"][-1]
    print(o["label"], "| last3 by fillers", {g: round(v, 1) for g, v in o["last3_by_fillers"].items()}, "|", " ".join(f"{q['lines']//1000}k " + "/".join(f"{v:.0f}" for v in q["never_by_fillers"].values()) for q in o["rows"]),
          "| place", r["place"], "| closed (filler digits, letters; of those that occur):", [(q["filler_digits_closed"][0], q["letters_closed"][0], len(q["others_closed"])) for q in o["rows"]], "of", r["filler_digits_closed"][1], r["letters_closed"][1], r["others_that_occur"], "| others closed at the end", r["others_closed"][:30], ("| after a byte passed over: " + str([(q["after_a_byte_passed_over"]["filler_digits_closed"], q["after_a_byte_passed_over"]["letters_closed"], len(q["after_a_byte_passed_over"]["others_closed"])) for q in o["rows"]]) + " " + str(r["after_a_byte_passed_over"]["others_closed"][:30])) if "after_a_byte_passed_over" in r else "", "|", o["cpu"], "s")
