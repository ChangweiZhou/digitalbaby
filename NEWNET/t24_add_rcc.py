"""t24_add_rcc.py (RCCARRY_PLAN.md): t24_add.py with the net imported by module name (net=net13 / net13rcc / store) and a
diagnostic of the rule cells against the true carry (read only; it changes nothing in the net). Original docstring:
Addition in one life, column by column (22 November 2026). Two numbers of n digits, lowest digit first; each column is written as
  fmt=col  : a_i b_i s_i            (s_i = (a_i + b_i + carry) mod 10; the carry is not written: it must be inferred - it is fixed by
                                     the column before: carry = 1 iff a + b >= s + 9 there - an intermediate quantity)
  fmt=colc : c_i a_i b_i s_i        (c_i, the carry into column i, is written first: every answer byte is a function of at most
                                     3-4 bytes at fixed lags - a procedure put outside, as a child writes the carry)
and a last column with a = b = 0 for the top digit; "\n" ends the line. Answer bytes (predicted, then read): s_i, and c_i in colc
(c_0 = 0 is not scored). Life: N0 lines with n in 1..4, then N1 lines with n in 1..8 (lengths 5-8 are new). Every pair is almost
surely new, so every answer byte is a first encounter.
Scores (prequential, learning on, the life's own answers): right share of s and c bytes by layer - old length / new length, and
carry in 0 / 1 - and the share of lines with every answer byte right. Free test every PROBE lines (learning off): 50 fresh pairs at
each length 1..8, the structure's own answer bytes fed back (its output is its input), whole line right.
Baseline net=store: the back-off n-gram table (orders 1-8). Usage: python3 t24_add_rcc.py net=net13|net13rcc|net12|store fmt=col|colc [struct seed N0 N1 ...]"""
import sys, json, time, numpy as np
from collections import defaultdict
def num(v):
    for f in (int, float):
        try: return f(v)
        except ValueError: pass
    return v
a = {k: num(v) for k, v in (x.split("=", 1) for x in sys.argv[1:])}
netn, sname, seed, fmt = a.pop("net", 10), a.pop("struct", "K8"), a.pop("seed", 1), a.pop("fmt", "col")
N0, N1, PROBE, out = a.pop("N0", 8000), a.pop("N1", 8000), a.pop("PROBE", 4000), a.pop("out", None)
STRUCT = dict(K8=dict(L=16, cC=64, joint=1, mesh=1, mesh_base=0, mesh_dag=5, c_k_win=4, c_th_on=0.9, c_th_match=0.75),
              N1=dict(L=8, cC=64, M=32, joint=1, mesh=2, mesh_base=0, mesh_dag=2, c_k_win=4))
rng, prng = np.random.default_rng(seed + 4000), np.random.default_rng(seed + 6000)
D = [ord(c) for c in "0123456789"]

class Store:
    def __init__(s, K=8): s.K, s.t, s.h = K, defaultdict(lambda: defaultdict(int)), []
    def probs(s):
        for k in range(min(s.K, len(s.h)), 0, -1):
            c = s.t.get(tuple(s.h[-k:]))
            if c:
                p = np.zeros(128)
                for b, n in c.items(): p[b] = n
                return p
        return np.ones(128)
    def step(s, b, learn=True):
        if learn:
            for k in range(1, min(s.K, len(s.h)) + 1): s.t[tuple(s.h[-k:])][b] += 1
        s.h = (s.h + [b])[-s.K:]
if netn == "store": net = Store()
else:
    import importlib
    opts = dict(STRUCT[sname]); opts.update(a); net = importlib.import_module(str(netn)).Tissue(seed=seed, n_values=128, **opts)
HASRC = hasattr(net, "sn"); DIAG = []                                # (phase, new, column, carry in, rule-cell state, winner at s before)
def guess():
    if netn == "store": return int(np.argmax(net.probs()))
    return net.answer()

def line(n, r):
    """the line's bytes and, for each, (kind, carry in) for answer bytes or None"""
    x, y = r.integers(0, 10, n).tolist() + [0], r.integers(0, 10, n).tolist() + [0]; c = 0; bs, tags = [], []
    for i in range(n + 1):
        s = (x[i] + y[i] + c) % 10
        if fmt == "colc": bs.append(D[c]); tags.append(("c", c) if i else None)
        bs += [D[x[i]], D[y[i]], D[s]]; tags += [None, None, ("s", c)]
        c = (x[i] + y[i] + c) // 10
    return bs + [10], tags + [None]

R = defaultdict(lambda: [0, 0]); LINES = defaultdict(lambda: [0, 0]); probes = []; t0 = time.time()
def free_test(ln):
    r = {}
    for n in range(1, 9):
        ok = 0
        for _ in range(50):
            bs, tags = line(n, prng); good = True
            for b, tg in zip(bs, tags):
                if tg is not None:
                    g = guess(); good &= g == b; net.step(g, False)          # its own answer byte is its input
                else: net.step(b, False)
            ok += good
        r[n] = round(ok / 50, 2)
    probes.append(dict(line=ln, **{str(k): v for k, v in r.items()})); print(json.dumps(probes[-1]), flush=True)
for ln in range(N0 + N1):
    n = int(rng.integers(1, 5)) if ln < N0 else int(rng.integers(1, 9)); new = "new" if n > 4 else "old"
    bs, tags = line(n, rng); good = True; swin = -1; col = 0
    for b, tg in zip(bs, tags):
        if HASRC and tg is not None and tg[0] == "s" and col >= 1:
            DIAG.append((int(ln >= N0), int(n > 4), col, tg[1], net.sn.copy(), swin))
        g = guess()
        if tg is not None:
            ok = g == b; good &= ok; rr = R[(ln >= N0, tg[0], new, tg[1])]; rr[0] += 1; rr[1] += ok
        net.step(b, True)
        if tg is not None and tg[0] == "s": swin = net.rwin if HASRC else -1; col += 1
    lr = LINES[(ln >= N0, new)]; lr[0] += 1; lr[1] += good
    if (ln + 1) % PROBE == 0: free_test(ln + 1)
    if (ln + 1) % 2000 == 0:
        print(f"line {ln + 1} {time.time() - t0:.0f}s " + " ".join(f"P{int(p)}{k}{l}c{c}:{v[1] / max(v[0], 1):.2f}" for (p, k, l, c), v in sorted(R.items())), flush=True)
res = dict(net=netn, struct=sname if netn != "store" else None, fmt=fmt, seed=seed, opts=a, N0=N0, N1=N1, s=round(time.time() - t0),
           bytes={f"P{int(p)}_{k}_{l}_carry{c}": dict(n=v[0], right=round(v[1] / max(v[0], 1), 3)) for (p, k, l, c), v in sorted(R.items())},
           lines={f"P{int(p)}_{l}": dict(n=v[0], right=round(v[1] / max(v[0], 1), 3)) for (p, l), v in sorted(LINES.items())}, free=probes)
def carry_diag():
    """per rule cell: Pearson corr of its activity (normalised, at the moment s_i is predicted) with the true carry into column i
    (columns i >= 1); persistence: share of s bytes at which the cell that won at s_(i-1) is still active; purity of the s_(i-1)
    winner as a carry code (each winner's majority carry on the first half, scored on the second half) against the base rate;
    and a held-out linear read of the carry from the state (fit on the first half by least squares, scored on the second half;
    a diagnostic probe, not part of the net)."""
    out = {}
    for name, sel in (("P1", lambda d: d[0] == 1), ("P1_new", lambda d: d[0] == 1 and d[1] == 1), ("P0", lambda d: d[0] == 0)):
        D = [d for d in DIAG if sel(d)]
        if len(D) < 50: continue
        S = np.array([d[4] for d in D]); c = np.array([d[3] for d in D], float); w = np.array([d[5] for d in D])
        sd = S.std(0); cs = c.std(); cor = np.where(sd > 0, ((S - S.mean(0)) * (c - c.mean())[:, None]).mean(0) / (sd * cs + 1e-12), 0.0)
        order = np.argsort(-np.abs(cor))
        pers = float(np.mean([d[4][d[5]] > 0 for d in D if d[5] >= 0]))
        h = len(D) // 2; maj = float(c[:h].mean() > 0.5); guessw = {}
        for r in np.unique(w[:h]): guessw[r] = float(c[:h][w[:h] == r].mean() > 0.5)
        pur = sum(guessw.get(r, maj) == ci for r, ci in zip(w[h:], c[h:])) * len(D) / (len(D) - h)
        X = np.c_[S, np.ones(len(D))]
        beta = np.linalg.lstsq(X[:h], c[:h], rcond=None)[0]; lin = float(np.mean((X[h:] @ beta > 0.5) == (c[h:] > 0.5)))
        out[name] = dict(n=len(D), carry_rate=round(float(c.mean()), 3), max_abs_corr=round(float(abs(cor[order[0]])), 3),
                         top_corr={int(r): round(float(cor[r]), 3) for r in order[:5]}, cells_abs_corr_over_0_2=int((np.abs(cor) > 0.2).sum()),
                         all_corr=[round(float(v), 3) for v in cor],
                         persist_prev_s_winner=round(pers, 3), winner_purity=round(pur / len(D), 3), base_rate=round(float(max(c.mean(), 1 - c.mean())), 3),
                         n_winner_cells=int(len(np.unique(w))), heldout_linear=round(lin, 3))
    return out
if HASRC: res["carry_diag"] = carry_diag(); res["nrecruit"] = int(net.nrecruit)
print(json.dumps(res), flush=True)
if out: open(out, "a").write(json.dumps(res) + "\n")
