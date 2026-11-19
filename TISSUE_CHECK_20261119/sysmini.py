"""The whole system in a small case (reviewer's task of 15:08 UTC, on Mr. Xu's terms): receiving cells H1, cells in between H2, an output
part; the output byte and the error go back to the receiving cells at the next byte. Written from the description; readings in
RESULTS.md section 18.
  H1: 64 cells, each with a sensitivity to every value of the things it receives: the text byte (256), with ins >= 2 the last output
      byte (256), with ins = 3 the error (2: right, wrong). At birth 8 entries drawn at random among those it receives, 1/8 each.
      Drive = sum of its sensitivities to the values present; the 8 cells of largest drive above 0 are active, output = drive.
  H2: 64 cells, each with 16 inputs from H1 cells drawn at random, strengths 1/16; drive = sum of strength x H1 output; the 8 of largest
      drive above 0 are active, output = drive.
  Output part: M cells (outpart.py's rules: cells taken in order of their best estimate, speak the best value if best - inhibition >=
      th, with chance eps a seen value at random; rule Q for the speakers on yes or no; th and G change on no); each cell reads
      U = [h2, sqrt(3) s_2, sqrt(7) s_4, 1], s_k += (h2 - s_k)/k once a byte; n = |U|^2. With M = 1: th = 0 fixed, eps 1/16.
      The system's output at byte t is the byte spoken (the first speaker's, if several; none if silent); right = (it equals byte t+1
      and every speaker spoke it).
  Local rule of H1 and H2 (rule=copy): an active cell moves each present input's strength toward 1 by r = 1/16 and rescales its
      strengths to sum 1 (for H1 the present entries of the things it receives; for H2 its inputs from active H1 cells).
  Other rules: see RULES below and section 18.
    python3 sysmini.py <label> seed=.. nbytes=.. ins=1|2|3 gen=.. [rule=copy] [M=1] [eps=1/16] [r=1/16]
"""
import json, sys, time
import numpy as np
sys.path.insert(0, "/home/user/prep_deps")
import life_stream as LS, score_emissions as SE

lab = sys.argv[1]; kw = dict(a.split("=") for a in sys.argv[2:]); t0 = time.time()
fl = lambda k, d: float(eval(kw.get(k, d)))
seed = int(kw["seed"]); NB = int(kw["nbytes"]); ins = int(kw["ins"]); gen = int(kw.get("gen", 0)); rule = kw.get("rule", "copy")
M = int(kw.get("M", 1)); eps = fl("eps", "1/16" if M == 1 else "1/64"); r = fl("r", "1/16"); eta = fl("eta", "1/16")
rg = fl("rg", "1/64"); rt = fl("rt", "1/1024"); th0 = 0.0 if M == 1 else fl("th0", "1/8")
b = np.asarray(LS.life(seed)["bytes"][:NB], dtype=np.int64)
rng = np.random.default_rng([seed, gen])
N1, N2, K = 64, 64, 8
NE = 256 + (256 if ins >= 2 else 0) + (2 if ins == 3 else 0)        # entries a receiving cell has
S1 = np.zeros((N1, NE))
for c in range(N1): S1[c, rng.choice(NE, 8, replace=False)] = 1 / 8
SRC = np.array([rng.choice(N1, 16, replace=False) for _ in range(N2)]); W2 = np.full((N2, 16), 1 / 16)
F = 3 * N2 + 1
W = np.zeros((M, 256, F)); G = np.zeros((M, M)); th = np.full(M, th0)
s2 = np.zeros(N2); s4 = np.zeros(N2)
right = np.zeros(NB, bool); seen = np.zeros(256, bool)
last_out, last_ok = -1, None                                         # nothing spoken before byte 0
# traces for the error rules: what each cell did at the previous byte
prev1 = None; prev2 = None
act1_count = np.zeros(N1); act2_count = np.zeros(N2)
freq1 = np.zeros(NE)                                                 # running presence of each entry (rule 'novel')
stats = dict(h1_text_share=[], silent=0)


def top(d):
    o = np.argsort(-d, kind="stable")[:K]
    return o[d[o] > 0]


for t in range(NB - 1):
    seen[b[t]] = True
    pres = [int(b[t])]
    if ins >= 2 and last_out >= 0: pres.append(256 + last_out)
    if ins == 3 and last_ok is not None: pres.append(512 + (0 if last_ok else 1))
    pres = np.array(pres)
    d1 = S1[:, pres].sum(1); a1 = top(d1); o1 = np.zeros(N1); o1[a1] = d1[a1]
    d2 = (W2 * o1[SRC]).sum(1); a2 = top(d2); o2 = np.zeros(N2); o2[a2] = d2[a2]
    # output part
    s2 += (o2 - s2) / 2; s4 += (o2 - s4) / 4
    u = np.concatenate([o2, np.sqrt(3.0) * s2, np.sqrt(7.0) * s4, np.ones(1)]); n = float(u @ u)
    A = W @ u                                                         # (M, 256)
    bv = A.argmax(1); best = A[np.arange(M), bv]
    order = sorted(range(M), key=lambda j: (-best[j], j)); inhib = np.zeros(M); spoke = []
    sv = np.flatnonzero(seen)
    for j in order:
        if rng.random() < eps: e = int(sv[rng.integers(0, len(sv))])
        elif best[j] - inhib[j] >= th[j]: e = int(bv[j])
        else: continue
        spoke.append((j, e)); inhib += G[:, j]; inhib[j] -= G[j, j]
    ok = len(spoke) > 0 and all(e == b[t + 1] for _, e in spoke)
    right[t + 1] = ok
    for j, e in spoke: W[j, e] += eta * (float(ok) - A[j, e]) / n * u
    if M > 1 and not ok:
        sp = np.zeros(M, bool); sp[[j for j, _ in spoke]] = True
        th = np.clip(th + np.where(sp, rt, -rt), 0.0, 1.0)
        if len(spoke) > 1:
            for x_ in [j for j, _ in spoke]:
                for y_ in [j for j, _ in spoke]:
                    if x_ != y_: G[x_, y_] += rg
    if not spoke: stats["silent"] += 1
    # local rules of H1 and H2
    if rule == "copy":
        for c in a1:
            S1[c, pres] += r * (1.0 - S1[c, pres]); S1[c] /= S1[c].sum()
        for c in a2:
            m = o1[SRC[c]] > 0; W2[c, m] += r * (1.0 - W2[c, m]); W2[c] /= W2[c].sum()
    elif rule == "copytext":
        # the cell copies only the text entry; the output and error entries drive it but are not strengthened
        for c in a1:
            S1[c, pres[:1]] += r * (1.0 - S1[c, pres[:1]]); S1[c] /= S1[c].sum()
        for c in a2:
            m = o1[SRC[c]] > 0; W2[c, m] += r * (1.0 - W2[c, m]); W2[c] /= W2[c].sum()
    elif rule == "novel":
        # the cell strengthens a present input in proportion to how rarely that input is present (a running presence at the synapse,
        # rate 1/1024): an input present at every byte gains nothing
        freq1 *= (1 - 1 / 1024); freq1[pres] += 1 / 1024
        for c in a1:
            S1[c, pres] += r * (1.0 - freq1[pres] * 16).clip(0, 1) * (1.0 - S1[c, pres]); S1[c] /= S1[c].sum()
        for c in a2:
            m = o1[SRC[c]] > 0; W2[c, m] += r * (1.0 - W2[c, m]); W2[c] /= W2[c].sum()
    elif rule == "errgate":
        # the error input as a gate on one kind of synapse: an active receiving cell copies the text entry as before; its entry for the
        # last output moves toward 1 by r if the error input it receives says 'right', toward 0 by r if 'wrong'; error entries are not
        # strengthened (they keep their birth strength, rescaled). The cell needs to know which present input is the error and which
        # is the output (two kinds of synapse). H2 copies as before.
        for c in a1:
            S1[c, pres[:1]] += r * (1.0 - S1[c, pres[:1]])
            if ins == 3 and len(pres) == 3:
                po = pres[1]
                if last_ok: S1[c, po] += r * (1.0 - S1[c, po])
                else: S1[c, po] -= r * S1[c, po]
            S1[c] /= S1[c].sum()
        for c in a2:
            m = o1[SRC[c]] > 0; W2[c, m] += r * (1.0 - W2[c, m]); W2[c] /= W2[c].sum()
    elif rule in ("tag", "tagH1"):
        # three-factor rule with a tag: an active cell tags its present inputs (the change it would make by copying) but does not make
        # it; at the next byte the error arrives as an input; a receiving cell that receives 'right' makes its tagged change, one that
        # receives 'wrong' makes the opposite change (moves the tagged strengths toward 0 by r). H2 cells (rule 'tag'): an H2 cell
        # takes the error from its inputs: it makes its tagged change if the active H1 cells among its inputs are, by their own
        # sensitivity, more sensitive to 'right' than to 'wrong' at this byte, the opposite if the reverse, nothing if equal.
        # The cell needs its own tag from the previous byte and the error as an input (receiving cells) or through its inputs (H2).
        if prev1 is not None and last_ok is not None:
            sgn1 = 1.0 if last_ok else -1.0
            for c, pp in prev1:
                if sgn1 > 0: S1[c, pp] += r * (1.0 - S1[c, pp])
                else: S1[c, pp] -= r * S1[c, pp]
                S1[c] /= S1[c].sum()
            if rule == "tag" and prev2 is not None:
                for c, m in prev2:
                    act_in = SRC[c][o1[SRC[c]] > 0]
                    ev = (S1[act_in, 512] - S1[act_in, 513]).sum() if ins == 3 else 0.0
                    if ev > 0: W2[c, m] += r * (1.0 - W2[c, m])
                    elif ev < 0: W2[c, m] -= r * W2[c, m]
                    W2[c] /= W2[c].sum()
            elif rule == "tagH1":
                for c in a2:
                    m = o1[SRC[c]] > 0; W2[c, m] += r * (1.0 - W2[c, m]); W2[c] /= W2[c].sum()
        prev1 = [(c, pres[:1].copy() if ins < 3 else pres[pres < 512].copy()) for c in a1]
        prev2 = [(c, (o1[SRC[c]] > 0).copy()) for c in a2]
    last_out = spoke[0][1] if spoke else -1; last_ok = ok
refs = SE.references(b)
Q = [(max(1, i * NB // 4), (i + 1) * NB // 4) for i in range(4)]
out = dict(label=lab, args=kw, quarters=[float(right[a:e].mean()) for a, e in Q],
           refs_last_quarter={k: float((refs[k] == b)[Q[3][0]:].mean()) for k in ("ref_1", "ref_2", "ref_3")},
           h1_mass=dict(text=float(S1[:, :256].sum() / N1), output=float(S1[:, 256:512].sum() / N1) if ins >= 2 else 0.0,
                        error=float(S1[:, 512:].sum() / N1) if ins == 3 else 0.0),
           silent=stats["silent"] / (NB - 1), seconds=time.time() - t0)
import os; os.makedirs("/home/user/tissue_check/out/sys", exist_ok=True)
json.dump(out, open(f"/home/user/tissue_check/out/sys/{lab}.json", "w"), indent=0)
print(lab, "right by quarter", " ".join(f"{100*x:.1f}" for x in out["quarters"]), "| H1 mass", {k: round(v, 3) for k, v in out["h1_mass"].items()},
      "| refs", {k: round(100 * v, 2) for k, v in out["refs_last_quarter"].items()}, f"| {out['seconds']:.0f} s", flush=True)
