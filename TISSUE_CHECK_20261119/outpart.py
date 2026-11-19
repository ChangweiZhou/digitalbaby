"""An output part of M cells on exact inputs (reviewer's description of 14:42 UTC, after Mr. Xu's ruling that the 256 output lines are
withdrawn). Readings O1-O8 of the places the description leaves open are in RESULTS.md section 17.
Inputs U = [one-hot of the present byte, sqrt(3) s_2, sqrt(7) s_4, 1] with s_k += (onehot - s_k)/k once a byte before use
(+ with fb=1 a one-hot of every byte spoken at the previous byte); with half=1 each cell sees a fixed random half of the 768 byte
features (default_rng(seed + 1)), the constant always; n_j = |U_j|^2.
Cell j's estimate of value v: a_j[v] = W_j[v] U_j. At each byte the cells are taken in order of their best estimate (max over v, ties to
the lower value; ties between cells to the lower index). In its turn a cell draws: with chance eps it speaks a seen value at random;
otherwise it speaks its best value if best - (inhibition received from cells that already spoke) >= th_j. A speaker adds G[k, j] to
the inhibition of every other cell k. Yes if at least one cell spoke and every spoken byte is the next byte. Every speaker:
W_j[e] += eta (ok - a_j[e]) U_j / n_j. On no (if thr=1): th_j -= r_t for silent cells, += r_t for speakers, clipped to [0, 1]; and if
more than one spoke (if inh=1), G[j, k] and G[k, j] += r_g for every pair of speakers.

    python3 outpart.py <label> seed=.. nbytes=.. M=.. eps=.. [rg=1/64] [th0=1/8] [thfix=0] [thr=1] [inh=1] [half=0] [fb=0]
           [eta=1/16] [rt=1/1024]
"""
import json, sys, time
import numpy as np
sys.path.insert(0, "/home/user/prep_deps")
import life_stream as LS, score_emissions as SE

lab = sys.argv[1]; kw = dict(a.split("=") for a in sys.argv[2:]); t0 = time.time()
f = lambda k, d: float(eval(kw.get(k, d)))
seed = int(kw["seed"]); NB = int(kw["nbytes"]); M = int(kw["M"]); eps = f("eps", "1/16"); rg = f("rg", "1/64"); th0 = f("th0", "1/8")
thfix = kw.get("thfix", "0") == "1"; thr = kw.get("thr", "1") == "1" and not thfix; inh = kw.get("inh", "1") == "1"
half = kw.get("half", "0") == "1"; fb = kw.get("fb", "0") == "1"; eta = f("eta", "1/16"); rt = f("rt", "1/1024")
b = np.asarray(LS.life(seed)["bytes"][:NB], dtype=np.int64)
F = 768 + (256 if fb else 0) + 1
W = np.zeros((M, 256, F)); G = np.zeros((M, M)); th = np.full(M, th0)
mask = np.ones((M, F))
if half:
    mr = np.random.default_rng(seed + 1)
    for j in range(M):
        mask[j, :768] = 0.0; mask[j, mr.choice(768, 384, replace=False)] = 1.0
rng = np.random.default_rng(int(kw.get("gen", seed)))
s2 = np.zeros(256); s4 = np.zeros(256); last_spoken = []
yes = np.zeros(NB, bool); nspeak = np.zeros(NB, np.int16); seen = np.zeros(256, bool)
for t in range(NB - 1):
    oh = np.zeros(256); oh[b[t]] = 1.0; seen[b[t]] = True
    s2 += (oh - s2) / 2; s4 += (oh - s4) / 4
    parts = [oh, np.sqrt(3.0) * s2, np.sqrt(7.0) * s4]
    if fb:
        fbv = np.zeros(256); fbv[last_spoken] = 1.0; parts.append(fbv)
    u = np.concatenate(parts + [np.ones(1)])
    nz = np.flatnonzero(u)
    Uj = mask[:, nz] * u[nz]                                   # (M, nnz)
    n = (Uj * Uj).sum(1)
    A = np.einsum("jvf,jf->jv", W[:, :, nz], Uj)               # (M, 256)
    best_v = A.argmax(1); best = A[np.arange(M), best_v]
    sv = np.flatnonzero(seen)
    inhib = np.zeros(M); spoke = []
    if kw.get("draw") == "after":
        # the reviewer's yq_out.py order (read after my figures): the threshold pass with inhibition first (cells by np.argsort(-best)),
        # then one draw for every cell; a random speaker speaks a seen value at random (replacing its value) and inhibits no one
        val = best_v.copy(); speak = np.zeros(M, bool)
        for j in np.argsort(-best):
            if best[j] - inhib[j] >= th[j]:
                speak[j] = True
                if inh: inhib += G[:, j]
        rr = rng.random(M) < eps
        for j in np.flatnonzero(rr): val[j] = int(sv[rng.integers(0, len(sv))]); speak[j] = True
        spoke = [(int(j), int(val[j])) for j in np.flatnonzero(speak)]
    order = [] if kw.get("draw") == "after" else sorted(range(M), key=lambda j: (-best[j], j))
    for j in order:
        if rng.random() < eps:
            e = int(sv[rng.integers(0, len(sv))])
        elif best[j] - inhib[j] >= th[j]:
            e = int(best_v[j])
        else:
            continue
        spoke.append((j, e))
        if inh: inhib += G[:, j]; inhib[j] -= G[j, j]
    ok = len(spoke) > 0 and all(e == b[t + 1] for _, e in spoke)
    yes[t + 1] = ok; nspeak[t + 1] = len(spoke)
    for j, e in spoke:
        W[j, e, nz] += eta * (float(ok) - A[j, e]) / n[j] * Uj[j]
    if not ok:
        if thr:
            sp = np.zeros(M, bool); sp[[j for j, _ in spoke]] = True
            th = np.clip(th + np.where(sp, rt, -rt), 0.0, 1.0)
        if inh and len(spoke) > 1:
            js = [j for j, _ in spoke]
            for a_ in js:
                for b_ in js:
                    if a_ != b_: G[a_, b_] += rg
    last_spoken = sorted({e for _, e in spoke})
refs = SE.references(b)
Q = [((0 if kw.get("draw") == "after" else max(1, i * NB // 4)) if i == 0 else i * NB // 4, (i + 1) * NB // 4) for i in range(4)]
rn = np.linalg.norm(W.reshape(M * 256, F), axis=1)
out = dict(label=lab, args=kw, quarters=[float(yes[a:e].mean()) for a, e in Q],
           speakers_mean_last_quarter=float(nspeak[Q[3][0]:].mean()), one_voice_last_quarter=float((nspeak[Q[3][0]:] == 1).mean()),
           silent_last_quarter=float((nspeak[Q[3][0]:] == 0).mean()), th=dict(min=float(th.min()), max=float(th.max())),
           G_max=float(G.max()), max_row_norm=float(rn.max()),
           refs_last_quarter={k: float((refs[k] == b)[Q[3][0]:].mean()) for k in ("ref_1", "ref_2", "ref_3")}, seconds=time.time() - t0)
json.dump(out, open(f"/home/user/tissue_check/out/op/{lab}.json", "w"), indent=0)
print(lab, "yes by quarter", " ".join(f"{100*x:.1f}" for x in out["quarters"]), f"| speakers {out['speakers_mean_last_quarter']:.2f}, one voice {100*out['one_voice_last_quarter']:.0f}%, silent {100*out['silent_last_quarter']:.0f}% | th {out['th']} G_max {out['G_max']:.3g} |W| {out['max_row_norm']:.3g} | {out['seconds']:.0f} s", flush=True)
