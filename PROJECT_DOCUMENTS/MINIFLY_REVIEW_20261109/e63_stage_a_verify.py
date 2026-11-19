"""Coordinator's verification of E63 stage A from the committed records (worker's branch claude/e63-tissue-step1 at 040d3cf0).
Read-only. Independent code for the stream, the shares, the classifier and the counts; nothing of the worker's is imported.
    python3.12 -I e63_stage_a_verify.py <folder with runs/ and STAGE_A_REPORT.md>
"""
import json, sys, re, hashlib
from pathlib import Path
from random import Random
import numpy as np
ROOT = Path(sys.argv[1]); SEED = 73129; N = 2048
# ---- the stream, written again from section 4 of the brief
SYM = {"add": "+", "sub": "-", "max": ">", "min": "<"}
DEV = [("add", 89, b"p"), ("sub", 89, b"q"), ("max", 89, b"v"), ("min", 89, b"w")]
LET = list(b"ABCDEFGHIJKLMNOPQRSTUVWXYZ")
def op(o, m, a, b): return (a + b) % m if o == "add" else (a - b) % m if o == "sub" else max(a, b) if o == "max" else min(a, b)
def stream(seed, n):
    flat = []
    for o, m, code in DEV:
        pairs = [(a, b) for a in range(m) for b in range(m)]
        Random(seed * 1000 + 50 + m * 7 + code[0]).shuffle(pairs)
        flat += [((o, m, code), pr) for pr in pairs[:round(.5 * len(pairs))]]
    rng = Random(seed + 77); out = []
    while len(out) < n:
        order = list(range(len(flat))); rng.shuffle(order)
        for i in order:
            (o, m, code), (a, b) = flat[i]
            k = rng.randint(0, 12)
            d = [rng.choice(LET) for _ in range(k)]
            out += list(code) + list(b"%02d" % a) + [ord(SYM[o])] + d + list(b"%02d" % b) + [ord("=")] + list(b"%02d" % op(o, m, a, b)) + [10]
            if len(out) >= n: break
    return np.array(out[:n], np.uint8)
ST = stream(SEED, 262144)
print("stream sha256 of 262,144 bytes:", hashlib.sha256(ST.tobytes()).hexdigest()[:8], "...", hashlib.sha256(ST.tobytes()).hexdigest()[-6:], "| values in first 16,384:", len(set(ST[:16384].tolist())))
# ---- my classifier (nearest centroid, balanced accuracy, four folds with a gap of 16)
def classify(F, y, pos, w0, wl, tie_rel=1e-9):
    q = wl // 4; right = {}; tot = {}; fb = []
    for f in range(4):
        lo, hi = w0 + f * q, w0 + (f + 1) * q
        te = (pos >= lo) & (pos < hi); fi = ~((pos >= lo - 16) & (pos < hi + 16))
        mu = F[fi].mean(0); sd = F[fi].std(0); use = sd >= 1e-9
        Zf = (F[fi][:, use] - mu[use]) / sd[use]; yf = y[fi]
        cl = np.unique(yf); cen = np.stack([Zf[yf == c].mean(0) for c in cl])
        Zt = (F[te][:, use] - mu[use]) / sd[use]; yt = y[te]
        if Zt.shape[1] == 0: pred = np.full(len(yt), cl[0])
        else:
            pred = np.empty(len(yt), cl.dtype)
            for s in range(0, len(yt), 128):
                b = Zt[s:s + 128]; d2 = ((b[:, None, :] - cen[None]) ** 2).sum(2)
                dm = d2.min(1, keepdims=True); pred[s:s + 128] = cl[np.argmax(d2 <= dm + tie_rel * (1 + np.abs(dm)), 1)]
        r = {}; t = {}
        for a, b in zip(yt.tolist(), pred.tolist()):
            t[a] = t.get(a, 0) + 1; r[a] = r.get(a, 0) + (a == b); tot[a] = tot.get(a, 0) + 1; right[a] = right.get(a, 0) + (a == b)
        fb.append(sum(r[c] / t[c] for c in t) / len(t))
    return sum(right[c] / tot[c] for c in tot) / len(tot), float(np.std(fb, ddof=1) / 2)
def onehot(cols):
    out = []
    for c in cols:
        v = sorted(set(c)); ix = {a: i for i, a in enumerate(v)}; M = np.zeros((len(c), len(v))); M[np.arange(len(c)), [ix[a] for a in c]] = 1; out.append(M)
    return np.concatenate(out, 1)
LAGS = (0, 1, 2, 4, 8); DS = (1, 2, 4)
def mode_distance(M):
    out = np.zeros(N, np.int16)
    for i in range(N):
        v, c = np.unique(M[:, i], return_counts=True); k = v > 0
        if k.any(): out[i] = v[k][np.argmax(c[k])]
    return out
res = {}
for tis in ("A_first_73129", "A_coincidence_73129", "A_fixed_73129"):
    rec = json.loads((ROOT / "runs" / tis / "receipt.json").read_text()); res[tis] = {}
    for wn in ("A1", "A2"):
        z = np.load(ROOT / "runs" / tis / f"{wn}.npz"); R = z["x"].astype(np.float64); w0 = int(z["positions"][0]); wl = R.shape[0]
        assert (z["bytes"] == ST[w0:w0 + wl]).all(), "the window's bytes are not the stream's"
        dist = mode_distance(z["block_distances"]); same_dist = bool((dist == z["distance"]).all())
        above = (R > 0.01).sum(0); silent = above < wl / 128; const = (~silent) & (R.std(0) < 0.01)
        m = dict(distance_recomputed_equal=same_dist, shares={}, S={}, C0={}, Clt={})
        for d in DS:
            c = dist == d; n = int(c.sum()); inf = int((c & ~silent & ~const).sum())
            m["shares"][d] = (n, inf, int((c & silent).sum()), int((c & const).sum()))
            cells = sorted(int(i) for i in np.nonzero(c)[0])
            samp = Random(SEED + 900000).sample(cells, 128) if len(cells) > 128 else cells
            for L in LAGS:
                idx = np.array([t for t in range(wl) if w0 + t - L >= 0]); F = R[idx][:, samp].copy(); F[F < 0.01] = 0
                m["S"][(d, L)] = classify(F, ST[w0 + idx - L].astype(np.int64), w0 + idx, w0, wl)
        if tis == "A_first_73129":
            for L in LAGS:
                pos = np.arange(w0, w0 + wl); m["C0"][L] = classify(onehot([ST[pos].tolist()]), ST[pos - L].astype(np.int64), pos, w0, wl)[0]
                if L: m["Clt"][L] = classify(onehot([ST[pos - j].tolist() for j in range(L)]), ST[pos - L].astype(np.int64), pos, w0, wl)[0]
        res[tis][wn] = m
        # against the receipt's measures
        mm = rec["measures"][wn]; worst = 0.0; nd = 0
        for d in DS:
            sh = mm["shares"][str(d)]
            if (sh["n"], sh["informative"], sh["silent"], sh["constant"]) != m["shares"][d]: nd += 1
            for L in LAGS:
                a = mm["scores"][str(d)]["by_lag"][str(L)]; worst = max(worst, abs(a["S"] - m["S"][(d, L)][0]), abs(a["se"] - m["S"][(d, L)][1]))
        if m["C0"]:
            for L in LAGS:
                worst = max(worst, abs(mm["C0"][str(L)] - m["C0"][L]))
                if L: worst = max(worst, abs(mm["C_lt"][str(L)] - m["Clt"][L]))
        print(f"{tis} {wn}: distances recomputed from the block distances equal {same_dist}; shares differing {nd} of 3; largest |receipt - recomputed| over S, se (and C0, C<L): {worst:.2e}")
        print("   shares (n, neither, silent, constant):", m["shares"], "| S(d,0):", {d: round(m["S"][(d, 0)][0], 4) for d in DS}, "| S(d,1), S(d,2):", {d: (round(m["S"][(d, 1)][0], 4), round(m["S"][(d, 2)][0], 4)) for d in DS})
        if m["C0"]: print("   C0:", {L: round(v, 4) for L, v in m["C0"].items()}, "C<L:", {L: round(v, 4) for L, v in m["Clt"].items()})
    # ---- the final state
    d = np.load(ROOT / "runs" / tis / "final_state.npz"); al = d["alive"]; snd = d["snd"][al]; rcv = d["rcv"][al]; w = d["w"][al]; k0 = d["k0"][al]
    ordinary = snd < N + 256; per = np.bincount(rcv[ordinary], minlength=N)
    em = np.load(ROOT / "runs" / tis / "emissions.npy")
    last = rec["counts"][-1]
    print(f"   final state: connections {int(ordinary.sum())} (receipt's last count {last['connections']}), mean a cell {per.mean():.2f}, starting remaining {int(((k0 == 0) & ordinary).sum())} "
          f"({last['starting_remaining']}), input-line connections {int(((snd >= N) & ordinary).sum())}; step {int(d['meta'][0])}; emitted share {float((em >= 0).mean()):.4f} (receipt {rec['emissions']['share']:.4f}); "
          f"cpu_total_s {rec['cpu_total_s']:.0f}; cost a byte over the last four stretches {np.mean([c['per_byte'] for c in rec['cpu_by_stretch'][-4:]]):.4f}; resumes {rec['resumes']}; program sha {rec['program_sha256'][:8]}")
    x = d["x_prev"]; print(f"   outputs at the last step: not zero {int((x != 0).sum())}, at least 1/128 {int((x >= 1/128).sum())}, at least 0.01 {int((x >= .01).sum())}; thresholds exactly zero {int((d['th'] == 0).sum())}; mean running average of activity {float(d['abar'].mean()):.4f}")
# ---- the outcome of section 11.3 at the level 0.1103
LEVEL = 0.0903 + 0.02
inf = {}
for tis in ("A_first_73129", "A_coincidence_73129"):
    m = res[tis]["A2"]; inf[tis] = {d: (m["shares"][d][1] / m["shares"][d][0] >= .5, m["S"][(d, 0)][0] >= LEVEL) for d in (2, 4)}
o1 = any(all(a and b for a, b in inf[t].values()) for t in inf); o2 = all(any(not a for a, _ in inf[t].values()) for t in inf)
print("outcome: (share >= 1/2, S >= level) at d = 2, 4:", inf, "-> outcome", 1 if o1 else 2 if o2 else 3)
# ---- lags 1 and 2 against C0 + 0.02
c0 = res["A_first_73129"]
for wn in ("A1", "A2"):
    best = max((res[t][wn]["S"][(d, L)][0] - c0[wn]["C0"][L], t, d, L) for t in res for d in DS for L in (1, 2))
    print(f"lags 1 and 2, {wn}: the largest S - C0 over tissues and distances is {best[0]:+.4f} ({best[1]}, d = {best[2]}, L = {best[3]}); any at or above +0.02: {best[0] >= 0.02}")
# ---- tissue 4
rec = json.loads((ROOT / "runs/A_first_f11written_73129/receipt.json").read_text())
for c in rec["counts"][:2]: print("tissue 4,", c["at"], ": connections", c["connections"], "starting remaining", c["starting_remaining"], "given up", c["given_up_total"], "made", c["made_total"])
# ---- the report's tables of scores against the recomputed values
rep = (ROOT / "STAGE_A_REPORT.md").read_text(); cmp = bad = 0
names = {"1. first form, E10": "A_first_73129", "2. coincidence, E10": "A_coincidence_73129", "3. fixed": "A_fixed_73129"}
for head, body in re.findall(r"#### (.+?), window (A[12]) .*?\n\n((?:\|.*\n)+)", rep) and [(h + "|" + w, b) for h, w, b in re.findall(r"#### (.+?), window (A[12]) .*?\n\n((?:\|.*\n)+)", rep)]:
    h, wn = head.split("|"); t = names[h]
    for line in body.splitlines():
        mm = re.match(r"\| S\(d = (\d)\).*?\|(.*)\|", line)
        if mm:
            d = int(mm.group(1)); vals = [float(v.split("±")[0]) for v in mm.group(2).split("|")]
            for L, v in zip(LAGS, vals):
                cmp += 1; bad += abs(v - res[t][wn]["S"][(d, L)][0]) > 5.1e-5
print(f"report's tables of S: {cmp} figures compared with the recomputed values, differing {bad}")
