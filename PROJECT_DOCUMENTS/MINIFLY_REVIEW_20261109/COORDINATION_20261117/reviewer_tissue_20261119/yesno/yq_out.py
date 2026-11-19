# Reviewer, 19 November 2026, after Mr. Xu's rulings of 14:24-14:33 UTC (WHAT_A_NEURON_HAS section 20): the output part is many
# cells, each of which can give a byte; nothing outside them decides which cell speaks; they are told only yes or no.
# A first small case, no tissue: M output cells; each sees the exact present byte (one-hot) with two slow copies (s_2, s_4) and a
# constant. Cell j holds, for each byte value v, an estimate a_j[v] = W_j[v] U of the chance that v is the next byte. At each byte a
# cell SPEAKS its best value if that estimate is at least its threshold theta, or, with chance eps, a seen value at random; else it
# is silent. The world hears every spoken byte. It answers yes if at least one cell spoke and every spoken byte is the next byte;
# otherwise no (silence of all is no). Every cell that spoke applies rule Q to the value it spoke with that common answer:
# W_j[e] += eta (ok - a_j[e]) U / n. Silent cells do not change. Nothing tells a cell whether it or another was the one in error.
#   python3 -I yq_out.py <label> seed=.. nbytes=.. M=8 theta=1/8 eps=1/64   -> res/<label>.json
import sys, os, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, 'deps'))
import life_stream as LS, score_emissions as SE
label = sys.argv[1]; kw = dict(a.split('=') for a in sys.argv[2:]); t0 = time.time()
SEED = int(kw.get('seed', 73129)); NB = int(kw.get('nbytes', 65536)); M = int(kw.get('M', 8)); TH = float(eval(kw.get('theta', '1/8'))); AD = float(eval(kw.get('adapt', '0'))); GR = float(eval(kw.get('inh', '0'))); EPS = float(eval(kw.get('eps', '1/64'))); ETA = float(eval(kw.get('eta', '1/16')))      # adapt=r: each cell's own threshold falls by r when the answer is no and it was silent, and rises by r when it spoke and the answer was no
L = LS.life(SEED); b = L['bytes'][:NB].astype(np.int64); TR = [2.0, 4.0]
F = 256 * 3 + 1; W = np.zeros((M, 256, F)); rng = np.random.default_rng(SEED)
th = np.full(M, TH); G = np.zeros((M, M)); right = np.zeros(NB, bool); nspoke = np.zeros(NB, np.int16); agree = np.zeros(NB, bool); seen = np.zeros(256, bool); tr = [np.zeros(256) for _ in TR]
for t in range(NB - 1):
    x = np.zeros(256); x[b[t]] = 1.0
    for j, k in enumerate(TR): tr[j] += (x - tr[j]) / k
    U = np.concatenate([x] + [tr[j] * np.sqrt(2 * k - 1) for j, k in enumerate(TR)] + [np.ones(1)]); n = float(U @ U)
    seen[b[t]] = True; yy = b[t + 1]; sv = np.flatnonzero(seen)
    A = W @ U                                  # M x 256
    best = A.argmax(1); am = A[np.arange(M), best]
    val = best.copy()
    if GR:                                     # inh=r: cells of the output part inhibit one another; the inhibition between two cells grows by r each time both spoke and the answer was no (it starts at 0). The more active cell speaks first.
        speak = np.zeros(M, bool); sup = np.zeros(M)
        for j in np.argsort(-am):
            if am[j] - sup[j] >= th[j]: speak[j] = True; sup += G[:, j]
    else: speak = am >= th
    r = rng.random(M) < EPS
    for j in np.flatnonzero(r): val[j] = int(sv[rng.integers(0, len(sv))]); speak[j] = True
    sp = np.flatnonzero(speak); ok = len(sp) > 0 and bool((val[sp] == yy).all())
    right[t + 1] = ok; nspoke[t + 1] = len(sp); agree[t + 1] = len(sp) > 0 and len(set(val[sp].tolist())) == 1
    for j in sp: W[j, val[j]] += ETA * (float(ok) - A[j, val[j]]) / n * U
    if GR and not ok and len(sp) > 1:
        for j in sp:
            for k2 in sp:
                if j != k2: G[j, k2] += GR
    if AD and not ok: th[~speak] -= AD; th[speak] += AD; np.clip(th, 0.0, 1.0, out=th)
refs = SE.references(L['bytes'][:NB]); q = [slice(i * NB // 4, (i + 1) * NB // 4) for i in range(4)]
out = dict(label=label, args=kw, right=[float(right[s].mean()) for s in q], spoke_mean=[float(nspoke[s].mean()) for s in q], none_spoke=[float((nspoke[s] == 0).mean()) for s in q],
           one_voice=[float(agree[s].mean()) for s in q], refs={k: float((refs[k] == b)[q[3]].mean()) for k in ('ref_1', 'ref_2', 'ref_3')}, seconds=time.time() - t0)
json.dump(out, open(os.path.join(HERE, 'res', f'{label}.json'), 'w'))
print(f"== {label} M={M} theta={TH:g} eps={EPS:g} eta={ETA:g} ({time.time() - t0:.0f} s); refs " + ' '.join(f"{k} {100 * v:.1f}" for k, v in out['refs'].items()))
for k in ('right', 'spoke_mean', 'none_spoke', 'one_voice'): print(f"   {k:11s} by quarter: " + ' '.join(f"{(100 if k != 'spoke_mean' else 1) * v:6.2f}" for v in out[k]))
