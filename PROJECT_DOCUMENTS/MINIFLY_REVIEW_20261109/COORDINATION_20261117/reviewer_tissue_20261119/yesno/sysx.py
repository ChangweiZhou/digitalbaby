# Reviewer, 19 November 2026. The system as Mr. Xu described it (WHAT_A_NEURON_HAS sections 15 and 20), in a small case: the
# cells that receive the world receive three things, each as it is: the byte of text, the byte the system gave out last, and
# whether that was right. Nothing carries the answer to any other cell except ordinary connections.
# H receiving cells; cell i has its own sensitivity to each value of each of the three (S_i: 256 + 256 + 2 entries; at birth K
# entries drawn at random among the stimuli it receives, 1/K each). Its drive g_i is the sum of its sensitivities to the three
# present stimuli; the A cells of largest drive (above 0) are active, h_i = g_i (inhibition among them).
# Local rule of a receiving cell (no signal from outside): when active it moves its sensitivity toward each stimulus present,
# S_i[s] += r (1 - S_i[s]), and rescales its sensitivities to sum 1.
# One acting cell reads [h, s_2(h), s_4(h), 1] and learns by rule Q from yes or no; its byte is the system's output.
# what=  t   : the receiving cells receive the text byte only (the case of mid.py mode=local)
#        to  : text byte and the last output
#        tok : text byte, the last output, and whether it was right
#        tk  : text byte and whether the last output was right
#   python3 -I sysx.py <label> seed=.. what=t|to|tok|tk [H=64 K=8 A=8 r=1/16 nbytes=65536]
import sys, os, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, 'deps'))
import life_stream as LS, score_emissions as SE
label = sys.argv[1]; kw = dict(a.split('=') for a in sys.argv[2:]); t0 = time.time()
SEED = int(kw.get('seed', 73129)); NB = int(kw.get('nbytes', 65536)); WHAT = kw.get('what', 't'); H = int(kw.get('H', 64)); K = int(kw.get('K', 8)); A = int(kw.get('A', 8)); R = float(eval(kw.get('r', '1/16')))
L = LS.life(SEED); b = L['bytes'][:NB].astype(np.int64); rng = np.random.default_rng(SEED + 7)
use_o = 'o' in WHAT; use_k = 'k' in WHAT
pool = list(range(256)) + (list(range(256, 512)) if use_o else []) + ([512, 513] if use_k else [])
S = np.zeros((H, 514))
for i in range(H): S[i, rng.choice(pool, K, replace=False)] = 1.0 / K
TR = [2.0, 4.0]; F = H * 3 + 1; W = np.zeros((256, F)); tr = [np.zeros(H) for _ in TR]; seen = np.zeros(256, bool); right = np.zeros(NB, bool)
last_out = -1; last_ok = 0
for t in range(NB - 1):
    stim = [int(b[t])] + ([256 + last_out] if use_o and last_out >= 0 else []) + ([512 + last_ok] if use_k else [])
    g = S[:, stim].sum(1); h = np.zeros(H)
    top = np.argsort(-g, kind='stable')[:A]; top = top[g[top] > 0]; h[top] = g[top]
    for j, k in enumerate(TR): tr[j] += (h - tr[j]) / k
    U = np.concatenate([h] + [tr[j] * np.sqrt(2 * k - 1) for j, k in enumerate(TR)] + [np.ones(1)]); n = float(U @ U)
    seen[b[t]] = True; yy = b[t + 1]; sv = np.flatnonzero(seen)
    a = W @ U; e = int(a.argmax())
    if rng.random() < 1 / 16: e = int(sv[rng.integers(0, len(sv))])
    ok = e == yy; right[t + 1] = ok
    d = np.zeros(256); d[e] = (float(ok) - a[e]) / 16
    if ok: d = -a / 16; d[e] = (1.0 - a[e]) / 16
    nz = np.flatnonzero(d); W[nz] += np.outer(d[nz] / n, U)
    on = h > 0
    for s in stim: S[on, s] += R * (1.0 - S[on, s])
    S[on] /= S[on].sum(1, keepdims=True)
    last_out = e; last_ok = int(ok)
q = [slice(i * NB // 4, (i + 1) * NB // 4) for i in range(4)]; refs = SE.references(L['bytes'][:NB])
out = dict(label=label, args=kw, right=[float(right[s].mean()) for s in q], refs={k: float((refs[k] == b)[q[3]].mean()) for k in ('ref_1', 'ref_2', 'ref_3')})
json.dump(out, open(os.path.join(HERE, 'res', f'{label}.json'), 'w'))
print(f"== {label} what={WHAT} seed={SEED} ({time.time() - t0:.0f} s): right by quarter " + ' '.join(f"{100 * v:5.2f}" for v in out['right']) + f" | ref_1 {100 * out['refs']['ref_1']:.1f} ref_2 {100 * out['refs']['ref_2']:.1f} ref_3 {100 * out['refs']['ref_3']:.1f}")
