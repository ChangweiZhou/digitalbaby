# Reviewer, 19 November 2026. Do cells in the middle keep what the acting cells need, when they too receive the right-or-wrong
# signal? A small case, no tissue program. H middle cells each receive the byte itself (WHAT_A_NEURON_HAS section 20) through
# their own sensitivity S_i[v] to each byte value v (at birth: K values drawn at random, strength 1/K each; the rest 0).
# Activity: g_i = S_i[byte]; the A most active cells with g_i > 0 are active (inhibition among them), h_i = g_i, the others 0.
# One acting cell reads [h, s_2(h), s_4(h), 1] (slow copies at its own connections) and learns by rule Q (yes or no only), as
# yq.py; emission the largest estimate, with chance 1/16 a seen value at random.
# The middle cells' rule (mode=):
#   fixed : S does not change.
#   local : no signal. An active cell moves its sensitivity toward the byte that made it active: S_i[byte] += r (1 - S_i[byte]),
#           and its sensitivities are rescaled to sum 1 (what fires together is kept, the rest fades).
#   fb    : the common answer reaches every middle cell one byte later (it is about the byte they were active for). A cell that
#           was active then changes its sensitivity to that byte by r (ok - m) h_i, m the running share of yes (rate 1/256, one
#           number released with the signal); sensitivities stay in [0, 1]; no rescaling.
#   fbx   : as fb, and a cell that was NOT active for that byte and has no sensitivity above 1/4 to any value, on "no", gains r/4
#           toward that byte (an idle cell is recruited where the animal fails).
# Reported: share of next bytes right by quarter; at the end, how many of the byte values in use give activity patterns that
# differ from every other value's ('told apart'), and how many give no activity at all ('silent').
#   python3 -I mid.py <label> seed=.. mode=fixed|local|fb|fbx [H=64 K=8 A=8 r=1/16 nbytes=65536]
import sys, os, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, 'deps'))
import life_stream as LS, score_emissions as SE
label = sys.argv[1]; kw = dict(a.split('=') for a in sys.argv[2:]); t0 = time.time()
SEED = int(kw.get('seed', 73129)); NB = int(kw.get('nbytes', 65536)); MODE = kw.get('mode', 'fixed'); H = int(kw.get('H', 64)); K = int(kw.get('K', 8)); A = int(kw.get('A', 8)); R = float(eval(kw.get('r', '1/16')))
L = LS.life(SEED); b = L['bytes'][:NB].astype(np.int64); rng = np.random.default_rng(SEED + 7)
S = np.zeros((H, 256))
for i in range(H): S[i, rng.choice(256, K, replace=False)] = 1.0 / K
def act(v):
    g = S[:, v].copy(); h = np.zeros(H)
    if (g > 0).any():
        top = np.argsort(-g, kind='stable')[:A]; top = top[g[top] > 0]; h[top] = g[top]
    return h
TR = [2.0, 4.0]; F = H * 3 + 1; W = np.zeros((256, F)); tr = [np.zeros(H) for _ in TR]; seen = np.zeros(256, bool); right = np.zeros(NB, bool); m = 0.1
prev_h = None; prev_b = None; prev_ok = None
for t in range(NB - 1):
    h = act(b[t])
    for j, k in enumerate(TR): tr[j] += (h - tr[j]) / k
    U = np.concatenate([h] + [tr[j] * np.sqrt(2 * k - 1) for j, k in enumerate(TR)] + [np.ones(1)]); n = float(U @ U)
    seen[b[t]] = True; yy = b[t + 1]; sv = np.flatnonzero(seen)
    a = W @ U; e = int(a.argmax())
    if rng.random() < 1 / 16: e = int(sv[rng.integers(0, len(sv))])
    ok = e == yy; right[t + 1] = ok
    d = np.zeros(256); d[e] = (float(ok) - a[e]) / 16
    if ok: d = -a / 16; d[e] = (1.0 - a[e]) / 16
    nz = np.flatnonzero(d); W[nz] += np.outer(d[nz] / n, U)
    # the middle cells
    if MODE == 'local':
        on = h > 0; S[on, b[t]] += R * (1.0 - S[on, b[t]]); S[on] /= S[on].sum(1, keepdims=True)
    elif MODE in ('fb', 'fbx'):
        on = h > 0; S[on, b[t]] = np.clip(S[on, b[t]] + R * (float(ok) - m) * h[on] * K, 0.0, 1.0)
        if MODE == 'fbx' and not ok:
            idle = (~on) & (S.max(1) <= 0.25); S[idle, b[t]] = np.clip(S[idle, b[t]] + R / 4, 0.0, 1.0)
        m += (float(ok) - m) / 256
used = sorted(set(b.tolist())); pats = {v: tuple(np.round(act(v), 6)) for v in used}
silent = sum(1 for v in used if not any(pats[v])); apart = sum(1 for v in used if any(pats[v]) and sum(1 for u in used if pats[u] == pats[v]) == 1)
q = [slice(i * NB // 4, (i + 1) * NB // 4) for i in range(4)]; refs = SE.references(L['bytes'][:NB])
out = dict(label=label, args=kw, right=[float(right[s].mean()) for s in q], used=len(used), told_apart=apart, silent=silent, refs={k: float((refs[k] == b)[q[3]].mean()) for k in ('ref_1', 'ref_2', 'ref_3')})
json.dump(out, open(os.path.join(HERE, 'res', f'{label}.json'), 'w'))
print(f"== {label} mode={MODE} seed={SEED} H={H} K={K} A={A} r={R:g} ({time.time() - t0:.0f} s): right by quarter " + ' '.join(f"{100 * v:5.2f}" for v in out['right']) + f" | of {len(used)} values: told apart {apart}, silent {silent} | ref_1 {100 * out['refs']['ref_1']:.1f} ref_2 {100 * out['refs']['ref_2']:.1f} ref_3 {100 * out['refs']['ref_3']:.1f}")
