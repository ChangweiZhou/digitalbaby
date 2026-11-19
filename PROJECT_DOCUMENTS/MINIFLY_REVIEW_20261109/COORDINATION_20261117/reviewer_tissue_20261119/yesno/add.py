# Reviewer, 19 November 2026. The smallest case of the real task: addition of two single digits, scored on the answer only.
# An item: the system is shown a and b (0..9 each) and must give a + b (one of 19 answers); it is told only yes or no.
# H cells in between. Each has its own sensitivity to each value of the first number and of the second (2 x 10 entries, at birth
# random in [0, 1/2]). Drive = sensitivity to a + sensitivity to b; the A cells of largest drive are active (inhibition), output 1.
# Rule of a cell in between (local, no signal): an active cell moves its two present sensitivities toward 1 by r and all its
# other sensitivities toward 0 by r/8 (mode=local); or it does not change (mode=fixed).
# One answering cell reads the active cells and learns by rule Q (estimate per answer; the largest, or with chance 1/16 a random
# answer; on yes or no the spoken answer's estimate moves toward 1 or 0; on yes the others toward 0).
# HELD of the 100 pairs are never shown in learning. Reported after each block of 2,000 items: share of answers right on the pairs
# shown (measured without the random choice and without learning), and on the pairs never shown; chance is about 1/19 to 1/10.
#   python3 -I add.py <label> seed=1 mode=local|fixed [H=200 A=4 r=1/8 held=20 items=40000]
import sys, os, json
import numpy as np
label = sys.argv[1]; kw = dict(a.split('=') for a in sys.argv[2:])
SEED = int(kw.get('seed', 1)); MODE = kw.get('mode', 'local'); H = int(kw.get('H', 200)); A = int(kw.get('A', 4)); R = float(eval(kw.get('r', '1/8'))); HELD = int(kw.get('held', 20)); N = int(kw.get('items', 40000))
rng = np.random.default_rng(SEED)
pairs = [(a, b) for a in range(10) for b in range(10)]; rng.shuffle(pairs); held = pairs[:HELD]; shown = pairs[HELD:]
S = rng.random((H, 2, 10)) * 0.5; W = np.zeros((19, H + 1))
def active(a, b):
    g = S[:, 0, a] + S[:, 1, b]; top = np.argsort(-g, kind='stable')[:A]; h = np.zeros(H + 1); h[top] = 1.0; h[H] = 1.0; return h, top
def test(ps): return float(np.mean([int((W @ active(a, b)[0]).argmax()) == a + b for a, b in ps]))
rows = []
for it in range(N):
    a, b = shown[rng.integers(len(shown))]; h, top = active(a, b); n = float(h @ h)
    est = W @ h; e = int(est.argmax())
    if rng.random() < 1 / 16: e = int(rng.integers(19))
    ok = e == a + b
    d = np.zeros(19); d[e] = (float(ok) - est[e]) / 4
    if ok: d = -est / 4; d[e] = (1.0 - est[e]) / 4
    W += np.outer(d / n, h)
    if MODE == 'local':
        S[top] *= (1 - R / 8); S[top, 0, a] += R * (1 - S[top, 0, a]) + S[top, 0, a] * (R / 8) / (1 - R / 8) * 0; S[top, 1, b] += R * (1 - S[top, 1, b])
    if (it + 1) % 2000 == 0: rows.append((it + 1, test(shown), test(held)))
os.makedirs('res', exist_ok=True); json.dump(dict(label=label, args=kw, rows=rows), open(f'res/{label}.json', 'w'))
print(f"== {label} mode={MODE} seed={SEED} H={H} A={A} held={HELD}: items / shown right / never-shown right")
for r_ in rows[::4] + [rows[-1]]: print(f"   {r_[0]:6d}  {100 * r_[1]:5.1f}  {100 * r_[2]:5.1f}")
