# Reviewer, 19 November 2026. Real addition as the text (Mr. Xu, 15:12 UTC): the stream is lines like "7+8=15\n", read one byte
# at a time; the system names the next byte and is told yes or no. Nothing is handed to it but the bytes.
# The system here is the smallest that exists so far: one answering cell that receives the present byte itself with slow copies of
# it at its own connections (s_2, s_4, s_8), learning by rule Q. Scored separately: the answer's bytes (those after '=' up to the
# end mark) and the other bytes; and, for the answer, sums that were shown in learning against sums never shown (HELD of the pairs
# a, b are kept out of the stream until the last block, in which only they appear and nothing is learned).
#   python3 -I addtext.py <label> seed=1 [maxn=9 held=20 items=60000]
import sys, os, json
import numpy as np
label = sys.argv[1]; kw = dict(a.split('=') for a in sys.argv[2:])
SEED = int(kw.get('seed', 1)); MAXN = int(kw.get('maxn', 9)); HELD = int(kw.get('held', 20)); N = int(kw.get('items', 60000))
rng = np.random.default_rng(SEED)
pairs = [(a, b) for a in range(MAXN + 1) for b in range(MAXN + 1)]; rng.shuffle(pairs); held = pairs[:HELD]; shown = pairs[HELD:]
TR = [2.0, 4.0, 8.0]; F = 256 * (1 + len(TR)) + 1; W = np.zeros((256, F)); tr = [np.zeros(256) for _ in TR]; seen = np.zeros(256, bool)
def run(ps, n_items, learn):
    ans = []; oth = []; whole = []
    for it in range(n_items):
        a, b = ps[rng.integers(len(ps))]; line = f"{a}+{b}={a + b}\n".encode(); eq = line.index(b'='); allok = True
        for t in range(len(line) - 1):
            x = np.zeros(256); x[line[t]] = 1.0
            for j, k in enumerate(TR): tr[j] += (x - tr[j]) / k
            U = np.concatenate([x] + [tr[j] * np.sqrt(2 * k - 1) for j, k in enumerate(TR)] + [np.ones(1)]); n = float(U @ U)
            seen[line[t]] = True; yy = line[t + 1]; est = W @ U; e = int(est.argmax())
            if learn and rng.random() < 1 / 16: sv = np.flatnonzero(seen); e = int(sv[rng.integers(len(sv))])
            ok = e == yy
            if t >= eq and yy != 10: ans.append(ok); allok = allok and ok
            else: oth.append(ok)
            if learn:
                d = np.zeros(256); d[e] = (float(ok) - est[e]) / 16
                if ok: d = -est / 16; d[e] = (1.0 - est[e]) / 16
                nz = np.flatnonzero(d); W[nz] += np.outer(d[nz] / n, U)
        # the last byte of the line is followed by the first of the next: not scored
        x = np.zeros(256); x[line[-1]] = 1.0
        for j, k in enumerate(TR): tr[j] += (x - tr[j]) / k
        whole.append(allok)
    return float(np.mean(ans)), float(np.mean(oth)), float(np.mean(whole))
rows = []
for blk in range(6): rows.append(('learning block %d' % (blk + 1),) + run(shown, N // 6, True))
rows.append(('sums shown, no learning',) + run(shown, 2000, False)); rows.append(('sums NEVER shown',) + run(held, 2000, False))
os.makedirs('res', exist_ok=True); json.dump(dict(label=label, args=kw, rows=rows), open(f'res/{label}.json', 'w'))
print(f"== {label} seed={SEED} numbers 0..{MAXN}, {HELD} of {len(pairs)} pairs never shown: answer bytes right % | other bytes right % | whole answer right %")
for r_ in rows: print(f"   {r_[0]:26s} {100 * r_[1]:5.1f} | {100 * r_[2]:5.1f} | {100 * r_[3]:5.1f}")
