# addtext_m.py, by E52's worker: addtext.py (beside this file, unchanged) with measurements added and nothing else changed (the same
# draws in the same order, so its rows equal addtext.py's):
#   - the whole-answer score split by the answer's length (number of digits of a + b);
#   - the ceilings from the text alone: for K = 1, 2, 3, 4 the most frequent next byte given the last K bytes of the stream (counts
#     taken online, updated in the learning blocks only, as the cell learns; a context never seen names nothing), scored like the
#     cell on the answer bytes and on the whole answer, shown and never shown, and by answer length.
# addtext.py's text follows.
# Reviewer, 19 November 2026. Real addition as the text (Mr. Xu, 15:12 UTC): the stream is lines like "7+8=15\n", read one byte
# at a time; the system names the next byte and is told yes or no. Nothing is handed to it but the bytes.
# The system here is the smallest that exists so far: one answering cell that receives the present byte itself with slow copies of
# it at its own connections (s_2, s_4, s_8), learning by rule Q. Scored separately: the answer's bytes (those after '=' up to the
# end mark) and the other bytes; and, for the answer, sums that were shown in learning against sums never shown (HELD of the pairs
# a, b are kept out of the stream until the last block, in which only they appear and nothing is learned).
#   python3 -I addtext.py <label> seed=1 [maxn=9 held=20 items=60000]
import sys, os, json
import numpy as np
from collections import defaultdict, Counter
KS = (1, 2, 3, 4); CT = {k: defaultdict(Counter) for k in KS}; HIST = []
label = sys.argv[1]; kw = dict(a.split('=') for a in sys.argv[2:])
SEED = int(kw.get('seed', 1)); MAXN = int(kw.get('maxn', 9)); HELD = int(kw.get('held', 20)); N = int(kw.get('items', 60000))
rng = np.random.default_rng(SEED)
pairs = [(a, b) for a in range(MAXN + 1) for b in range(MAXN + 1)]; rng.shuffle(pairs); held = pairs[:HELD]; shown = pairs[HELD:]
TR = [2.0, 4.0, 8.0]; F = 256 * (1 + len(TR)) + 1; W = np.zeros((256, F)); tr = [np.zeros(256) for _ in TR]; seen = np.zeros(256, bool)
def run(ps, n_items, learn):
    ans = []; oth = []; whole = []; wlen = defaultdict(list); cans = {k: [] for k in KS}; cwhole = {k: [] for k in KS}; cwlen = {k: defaultdict(list) for k in KS}
    for it in range(n_items):
        a, b = ps[rng.integers(len(ps))]; line = f"{a}+{b}={a + b}\n".encode(); eq = line.index(b'='); allok = True; call = {k: True for k in KS}; L_ = len(str(a + b))
        for t in range(len(line) - 1):
            x = np.zeros(256); x[line[t]] = 1.0
            for j, k in enumerate(TR): tr[j] += (x - tr[j]) / k
            U = np.concatenate([x] + [tr[j] * np.sqrt(2 * k - 1) for j, k in enumerate(TR)] + [np.ones(1)]); n = float(U @ U)
            HIST.append(line[t])
            seen[line[t]] = True; yy = line[t + 1]; est = W @ U; e = int(est.argmax())
            if learn and rng.random() < 1 / 16: sv = np.flatnonzero(seen); e = int(sv[rng.integers(len(sv))])
            ok = e == yy
            for k in KS:
                if len(HIST) >= k:
                    c = tuple(HIST[-k:]); okc = c in CT[k] and CT[k][c].most_common(1)[0][0] == yy
                    if t >= eq and yy != 10: cans[k].append(okc); call[k] = call[k] and okc
                    if learn: CT[k][c][yy] += 1
                elif t >= eq and yy != 10: call[k] = False
            if t >= eq and yy != 10: ans.append(ok); allok = allok and ok
            else: oth.append(ok)
            if learn:
                d = np.zeros(256); d[e] = (float(ok) - est[e]) / 16
                if ok: d = -est / 16; d[e] = (1.0 - est[e]) / 16
                nz = np.flatnonzero(d); W[nz] += np.outer(d[nz] / n, U)
        # the last byte of the line is followed by the first of the next: not scored
        HIST.append(line[-1])
        x = np.zeros(256); x[line[-1]] = 1.0
        for j, k in enumerate(TR): tr[j] += (x - tr[j]) / k
        whole.append(allok); wlen[L_].append(allok)
        for k in KS: cwhole[k].append(call[k]); cwlen[k][L_].append(call[k])
    EXTRA.append(dict(by_len={str(l): [float(np.mean(v)), len(v)] for l, v in sorted(wlen.items())},
                      ceiling={str(k): dict(answer=float(np.mean(cans[k])), whole=float(np.mean(cwhole[k])), by_len={str(l): float(np.mean(v)) for l, v in sorted(cwlen[k].items())}) for k in KS}))
    return float(np.mean(ans)), float(np.mean(oth)), float(np.mean(whole))
rows = []; EXTRA = []
for blk in range(6): rows.append(('learning block %d' % (blk + 1),) + run(shown, N // 6, True))
rows.append(('sums shown, no learning',) + run(shown, 2000, False)); rows.append(('sums NEVER shown',) + run(held, 2000, False))
os.makedirs('res', exist_ok=True); json.dump(dict(label=label, args=kw, rows=rows, extra=EXTRA), open(f'res/{label}.json', 'w'))
print(f"== {label} seed={SEED} numbers 0..{MAXN}, {HELD} of {len(pairs)} pairs never shown: answer bytes right % | other bytes right % | whole answer right %")
for r_ in rows: print(f"   {r_[0]:26s} {100 * r_[1]:5.1f} | {100 * r_[2]:5.1f} | {100 * r_[3]:5.1f}")
