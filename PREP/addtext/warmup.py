# warmup.py, by E52's worker (19 November 2026): levels.py (beside this file) with a warm-up before the life, and three additions each
# by a local rule. addtext.py's text and score (the reviewer's, unchanged beside this file): lines "a+b=c\n", one answering cell names
# the next byte; whole answer right on sums shown and never shown.
#
# Cells (NC = 256 by default): the level input of levels.py, u(t) = value(t)/255; cell i takes u at delays d1_i < d2_i drawn at birth from
# 0..5; raw drive r_i = g1_i u(t - d1_i) + g2_i u(t - d2_i), gains drawn N(0, 1).
#   add=none  adaptive cells as levels.py: running mean m_i and mean square q_i of r_i (rate 1/1024, after use), y = max(0, (r - m)/sd - c_i).
#   add=a     as none, and the two strengths brought to equal mean drive: each connection k keeps a running mean D_k of its drive g_k u_k
#             (rate 1/1024); |g_k| *= 1 + (1/1024)(mean(|D_1|, |D_2|) / |D_k| - 1); then |g_1| + |g_2| is set to 2 (signs kept); every byte.
#   add=b     in place of the soft threshold, a band: cell i is made at a step t_i drawn at birth uniformly in the first 2,048 steps of the
#             life (warm-up included); its centre r0_i is its drive at that step; from then on y = 1 when |r - r0| <= unit_i / 2 and 0
#             otherwise, unit_i = (|g1| + |g2|) / 2 / 255 (the drive's change for one unit of byte value at one input); 0 before it is made.
#   add=c     as none, and each cell gated by the present byte: cell i is made at t_i as in b; its gate byte is the byte present then;
#             from then on its output is multiplied by 1 when the present byte is its gate byte and 0 otherwise; 0 before it is made.
# The answering cell: U = [y, 1], n = U.U, rule Q exactly as addtext.py (largest estimate; chance 1/16 of a seen value at random while
# learning; emitter W[e] += (1/16)(ok - est_e) U/n; on yes the others W[v] += (1/16)(0 - est_v) U/n; on no the others nothing).
# Warm-up: a set of 1,000 lines drawn once from the pairs shown, gone through R times (each pass in a new random order), in one form:
#   W1 under yes or no (rule Q, as the life); W2 the answering cell told the right byte: every estimate toward 1 for the right byte and 0
#   for the others, W += (1/16)(target - est) U/n; W3 no answer asked: the cells run and adapt, W is not changed.
# Then the life: LIFE lines (60,000) under rule Q. Control: warm=none, a life of LIFE + 20,000 lines, read at every count of lines the
# warm runs are read at, so that each warm run can be set against no warm-up at equal total lines.
# Reading (no learning): 2,000 lines of sums shown and 2,000 of sums never shown, drawn by a generator of their own; the state of the cells
# (and everything else) is saved before and put back after, so that reading changes nothing.
#   python3 -I warmup.py <label> seed=.. warm=W1|W2|W3|none reps=1|5|20 add=none|a|b|c [nc=256 maxn=9 held=20 life=60000]
import sys, os, json
import numpy as np
label = sys.argv[1]; kw = dict(a.split('=') for a in sys.argv[2:])
SEED = int(kw.get('seed', 1)); MAXN = int(kw.get('maxn', 9)); HELD = int(kw.get('held', 20)); LIFE = int(kw.get('life', 60000))
WARM = kw.get('warm', 'none'); REPS = int(kw.get('reps', 1)); ADD = kw.get('add', 'none'); NC = int(kw.get('nc', 256))
rt = np.random.default_rng(SEED); rb = np.random.default_rng(SEED + 1); rc = np.random.default_rng(SEED + 2); rr = np.random.default_rng(SEED + 3)
pairs = [(a, b) for a in range(MAXN + 1) for b in range(MAXN + 1)]; rt.shuffle(pairs); held = pairs[:HELD]; shown = pairs[HELD:]
warmset = [shown[i] for i in rt.integers(len(shown), size=1000)]
DMAX = 6; RATE = 1 / 1024
d1 = rb.integers(0, DMAX - 1, NC); d2 = np.array([rb.integers(a + 1, DMAX) for a in d1])
S = dict(g1=rb.normal(size=NC), g2=rb.normal(size=NC), c=rb.normal(size=NC), m=np.zeros(NC), q=np.zeros(NC), D1=np.zeros(NC), D2=np.zeros(NC),
         hist=np.zeros(DMAX, np.int64), step=np.zeros(1, np.int64), made=np.zeros(NC, bool), r0=np.zeros(NC), gate=np.full(NC, -1, np.int64),
         W=np.zeros((256, NC + 1)), seen=np.zeros(256, bool))
tmade = rb.integers(0, 2048, NC) if ADD in ('b', 'c') else np.zeros(NC, np.int64)
F = np.arange(256) / 255.0


def cells():
    h = S['hist']; u1 = F[h[d1]]; u2 = F[h[d2]]
    r = S['g1'] * u1 + S['g2'] * u2
    k = int(S['step'][0])
    new = (~S['made']) & (tmade <= k)
    if new.any():
        S['made'][new] = True; S['r0'][new] = r[new]; S['gate'][new] = h[0]
    if ADD == 'b':
        unit = (np.abs(S['g1']) + np.abs(S['g2'])) / 2 / 255
        y = ((np.abs(r - S['r0']) <= unit / 2) & S['made']).astype(float)
    else:
        sd = np.sqrt(np.maximum(S['q'] - S['m'] ** 2, 1e-8))
        y = np.maximum(0.0, (r - S['m']) / sd - S['c'])
        S['m'] += RATE * (r - S['m']); S['q'] += RATE * (r * r - S['q'])
        if ADD == 'c':
            y = y * (S['made'] & (S['gate'] == h[0]))
    if ADD == 'a':
        S['D1'] += RATE * (S['g1'] * u1 - S['D1']); S['D2'] += RATE * (S['g2'] * u2 - S['D2'])
        a1 = np.abs(S['D1']) + 1e-12; a2 = np.abs(S['D2']) + 1e-12; mn = (a1 + a2) / 2
        G1 = np.abs(S['g1']) * (1 + RATE * (mn / a1 - 1)); G2 = np.abs(S['g2']) * (1 + RATE * (mn / a2 - 1)); tot = (G1 + G2) / 2
        S['g1'] = np.sign(S['g1']) * G1 / tot; S['g2'] = np.sign(S['g2']) * G2 / tot
    S['step'][0] += 1
    return y


def push(v):
    h = S['hist']; h[1:] = h[:-1]; h[0] = v


def line(a, b, mode, rng):
    """mode: Q, D (told the right byte), none (cells only), read (no learning). Returns (whole answer right, answer length)."""
    s = f"{a}+{b}={a + b}\n".encode(); eq = s.index(b'='); allok = True; W = S['W']
    for t in range(len(s) - 1):
        push(s[t]); y = cells(); U = np.concatenate([y, np.ones(1)]); n = float(U @ U)
        S['seen'][s[t]] = True; yy = s[t + 1]
        if mode == 'none':
            continue
        est = W @ U; e = int(est.argmax())
        if mode == 'Q' and rng.random() < 1 / 16: sv = np.flatnonzero(S['seen']); e = int(sv[rng.integers(len(sv))])
        ok = e == yy
        if t >= eq and yy != 10: allok = allok and ok
        if mode == 'Q':
            d = np.zeros(256); d[e] = (float(ok) - est[e]) / 16
            if ok: d = -est / 16; d[e] = (1.0 - est[e]) / 16
            nz = np.flatnonzero(d); W[nz] += np.outer(d[nz] / n, U)
        elif mode == 'D':
            tg = np.zeros(256); tg[yy] = 1.0; W += np.outer((tg - est) / 16 / n, U)
    push(s[-1]); cells()
    return allok


def read():
    keep = {k: v.copy() for k, v in S.items()}
    res = {}
    for nm, ps in (('shown', shown), ('never', held)):
        res[nm] = float(np.mean([line(*ps[rr.integers(len(ps))], 'read', None) for _ in range(2000)]))
    for k, v in keep.items(): S[k] = v
    return res


out = dict(label=label, args=kw, reads=[])
total = 0
if WARM != 'none':
    mode = {'W1': 'Q', 'W2': 'D', 'W3': 'none'}[WARM]
    for rep in range(REPS):
        for j in rt.permutation(len(warmset)):
            line(*warmset[j], mode, rc); total += 1
    out['reads'].append(dict(at='end of warm-up', lines=total, **read()))
    marks = {10000: '10,000', 30000: '30,000', 60000: '60,000'}
    life = LIFE
else:
    marks = {m: str(m) for m in sorted({1000 * R + x for R in (1, 5, 20) for x in (0, 10000, 30000, 60000)} | {10000, 30000, 60000})}
    life = LIFE + 20000
for i in range(1, life + 1):
    line(*shown[rt.integers(len(shown))], 'Q', rc); total += 1
    if i in marks:
        out['reads'].append(dict(at=f'life line {i}', lines=total, **read()))
os.makedirs('resW', exist_ok=True); json.dump(out, open(f'resW/{label}.json', 'w'))
print(f"== {label}: " + " | ".join(f"{r['at']} ({r['lines']}): {100 * r['shown']:.1f} / {100 * r['never']:.1f}" for r in out['reads']), flush=True)
