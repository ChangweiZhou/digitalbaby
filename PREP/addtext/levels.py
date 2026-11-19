# levels.py, by E52's worker (19 November 2026): the byte given to cells as a LEVEL, on addtext.py's text and score (PREP/addtext/addtext.py,
# the reviewer's, unchanged beside this file). Lines "a+b=c\n" read one byte at a time; one answering cell names the next byte and is told
# yes or no (rule Q, as addtext.py); scored on the whole answer, for sums shown in learning and sums never shown (HELD pairs kept out until
# the last block, in which nothing is learned).
#
# Receiving cells (NC of them). Cell i takes two of the stream's recent bytes, at delays d1_i < d2_i drawn at birth from 0..5, so that the
# two numbers of a line can meet in one cell's drive:
#   input  = "level":    u(t) = value(t) / 255, the byte's value itself as a level;
#            "scrambled": u(t) = perm(value(t)) / 255, a fixed random permutation of the 256 values drawn at birth: every byte stays
#                         distinguishable, the order of the values is destroyed;
#            "onehot":   the byte as one line per value (as addtext.py): the cell has a strength for each value at each of its two
#                         delays, drawn at birth, N(0, 1), and its raw drive is the sum of the two strengths of the values present.
#   raw drive (level, scrambled) r_i(t) = g1_i u(t - d1_i) + g2_i u(t - d2_i), gains N(0, 1) drawn at birth.
#   cells = "fixed":  output y_i = max(0, G (r_i - theta_i)), G = 8; theta_i drawn at birth uniformly in the range r_i takes over all
#                     byte values 0..255 at both delays (for onehot: N(0, 1)).
#           "adapt":  the cell follows its own drive: a running mean m_i and running mean square q_i of r_i (rate 1/1024, updated after use;
#                     sd_i = sqrt(max(q_i - m_i^2, 1e-8))), and y_i = max(0, (r_i - m_i) / sd_i - c_i), c_i drawn at birth N(0, 1): the
#                     cell's gain becomes 1 / sd_i and its threshold m_i + c_i sd_i, both from its own drive alone. This is the local
#                     rule of part (b).
# The answering cell: inputs U = [y_1 .. y_NC, 1], n = U.U; 256 strengths for each input; estimate W U; rule Q exactly as addtext.py
# (emission the largest, with chance 1/16 a seen value at random while learning; the emitter W[e] += (1/16)(ok - est_e) U / n; on yes
# every other W[v] += (1/16)(0 - est_v) U / n; on no the others nothing).
# Three generators: the text (seed; pairs and items drawn as addtext.py does), the birth (seed + 1), the choices (seed + 2), so that every
# variant on a seed reads the same text.
#   python3 -I levels.py <label> seed=.. input=level|scrambled|onehot cells=fixed|adapt nc=64 [maxn=9 held=20 items=60000]
import sys, os, json
import numpy as np
label = sys.argv[1]; kw = dict(a.split('=') for a in sys.argv[2:])
SEED = int(kw.get('seed', 1)); MAXN = int(kw.get('maxn', 9)); HELD = int(kw.get('held', 20)); NIT = int(kw.get('items', 60000))
INP = kw.get('input', 'level'); CEL = kw.get('cells', 'fixed'); NC = int(kw.get('nc', 64))
rt = np.random.default_rng(SEED); rb = np.random.default_rng(SEED + 1); rc = np.random.default_rng(SEED + 2)
pairs = [(a, b) for a in range(MAXN + 1) for b in range(MAXN + 1)]; rt.shuffle(pairs); held = pairs[:HELD]; shown = pairs[HELD:]
DMAX = 6
d1 = rb.integers(0, DMAX - 1, NC); d2 = np.array([rb.integers(a + 1, DMAX) for a in d1])
perm = rb.permutation(256)
if INP == 'onehot':
    S1 = rb.normal(size=(NC, 256)); S2 = rb.normal(size=(NC, 256))
    theta = rb.normal(size=NC)
else:
    g1 = rb.normal(size=NC); g2 = rb.normal(size=NC)
    lo_ = np.minimum(g1, 0) + np.minimum(g2, 0); hi_ = np.maximum(g1, 0) + np.maximum(g2, 0)      # range of r over u in [0, 1]
    theta = lo_ + rb.random(NC) * (hi_ - lo_)
c = rb.normal(size=NC); m = np.zeros(NC); q = np.zeros(NC); RATE = 1 / 1024; GF = 8.0
hist = np.zeros(DMAX, np.int64)                                                                    # hist[d] = value at t - d
W = np.zeros((256, NC + 1)); seen = np.zeros(256, bool)
ar = np.arange(NC)


def cells():
    v1 = hist[d1]; v2 = hist[d2]
    if INP == 'onehot':
        r = S1[ar, v1] + S2[ar, v2]
    else:
        f = (perm if INP == 'scrambled' else np.arange(256)) / 255.0
        r = g1 * f[v1] + g2 * f[v2]
    if CEL == 'fixed':
        y = np.maximum(0.0, GF * (r - theta))
    else:
        global m, q
        sd = np.sqrt(np.maximum(q - m * m, 1e-8))
        y = np.maximum(0.0, (r - m) / sd - c)
        m += RATE * (r - m); q += RATE * (r * r - q)
    return y


def push(v):
    hist[1:] = hist[:-1]; hist[0] = v


def run(ps, n_items, learn):
    global W
    ans = []; whole = []; wlen = {}
    for it in range(n_items):
        a, b = ps[rt.integers(len(ps))]; line = f"{a}+{b}={a + b}\n".encode(); eq = line.index(b'='); allok = True
        for t in range(len(line) - 1):
            push(line[t]); y = cells()
            U = np.concatenate([y, np.ones(1)]); n = float(U @ U)
            seen[line[t]] = True; yy = line[t + 1]; est = W @ U; e = int(est.argmax())
            if learn and rc.random() < 1 / 16: sv = np.flatnonzero(seen); e = int(sv[rc.integers(len(sv))])
            ok = e == yy
            if t >= eq and yy != 10: ans.append(ok); allok = allok and ok
            if learn:
                d = np.zeros(256); d[e] = (float(ok) - est[e]) / 16
                if ok: d = -est / 16; d[e] = (1.0 - est[e]) / 16
                nz = np.flatnonzero(d); W[nz] += np.outer(d[nz] / n, U)
        push(line[-1]); cells()
        whole.append(allok); wlen.setdefault(len(str(a + b)), []).append(allok)
    return dict(answer=float(np.mean(ans)), whole=float(np.mean(whole)), by_len={str(k): [float(np.mean(v)), len(v)] for k, v in sorted(wlen.items())})


rows = []
for blk in range(6): rows.append(['learning block %d' % (blk + 1), run(shown, NIT // 6, True)])
rows.append(['sums shown, no learning', run(shown, 2000, False)]); rows.append(['sums NEVER shown', run(held, 2000, False)])
os.makedirs('resL', exist_ok=True); json.dump(dict(label=label, args=kw, rows=rows), open(f'resL/{label}.json', 'w'))
print(f"== {label}: whole answer right % (answer bytes %)  " + " | ".join(f"{r[0].split(',')[0][:14]} {100 * r[1]['whole']:.1f} ({100 * r[1]['answer']:.1f})" for r in rows), flush=True)
