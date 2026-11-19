# Is the shortfall of yes-or-no learning a property of the RULE or of the INPUTS? The output end's rules of E55's worker (measure.py of
# output_reach4_20261119 on claude/e55-output-reach-017dSEpW: yes/no A, yes/no B, supervised D, settings as there) given EXACT inputs,
# no tissue: (i) the present byte as a one-hot; (ii) present and previous bytes, two one-hots; (iii) three one-hots; (iv) the three
# plus a one-hot of the class of the present byte in its item (life_stream's labels: TASK, A1, A2, SIGN, LET, B1, B2, EQ, R1, R2, END,
# or none outside the first family), which is the stream's own position in the item.
#
# Every output unit v (one per byte value) sees the same input vector U (the one-hots and a constant 1) and has its weights W[v];
# activity a = W U; the emission is the unit of largest activity (ties to the lower byte value), or, with probability eps, a byte value
# drawn among those seen so far. The rules, as in measure.py:
#   A (kappa, eps): on yes W[e] += 0.25 / n * U; on no W[e] -= kappa * 0.25 / n * U   (e the emitter)
#   B (eps): W[e] += (ok - rbar[e]) * 0.25 / n * U; rbar[e] += (ok - rbar[e]) / 64
#   D (eta): every unit, W[v] += eta * (target_v - a_v) / |U|^2 * U     (told the right byte)
# measure.py scales A and B by 0.25 / P, P the number of a driver's inputs plus one, with inputs standardised to unit variance, so that
# |U|^2 is about P. With one-hot inputs |U|^2 = n = (number of one-hots) + 1 exactly; this program uses n there (the worker's choice).
# The random choices use numpy default_rng(seed), as measure.py.
# Step 2: a yes/no output end with four changes, each local to an output unit told only yes or no (rule Y, fields in order):
#   Y:choice:up:base:decay:s0
#   choice  max  the largest activity, as A and B (with eps 1/16 a random seen value: 'eps');
#           prop the emitter drawn with probability in proportion to the positive part of the activities (all zero: a seen value at
#                random) -- the unit needs nothing more: its firing is a chance event whose rate grows with its activity;
#   up      on "no", every unit that did not emit adds up * step * U (the right one is among them); the emitter subtracts as below --
#           the unit needs to know only that it did not fire and that the answer was no;
#   base    own: the emitter's change is (ok - r_e) step U, r_e the unit's own running share of yes when it emitted (rate 1/64), as B;
#           glob: (ok - r) step U, r one running share of yes over the whole output end (rate 1/64), a signal common to all units;
#           none: +step U on yes, -step U / 8 on no, as A;
#   decay   c: step = s0 * 0.25 / n; s: step = s0 * 0.25 / n / sqrt(1 + m_e / 16), m_e the number of times unit e has emitted -- the
#           unit holds a count of its own firings;
#   s0      the scale of the step.
#   python3 -I yn.py <label> seed=.. nbytes=.. inputs=1|2|3|3c [rules=...]   -> res/<label>.json
import sys, os, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, 'deps'))
import life_stream as LS, score_emissions as SE
label = sys.argv[1]; kw = dict(a.split('=') for a in sys.argv[2:]); t0 = time.time()
SEED = int(kw.get('seed', 73129)); NB = int(kw.get('nbytes', 65536)); INP = kw.get('inputs', '1')
L = LS.life(SEED); b = L['bytes'][:NB].astype(np.int64); cls = L['cls'][:NB].astype(np.int64)
h = int(INP[0]); usecls = INP.endswith('c')
F = 256 * h + (12 if usecls else 0) + 1                                  # features, the last the constant
def active(t):
    idx = [256 * j + b[t - j] for j in range(h) if t - j >= 0]
    if usecls: idx.append(256 * h + cls[t])
    idx.append(F - 1)
    return np.array(idx)
RULES = [r.split(':') for r in kw.get('rules', 'A:0:1/8,A:1/16:1/8,B:1/16:0,D:0:1/64,D:0:1/16').split(',')]
RULES = [(r[0], float(eval(r[1])), float(eval(r[2]))) if r[0] != 'Y' else ('Y', r[1], (float(eval(r[2])), r[3], r[4], float(eval(r[5])))) for r in RULES]
C = len(RULES); W = np.zeros((C, 256, F)); rbar = np.zeros((C, 256)); rglob = np.zeros(C); nemit = np.zeros((C, 256)); rng = np.random.default_rng(SEED)
right = np.zeros((C, NB), bool); seen = np.zeros(256, bool)
for t in range(NB - 1):
    act = active(t); n = len(act); seen[b[t]] = True; yy = b[t + 1]
    for c, (rule, eps, par) in enumerate(RULES):
        a = W[c][:, act].sum(1)
        if rule == 'Y' and eps == 'prop':
            pa = np.maximum(a, 0.0); pa[~seen] = 0.0; tot = pa.sum()
            if tot > 0: e = int(np.searchsorted(np.cumsum(pa), rng.random() * tot, side='right')); e = min(e, 255)
            else: sv = np.flatnonzero(seen); e = int(sv[rng.integers(0, len(sv))])
        else:
            e = int(a.argmax())
        if rule == 'Y' and eps == 'eps' and rng.random() < 1 / 16:
            sv = np.flatnonzero(seen); e = int(sv[rng.integers(0, len(sv))])
        elif rule != 'Y' and eps and rng.random() < eps:
            sv = np.flatnonzero(seen); e = int(sv[rng.integers(0, len(sv))])
        ok = e == yy; right[c, t + 1] = ok
        if rule == 'A':
            W[c, e, act] += (1.0 if ok else -par) * 0.25 / n
        elif rule == 'Y':
            up, base, decay, s0 = par
            st = s0 * 0.25 / n / (np.sqrt(1.0 + nemit[c, e] / 16.0) if decay == 's' else 1.0); nemit[c, e] += 1
            if base == 'own': g = float(ok) - rbar[c, e]; rbar[c, e] += (float(ok) - rbar[c, e]) / 64
            elif base == 'glob': g = float(ok) - rglob[c]; rglob[c] += (float(ok) - rglob[c]) / 64
            else: g = 1.0 if ok else -1.0 / 8
            W[c, e, act] += g * st
            if up and not ok:
                msk = np.ones(256, bool); msk[e] = False
                W[c][np.ix_(msk, act)] += up * s0 * 0.25 / n
        elif rule == 'B':
            W[c, e, act] += (float(ok) - rbar[c, e]) * 0.25 / n; rbar[c, e] += (float(ok) - rbar[c, e]) / 64
        else:
            tg = np.zeros(256); tg[yy] = 1.0; W[c][:, act] += (par * (tg - a) / n)[:, None]
refs = SE.references(L['bytes'][:NB])
wins = [(a_, min(a_ + 4096, NB)) for a_ in range(0, NB, 4096)]
wsh = lambda ok: [float(ok[max(a_, 1):e_].mean()) for a_, e_ in wins]
lv = {k: wsh(refs[k] == b) for k in ('ref_1', 'ref_2', 'ref_3')}; lv['end mark'] = [float((b[a_:e_] == 10).mean()) for a_, e_ in wins]
q0 = (3 * NB) // 4
res = {}
for c, (rule, eps, par) in enumerate(RULES):
    nm = (f'Y {eps} up={par[0]:g} base={par[1]} decay={par[2]} s0={par[3]:g}' if rule == 'Y' else None) or f'{rule}' + (f' eps={eps:g}' if rule != 'D' else '') + (f' kappa={par:g}' if rule == 'A' else '') + (f' eta={par:g}' if rule == 'D' else '')
    w = wsh(right[c])
    res[nm] = dict(windows=w, last_window=w[-1], last_quarter=float(right[c, q0:].mean()),
                   first_window_reaching={k: next((wins[i][1] for i in range(len(w)) if w[i] >= l[i]), None) for k, l in lv.items()})
out = dict(label=label, seed=SEED, bytes=NB, inputs=INP, features=F, rules=res, levels=lv, last_quarter_refs={k: float((refs[k] == b)[q0:].mean()) for k in ('ref_1', 'ref_2', 'ref_3')},
           windows=wins, seconds=time.time() - t0)
os.makedirs(os.path.join(HERE, 'res'), exist_ok=True); json.dump(out, open(os.path.join(HERE, 'res', f'{label}.json'), 'w'), indent=0)
print(f"== {label} (seed {SEED}, {NB} bytes, inputs {INP}; refs last quarter " + ' '.join(f"{k} {100 * v:.2f}" for k, v in out['last_quarter_refs'].items()) + f"; {time.time() - t0:.0f} s)")
for k, v in res.items():
    print(f"   {k:22s} last window {100 * v['last_window']:5.2f}  last quarter {100 * v['last_quarter']:5.2f}  first window reaching {v['first_window_reaching']}")
sys.stdout.flush()
