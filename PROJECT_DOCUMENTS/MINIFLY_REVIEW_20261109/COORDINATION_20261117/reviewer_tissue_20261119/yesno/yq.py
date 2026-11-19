# Reviewer, 19 November 2026. Yes-or-no rules for the output end on EXACT inputs (no tissue), after the main worker's yn.py (branch
# claude/e63-yesno, YESNO/yn.py; deps/ copied from there unchanged). Stream, inputs (one, two or three one-hots of the last bytes
# and a constant), the choice of the emitter and the references are as in yn.py. What is new is the rule Q and its additions:
# a unit's activity a_v = W[v] U is treated as the unit's own estimate of the chance that its byte value is the next byte.
#   Q        only the emitter e changes: W[e] += eta * (ok - a_e) * U / n  (ok 1 on yes, 0 on no). The unit needs its own activity,
#            the fact that it emitted, and yes or no.
#   +z       on yes, every unit that did not emit also moves toward 0: W[v] += eta * (0 - a_v) * U / n (it knows that the answer
#            was yes and that it was not the emitter, hence that its byte value was not the next byte).
#   +u:c     on no, every unit that did not emit raises itself in proportion to its own activity (clipped to [0, 1]):
#            W[v] += eta * c * a_v * U / n, c a constant; with 'un' the factor is a_e / (sum of the others' activities), at most 1,
#            so that what the emitter loses the others gain: it needs the summed activity of the output end, a common signal.
#            (The first form tried, factor a_e / (1 - a_e), ran away to infinity and is removed.)
#   init     i0: all weights start at 0; i1/4: every unit's weight from the constant input starts at 1/4, so that every unit starts
#            with activity 1/4 and is tried until a "no" lowers it.
#   choice   eps: the largest activity, with chance 1/16 a seen byte value at random; prop: drawn in proportion to the positive part of
#            the activities; mixK: with chance 1/K drawn as prop, otherwise the largest.
# Rule D (told the right byte; eta 1/16) and rule B of yn.py are run beside them on the same stream.
#   python3 -I yq.py <label> seed=.. nbytes=.. inputs=1|2|3 rules=Q:eps:1/4:z0:u0:i0,...   -> res/<label>.json
import sys, os, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, 'deps'))
import life_stream as LS, score_emissions as SE
label = sys.argv[1]; kw = dict(a.split('=') for a in sys.argv[2:]); t0 = time.time()
SEED = int(kw.get('seed', 73129)); NB = int(kw.get('nbytes', 65536)); h = int(kw.get('inputs', '1')); GEN = int(kw.get('gen', SEED))
L = LS.life(SEED); b = L['bytes'][:NB].astype(np.int64)
F = 256 * h + 1
RULES = [r.split(':') for r in kw['rules'].split(',')]
C = len(RULES) + 2; W = np.zeros((C, 256, F)); rbar = np.zeros(256); rng = np.random.default_rng(GEN)
right = np.zeros((C, NB), bool); seen = np.zeros(256, bool); nem = np.zeros((C, 256))
for c, r in enumerate(RULES, start=2): W[c][:, F - 1] = float(eval(r[5][1:]))
def pick_prop(a, sv):
    pa = np.maximum(a, 0.0); pa[~seen] = 0.0; tot = pa.sum()
    if tot > 0: return min(int(np.searchsorted(np.cumsum(pa), rng.random() * tot, side='right')), 255)
    return int(sv[rng.integers(0, len(sv))])
for t in range(NB - 1):
    act = np.array([256 * j + b[t - j] for j in range(h) if t - j >= 0] + [F - 1]); n = len(act); seen[b[t]] = True; yy = b[t + 1]; sv = np.flatnonzero(seen)
    # D
    a = W[0][:, act].sum(1); right[0, t + 1] = int(a.argmax()) == yy; tg = np.zeros(256); tg[yy] = 1.0; W[0][:, act] += ((tg - a) / 16 / n)[:, None]
    # B
    a = W[1][:, act].sum(1); e = int(a.argmax())
    if rng.random() < 1 / 16: e = int(sv[rng.integers(0, len(sv))])
    ok = e == yy; right[1, t + 1] = ok; W[1, e, act] += (float(ok) - rbar[e]) * 0.25 / n; rbar[e] += (float(ok) - rbar[e]) / 64
    for c, (rule, choice, eta, z, u, ini) in enumerate(RULES, start=2):
        if '>' in eta:                                                    # hi>lo@K: a unit's step falls with the number of its own emissions m: max(lo, hi / sqrt(1 + m / K))
            hi, r_ = eta.split('>'); lo, K = r_.split('@'); eta = np.maximum(float(eval(lo)), float(eval(hi)) / np.sqrt(1.0 + nem[c] / float(K)))
        else: eta = float(eval(eta))
        a = W[c][:, act].sum(1)
        if choice == 'eps':
            e = int(a.argmax())
            if rng.random() < 1 / 16: e = int(sv[rng.integers(0, len(sv))])
        elif choice == 'prop': e = pick_prop(a, sv)
        else:
            e = pick_prop(a, sv) if rng.random() < 1.0 / float(choice[3:]) else int(a.argmax())
        ok = e == yy; right[c, t + 1] = ok; nem[c, e] += 1
        d = np.zeros(256); d[e] = float(ok) - a[e]
        if ok and z == 'z1':
            d = -a.copy(); d[e] = 1.0 - a[e]
        if (not ok) and u != 'u0':
            ap = np.clip(a, 0.0, 1.0); ap[~seen] = 0.0; rest = ap.sum() - ap[e]
            f = (ap[e] / rest if rest > 1e-9 else 0.0) if u == 'un' else float(eval(u[1:]))
            ae = d[e]; d = min(f, 1.0) * ap; d[e] = ae
        W[c][:, act] += (eta * d / n)[:, None]
refs = SE.references(L['bytes'][:NB]); q0 = (3 * NB) // 4
names = ['D eta=1/16', 'B eps=1/16'] + [':'.join(r) for r in RULES]
res = {nm: dict(last_quarter=float(right[c, q0:].mean()), last_window=float(right[c, NB - 4096:].mean()), quarters=[float(right[c, i * NB // 4:(i + 1) * NB // 4].mean()) for i in range(4)]) for c, nm in enumerate(names)}
out = dict(label=label, seed=SEED, gen=GEN, bytes=NB, inputs=h, rules=res, last_quarter_refs={k: float((refs[k] == b)[q0:].mean()) for k in ('ref_1', 'ref_2', 'ref_3')}, seconds=time.time() - t0)
os.makedirs(os.path.join(HERE, 'res'), exist_ok=True); json.dump(out, open(os.path.join(HERE, 'res', f'{label}.json'), 'w'), indent=0)
print(f"== {label} (seed {SEED}, {NB} bytes, inputs {h}; refs last quarter " + ' '.join(f"{k} {100 * v:.2f}" for k, v in out['last_quarter_refs'].items()) + f"; {time.time() - t0:.0f} s)")
for k, v in res.items(): print(f"   {k:26s} last quarter {100 * v['last_quarter']:5.2f}   quarters " + ' '.join(f"{100 * x:5.2f}" for x in v['quarters']))
