# Reviewer, 19 November 2026. The ceiling of memory held at the output unit's own connections: the EXACT present byte as a one-hot
# (no tissue), each of the 256 input lines carrying beside its value x slow copies s_k += (x - s_k) / k once a byte (k from traces=),
# scaled by sqrt(2k - 1); a constant 1; n = |U|^2. Rules D (told the right byte, eta 1/64 and 1/16) and Q (yes or no, eta 1/16 and
# 1/4; the emitter toward ok, the others toward 0 on yes), as yq.py and outq.py. It says what the output end would learn if the
# tissue brought the present byte to it whole.
#   python3 -I yq_tr.py <label> seed=.. nbytes=.. traces=2,4   -> res/<label>.json
import sys, os, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, 'deps'))
import life_stream as LS, score_emissions as SE
label = sys.argv[1]; kw = dict(a.split('=') for a in sys.argv[2:]); t0 = time.time()
SEED = int(kw.get('seed', 73129)); NB = int(kw.get('nbytes', 65536)); TR = [float(k) for k in kw.get('traces', '2,4').split(',') if k and k != 'none']
L = LS.life(SEED); b = L['bytes'][:NB].astype(np.int64)
RULES = [('D', 1 / 64), ('D', 1 / 16), ('Q', 1 / 16), ('Q', 1 / 4)]; names = ['D 1/64', 'D 1/16', 'Q 1/16', 'Q 1/4']
F = 256 * (1 + len(TR)) + 1; C = len(RULES); W = np.zeros((C, 256, F)); rng = np.random.default_rng(SEED)
right = np.zeros((C, NB), bool); seen = np.zeros(256, bool); tr = [np.zeros(256) for _ in TR]
for t in range(NB - 1):
    x = np.zeros(256); x[b[t]] = 1.0
    for j, k in enumerate(TR): tr[j] += (x - tr[j]) / k
    U = np.concatenate([x] + [tr[j] * np.sqrt(2 * k - 1) for j, k in enumerate(TR)] + [np.ones(1)]); n = float(U @ U)
    seen[b[t]] = True; yy = b[t + 1]; sv = np.flatnonzero(seen)
    for c, (rule, eta) in enumerate(RULES):
        a = W[c] @ U; e = int(a.argmax())
        if rule == 'Q' and rng.random() < 1 / 16: e = int(sv[rng.integers(0, len(sv))])
        ok = e == yy; right[c, t + 1] = ok
        if rule == 'D': d = -a; d[yy] += 1.0; d *= eta
        else:
            d = np.zeros(256); d[e] = eta * (float(ok) - a[e])
            if ok: d = -eta * a; d[e] = eta * (1.0 - a[e])
        nz = np.flatnonzero(d); W[c][nz] += np.outer(d[nz] / n, U)
refs = SE.references(L['bytes'][:NB]); q0 = 3 * NB // 4
res = {nm: dict(last_quarter=float(right[c, q0:].mean()), quarters=[float(right[c, i * NB // 4:(i + 1) * NB // 4].mean()) for i in range(4)]) for c, nm in enumerate(names)}
out = dict(label=label, seed=SEED, bytes=NB, traces=TR, rules=res, last_quarter_refs={k: float((refs[k] == b)[q0:].mean()) for k in ('ref_1', 'ref_2', 'ref_3')}, seconds=time.time() - t0)
json.dump(out, open(os.path.join(HERE, 'res', f'{label}.json'), 'w'), indent=0)
print(f"== {label} (seed {SEED}, {NB} bytes, traces {TR}; refs " + ' '.join(f"{k} {100 * v:.2f}" for k, v in out['last_quarter_refs'].items()) + f"; {time.time() - t0:.0f} s)")
for k, v in res.items(): print(f"   {k:8s} last quarter {100 * v['last_quarter']:5.2f}   quarters " + ' '.join(f"{100 * x:5.2f}" for x in v['quarters']))
