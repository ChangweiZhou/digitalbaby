# The output end alone, learning online, on the cells near each driver of any tissue of tis.make (as yesno.py and yesno2.py of
# output_reach_20261119, whose headers give the emission, the windows and the references). The tissue is run and not changed.
#
# usage: yesno3.py <label> [tissue options of drive.py: rule=.. eta_s=.. no_make=.. norm=.. we=.. wi=.. lines_learn=.. seed=.. nbytes=..]
#
# Inputs of driver v: the set-(a) cells in driver v's own neighbourhood (the driver included, feedback cells excluded) and a constant 1.
# Two ways of preparing them, each run with every rule:
#   run     each cell's output x_j standardised by a running mean and deviation the cell could hold: m += r (x - m),
#           s2 += r ((x - m)^2 - s2), r = 1 / min(n, 1024); input (x - m) / sqrt(s2) (0 where s2 = 0).
#   shared  driver v first removes what its inputs share: x'_j = x_j - mean over its own cells of x at that step; then each x'_j is
#           standardised as above by a running mean and deviation held at driver v's connection j.
# Rules (strengths start at 0; eta_eff = eta / P, P the number of inputs of the unit, constant included):
#   A   +eta_eff u after yes, -kappa eta_eff u after no, kappa 1/8; eps 0 and 1/16 (share of emissions by a driver drawn at random among
#       the bytes seen so far).
#   B   eta_eff (r - rbar_e) u, r = 1 after yes, 0 after no, rbar_e the emitter's own running share of yes (rate 1/64, from 0); eps 1/16.
#   D   supervised delta rule told the right byte: every driver moves by eta (target - activity) u / |u|^2; eta 1/64 and 1/16.
# Output: yesno3/<label>.json: share right in windows of 4,096 bytes and over the last quarter of the positions (the positions on which
# reach.py and reach_nbhd.py score their fitted readers), for every rule and preparation; the references of the scorer.
import sys, os, json, time, ctypes, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, 'deps', 'PREP'))
import tis, life_stream as LS, score_emissions as SE
label = sys.argv[1]; kw = dict(a.split('=') for a in sys.argv[2:]); t0 = time.time()
T = tis.make(kw); NB = T['NB']; N = T['N']; ts = tis.ts; B = T['B']; b = T['b'].astype(np.int64); isfb = B['kind'] == ts.FBK
L = LS.life(T['SEED']); assert np.array_equal(L['bytes'][:NB], T['b'])
drivers = list(B['driver']); near = set(drivers)
for i in drivers: near.update(int(m) for m in B['members'][i] if m < N)
near = np.array(sorted(i for i in near if not isfb[i])); col = {c: k for k, c in enumerate(near)}; K = len(near); D = K + 1
mask = np.zeros((256, D)); mask[:, -1] = 1
for v, i in enumerate(drivers):
    for c in [i] + [int(m) for m in B['members'][i] if m < N]:
        if c in col: mask[v, col[c]] = 1
mc = mask[:, :-1]; cnt = mc.sum(1); P = mask.sum(1)
PREPS = ('run', 'shared')
RULES = [('A', 1 / 16 * 0, 1 / 8), ('A', 1 / 16, 1 / 8), ('B', 1 / 16, 0.0), ('D', 0.0, 1 / 64), ('D', 0.0, 1 / 16)]
CFG = [(p, r) for p in PREPS for r in RULES]; C = len(CFG)
Wt = np.zeros((C, 256, D)); rbar = np.zeros((C, 256)); rng = np.random.default_rng(T['SEED'])
right = np.zeros((C, NB), bool); seen = np.zeros(256, bool); cidx = np.arange(C)
m1 = np.zeros(K); s1 = np.zeros(K); m2 = np.zeros((256, K)); s2 = np.zeros((256, K)); n = 0
CH = 4096
for lo in range(0, NB, CH):
    hi = min(NB, lo + CH); rec = np.zeros((hi - lo, N), np.float32)
    T['L'].tis_run(ctypes.byref(T['t']), ctypes.c_void_p(T['b'].ctypes.data), ctypes.c_int64(lo), ctypes.c_int64(hi), ctypes.c_int64(lo), ctypes.c_int64(hi), ctypes.c_void_p(rec.ctypes.data))
    X = rec[:, near].astype(np.float64); del rec
    for k in range(hi - lo):
        t = lo + k; n += 1; r = 1.0 / min(n, 1024); x = X[k]; seen[b[t]] = True
        m1 += r * (x - m1); s1 += r * ((x - m1) ** 2 - s1); sd = np.sqrt(s1)
        u1 = np.where(sd > 0, (x - m1) / np.where(sd > 0, sd, 1), 0.0)
        xs = (x[None, :] - ((mc @ x) / cnt)[:, None]) * mc             # driver v's inputs less their mean, 256 x K
        m2 += r * (xs - m2); s2 += r * ((xs - m2) ** 2 - s2); sd2 = np.sqrt(s2)
        u2 = np.where(sd2 > 0, (xs - m2) / np.where(sd2 > 0, sd2, 1), 0.0) * mc
        if t + 1 >= NB: break
        Um = {'run': np.concatenate([mc * u1[None, :], np.ones((256, 1))], 1), 'shared': np.concatenate([u2, np.ones((256, 1))], 1)}
        y = b[t + 1]
        for c, (prep, (rule, eps, par)) in enumerate(CFG):
            U = Um[prep]; a = (Wt[c] * U).sum(1); e = int(a.argmax())
            if eps and rng.random() < eps:
                sv = np.flatnonzero(seen); e = int(sv[rng.integers(0, len(sv))])
            ok = e == y; right[c, t + 1] = ok
            if rule == 'A':
                Wt[c, e] += (1.0 if ok else -par) * (0.25 / P[e]) * U[e]
            elif rule == 'B':
                Wt[c, e] += (float(ok) - rbar[c, e]) * (0.25 / P[e]) * U[e]; rbar[c, e] += (float(ok) - rbar[c, e]) / 64
            else:
                tg = np.zeros(256); tg[y] = 1; nr = (U * U).sum(1)
                Wt[c] += (par * (tg - a) / nr)[:, None] * U
W4 = 4096; wins = [(a_, min(a_ + W4, NB)) for a_ in range(0, NB, W4)]
nf = (3 * (NB - 1)) // 4; q0 = nf + 1                                   # reach.py's scored positions: nf + 1 .. NB - 1
wsh = lambda ok: [float(ok[max(a_, 1):e_].mean()) for a_, e_ in wins]
refs = SE.references(L['bytes'][:NB])
R = {}
for c, (prep, (rule, eps, par)) in enumerate(CFG):
    nm = f'{prep} {rule}' + (f' eps={eps:g}' if rule != 'D' else '') + (f' kappa={par:g}' if rule == 'A' else '') + (f' eta={par:g}' if rule == 'D' else '')
    R[nm] = dict(windows=wsh(right[c]), last_quarter=float(right[c, q0:].mean()))
ref = {k: dict(windows=wsh(refs[k] == b), last_quarter=float((refs[k] == b)[q0:].mean())) for k in ('ref_1', 'ref_2', 'ref_3')}
ref['end mark (byte 10)'] = dict(windows=[float((b[a_:e_] == 10).mean()) for a_, e_ in wins], last_quarter=float((b[q0:] == 10).mean()))
out = dict(label=label, args=kw, bytes=NB, cells=int(K), inputs_per_driver=dict(min=int(cnt.min()), median=float(np.median(cnt)), max=int(cnt.max())),
           windows=wins, rules=R, references=ref, seconds=time.time() - t0)
os.makedirs(os.path.join(HERE, 'yesno3'), exist_ok=True); json.dump(out, open(os.path.join(HERE, 'yesno3', f'{label}.json'), 'w'), indent=0)
print(f'== {label} ({NB} bytes, {time.time() - t0:.0f} s); share right %, last window | last quarter')
for k, v in list(R.items()) + list(ref.items()):
    print(f"   {k:32s} {100 * v['windows'][-1]:6.2f} | {100 * v['last_quarter']:6.2f}")
sys.stdout.flush()
