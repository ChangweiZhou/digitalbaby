# Is what the next byte depends on present in the cells near the output lines?  A linear reader (a measuring device, not a part of the
# tissue) predicts the NEXT byte of the stream from the cells' outputs at the reading step of the present byte.
#
# usage: reach.py <label> rule=fixed [options of drive.py: gain=2 rho=1 gmax=4 seed=.. nbytes=..]
#
# Cell sets:
#   (a) near the output lines: the 256 driver cells of the output lines and every cell a driver can receive from (its neighbourhood,
#       the cells within the radius; input lines are not cells), feedback cells left out;
#   (b) every cell that is not a feedback cell.
# Features: the output x of each cell at the reading step (step 32 of the byte) of byte t, divided by the cell's own standard deviation
#   over the fitting part (no floor) and centred on its mean over the fitting part; a cell whose output does not vary over the fitting part
#   (standard deviation 0) is left out and counted. Target: the byte at t + 1.
# Reader: ridge regression onto the 256 one-hot byte values, plus a constant; the byte given is the value of largest output. Fitted on the
#   first three quarters of the positions t (t = 0 .. NB-2), scored on the last quarter. The ridge constant is chosen from
#   10^-6 .. 10^4 (per position, times the number of fitting positions) on the last fifth of the fitting part with the first four fifths,
#   then the reader is fitted again on the whole fitting part with that constant; the scored quarter is not used in any choice.
# The same reader is fed with the present byte alone (one-hot, 256) and with the present and the previous byte (two one-hots, 512).
# References on the same scored positions, from the main worker's scorer (deps/PREP/score_emissions.py, 10e3ab9c on
#   claude/e63-score-emissions): ref_mode, ref_1, ref_2, ref_3, each fitted on all earlier positions of the stream. The tissue's stream is
#   the first NB bytes of life(seed) (checked: the same bytes), so the scorer's kinds apply; in the first 65,536 bytes only R1 and R2 occur.
import sys, os, json, time, ctypes, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, 'deps', 'PREP'))
import tis, life_stream as LS, score_emissions as SE
label = sys.argv[1]; kw = dict(a.split('=') for a in sys.argv[2:])
t0 = time.time()
T = tis.make(kw); NB = T['NB']; N = T['N']; ts = tis.ts; t = T['t']; L_ = T['L']; b = T['b']; B = T['B']; A = T['A']
isfb = B['kind'] == ts.FBK
drivers = list(B['driver'])
near = set(drivers)
for i in drivers:
    near.update(int(m) for m in B['members'][i] if m < N)
near = np.array(sorted(i for i in near if not isfb[i]))
allc = np.flatnonzero(~isfb)
# record every cell's output at the reading step of every byte
rec = np.zeros((NB, N), np.float32)
L_.tis_run(ctypes.byref(t), ctypes.c_void_p(b.ctypes.data), ctypes.c_int64(0), ctypes.c_int64(NB), ctypes.c_int64(0), ctypes.c_int64(NB), ctypes.c_void_p(rec.ctypes.data))
t_run = time.time() - t0
# distances from the input lines (lines of the byte values present in the stream), as in diag.py
lines = set(ts.LINE0 + int(v) for v in set(b.tolist()))
deg = A['deg']; src = A['c_src']
dist = np.zeros(N, np.int16)
fr = np.array([(not isfb[i]) and any(int(s) in lines for s in src[i, :deg[i]]) for i in range(N)]); dist[fr] = 1; d = 1
while fr.any():
    nxt = np.array([(dist[i] == 0) and (not isfb[i]) and bool(fr[src[i, :deg[i]][src[i, :deg[i]] < N]].any()) for i in range(N)])
    d += 1; dist[nxt] = d; fr = nxt
hist = lambda cs: {str(k): int(v) for k, v in zip(*np.unique(dist[cs], return_counts=True))}   # key 0: not reached from the input lines

# positions
tt = np.arange(NB - 1); y = b[1:].astype(np.int64)                     # feature at byte t, target the byte at t + 1
nf = (3 * len(tt)) // 4; fit = tt[:nf]; test = tt[nf:]
nin = (4 * nf) // 5
Y = np.zeros((len(tt), 256)); Y[np.arange(len(tt)), y] = 1.0

def ridge(X, label_):
    """X: features at positions tt. Returns the bytes given on the scored quarter, the ridge constant chosen, and the shares on the inner check."""
    def solve(Xa, Ya, lam):
        mu = Xa.mean(0); my = Ya.mean(0); Xc = Xa - mu
        G = Xc.T @ Xc; G[np.diag_indices_from(G)] += lam * len(Xa)
        W = np.linalg.solve(G, Xc.T @ (Ya - my))
        return lambda Z: (Z - mu) @ W + my
    inner = {}
    for e in range(-6, 5):
        f = solve(X[:nin], Y[:nin], 10.0 ** e)
        inner[e] = float((f(X[nin:nf]).argmax(1) == y[nin:nf]).mean())
    best = max(inner, key=lambda e: (inner[e], e))
    f = solve(X[:nf], Y[:nf], 10.0 ** best)
    return f(X[nf:]).argmax(1), 10.0 ** best, inner

def cells_features(cs):
    X = rec[:NB - 1][:, cs].astype(np.float64)
    sd = X[:nf].std(0); keep = sd > 0
    return X[:, keep] / sd[keep], int((~keep).sum())

L = LS.life(T['SEED'])
assert np.array_equal(L['bytes'][:NB], b), 'the tissue stream is not the life of this seed'
refs = SE.references(L['bytes'][:NB])
K = SE.kinds(L)
pos = test + 1                                                            # stream positions scored
kmask = {'all': np.ones(len(pos), bool)}
for kn, m in K.items():
    mm = m[:NB][pos]
    if mm.any(): kmask[kn] = mm
r12 = np.zeros(len(pos), bool)
for kn in kmask:
    if kn.startswith('R'): r12 |= kmask[kn]
kmask['rest (not R1, R2)'] = ~r12

def shares(given):
    right = given == b[pos]
    return {kn: dict(n=int(m.sum()), right=float(right[m].mean())) for kn, m in kmask.items()}

out = dict(label=label, args=kw, bytes=NB, fit_positions=int(nf), scored_positions=int(len(pos)), scored_stream_positions=[int(pos[0]), int(pos[-1])],
           seconds_tissue=t_run, mean_output_all=float(rec[:, allc].mean()))
out['sets'] = {'a_near_output': dict(cells=int(len(near)), drivers=len(drivers), distance_from_input_lines=hist(near),
                                     mean_output=float(rec[:, near].mean())),
               'b_all_non_feedback': dict(cells=int(len(allc)), distance_from_input_lines=hist(allc), mean_output=float(rec[:, allc].mean()))}
res = {}
for name, cs in (('a_near_output', near), ('b_all_non_feedback', allc)):
    X, dropped = cells_features(cs)
    g, lam, inner = ridge(X, name)
    res[name] = dict(features=int(X.shape[1]), cells_constant_left_out=dropped, ridge=lam, inner_check=inner, shares=shares(g))
OH1 = np.zeros((NB - 1, 256)); OH1[tt, b[tt]] = 1
OH2 = np.zeros((NB - 1, 512)); OH2[tt, b[tt]] = 1; OH2[tt[1:], 256 + b[tt[1:] - 1].astype(np.int64)] = 1
for name, X in (('onehot_present', OH1), ('onehot_present_previous', OH2)):
    g, lam, inner = ridge(X, name)
    res[name] = dict(features=int(X.shape[1]), ridge=lam, inner_check=inner, shares=shares(g))
for rn, rv in refs.items():
    res[rn] = dict(shares=shares(rv[pos].astype(np.int64)))
out['readers'] = res; out['seconds_all'] = time.time() - t0
os.makedirs('reach', exist_ok=True); json.dump(out, open(os.path.join(HERE, 'reach', f'{label}.json'), 'w'), indent=1, default=float)
order = ['a_near_output', 'b_all_non_feedback', 'onehot_present', 'onehot_present_previous', 'ref_mode', 'ref_1', 'ref_2', 'ref_3']
ks = list(kmask)
print(f"== {label} ({NB} bytes, scored {len(pos)}; set a {len(near)} cells, distances {hist(near)}; tissue {t_run:.0f} s)")
print('   ' + ' | '.join(f'{k.split(" (")[0]} n={kmask[k].sum()}' for k in ks))
for nm in order:
    print(f'   {nm:24s} ' + ' '.join(f"{100 * res[nm]['shares'][k]['right']:6.2f}" for k in ks)
          + (f"   [features {res[nm]['features']}, ridge {res[nm]['ridge']:.0e}]" if 'ridge' in res[nm] else ''))
sys.stdout.flush()
