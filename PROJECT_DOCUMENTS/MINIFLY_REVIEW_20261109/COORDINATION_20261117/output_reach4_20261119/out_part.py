# The output part as many cells (Mr. Xu's rulings of 19 November, 14:24-14:33 UTC; the reviewer's task of 14:42 UTC), on a real tissue's
# activity. The tissue of tis.make is run once (65,536 bytes) and its cells' outputs at the reading step of every byte are kept; the output
# part only reads them (the tissue is not changed by it). The rule follows the reviewer's yesno/yq_out.py (inh and adapt as there).
#   Output cells: M at random coordinates (numpy.random.default_rng(seed * 7 + M + 1000 * place)), x uniform in [0, 1] ("anywhere") or in
#   [0, 1/4] ("input side", within 1/4 of the input side x = 0), y uniform in [0, 1]. Output cell j reads the non-feedback cells within 1/8
#   of its coordinate; each input is standardised by a running mean and deviation (as measure.py) and clipped at +-8; two slow copies
#   s_k += (u - s_k) / k (k = 2, 4) of each input, scaled by sqrt(2k - 1) as in yq_out.py; and a constant 1. U_j has 3 K_j + 1 entries.
#   Cell j holds a_j[v] = W_j[v] U_j for every byte value v. At each byte the cells are taken in order of their best estimate, largest first;
#   cell j speaks its best value if (best estimate - inhibition received at this byte from cells that already spoke) >= th_j; a speaking
#   cell adds G[k, j] to the inhibition of every other cell k. Besides, each cell with chance eps (1/64 for M = 8, 1/256 for M = 32) speaks a
#   value seen so far at random. The world answers yes if at least one cell spoke and every spoken byte is the next byte, otherwise no.
#   Every cell that spoke: W_j[e] += (1/16)(ok - a_j[e]) U_j / n_j, n_j = |U_j|^2. On no: th falls by 1/1024 for silent cells and rises by
#   1/1024 for cells that spoke (th held in [0, 1], as yq_out.py); if more than one spoke, G between each pair of them grows by 1/64.
#   Start W = 0, G = 0, th = 1/8.
# Reported by quarter of the 65,536 bytes: the share answered yes; ref_1, ref_2 of the scorer on the same positions; mean number of cells
# speaking; share of bytes at which no cell spoke; share with one voice (at least one cell spoke and all spoken bytes are the same value).
# usage: out_part.py <label> [tissue options] seed=.. nbytes=..   -> res/outpart_<label>.json
import sys, os, json, time, ctypes, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, 'deps', 'PREP'))
import tis, life_stream as LS, score_emissions as SE
label = sys.argv[1]; kw = dict(a.split('=') for a in sys.argv[2:]); t0 = time.time()
T = tis.make(kw); NB = T['NB']; N = T['N']; ts = tis.ts; B = T['B']; b = T['b'].astype(np.int64); isfb = B['kind'] == ts.FBK; SEED = T['SEED']
L = LS.life(SEED); assert np.array_equal(L['bytes'][:NB], T['b'])
Pxy = np.array(B['P'])
R = np.zeros((NB, N), np.float32)
for lo in range(0, NB, 4096):
    hi = min(NB, lo + 4096)
    T['L'].tis_run(ctypes.byref(T['t']), ctypes.c_void_p(T['b'].ctypes.data), ctypes.c_int64(lo), ctypes.c_int64(hi), ctypes.c_int64(lo), ctypes.c_int64(hi), ctypes.c_void_p(R[lo:hi].ctypes.data))
t_tis = time.time() - t0
refs = SE.references(L['bytes'][:NB]); q = [slice(i * NB // 4, (i + 1) * NB // 4) for i in range(4)]
TR = (2.0, 4.0)
out = dict(label=label, args=kw, bytes=NB, seconds_tissue=t_tis, refs={k: [float((refs[k] == b)[s].mean()) for s in q] for k in ('ref_1', 'ref_2', 'ref_3')},
           end_mark=[float((b[s] == 10).mean()) for s in q], runs={})
for place, x_hi in (('anywhere', 1.0), ('input side', 0.25)):
    for M in (8, 32):
        t1 = time.time(); EPS = 1 / 64 if M == 8 else 1 / 256
        rc = np.random.default_rng(SEED * 7 + M + 1000 * (place == 'input side'))
        xy = np.stack([rc.random(M) * x_hi, rc.random(M)], 1)
        ins = [np.flatnonzero((((Pxy - xy[j]) ** 2).sum(1) <= 1 / 64) & ~isfb) for j in range(M)]
        cells = np.unique(np.concatenate(ins)); col = {c: k for k, c in enumerate(cells)}; Kc = len(cells)
        idx = [np.array([col[c] for c in i], np.int64) for i in ins]
        W = [np.zeros((256, 3 * len(i) + 1)) for i in ins]
        th = np.full(M, 1 / 8); G = np.zeros((M, M)); rng = np.random.default_rng(SEED)
        m1 = np.zeros(Kc); s1 = np.zeros(Kc); sl = [np.zeros(Kc) for _ in TR]
        right = np.zeros(NB, bool); nsp = np.zeros(NB, np.int16); one = np.zeros(NB, bool); seen = np.zeros(256, bool)
        for t in range(NB - 1):
            x = R[t, cells].astype(np.float64); r = 1.0 / min(t + 1, 1024)
            m1 += r * (x - m1); s1 += r * ((x - m1) ** 2 - s1); sd = np.sqrt(s1)
            u = np.clip(np.where(sd > 0, (x - m1) / np.where(sd > 0, sd, 1), 0.0), -8, 8)
            for i, k in enumerate(TR): sl[i] += (u - sl[i]) / k
            seen[b[t]] = True; yy = b[t + 1]; sv = np.flatnonzero(seen)
            Us, As = [], np.zeros((M, 256))
            for j in range(M):
                Uj = np.concatenate([u[idx[j]]] + [sl[i][idx[j]] * np.sqrt(2 * k - 1) for i, k in enumerate(TR)] + [np.ones(1)])
                Us.append(Uj); As[j] = W[j] @ Uj
            best = As.argmax(1); am = As[np.arange(M), best]; val = best.copy()
            speak = np.zeros(M, bool); sup = np.zeros(M)
            for j in np.argsort(-am, kind='stable'):
                if am[j] - sup[j] >= th[j]: speak[j] = True; sup += G[:, j]
            rr = rng.random(M) < EPS
            for j in np.flatnonzero(rr): val[j] = int(sv[rng.integers(0, len(sv))]); speak[j] = True
            sp = np.flatnonzero(speak); ok = len(sp) > 0 and bool((val[sp] == yy).all())
            right[t + 1] = ok; nsp[t + 1] = len(sp); one[t + 1] = len(sp) > 0 and len(set(val[sp].tolist())) == 1
            for j in sp:
                Uj = Us[j]; W[j][val[j]] += (1 / 16) * (float(ok) - As[j, val[j]]) / float(Uj @ Uj) * Uj
            if not ok:
                if len(sp) > 1:
                    for j in sp:
                        for k2 in sp:
                            if j != k2: G[j, k2] += 1 / 64
                th[~speak] -= 1 / 1024; th[speak] += 1 / 1024; np.clip(th, 0.0, 1.0, out=th)
        key = f'{place}, M={M}'
        out['runs'][key] = dict(coordinates=xy.tolist(), inputs_per_cell=[int(len(i)) for i in ins], eps=EPS, yes=[float(right[s].mean()) for s in q],
                                spoke_mean=[float(nsp[s].mean()) for s in q], none_spoke=[float((nsp[s] == 0).mean()) for s in q],
                                one_voice=[float(one[s].mean()) for s in q], th_final=th.tolist(), G_max=float(G.max()), seconds=time.time() - t1)
        json.dump(out, open(os.path.join(HERE, 'res', f'outpart_{label}.json'), 'w'), indent=0)
        v = out['runs'][key]
        print(f"== {label} {key} (inputs a cell {min(v['inputs_per_cell'])}-{max(v['inputs_per_cell'])}; {time.time() - t1:.0f} s): yes by quarter "
              + ' '.join(f'{100 * z:.1f}' for z in v['yes']) + ' | ref_1 ' + ' '.join(f'{100 * z:.1f}' for z in out['refs']['ref_1'])
              + ' | ref_2 ' + ' '.join(f'{100 * z:.1f}' for z in out['refs']['ref_2']) + ' | cells speaking ' + ' '.join(f'{z:.2f}' for z in v['spoke_mean'])
              + ' | none ' + ' '.join(f'{100 * z:.0f}' for z in v['none_spoke']) + ' | one voice ' + ' '.join(f'{100 * z:.0f}' for z in v['one_voice']), flush=True)
