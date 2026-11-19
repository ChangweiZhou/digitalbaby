# What is still laid down in addcheck2.py's system (variant carry), removed one at a time (the reviewer's message of 15:40 UTC).
# One general program; the options:
#   base           as addcheck2.py variant carry: four holding cells laid down (the byte 1 and 2 before '+', the same before '='; a byte that
#                  is not a digit holds 0; released at the end of line); a units adding cell (a1 + b1) and a tens adding cell
#                  (a10 + b10 + its carry cell, which fires when the units adding cell's total > 9); tuned cells (one total each: the drive's
#                  bin floor(drive + 1/2), 0..19) gated by (present byte, byte 1 back); byte cells; a constant; one answering cell, rule Q.
#                  Difference from addcheck2.py: the answering cell holds estimates for the 13 byte values of the text only (memory).
#   carry=learned  (a) no carry cell. The units adding cell's tuned cells, ungated (one for each units total T), feed the tens adding cell's
#                  drive through strengths s_T that start at 0. While learning, the tens drive also carries a fluctuation xi of its own,
#                  drawn once a line from N(0, sigma), sigma = 1/2. Local rule (reward-modulated perturbation): at every byte after '=',
#                  s_T += eta_s (ok - rbar_T) xi for the active T, rbar_T += (ok - rbar_T) / 64 (the connection's own running share of yes);
#                  eta_s = 1/4. What each strength has: its sender's activity, the drive's own fluctuation, the yes or no. Scored with xi = 0.
#   hold=pool      (b) no holding cells chosen by me: a pool of the 39 holding cells (each of the 13 bytes as a gate, the byte 1, 2 or 3 before
#                  it, digits held, others 0, released at the end of line); adding cells over pairs of them with equal strengths: all 741
#                  pairs (pairs=all) or a random number of them (pairs=N); and carry adding cells: a random pair plus the carry cell (total > 9)
#                  of a random pair adding cell, number cpairs=N. Every adding cell has its tuned cells gated as in base.
#   strength=x     (c) the two strengths of each adding cell (on its two holding inputs) start at 1 and 1 - x. eq=1: local rule bringing an
#                  adding cell's input strengths to equal mean contribution: when both inputs are non-zero, each input's running mean
#                  m_i += (h_i - m_i) / 1024 and its strength w_i += eta_c (cbar - w_i m_i) / m_i, cbar the mean of w_j m_j over the cell's two
#                  inputs (what the cell has: its own inputs and its own strengths); eta_c = 1/256.
#   M=8            (2) the output part of several cells as out_part.py: M cells each reading all the features, taken in order of their best
#                  estimate, a cell speaks if (estimate - inhibition received) >= its threshold; inhibition G grows by 1/64 between cells that
#                  spoke together on a no; thresholds move by 1/1024 on a no (silent down, speakers up); eps 1/64 a cell; rule Q for speakers.
#                  M=1: the single answering cell as before (largest estimate; eps 1/16).
# Text and score as addcheck2.py (whole answer, by part, carry cases apart; learning frozen, no random choice; for M > 1 the part's speech
# with the same order and thresholds, no random speaker: right if exactly the spoken values are all the true byte and at least one spoke).
# Scores are also taken after 60,000, 120,000 (and 240,000 when run) lines.
# usage: addcheck3.py <label> D=100 lines=240000 carry=cell|learned hold=laid|pool pairs=all|N cpairs=N strength=0 eq=0 M=1 seeds=.. orders=5
import sys, os, json, time, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
label = sys.argv[1]; kw = dict(a.split('=') for a in sys.argv[2:]); t0 = time.time()
D = int(kw.get('D', 100)); LINES = int(kw.get('lines', 240000)); SEEDS = [int(s) for s in kw.get('seeds', '73129,93683,116089').split(',')]
ORDERS = int(kw.get('orders', 5)); CARRY = kw.get('carry', 'cell'); HOLD = kw.get('hold', 'laid'); PAIRS = kw.get('pairs', 'all'); CPAIRS = int(kw.get('cpairs', 0))
STR = float(eval(kw.get('strength', '0'))); EQ = int(kw.get('eq', 0)); M = int(kw.get('M', 1)); ETA = 1 / 16
CHECK = [c for c in (60000, 120000, 240000, 480000, 960000) if c <= LINES]
NHIDE = 20 if D == 10 else 2000
ALPH = [ord(c) for c in '0123456789+=\n']; AI = {c: i for i, c in enumerate(ALPH)}; NA = len(ALPH)
DIG = {ord('0') + d: d for d in range(10)}; NG = NA * NA; NT = 20
def line(a, b): return [ord(c) for c in f'{a}+{b}={a + b}\n']
def build(rs):
    """holding cells: list of (gate byte, distance); adding cells: arrays in1, in2, cin (-1 none), and which are units / tens in base."""
    if HOLD == 'laid':
        H = [(ord('+'), 1), (ord('+'), 2), (ord('='), 1), (ord('='), 2)]
        in1 = [0, 1]; in2 = [2, 3]; cin = [-1, 0 if CARRY == 'cell' else -1]
    else:
        H = [(g, k) for g in ALPH for k in (1, 2, 3)]
        allp = [(i, j) for i in range(len(H)) for j in range(i + 1, len(H))]
        sel = allp if PAIRS == 'all' else [allp[i] for i in rs.choice(len(allp), int(PAIRS), replace=False)]
        in1 = [p[0] for p in sel]; in2 = [p[1] for p in sel]; cin = [-1] * len(sel)
        for _ in range(CPAIRS):
            i, j = allp[rs.integers(len(allp))]; in1.append(i); in2.append(j); cin.append(int(rs.integers(len(sel))))
    return H, np.array(in1), np.array(in2), np.array(cin)
class Tissue:
    def __init__(self, H, in1, in2, cin, rs):
        self.H = H; self.in1, self.in2, self.cin = in1, in2, cin; self.A = len(in1)
        self.w1 = np.ones(self.A); self.w2 = np.full(self.A, 1.0 - STR); self.m1 = np.full(self.A, 5.0); self.m2 = np.full(self.A, 5.0)
        self.s = np.zeros(NT); self.rbar = np.zeros(NT); self.xi = 0.0
        self.reset()
    def reset(self): self.hist = [ord('\n')] * 4; self.h = np.zeros(len(self.H)); self.after_eq = False
    def read(self, c):
        self.hist = self.hist[1:] + [c]
        if c == ord('\n'): self.h[:] = 0; self.after_eq = False
        for i, (g, k) in enumerate(self.H):
            if c == g: self.h[i] = DIG.get(self.hist[-1 - k], 0)
        if c == ord('='): self.after_eq = True
    def drives(self):
        d = self.w1 * self.h[self.in1] + self.w2 * self.h[self.in2]
        hc = self.cin >= 0
        if hc.any(): d[hc] += (d[self.cin[hc]] > 9.5).astype(float)          # carry cell: the sender's total > 9 (its drive's bin >= 10)
        if CARRY == 'learned':
            tu = int(np.clip(np.floor(d[0] + 0.5), 0, NT - 1)); d[1] += self.s[tu] + self.xi; self.tu = tu
        return d
    def feats(self):
        d = self.drives(); bins = np.clip(np.floor(d + 0.5), 0, NT - 1).astype(np.int64)
        g = AI[self.hist[-1]] * NA + AI[self.hist[-2]]
        f = np.arange(self.A) * (NT * NG) + bins * NG + g
        return np.concatenate([f, [self.A * NT * NG + AI[self.hist[-1]], self.A * NT * NG + NA]])
    def learn_local(self, ok):
        if CARRY == 'learned' and self.after_eq:
            t = self.tu; self.s[t] += 0.25 * (float(ok) - self.rbar[t]) * self.xi; self.rbar[t] += (float(ok) - self.rbar[t]) / 64
        if EQ:
            a, b = self.h[self.in1], self.h[self.in2]; on = (a > 0) & (b > 0)
            if on.any():
                self.m1[on] += (a[on] - self.m1[on]) / 1024; self.m2[on] += (b[on] - self.m2[on]) / 1024
                cb = (self.w1 * self.m1 + self.w2 * self.m2) / 2
                self.w1[on] += (1 / 256) * (cb[on] - self.w1[on] * self.m1[on]) / self.m1[on]
                self.w2[on] += (1 / 256) * (cb[on] - self.w2[on] * self.m2[on]) / self.m2[on]
runs = []
for seed in SEEDS:
    rs = np.random.default_rng(seed)
    pairs = [(a, b) for a in range(D) for b in range(D)]
    hide = set(map(tuple, np.array(pairs)[rs.choice(len(pairs), NHIDE, replace=False)].tolist()))
    shown = [p for p in pairs if p not in hide]
    for order in range(ORDERS):
        ro = np.random.default_rng(seed * 100 + order)
        H, in1, in2, cin = build(np.random.default_rng(seed * 31 + order))
        tis = Tissue(H, in1, in2, cin, ro); NF = tis.A * NT * NG + NA + 1
        W = np.zeros((M, NA, NF), np.float32); th = np.full(M, 1 / 8); G = np.zeros((M, M)); EPS = 1 / 16 if M == 1 else (1 / 64 if M <= 8 else 1 / 256)
        seen = np.zeros(NA, bool); yes = steps = 0; curve = {}
        def speak(A, explore):
            if M == 1:
                e = int(A[0].argmax())
                if explore and ro.random() < EPS and seen.any():
                    sv = np.flatnonzero(seen); e = int(sv[ro.integers(len(sv))])
                return np.array([0]), np.array([e])
            best = A.argmax(1); am = A[np.arange(M), best]; val = best.copy(); sp = np.zeros(M, bool); sup = np.zeros(M)
            for j in np.argsort(-am, kind='stable'):
                if am[j] - sup[j] >= th[j]: sp[j] = True; sup += G[:, j]
            if explore:
                for j in np.flatnonzero(ro.random(M) < EPS):
                    if seen.any(): sv = np.flatnonzero(seen); val[j] = int(sv[ro.integers(len(sv))]); sp[j] = True
            return np.flatnonzero(sp), val[np.flatnonzero(sp)]
        def score(ps):
            whole = 0; part = {k: [0, 0] for k in ('hundreds', 'tens', 'units', 'end of line')}; carry = [0, 0]; nocarry = [0, 0]
            xi0 = tis.xi; tis.xi = 0.0
            for a, b in ps:
                tis.reset()
                for c in [ord(x) for x in f'\n{a}+{b}=']: tis.read(c)
                ans = f'{a + b}\n'; roles = {2: ['units', 'end of line'], 3: ['tens', 'units', 'end of line'], 4: ['hundreds', 'tens', 'units', 'end of line']}[len(ans)]
                good = True
                for ch, role in zip(ans, roles):
                    f = tis.feats(); A = W[:, :, f].sum(2); sp, vals = speak(A, False)
                    r = len(sp) > 0 and bool((vals == AI[ord(ch)]).all()); part[role][0] += r; part[role][1] += 1; good &= r; tis.read(ord(ch))
                whole += good; k = carry if a % 10 + b % 10 > 9 else nocarry; k[0] += good; k[1] += 1
            tis.xi = xi0; tis.reset()
            return dict(whole=whole / len(ps), parts={k: v[0] / v[1] for k, v in part.items() if v[1]}, carry=carry[0] / max(carry[1], 1), no_carry=nocarry[0] / max(nocarry[1], 1))
        for li in range(LINES):
            a, b = shown[ro.integers(len(shown))]
            if CARRY == 'learned': tis.xi = float(ro.normal(0, 0.5))
            for nxt in line(a, b):
                f = tis.feats(); n = float(len(f)); A = W[:, :, f].sum(2); y = AI[nxt]
                sp, vals = speak(A, True); ok = len(sp) > 0 and bool((vals == y).all()); yes += ok; steps += 1
                for j, e in zip(sp, vals):
                    if M == 1 and ok:
                        dd = ETA * (0.0 - A[j]) / n; dd[e] = ETA * (1.0 - A[j, e]) / n; W[j][:, f] += dd[:, None]
                    else:
                        W[j, e, f] += ETA * (float(ok) - A[j, e]) / n
                if M > 1 and not ok:
                    if len(sp) > 1:
                        for j in sp:
                            for k2 in sp:
                                if j != k2: G[j, k2] += 1 / 64
                    spk = np.zeros(M, bool); spk[sp] = True; th[~spk] -= 1 / 1024; th[spk] += 1 / 1024; np.clip(th, 0.0, 1.0, out=th)
                tis.learn_local(ok); seen[y] = True; tis.read(nxt)
            if li + 1 in CHECK:
                curve[li + 1] = dict(shown=score(shown), never_shown=score(sorted(hide)))
        fin = curve[max(curve)]
        runs.append(dict(seed=seed, order=order, adding_cells=int(tis.A), shown=fin['shown'], never_shown=fin['never_shown'], curve=curve, yes_share=yes / steps,
                         s=tis.s.tolist() if CARRY == 'learned' else None, w_ratio=[float(np.median(tis.w2 / tis.w1)), float((tis.w2 / tis.w1).min()), float((tis.w2 / tis.w1).max())]))
        print(f"{label} seed {seed} order {order}: adding cells {tis.A}; " + ' | '.join(f"{c}: shown {100 * v['shown']['whole']:.1f} never {100 * v['never_shown']['whole']:.1f} (units {100 * v['never_shown']['parts']['units']:.1f}, tens {100 * v['never_shown']['parts'].get('tens', 0):.1f})" for c, v in curve.items())
              + (f" | s_T {np.round(tis.s, 2).tolist()}" if CARRY == 'learned' else '') + (f" | w2/w1 median {np.median(tis.w2 / tis.w1):.3f}" if STR or EQ else ''), flush=True)
q = lambda f: [100 * min(f(r) for r in runs), 100 * float(np.median([f(r) for r in runs])), 100 * max(f(r) for r in runs)]
summ = {'shown whole': q(lambda r: r['shown']['whole']), 'never shown whole': q(lambda r: r['never_shown']['whole']), 'never shown, carry': q(lambda r: r['never_shown']['carry']),
        'never shown, no carry': q(lambda r: r['never_shown']['no_carry'])}
for k in ('hundreds', 'tens', 'units', 'end of line'):
    if k in runs[0]['never_shown']['parts']: summ[f'never shown, {k}'] = q(lambda r: r['never_shown']['parts'][k])
json.dump(dict(label=label, args=kw, summary=summ, runs=runs, seconds=time.time() - t0), open(os.path.join(HERE, 'res', f'add3_{label}.json'), 'w'), indent=1)
print(f'== {label} ({len(runs)} runs, {time.time() - t0:.0f} s): ' + '; '.join(f'{k} ' + ' / '.join(f'{v:.1f}' for v in vs) for k, vs in summ.items()), flush=True)
