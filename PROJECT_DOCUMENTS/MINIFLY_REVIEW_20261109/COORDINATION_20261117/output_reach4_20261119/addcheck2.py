# Carrying addcheck.py's system to 0..D-1 with numbers of one or two digits (the reviewer's message of 15:33 UTC). Three changes, in order,
# each run with those before it (variant=hold, carry, place). What is LAID DOWN (fixed by this program) and what is LEARNED:
#   hold  [laid down] holding cells that take a byte's value at an EVENT, not at a fixed delay: when the present byte is '+', cell Ha1 takes
#         the value of the byte just before it and Ha10 the value of the byte two before it (0 if that byte is not a digit); when the present
#         byte is '=', Hb1 and Hb10 the same for the second number; all release (return to 0) at the end of line. Two drives: units
#         u = Ha1 + Hb1, tens w = Ha10 + Hb10. Tuned cells for each total (active exactly when the drive equals it), gated as A7's kind 2 by
#         the present byte and the byte 1 back: one cell for each (total, present byte, byte 1 back), for u and for w.
#   carry [laid down] a cell that fires when u > 9 and adds one to the tens drive: w = Ha10 + Hb10 + [u > 9].
#   place [laid down] the tuned cells gated by the place in the answer instead of by the bytes: a counter cell set at '=' (place 0: the present
#         byte is '='), one more at each byte after it, released at the end of line (place "none" elsewhere); one tuned cell for each (total,
#         place) for u and for w.
#   Besides, as in addcheck.py: a byte cell for each value of the present byte, and a constant. Four active inputs of value 1, n = 4.
#   [learned] the answering cell's estimates W[v] over these inputs, by rule Q from yes or no only (eta 1/16, eps 1/16 while learning), as
#         addcheck.py: which byte each tuned cell's activity stands for at each place, which is the digit table and its carry.
# Text and score as addcheck.py: lines "a+b=c\n"; pairs never shown fixed by the seed (20 of 100 for D = 10, 2,000 of 10,000 for D = 100);
#   60,000 lines drawn from the shown pairs; score with learning frozen and no random choice, every byte after '=' up to and including the
#   end of line named right (teacher forcing). Also by part: the units digit, the tens digit, the hundreds digit, and the end of line, each
#   named right where it exists; and the pairs with a carry (units total > 9) apart. And the coverage: for each pair never shown, whether
#   its units total and its tens total (with carry) occurred together in some shown pair, and whether each occurred in some shown pair.
# usage: addcheck2.py <label> D=100 variant=hold|carry|place lines=60000 seeds=.. orders=5   -> res/add2_<label>.json
import sys, os, json, time, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
label = sys.argv[1]; kw = dict(a.split('=') for a in sys.argv[2:]); t0 = time.time()
D = int(kw.get('D', 100)); LINES = int(kw.get('lines', 60000)); SEEDS = [int(s) for s in kw.get('seeds', '73129,93683,116089').split(',')]
ORDERS = int(kw.get('orders', 5)); VAR = kw.get('variant', 'place'); ETA = float(eval(kw.get('eta', '1/16'))); EPS = float(eval(kw.get('eps', '1/16')))
NHIDE = 20 if D == 10 else 2000
ALPH = [ord(c) for c in '0123456789+=\n']; AI = {c: i for i, c in enumerate(ALPH)}; NA = len(ALPH)
DIG = {ord('0') + d: d for d in range(10)}
NT = 20                                           # totals 0..19 (u up to 18; w up to 18 + 1)
NPL = 6                                           # places: none, 0..4
CARRY = VAR in ('carry', 'place'); PLACE = VAR == 'place'
if PLACE: NG = NPL
else: NG = NA * NA
NF = 2 * NT * NG + NA + 1
class State:
    def __init__(self): self.h = [ord('\n')] * 3; self.Ha1 = self.Ha10 = self.Hb1 = self.Hb10 = 0; self.place = -1
    def read(self, c):
        self.h = self.h[1:] + [c]
        if c == ord('+'): self.Ha1 = DIG.get(self.h[-2], 0); self.Ha10 = DIG.get(self.h[-3], 0) if self.h[-2] in DIG else 0
        elif c == ord('='): self.Hb1 = DIG.get(self.h[-2], 0); self.Hb10 = DIG.get(self.h[-3], 0) if self.h[-2] in DIG else 0; self.place = 0
        elif c == ord('\n'): self.Ha1 = self.Ha10 = self.Hb1 = self.Hb10 = 0; self.place = -1
        elif self.place >= 0: self.place = min(self.place + 1, NPL - 2)
    def feats(self):
        u = self.Ha1 + self.Hb1; w = self.Ha10 + self.Hb10 + (int(u > 9) if CARRY else 0)
        g = (self.place + 1) if PLACE else (AI[self.h[-1]] * NA + AI[self.h[-2]])
        return (u * NG + g, NT * NG + w * NG + g, 2 * NT * NG + AI[self.h[-1]], 2 * NT * NG + NA)
def line(a, b): return [ord(c) for c in f'{a}+{b}={a + b}\n']
runs = []
for seed in SEEDS:
    rs = np.random.default_rng(seed)
    pairs = [(a, b) for a in range(D) for b in range(D)]
    hide = set(map(tuple, np.array(pairs)[rs.choice(len(pairs), NHIDE, replace=False)].tolist()))
    shown = [p for p in pairs if p not in hide]
    tot = lambda a, b: (a % 10 + b % 10, a // 10 + b // 10 + int(a % 10 + b % 10 > 9))
    sh_both = {tot(a, b) for a, b in shown}; sh_u = {tot(a, b)[0] for a, b in shown}; sh_w = {tot(a, b)[1] for a, b in shown}
    cov = dict(never_shown=len(hide), both_totals_seen_together=sum(tot(a, b) in sh_both for a, b in hide),
               units_total_seen=sum(tot(a, b)[0] in sh_u for a, b in hide), tens_total_seen=sum(tot(a, b)[1] in sh_w for a, b in hide))
    for order in range(ORDERS):
        ro = np.random.default_rng(seed * 100 + order)
        W = np.zeros((256, NF)); seen = np.zeros(256, bool); st = State(); yes = steps = 0
        for li in range(LINES):
            a, b = shown[ro.integers(len(shown))]
            for nxt in line(a, b):
                f = st.feats(); A = W[:, f].sum(1); e = int(A.argmax())
                if ro.random() < EPS and seen.any():
                    sv = np.flatnonzero(seen); e = int(sv[ro.integers(len(sv))])
                ok = e == nxt; yes += ok; steps += 1
                if ok:
                    d = ETA * (0.0 - A) / 4.0; d[e] = ETA * (1.0 - A[e]) / 4.0; W[:, f] += d[:, None]
                else:
                    W[e, f] += ETA * (0.0 - A[e]) / 4.0
                seen[nxt] = True; st.read(nxt)
        def score(ps):
            whole = 0; part = {k: [0, 0] for k in ('hundreds', 'tens', 'units', 'end of line')}; carry = [0, 0]; nocarry = [0, 0]; wrong = []
            for a, b in ps:
                s = State()
                for c in [ord(x) for x in f'\n{a}+{b}=']: s.read(c)
                ans = f'{a + b}\n'; roles = {3: ['units', 'end of line'], 4: ['tens', 'units', 'end of line'], 5: ['hundreds', 'tens', 'units', 'end of line']}[len(ans) + 1]
                good = True
                for ch, role in zip(ans, roles):
                    nxt = ord(ch); e = int(W[:, s.feats()].sum(1).argmax()); r = e == nxt
                    part[role][0] += r; part[role][1] += 1; good &= r; s.read(nxt)
                whole += good
                k = carry if a % 10 + b % 10 > 9 else nocarry; k[0] += good; k[1] += 1
                if not good: wrong.append(f'{a}+{b}')
            return dict(whole=whole / len(ps), parts={k: v[0] / v[1] for k, v in part.items() if v[1]}, carry=carry[0] / max(carry[1], 1),
                        no_carry=nocarry[0] / max(nocarry[1], 1), n_carry=carry[1], wrong=wrong[:40])
        s_sh = score(shown); s_hd = score(sorted(hide))
        runs.append(dict(seed=seed, order=order, shown=s_sh, never_shown=s_hd, yes_share_learning=yes / steps, coverage=cov))
        print(f"{label} seed {seed} order {order}: whole shown {100 * s_sh['whole']:.1f} never {100 * s_hd['whole']:.1f} | never shown by part "
              + ' '.join(f"{k} {100 * v:.1f}" for k, v in s_hd['parts'].items()) + f" | never shown carry {100 * s_hd['carry']:.1f} (n {s_hd['n_carry']}) no carry {100 * s_hd['no_carry']:.1f}"
              + f" | coverage {cov} | yes {100 * yes / steps:.1f}", flush=True)
q = lambda f: [100 * min(f(r) for r in runs), 100 * float(np.median([f(r) for r in runs])), 100 * max(f(r) for r in runs)]
summ = {'shown whole': q(lambda r: r['shown']['whole']), 'never shown whole': q(lambda r: r['never_shown']['whole']),
        'never shown, carry': q(lambda r: r['never_shown']['carry']), 'never shown, no carry': q(lambda r: r['never_shown']['no_carry'])}
for k in ('hundreds', 'tens', 'units', 'end of line'):
    if k in runs[0]['never_shown']['parts']: summ[f'never shown, {k}'] = q(lambda r: r['never_shown']['parts'][k])
out = dict(label=label, args=kw, features=NF, summary=summ, runs=runs, seconds=time.time() - t0)
json.dump(out, open(os.path.join(HERE, 'res', f'add2_{label}.json'), 'w'), indent=1)
print(f'== {label} (variant {VAR}, D {D}, {len(runs)} runs, {time.time() - t0:.0f} s): ' + '; '.join(f'{k} ' + ' / '.join(f'{v:.1f}' for v in vs) for k, vs in summ.items()), flush=True)
