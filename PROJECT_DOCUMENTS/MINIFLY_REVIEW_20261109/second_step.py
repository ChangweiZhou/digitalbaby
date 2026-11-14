"""Second-step mistakes on composition items, for lives under each update. Greedy decoding, independent numpy GRU, no training."""
import sys, collections
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from np_gru import load_weights, comp_panels, forward, apply, SYM_OP, P, EOS, BOS
SEEDS = (73129, 93683, 116089)
CELLS = {'SS': 'e26/e26_life_S_{s}', 'BS': 'e26x/e26x_life_BS_{s}', 'BB': 'e26/e26_life_B_{s}', 'SB': 'e26x/e26x_life_SB_{s}'}
def greedy(W, prompts, n=6):
    out = np.zeros((len(prompts), 0), dtype=np.int64); base = np.array([[BOS] + p for p in prompts], dtype=np.int64)
    for _ in range(n):
        out = np.concatenate([out, forward(W, np.concatenate([base, out], axis=1))[3][:, -1, :].argmax(-1)[:, None]], axis=1)
    return out
def num(d1, d2): return (d1 - 48) * 10 + (d2 - 48) if 48 <= d1 <= 57 and 48 <= d2 <= 57 else None
KINDS = ['tens digit right, units wrong', 'units digit right, tens wrong', 'neither digit right', 'off by one (mod 97)', 'off by ten (mod 97)',
         'right answer for a c that differs in one digit', 'right answer with the opposite sign of c (i - c for i + c, or the reverse)',
         'equals the intermediate itself', 'equals i op2 b (used b for c)', 'equals i op2 a']
tab = {}
for s in SEEDS:
    comp, _ = comp_panels(s)
    for name, items in (('untaught', comp['+-'] + comp['-+']), ('taught', comp['within'])):
        for cell, stem in CELLS.items():
            W, _ = load_weights(P + stem.format(s=s) + '.pt'); outs = greedy(W, [p for p, _ in items])
            c = collections.Counter(); first_right = 0
            for (p, t), o in zip(items, outs):
                a = (p[1]-48)*10 + (p[2]-48); o1 = chr(p[3]); b = (p[4]-48)*10 + (p[5]-48); o2 = chr(p[6]); cc = (p[7]-48)*10 + (p[8]-48)
                if o2 not in '+-': continue                          # second step an addition or subtraction, so that "off by" is meaningful
                i_true = apply(SYM_OP[o1], 97, a, b); f_true = apply(SYM_OP[o2], 97, i_true, cc)
                ii, ff = num(o[0], o[1]), num(o[3], o[4])
                if ii != i_true or ff is None: continue
                first_right += 1
                if ff == f_true: continue
                c['n'] += 1
                t_ok, u_ok = ff // 10 == f_true // 10, ff % 10 == f_true % 10
                c['tens digit right, units wrong'] += t_ok and not u_ok; c['units digit right, tens wrong'] += u_ok and not t_ok; c['neither digit right'] += not t_ok and not u_ok
                d = (ff - f_true) % 97
                c['off by one (mod 97)'] += d in (1, 96); c['off by ten (mod 97)'] += d in (10, 87)
                alt = [x for x in range(100) if x != cc and ((x // 10 == cc // 10) != (x % 10 == cc % 10))]
                c['right answer for a c that differs in one digit'] += any(apply(SYM_OP[o2], 97, i_true, x % 97) == ff for x in alt if x < 97)
                other = '-' if o2 == '+' else '+'
                c['right answer with the opposite sign of c (i - c for i + c, or the reverse)'] += apply(SYM_OP[other], 97, i_true, cc) == ff
                c['equals the intermediate itself'] += ff == i_true
                c['equals i op2 b (used b for c)'] += ff == apply(SYM_OP[o2], 97, i_true, b) and b != cc
                c['equals i op2 a'] += ff == apply(SYM_OP[o2], 97, i_true, a) and a != cc
            tab[(cell, s, name)] = (c, first_right)
for name in ('untaught', 'taught'):
    print(f"=== {name} pairs whose second operator is + or -: items with the first step right, and how the second step fails (seeds 73129 / 93683 / 116089)")
    for cell in ('SS', 'BS', 'BB', 'SB'):
        row = [f"{tab[(cell, s, name)][0]['n']} of {tab[(cell, s, name)][1]} ({100*tab[(cell, s, name)][0]['n']/max(tab[(cell, s, name)][1],1):.0f}%)" for s in SEEDS]
        print(f"   {cell} ({'sweep' if cell[1]=='S' else 'backprop'} life) second step wrong: " + ' / '.join(row))
    for k in KINDS:
        print(f"   share of those mistakes: {k}")
        for cell in ('SS', 'BS', 'BB', 'SB'):
            print(f"        {cell}: " + ' / '.join(f"{100*tab[(cell, s, name)][0][k]/max(tab[(cell, s, name)][0]['n'],1):4.0f}%" for s in SEEDS))
    print()
