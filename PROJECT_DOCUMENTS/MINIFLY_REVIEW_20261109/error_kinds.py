"""What kind of mistake does each life make on composition items? Greedy decoding with an independent numpy GRU; no training.
Items 'c aa o1 bb o2 cc =' -> 'ii;ff<EOS>' with i = a o1 b, f = i o2 c (mod 97). Read-only (weights via git show)."""
import json, sys, collections
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from np_gru import load_weights, comp_panels, forward, git_bytes, apply, SYM_OP, P, EOS, BOS, TAUGHT_PAIRS
SEEDS = (73129, 93683, 116089)
CELLS = {'SS': 'e26/e26_life_S_{s}', 'BS': 'e26x/e26x_life_BS_{s}', 'BB': 'e26/e26_life_B_{s}', 'SB': 'e26x/e26x_life_SB_{s}'}

def greedy(W, prompts, n=6):
    out = np.zeros((len(prompts), 0), dtype=np.int64)
    base = np.array([[BOS] + p for p in prompts], dtype=np.int64)
    for _ in range(n):
        x = np.concatenate([base, out], axis=1)
        logits = forward(W, x)[3][:, -1, :]
        out = np.concatenate([out, logits.argmax(-1)[:, None]], axis=1)
    return out

def num(d1, d2):
    return (d1 - 48) * 10 + (d2 - 48) if 48 <= d1 <= 57 and 48 <= d2 <= 57 else None

def classify(items, outs):
    c = collections.Counter(); near = collections.Counter()
    for (p, t), o in zip(items, outs):
        a = (p[1] - 48) * 10 + (p[2] - 48); o1 = chr(p[3]); b = (p[4] - 48) * 10 + (p[5] - 48); o2 = chr(p[6]); cc = (p[7] - 48) * 10 + (p[8] - 48)
        i_true = apply(SYM_OP[o1], 97, a, b); f_true = apply(SYM_OP[o2], 97, i_true, cc)
        ii, ff = num(o[0], o[1]), num(o[3], o[4])
        if ii is None or ff is None or o[2] != ord(';') or o[5] != EOS:
            c['malformed'] += 1; continue
        A = ii == i_true; B = ii == apply(SYM_OP[o2], 97, a, b)
        C = ff == apply(SYM_OP[o2], 97, ii, cc); D = ff == apply(SYM_OP[o1], 97, ii, cc)
        if A and C: c['right'] += 1
        elif A and D: c['first step right, second step done with the FIRST operator'] += 1
        elif A: c['first step right, second step wrong otherwise'] += 1
        elif B and C: c['first step done with the SECOND operator, second step consistent'] += 1
        elif B and D: c['operators swapped'] += 1
        elif C: c['first step wrong, second step consistent with what it wrote'] += 1
        else: c['first step wrong and second step wrong'] += 1
        if not A:
            d = (ii - i_true) % 97
            near['one digit of the intermediate right'] += int((ii // 10 == i_true // 10) != (ii % 10 == i_true % 10))
            near['off by one'] += int(d in (1, 96)); near['total'] += 1
    return c, near

ORDER = ['right', 'first step right, second step done with the FIRST operator', 'first step right, second step wrong otherwise',
         'first step done with the SECOND operator, second step consistent', 'operators swapped',
         'first step wrong, second step consistent with what it wrote', 'first step wrong and second step wrong', 'malformed']
res = {}
for s in SEEDS:
    comp, _ = comp_panels(s)
    panels = {'untaught +-/-+': comp['+-'] + comp['-+'], 'taught pairs, new instances': comp['within']}
    for cell, stem in CELLS.items():
        W, _ = load_weights(P + stem.format(s=s) + '.pt')
        rec = json.loads(git_bytes(P + stem.format(s=s) + '.json'))['stages'][-1]['scores']['compose']
        for name, items in panels.items():
            outs = greedy(W, [p for p, _ in items])
            c, near = classify(items, outs)
            want = rec['+-']['exact'] + rec['-+']['exact'] if name.startswith('untaught') else rec['within']['exact']
            assert c['right'] == want, (cell, s, name, c['right'], want)      # my greedy decoding must reproduce the receipt's count
            res[(cell, s, name)] = (c, near, len(items))
print('greedy decoding reproduces the receipts\' whole-answer counts in all 24 cases\n')
for name in ('untaught +-/-+', 'taught pairs, new instances'):
    print(f'=== {name}: percent of items, seeds 73129 / 93683 / 116089')
    for k in ORDER:
        print(f'  {k}')
        for cell in ('SS', 'BS', 'BB', 'SB'):
            vals = [100 * res[(cell, s, name)][0][k] / res[(cell, s, name)][2] for s in SEEDS]
            print(f"      {cell} ({'sweep' if cell[1] == 'S' else 'backprop'} life): " + ' / '.join(f'{v:5.1f}' for v in vals))
    print('  among items with the first step wrong: share with exactly one digit of the intermediate right; share off by one')
    for cell in ('SS', 'BS', 'BB', 'SB'):
        row = []
        for s in SEEDS:
            n = res[(cell, s, name)][1]; t = max(n['total'], 1)
            row.append(f"{100 * n['one digit of the intermediate right'] / t:4.0f}% ; {100 * n['off by one'] / t:4.0f}% (n {n['total']})")
        print(f"      {cell}: " + ' / '.join(row))
    print()
