# Builds TABLE.md from reach/*.json (reach.py), reach/nbhd_*.json (reach_nbhd.py) and yesno3/*.json (yesno3.py).
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = (73129, 93683, 116089); TIS = ('U', 'BAL', 'B1')
def J(p):
    p = os.path.join(HERE, p); return json.load(open(p)) if os.path.exists(p) else None
cols = [(t, s) for t in TIS for s in SEEDS]
L = ['# The output end on the new tissues: share right, %, 65,536 bytes', '',
     'U: rule=fixed. BAL: rule=timing eta_s=0 no_make=1 norm=1 we=1/2 wi=1/2. B1: rule=timing norm=1 lines_learn=1 eta_s=1 no_make=1 we=1 wi=1/4.', '',
     '## A. Fitted readers (fitted on the first three quarters, scored on the last quarter): all / R1 / R2', '',
     '| reader | ' + ' | '.join(f'{t} {s}' for t, s in cols) + ' |', '|---|' + '---|' * len(cols)]
KS = ('all', 'R1 (first family, phase FIRST)', 'R2 (first family, phase FIRST)')
def cell_reach(d, key):
    if not d: return ''
    sh = d['readers'][key]['shares']; return ' / '.join(f"{100 * sh[k]['right']:.1f}" for k in KS)
for key, name in (('a_near_output', 'ridge on set (a), near the output lines'), ('b_all_non_feedback', 'ridge on set (b), every non-feedback cell'),
                  ('onehot_present', 'ridge on the present byte (one-hot)'), ('onehot_present_previous', 'ridge on present + previous byte'),
                  ('ref_1', 'ref_1'), ('ref_2', 'ref_2'), ('ref_3', 'ref_3')):
    L.append(f'| {name} | ' + ' | '.join(cell_reach(J(f'reach/{t}_s{s}.json'), key) for t, s in cols) + ' |')
row = []
for t, s in cols:
    d = J(f'reach/nbhd_{t}_s{s}.json'); row.append(f"{100 * d['readers']['own_neighbourhood']['right']:.1f}" if d else '')
L.append('| ridge, each byte on its own driver\'s neighbourhood (all) | ' + ' | '.join(row) + ' |')
L += ['', '## B and C. Online one-pass rules on the cells near each driver: last window of 4,096 / last quarter (the positions of A)', '',
      '| rule | ' + ' | '.join(f'{t} {s}' for t, s in cols) + ' |', '|---|' + '---|' * len(cols)]
first = next((J(f'yesno3/{t}_s{s}.json') for t, s in cols if J(f'yesno3/{t}_s{s}.json')), None)
if first:
    for k in list(first['rules']) + list(first['references']):
        c = []
        for t, s in cols:
            d = J(f'yesno3/{t}_s{s}.json')
            if not d: c.append(''); continue
            v = d['rules'].get(k) or d['references'].get(k); c.append(f"{100 * v['windows'][-1]:.1f} / {100 * v['last_quarter']:.1f}")
        L.append(f'| {k} | ' + ' | '.join(c) + ' |')
L.append('')
open(os.path.join(HERE, 'TABLE.md'), 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
