# Builds TABLE.md from reach/*.json (reach.py) and reach/onehot3_*.json (onehot3.py).
import json, os, glob
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, 'reach')
SEEDS = (73129, 93683, 116089)
KS = [('all', 'all'), ('R1 (first family, phase FIRST)', 'R1'), ('R2 (first family, phase FIRST)', 'R2'), ('rest (not R1, R2)', 'rest')]
ROWS = [('U', 'a_near_output', 'U, (a) near the output lines'), ('U', 'b_all_non_feedback', 'U, (b) every non-feedback cell'),
        ('M', 'a_near_output', 'M, (a) near the output lines'), ('M', 'b_all_non_feedback', 'M, (b) every non-feedback cell'),
        ('U', 'onehot_present', 'reader on the present byte (one-hot)'), ('U', 'onehot_present_previous', 'reader on present + previous byte (two one-hots)'),
        (None, 'onehot3', 'reader on the last three bytes (three one-hots)'),
        ('U', 'ref_mode', 'ref_mode (scorer)'), ('U', 'ref_1', 'ref_1 (scorer)'), ('U', 'ref_2', 'ref_2 (scorer)'), ('U', 'ref_3', 'ref_3 (scorer)')]
def load(p):
    return json.load(open(p)) if os.path.exists(p) else None
lines = ['# Next-byte reader on the cells (reach.py): share right on the scored quarter, %', '',
         'U: rule=fixed, no control. M: rule=fixed gain=2 rho=1 gmax=4. Each cell: R1 / R2 / rest / all (n of the scored positions in the first row of each seed).', '']
for NB, suf in ((16384, ''), (65536, '_64k')):
    if not os.path.exists(os.path.join(R, f'U_s{SEEDS[0]}{suf}.json')):
        continue
    lines += [f'## {NB:,} bytes (fitted on the first three quarters, scored on the last quarter)', '']
    sets = []
    for s in SEEDS:
        u = load(os.path.join(R, f'U_s{s}{suf}.json'))
        if u:
            a = u['sets']['a_near_output']
            sets.append(f"seed {s}: (a) {a['cells']} cells ({a['drivers']} drivers and their neighbourhoods), distance from the input lines "
                        + ', '.join(f'{k}: {v}' for k, v in a['distance_from_input_lines'].items()) + f"; (b) {u['sets']['b_all_non_feedback']['cells']} cells")
    lines += ['- ' + x for x in sets] + ['']
    lines += ['| reader | ' + ' | '.join(f'seed {s}' for s in SEEDS) + ' |', '|---|' + '---|' * len(SEEDS)]
    first = True
    for arm, key, name in ROWS:
        cells = []
        for s in SEEDS:
            d = load(os.path.join(R, f'onehot3_{s}_{NB}.json')) if key == 'onehot3' else load(os.path.join(R, f'{arm}_s{s}{suf}.json'))
            if d is None:
                cells.append('-'); continue
            sh = d['shares'] if key == 'onehot3' else d['readers'][key]['shares']
            c = ' / '.join(f"{100 * sh[k]['right']:.1f}" for k, _ in KS[1:]) + f" / **{100 * sh['all']['right']:.1f}**"
            if first:
                c += ' (n ' + '/'.join(str(sh[k]['n']) for k, _ in KS[1:]) + ')'
            cells.append(c)
        first = False
        lines.append(f'| {name} | ' + ' | '.join(cells) + ' |')
    lines.append('')
open(os.path.join(HERE, 'TABLE.md'), 'w').write('\n'.join(lines) + '\n'); print('\n'.join(lines))
