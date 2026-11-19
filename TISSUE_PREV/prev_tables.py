# Tables of the previous-byte runs, from prev/<label>.json. Writes PREV_TABLES.md.
#   python3 -I prev_tables.py
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
LEVEL = {'0': 0.1103, '1': 0.090, '2': 0.076}
ROWS = [('base', 'base: rule=fixed gain=2 rho=1'),
        ('a_rho2_q4', '(a) rho 2 on 1/4 of the cells'), ('a_rho2_q2', '(a) rho 2 on 1/2 of the cells'),
        ('b_tau8_q4', '(b) potential time constant 8 on 1/4'), ('b_tau16_q4', '(b) time constant 16 on 1/4'), ('b_tau32_q4', '(b) time constant 32 on 1/4'),
        ('b_tau8_q2', '(b) time constant 8 on 1/2'), ('b_tau16_q2', '(b) time constant 16 on 1/2'), ('b_tau32_q2', '(b) time constant 32 on 1/2'),
        ('c_del8_q4', '(c) delay 8 steps on 1/4 of the connections between cells'), ('c_del16_q4', '(c) delay 16 on 1/4'), ('c_del32_q4', '(c) delay 32 on 1/4'),
        ('c_del8_q2', '(c) delay 8 on 1/2'), ('c_del16_q2', '(c) delay 16 on 1/2'), ('c_del32_q2', '(c) delay 32 on 1/2'),
        ('d_input8', '(d) input line on for the first 8 steps of a byte (the input end, not a cell)')]
SEED2 = [('s93683_base', 'base'), ('s93683_c_del8_q4', '(c) delay 8 on 1/4'), ('s93683_c_del32_q4', '(c) delay 32 on 1/4'),
         ('s93683_b_tau16_q4', '(b) time constant 16 on 1/4')]


def load(l):
    return json.load(open(os.path.join(HERE, 'prev', f'{l}.json')))


def m(v, L):
    if v is None:
        return '—'
    return f'**{v:.3f}**' if L is not None and v >= LEVEL[L] else f'{v:.3f}'


def table(rows, mark=True):
    out = ['| Tissue | S(1,0) | S(2,0) | S(4,0) | S(1,1) | S(2,1) | S(4,1) | S(1,2) | S(2,2) | S(4,2) | c0 L1 | c0 L2 | share d=1/2/4 | divergence after 512 bytes | gain median | CPU s |',
           '|---|' + '---|' * 15]
    for lab, name in rows:
        r = load(lab); S = r['table']['S']
        cells = [m(S[d][L], L if mark else None) for L in ('0', '1', '2') for d in ('1', '2', '4')]
        sh = ' / '.join(f"{r['table']['share'][d]:.3f}" for d in ('1', '2', '4'))
        out.append(f"| {name} | " + ' | '.join(cells) + f" | {r['table']['c0']['1']:.3f} | {r['table']['c0']['2']:.3f} | {sh} | "
                   f"{r['table']['divergence']['512']:.2g} | {r['gain_median']:.3g} | {r['cpu_s']:.0f} |")
    return out


out = ['# The previous byte at the reading step: tables (written by prev_tables.py from prev/*.json)\n',
       'Seed 73129 unless said, 16,384 bytes, window A2, S(d, L) by the readings of the brief (readings.py, as drive.py). **Bold**: at or above '
       'the level (L = 0: 0.1103; L = 1: 0.090; L = 2: 0.076; levels of seed 73129). Divergence: the largest difference of outputs of two copies, '
       'the second with one cell\'s potential changed by 1e-9, 512 bytes later (diag.py\'s method).\n']
out += table(ROWS)
out += ['\n## Item 1: the share of a cell\'s output variance explained by the previous byte, median, after steps 1 / 2 / 4 / 8 / 16 / 32 of the byte\n']
for lab, name in [('base', 'base'), ('c_del32_q4', 'delay 32 on 1/4'), ('b_tau16_q4', 'time constant 16 on 1/4'), ('c_del8_q4', 'delay 8 on 1/4')]:
    r = load(lab)
    out += [f'\n### {name}\n', '| d | cells | previous byte, median, steps 1/2/4/8/16/32 | upper decile at step 32 | present byte, median, steps 1/2/4/8/16/32 | mean output at step 32 |',
            '|---|---|---|---|---|---|']
    for d, v in r['item1'].items():
        f = lambda x: '—' if x is None else f'{x:.3f}'
        out.append(f"| {d} | {v['cells']} | " + ' / '.join(f(v[f'step_{s}']['previous_median']) for s in (1, 2, 4, 8, 16, 32))
                   + f" | {f(v['step_32']['previous_p90'])} | " + ' / '.join(f(v[f'step_{s}']['present_median']) for s in (1, 2, 4, 8, 16, 32))
                   + f" | {v['step_32']['mean_output']:.4f} |")
out += ['\n## A second seed (93683): levels of seed 73129 do not apply; compare with this seed\'s base and c0\n']
out += table(SEED2, mark=False)
open(os.path.join(HERE, 'PREV_TABLES.md'), 'w').write('\n'.join(out) + '\n')
print('\n'.join(out[:22]))
