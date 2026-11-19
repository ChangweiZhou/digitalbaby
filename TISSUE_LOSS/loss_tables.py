# Tables of the measurement of where the present byte is lost, from out/measure_<label>.json. Writes LOSS_TABLES.md.
#   python3 -I loss_tables.py
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
T = (("fixed", "unchanging, no gain"), ("fixed_gain_a32", "unchanging, gain, a*=1/32"),
     ("coinc_gain_a32_F", "coincidence, gain, a*=1/32, floor 1/128"))
CH = (("fixed_notheta", "unchanging, no gain, threshold held at 0"), ("fixed_gain_a32_notheta", "unchanging, gain, a*=1/32, threshold held at 0"),
      ("coinc_gain_a32_F_notheta", "coincidence, gain, a*=1/32, floor 1/128, threshold held at 0"))


def load(l):
    p = os.path.join(HERE, 'out', f'measure_{l}.json')
    return json.load(open(p)) if os.path.exists(p) else None


def f(v, n=3):
    return '—' if v is None else (f'{v:.{n}f}' if abs(v) >= 10 ** -n or v == 0 else f'{v:.2g}')


out = []
P = out.append
P('# Where the present byte is lost: tables (written by loss_tables.py from out/measure_*.json)\n')
P('Seed 73129, 16,384 bytes, window A2 (bytes 12,288 to 16,383), outputs at the reading step unless said, raw (nothing below 0.01 set to 0). '
  'd: distance from the input lines of the bytes present in A2 along the connections present at the end of the window.\n')
for lab, name in T:
    r = load(lab)
    P(f'## {name}\n')
    P(f"CPU {r['cpu_s']:.0f} s; connections a cell {r['conns_per_cell']:.1f}; S(d, 0) at d = 1 / 2 / 4: "
      + ' / '.join(f(r['A2']['S'][d]['0']) for d in ('1', '2', '4')) + '\n')
    P('| d | cells | 1. mean output | 1. share > 0.01 | 2. eta2 present, median | upper decile | 2. eta2 previous, median | upper decile | 3. eta2 present median at step 4 / 8 / 16 / 32 |')
    P('|---|---|---|---|---|---|---|---|---|')
    for d in range(1, 9):
        d = str(d)
        if d not in r['item1']:
            continue
        a, b = r['item1'][d], r['item2'][d]
        ph = ' / '.join(f(r['item3'][d][f'step_{s}']['present']['median']) for s in (4, 8, 16, 32))
        P(f"| {d} | {a['cells']} | {f(a['mean_output'], 4)} | {f(a['share_above_001'])} | {f(b['present']['median'])} | {f(b['present']['upper_decile'])} | "
          f"{f(b['previous']['median'])} | {f(b['previous']['upper_decile'])} | {ph} |")
    i4 = r['item4']
    P(f"\n4. Potential v changed by 1e-9 at byte 8,192: (one cell) cell {i4['cell']}, the most active at d = 1 (output {i4['cell_output_at_8192']:.3f}); "
      "(every cell) every cell that is not a feedback cell. Largest difference of outputs, and of gains and thresholds, after n bytes:\n")
    P('| n | one cell: outputs | cells differing | every cell: outputs | cells differing | gains | thresholds | strengths |')
    P('|---|---|---|---|---|---|---|---|')
    for n in ('1', '4', '16', '64', '256'):
        o, e = i4['one_cell']['after'][n], i4['every_cell']['after'][n]
        P(f"| {n} | {f(o['largest'])} | {o['cells_differing']} | {f(e['largest'])} | {e['cells_differing']} | {f(e['state_largest']['gain'])} | "
          f"{f(e['state_largest']['th'])} | {f(e['state_largest']['c_w'])} |")
    P('\n5. Share of the absolute arrival at the reading step, mean over the cells at d (lines / nearer / same distance / farther / not reached):\n')
    P('| d | input lines | nearer | same distance | farther | not reached |')
    P('|---|---|---|---|---|---|')
    for d, v in r['item5'].items():
        s = v['mean_share_of_abs_arrival']
        P(f"| {d} | {f(s['input lines'])} | {f(s['nearer'])} | {f(s['same distance'])} | {f(s['farther'])} | {f(s['not reached'])} |")
    P('')
P('## The one change: the threshold held at 0 (a cell with the gain on controls its activity by its gain alone)\n')
P('| Tissue | change | S(1,0) | S(1,1) | S(1,2) | S(2,0) | S(2,1) | S(2,2) | S(4,0) | S(4,1) | S(4,2) | eta2 present median d = 1 / 2 / 4 / 8 | mean output d = 1 / 4 |')
P('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
for (lab, name), (lab2, _) in zip(T, CH):
    for l, ch in ((lab, 'without'), (lab2, 'with')):
        r = load(l)
        if r is None:
            P(f'| {name} | {ch} | not run |' + ' |' * 10); continue
        S = r['A2']['S']
        e = ' / '.join(f(r['item2'][d]['present']['median']) for d in ('1', '2', '4', '8'))
        P(f"| {name} | {ch} | " + ' | '.join(f(S[d][L]) for d in ('1', '2', '4') for L in ('0', '1', '2')) + f" | {e} | {f(r['item1']['1']['mean_output'], 4)} / {f(r['item1']['4']['mean_output'], 4)} |")
open(os.path.join(HERE, 'LOSS_TABLES.md'), 'w').write('\n'.join(out) + '\n')
print('\n'.join(out))
