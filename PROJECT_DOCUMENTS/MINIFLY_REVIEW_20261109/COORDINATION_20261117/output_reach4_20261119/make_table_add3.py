# Table of addcheck3.py results (res/add3_*.json) -> TABLE_ADD3.md. Rows not run are marked NOT_RUN.
import os, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = [('base', 'base (carry cell, holding cells laid down)'), ('learned', '(a) carry=learned'), ('pool', '(b) hold=pool, all 741 pairs'),
         ('poolc', '(b) hold=pool, 741 pairs + cpairs=2000'), ('pool10', '(b) hold=pool, D=10, 60,000 lines'), ('str10', '(c) strength=0.1, eq=0'),
         ('str10eq', '(c) strength=0.1, eq=1'), ('M8', '(2) M=8 output part')]
SEEDS = [73129, 93683, 116089]
f3 = lambda v: ' / '.join(f'{x:.1f}' for x in v)
L = ['# addcheck3.py: removals from addcheck2.py variant carry (D = 100 unless stated; 240,000 lines; 5 orders a seed)', '',
     'Values: smallest / median / largest over the 5 orders of a seed, in %. Never shown = the pairs held out (2,000 of 10,000; 20 of 100 for D = 10).',
     'Curve: median over the 5 orders of the never-shown whole score after 60k / 120k / 240k lines.', '',
     '| Variant | Seed | Shown whole | Never shown whole | Never shown tens | Never shown units | Curve (never shown whole) | Yes while learning |',
     '|---|---|---|---|---|---|---|---|']
for k, desc in NAMES:
    for s in SEEDS:
        p = os.path.join(HERE, 'res', f'add3_{k}_s{s}.json')
        if not os.path.exists(p):
            L.append(f'| {desc} | {s} | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |'); continue
        d = json.load(open(p)); sm = d['summary']; rs = d['runs']
        cur = ' / '.join(f"{100 * np.median([r['curve'][c]['never_shown']['whole'] for r in rs]):.1f}" for c in rs[0]['curve'])
        L.append(f"| {desc} | {s} | {f3(sm['shown whole'])} | {f3(sm['never shown whole'])} | {f3(sm['never shown, tens'])} | "
                 f"{f3(sm['never shown, units'])} | {cur} | {100 * np.median([r['yes_share'] for r in rs]):.1f} |")
L += ['', 'NOT_RUN: base and learned on every seed (the batch file `batch_add3.txt` had a trailing space on the base lines, so `xargs -L1`',
      'joined each base line to the next learned line; base stopped with a ValueError on its arguments and learned never started);',
      'every variant on seed 116089 (not started before the stop instruction of 16:13 UTC).']
open(os.path.join(HERE, 'TABLE_ADD3.md'), 'w').write('\n'.join(L) + '\n')
