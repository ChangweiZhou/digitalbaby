# Summary of the addition runs: for each system (tag), over the seeds and orders run, the whole answer right on sums shown (no learning)
# and on sums never shown; and on the never-shown pairs whose sum occurs among the shown pairs, and those whose sum does not.
# Writes TABLES.md from res/<tag>_s<seed>o<order>.json.
#   python3 -I tables.py
import glob
import json
import os
import re
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
runs = {}
for p in sorted(glob.glob(os.path.join(HERE, 'res', '*_s*o*.json'))):
    m = re.match(r'(.+)_s(\d+)o(\d+)\.json$', os.path.basename(p))
    if m:
        r = json.load(open(p))
        if not isinstance(r['rows'][0][1], dict):        # addsys.py's format (addpool.py's runs are in TABLES2.md)
            runs.setdefault(m.group(1), []).append(r)
f = lambda v: f'{100 * min(v):.1f} / {100 * st.median(v):.1f} / {100 * max(v):.1f}'
out = ['# Addition: whole answer right, % (written by tables.py from res/*.json)\n',
       'Each entry: smallest / median / largest over the runs (seeds 1-3 x orders 0-4). "held, sum shown": the never-shown pairs whose sum '
       'occurs among shown pairs, each given 20 times with no learning; "held, sum not shown": those whose sum occurs in no shown pair.\n',
       '| System | runs | options | cells that fired (median) | last learning block | sums shown | sums never shown | held, sum shown | held, sum not shown |',
       '|---|---|---|---|---|---|---|---|---|']
for tag, rs in runs.items():
    opts = ' '.join(f'{k}={v}' for k, v in rs[0]['args'].items() if k not in ('seed', 'order'))
    hs = [st.mean([v['whole'] for v in r['per_held'].values() if v['sum_shown']]) for r in rs]
    hn = [x for r in rs for x in [v['whole'] for v in r['per_held'].values() if not v['sum_shown']]]
    out.append(f"| {tag} | {len(rs)} | `{opts}` | {st.median([r['cells'] for r in rs]):.0f} | {f([r['rows'][5][3] for r in rs])} | "
               f"{f([r['rows'][6][3] for r in rs])} | {f([r['rows'][7][3] for r in rs])} | {f(hs)} | "
               + (f'{f(hn)} ({len(hn)} pairs)' if hn else '—') + ' |')
out += ['\n## Every run\n', '| System | seed | order | sums shown | sums never shown | held pairs not right (whole answer < 100%) |', '|---|---|---|---|---|---|']
for tag, rs in runs.items():
    for r in rs:
        bad = ', '.join(f"{k} {100 * v['whole']:.0f}%" + ('' if v['sum_shown'] else ' (sum never shown)') for k, v in r['per_held'].items() if v['whole'] < 1)
        out.append(f"| {tag} | {r['args'].get('seed', 1)} | {r['args'].get('order', 0)} | {100 * r['rows'][6][3]:.1f} | {100 * r['rows'][7][3]:.1f} | {bad or 'none'} |")
open(os.path.join(HERE, 'TABLES.md'), 'w').write('\n'.join(out) + '\n')
print('\n'.join(out[:4 + len(runs)]))
