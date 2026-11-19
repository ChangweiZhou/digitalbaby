# Summary of the addpool.py runs (items 1-3 of the reviewer's message of 15:42 UTC). For each tag over its runs: the whole answer right on
# pairs shown and never shown, for each operation, smallest / median / largest; the never-shown pairs whose result occurs among shown pairs;
# the pool at the end (the kinds with the most weight). Writes TABLES2.md from res/<tag>_s<seed>o<order>.json made by addpool.py.
#   python3 -I tables2.py
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
        if isinstance(r['rows'][0][1], dict):            # addpool.py's format
            runs.setdefault(m.group(1), []).append(r)
f = lambda v: f'{100 * min(v):.1f} / {100 * st.median(v):.1f} / {100 * max(v):.1f}' if v else '—'
out = ['# Addition (and subtraction): runs of addpool.py (written by tables2.py from res/*.json)\n',
       'Whole answer right, %: smallest / median / largest over the runs (seeds 1-3 x orders 0-4). "held, result shown": never-shown pairs '
       'whose result occurs among the shown pairs of that operation, each given 20 times with no learning.\n',
       '| Tag | runs | options | op | shown | never shown | held, result shown | held, result not shown | largest w1-w2 difference / mean w (median) | cells withdrawn (median) |',
       '|---|---|---|---|---|---|---|---|---|---|']
for tag, rs in runs.items():
    opts = ' '.join(f'{k}={v}' for k, v in rs[0]['args'].items() if k not in ('seed', 'order'))
    for op in rs[0]['rows'][6][1]:
        hs = [st.mean([v['whole'] for k, v in r['per_held'].items() if op in k and v['result_shown']]) for r in rs]
        hn = [v['whole'] for r in rs for k, v in r['per_held'].items() if op in k and not v['result_shown']]
        sd = [r['strengths']['largest_relative_difference'] for r in rs if r['strengths']['largest_relative_difference'] is not None]
        out.append(f"| {tag} | {len(rs)} | `{opts}` | {op} | {f([r['rows'][6][1][op] for r in rs])} | {f([r['rows'][7][1][op] for r in rs])} | "
                   f"{f(hs)} | " + (f'{f(hn)} ({len(hn)})' if hn else '—') + f" | {st.median(sd):.4f} | {st.median([r['withdrawn'] for r in rs]):.0f} |")
out += ['\n## The pool at the end: for each run, the five kinds (D1-D2-G) with the most weight to the output units: cells of the kind, weight\n']
for tag, rs in runs.items():
    if len(rs[0]['pool_kinds']) <= 2:
        continue
    for r in rs:
        top = sorted(r['pool_kinds'].items(), key=lambda kv: -kv[1]['weight'])[:5]
        out.append(f"- {tag} seed {r['args'].get('seed')} order {r['args'].get('order')}: " + '; '.join(f"{k} ({v['n']}, {v['weight']:.1f})" for k, v in top))
open(os.path.join(HERE, 'TABLES2.md'), 'w').write('\n'.join(out) + '\n')
print('\n'.join(out[:4 + 2 * len(runs)]))
