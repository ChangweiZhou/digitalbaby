# Item 3 (the reviewer's message after 40d93aea2): the output end on tissue (vi) (no long axons) and on the tissue with no memory.
# Writes OUTQ_TABLES.md from outq/<label>.json: the share right in the last quarter (and the four quarters) beside ref_1 and ref_2.
#   python3 -I outq_tables.py
import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = (73129, 93683, 116089, 465973, 614332, 406364, 545353)
TIS = [('vi', '(vi) delays 0–8, one cell in 4 at 64 steps, no long axons'), ('none', 'no memory: no delays, no slow cells, no long axons')]
SETS = [('own', 'input alone'), ('own traces 2,4', 'with copies 2,4')]
RULES = ['D eta=1/64', 'Q 1/16 z1']

out = ['# The output end on tissue (vi) and on the tissue with no memory (written by outq_tables.py from outq/*.json)\n',
       '65,536 bytes; inputs: unit v sees the cells of the neighbourhood of the v-th output-line cell (\'own\'); the reviewer\'s outq.py rules '
       'D (eta 1/64) and Q (eta 1/16, z1). Share right in the last quarter, %.\n',
       '| Seed | ref_1 | ref_2 | ' + ' | '.join(f'{t} {r[0]} {s}' for t, _ in TIS for _, s in SETS for r in RULES) + ' |',
       '|---' * (3 + len(TIS) * len(SETS) * len(RULES)) + '|']
col = {}
for s in SEEDS:
    rs = {t: (json.load(open(p)) if os.path.exists(p := os.path.join(HERE, 'outq', f'{t}_{s}.json')) else None) for t, _ in TIS}
    any_r = next((r for r in rs.values() if r), None)
    if any_r is None:
        continue
    ref = any_r['last_quarter_refs']
    cells = []
    for t, _ in TIS:
        for sk, sn in SETS:
            for r in RULES:
                v = rs[t]['sets'][sk]['rules'][r]['last_quarter'] * 100 if rs[t] else None
                cells.append('—' if v is None else f'{v:.2f}')
                if v is not None:
                    col.setdefault((t, sn, r[0]), []).append(v)
    out.append(f"| {s} | {100 * ref['ref_1']:.2f} | {100 * ref['ref_2']:.2f} | " + ' | '.join(cells) + ' |')
out += ['\n## Over the seeds run: smallest / median / largest\n', '| Tissue | inputs | rule | seeds | smallest | median | largest |', '|---|---|---|---|---|---|---|']
for (t, sn, r), v in col.items():
    out.append(f'| {t} | {sn} | {r} | {len(v)} | {min(v):.2f} | {st.median(v):.2f} | {max(v):.2f} |')
out += ['\n## Quarters (%, first to last)\n', '| Tissue | Seed | inputs | D quarters | Q quarters |', '|---|---|---|---|---|']
for t, _ in TIS:
    for s in SEEDS:
        p = os.path.join(HERE, 'outq', f'{t}_{s}.json')
        if not os.path.exists(p):
            continue
        r = json.load(open(p))
        for sk, sn in SETS:
            q = r['sets'][sk]['rules']
            out.append(f"| {t} | {s} | {sn} | " + ' / '.join(f'{100 * x:.2f}' for x in q['D eta=1/64']['quarters']) + ' | '
                       + ' / '.join(f'{100 * x:.2f}' for x in q['Q 1/16 z1']['quarters']) + ' |')
open(os.path.join(HERE, 'OUTQ_TABLES.md'), 'w').write('\n'.join(out) + '\n')
print('\n'.join(out))
