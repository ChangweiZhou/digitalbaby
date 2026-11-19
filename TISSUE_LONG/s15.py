# Summary of the 15-seed runs (the reviewer's message of 10:45 UTC, 19 November): for each memory, on how many of the 15 seeds each of lags
# 0, 1, 2 is above its level among the cells near the output lines (the enlarged group: drivers and every member of their neighbourhoods,
# long-axon members included; and the cells within 1/8 of a driver), with the smallest and the median; the nine entries (S at d = 1, 2, 4,
# lags 0, 1, 2, by the tool); the largest output near; divergence. Reading (a): the tool as it is. Levels: lag 0 0.1103; lags 1, 2:
# c0(L) + 0.02 of the window. Writes S15_TABLES.md.
#   python3 -I s15.py
import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = (73129, 93683, 116089, 465973, 614332, 545353, 406364, 308924, 754734, 224268, 746958, 146821, 438778, 112290, 969620)
MEM = [('d8', '(ii) delays 0–8'), ('d16', "(ii') delays 0–16"), ('t64', '(iii) time constants up to 64'),
       ('d8f64', '(vi) delays 0–8, one cell in 4 at 64 steps'), ('d16f64', '(vii) delays 0–16, one cell in 4 at 64 steps')]


def levels(c0):
    return {'0': 0.1103, '1': c0['1'] + 0.02, '2': c0['2'] + 0.02}


out = ['# Long axons with memories on 15 seeds: summary (written by s15.py from delay/S_*.json)\n',
       'long=8-1/2, 16,384 bytes, last 4,096 bytes; the tool as it is. near: drivers and every member of their neighbourhoods (long-axon members '
       'included); near8: the cells within 1/8 of a driver. Levels: lag 0 0.1103; lags 1, 2: c0(L) + 0.02 of the window.\n',
       '| Memory | seeds run | group | lag 0: above / smallest / median | lag 1 | lag 2 | all three above | nine entries all above | near largest output (smallest) | divergence 64/512 not 0 |',
       '|---|---|---|---|---|---|---|---|---|---|']
rows_all = {}
for key, name in MEM:
    runs = {}
    for s in SEEDS:
        p = os.path.join(HERE, 'delay', f'S_{key}_{s}.json')
        if os.path.exists(p):
            runs[s] = json.load(open(p))
    if not runs:
        continue
    per = {}
    for s, r in runs.items():
        w = r['windows'][-1]; lev = levels(w['c0']); g = {x['d']: x for x in w['rows']}
        nine = all(d in g and g[d]['a'][L] is not None and g[d]['a'][L] >= lev[L] for d in ('1', '2', '4') for L in ('0', '1', '2'))
        per[s] = dict(lev=lev, g=g, nine=nine, div=(r['divergence']['64'], r['divergence']['512']))
    rows_all[key] = per
    for grp in ('near', 'near8'):
        cells = []
        for L in ('0', '1', '2'):
            v = [per[s]['g'][grp]['a'][L] for s in per]; a = [per[s]['g'][grp]['a'][L] >= per[s]['lev'][L] for s in per]
            cells.append(f"{sum(a)} / {min(v):.3f} / {st.median(v):.3f}")
        allthree = sum(all(per[s]['g'][grp]['a'][L] >= per[s]['lev'][L] for L in ('0', '1', '2')) for s in per)
        nine = sum(per[s]['nine'] for s in per)
        mx = min(per[s]['g'][grp]['max_output'] for s in per)
        dv = sum(1 for s in per if per[s]['div'] != (0.0, 0.0))
        out.append(f"| {name} | {len(per)} | {grp} | {cells[0]} | {cells[1]} | {cells[2]} | {allthree} | {nine} | {mx:.3f} | {dv} |")
out += ['\n## Every seed (near and near8, lags 0 / 1 / 2, with the levels; nine entries)\n',
        '| Memory | Seed | near (a) L0/L1/L2 | near8 (a) L0/L1/L2 | levels L1/L2 | nine all above | S0 at d = 1/2/4 | divergence 64/512 |', '|---|---|---|---|---|---|---|---|']
for key, name in MEM:
    for s, v in rows_all.get(key, {}).items():
        g, lev = v['g'], v['lev']
        out.append(f"| {name} | {s} | " + ' / '.join(f"{g['near']['a'][L]:.3f}" for L in '012') + ' | ' + ' / '.join(f"{g['near8']['a'][L]:.3f}" for L in '012')
                   + f" | {lev['1']:.3f} / {lev['2']:.3f} | {'yes' if v['nine'] else 'no'} | " + ' / '.join(f"{g[d]['a']['0']:.3f}" for d in ('1', '2', '4'))
                   + f" | {v['div'][0]:.2g} / {v['div'][1]:.2g} |")
open(os.path.join(HERE, 'S15_TABLES.md'), 'w').write('\n'.join(out) + '\n')
print('\n'.join(out[:4 + 2 * len(rows_all)]))
