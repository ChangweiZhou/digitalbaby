# Item 2 of the reviewer's message of 10:45 UTC, 19 November: memory (vi) (delays 0-8, one cell in 4 at 64 steps) on six lives
# (life 0: unchanged; lives 1-5: the potential of the 1st to 5th non-feedback cell set to 1e-9 before step 0) on seeds 93683 and 112290,
# 16,384 bytes; and 65,536 bytes on those two seeds, four windows. Reading (a), the tool as it is; levels as in s15.py.
# Writes LIVES_TABLES.md from delay/LIFE_d8f64_*.json and delay/LONG_d8f64_64k_*.json.
#   python3 -I lives.py
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = (93683, 112290)


def levels(c0):
    return {'0': 0.1103, '1': c0['1'] + 0.02, '2': c0['2'] + 0.02}


def row(w):
    lev = levels(w['c0']); g = {x['d']: x for x in w['rows']}
    nine = all(d in g and g[d]['a'][L] is not None and g[d]['a'][L] >= lev[L] for d in ('1', '2', '4') for L in ('0', '1', '2'))
    above = {grp: all(g[grp]['a'][L] >= lev[L] for L in '012') for grp in ('near', 'near8')}
    return g, lev, nine, above


yn = lambda x: 'yes' if x else 'no'
out = ['# Memory (vi) on six lives and over 65,536 bytes (written by lives.py from delay/LIFE_d8f64_*.json, delay/LONG_d8f64_64k_*.json)\n',
       'Line L with long=8-1/2 and `cand=rand dmax=8 tau=f4-64`; the tool as it is. Levels: lag 0 0.1103; lags 1, 2: c0(L) + 0.02 of the window. '
       'Life 0: unchanged; life k (1-5): the potential of the k-th non-feedback cell set to 1e-9 before step 0.\n',
       '## Six lives, 16,384 bytes, last 4,096 bytes\n',
       '| Seed | Life | near (a) L0/L1/L2 | near8 (a) L0/L1/L2 | levels L1/L2 | all three near / near8 | nine all above | S0 at d = 1/2/4 | near max | divergence 64/512 | CPU s |',
       '|---|---|---|---|---|---|---|---|---|---|---|']
cnt = {s: [0, 0, 0] for s in SEEDS}
for s in SEEDS:
    for life in range(6):
        p = os.path.join(HERE, 'delay', f'LIFE_d8f64_{s}_l{life}.json')
        if not os.path.exists(p):
            continue
        r = json.load(open(p)); g, lev, nine, above = row(r['windows'][-1])
        cnt[s][0] += 1; cnt[s][1] += above['near']; cnt[s][2] += above['near8']
        out.append(f"| {s} | {life} | " + ' / '.join(f"{g['near']['a'][L]:.3f}" for L in '012') + ' | '
                   + ' / '.join(f"{g['near8']['a'][L]:.3f}" for L in '012') + f" | {lev['1']:.3f} / {lev['2']:.3f} | "
                   + f"{yn(above['near'])} / {yn(above['near8'])} | {yn(nine)} | " + ' / '.join(f"{g[d]['a']['0']:.3f}" for d in ('1', '2', '4'))
                   + f" | {g['near']['max_output']:.3f} | {r['divergence']['64']:.2g} / {r['divergence']['512']:.2g} | {r['cpu_s']:.0f} |")
out += ['\nLives with all three above: ' + '; '.join(f"seed {s}: near {c[1]} of {c[0]}, near8 {c[2]} of {c[0]}" for s, c in cnt.items()) + '\n',
        '## 65,536 bytes, four windows (the last 4,096 bytes of each quarter)\n',
        '| Seed | Window | near (a) L0/L1/L2 | near8 (a) L0/L1/L2 | levels L1/L2 | all three near / near8 | nine all above | S0 at d = 1/2/4 | near max | connections a cell |',
        '|---|---|---|---|---|---|---|---|---|---|']
for s in SEEDS:
    p = os.path.join(HERE, 'delay', f'LONG_d8f64_64k_{s}.json')
    if not os.path.exists(p):
        continue
    r = json.load(open(p))
    for w in r['windows']:
        g, lev, nine, above = row(w)
        out.append(f"| {s} | {w['window'][0]}–{w['window'][1]} | " + ' / '.join(f"{g['near']['a'][L]:.3f}" for L in '012') + ' | '
                   + ' / '.join(f"{g['near8']['a'][L]:.3f}" for L in '012') + f" | {lev['1']:.3f} / {lev['2']:.3f} | "
                   + f"{yn(above['near'])} / {yn(above['near8'])} | {yn(nine)} | " + ' / '.join(f"{g[d]['a']['0']:.3f}" for d in ('1', '2', '4'))
                   + f" | {g['near']['max_output']:.3f} | {w['connections_a_cell']:.1f} |")
    out.append(f"\nSeed {s}, 65,536 bytes: divergence 64/512 {r['divergence']['64']:.2g} / {r['divergence']['512']:.2g}; CPU {r['cpu_s']:.0f} s.\n")
open(os.path.join(HERE, 'LIVES_TABLES.md'), 'w').write('\n'.join(out) + '\n')
print('\n'.join(out))
