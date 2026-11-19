# Tables of the runs of TISSUE_PREV2, from prev2/<label>.json. Writes PREV2_TABLES.md.
#   python3 -I prev2_tables.py
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
LEVEL = {'0': 0.1103, '1': 0.090, '2': 0.076}
STEP1 = [('M_base', 'M base (gain=2 rho=1 gmax=4)'), ('U_base', 'U base (no gain)')] + \
    [(f'{b}_del{D}_{q}', f'{b} delay {D} on {sh}') for b in ('M', 'U') for q, sh in (('q4', '1/4'), ('q2', '1/2')) for D in (8, 16, 32)]
STEP2 = [(f'{b}_dist{D}', f'{b} delay proportional to distance, dmax {D}') for b in ('M', 'U') for D in (8, 16, 32)] + \
    [(f'{b}_rand32', f'{b} delay drawn from 0 to 32') for b in ('M', 'U')]
SEEDS = (73129, 93683, 116089)
FULL = [('F_M_base', 'M base'), ('F_M_rand32', 'M, delay drawn from 0 to 32 on every connection between cells'), ('F_M_del32_q4', 'M, delay 32 on 1/4')]


def load(l):
    p = os.path.join(HERE, 'prev2', f'{l}.json')
    return json.load(open(p)) if os.path.exists(p) else None


def mk(v, L, lev=None):
    lev = lev or LEVEL
    if v is None:
        return '—'
    return f'**{v:.3f}**' if L in lev and v >= lev[L] else f'{v:.3f}'


def seed_levels(c0):
    """Lag 0: 0.1103 (the reviewer's level of seed 73129, used for every seed); lags 1 to 4: c0(L) + 0.02 of the seed, the rule that gives
    the reviewer's levels 0.090 and 0.076 at lags 1 and 2 on seed 73129 (c0 0.070 and 0.056)."""
    return {'0': 0.1103, **{k: c0[k] + 0.02 for k in ('1', '2', '3', '4')}}


def rows(lst):
    o = ['| Tissue | S(1,0) | S(2,0) | S(4,0) | S(1,1) | S(2,1) | S(4,1) | S(1,2) | S(2,2) | S(4,2) | divergence after 64 / 512 bytes | gain median | mean delay of a connection between cells | CPU s |',
         '|---|' + '---|' * 13]
    for lab, name in lst:
        r = load(lab)
        if r is None:
            o.append(f'| {name} | not run |'); continue
        S = r['S']
        ds = r.get('delay_stats') or {}
        o.append(f'| {name} | ' + ' | '.join(mk(S[d][L], L) for L in ('0', '1', '2') for d in ('1', '2', '4'))
                 + f" | {r['divergence']['64']:.2g} / {r['divergence']['512']:.2g} | {r['gain_median']:.2f} | {ds.get('mean_delay', 0):.1f} | {r['cpu_s']:.0f} |")
    return o


out = ['# Bytes back at the reading step on bases that cannot depend on their past: tables (written by prev2_tables.py from prev2/*.json)\n',
       'Seed 73129 unless said, 16,384 bytes, window A2, S(d, L) by the readings tool as it is. **Bold**: at or above the level of seed 73129 '
       '(L = 0: 0.1103; 1: 0.090; 2: 0.076). M: rule=fixed gain=2 rho=1 gmax=4. U: rule=fixed, no gain. Divergence: two copies, one cell\'s '
       'potential changed by 1e-9, the largest difference of outputs 64 and 512 bytes later.\n', '## Step 1: a share of the connections between cells delayed\n']
out += rows(STEP1)
out += ['\n## Step 2: every connection between cells with its own delay\n']
out += rows(STEP2)
out += ['\n## Step 3: three seeds, every distance, lags 0 to 4\n',
        '(a) the readings tool as it is (outputs below 0.01 set to 0); (b) no floor, each cell divided by its own standard deviation (where.py\'s '
        'method). "pool": all cells that are not feedback cells (a sample of 128). **Bold** here: at or above the seed\'s own level, lag 0 '
        '0.1103 and lags 1 to 4 c0(L) + 0.02 of the seed (the rule that gives the levels 0.090 and 0.076 of lags 1 and 2 on seed 73129). '
        'c0 at lag 3 is high (0.16 on seed 73129): the stream\'s structure makes the byte three back partly predictable from the present byte. '
        'Entries: S at lags 0 / 1 / 2 / 3 / 4.\n']
for lab, name in FULL:
    for s in SEEDS:
        r = load(f'{lab}_{s}')
        if r is None:
            out.append(f'\n### {name}, seed {s}: not run\n'); continue
        F = r['full']
        out += [f'\n### {name}, seed {s} (divergence after 512 bytes {r["divergence"]["512"]:.2g}; c0 at lags 0–4: '
                + ' / '.join(f"{F['c0'][k]:.3f}" for k in ('0', '1', '2', '3', '4')) + ')\n',
                '| d | cells | mean output | (a) lags 0/1/2/3/4 | (b) lags 0/1/2/3/4 |', '|---|---|---|---|---|']
        for row in F['rows']:
            lev = seed_levels(F['c0'])
            fa = ' / '.join(mk(row['a'][k], k, lev) for k in ('0', '1', '2', '3', '4'))
            fb = ' / '.join(mk(row['b'][k], k, lev) for k in ('0', '1', '2', '3', '4'))
            out.append(f"| {row['d']} | {row['cells']} | {row['mean_output']:.4f} | {fa} | {fb} |")
open(os.path.join(HERE, 'PREV2_TABLES.md'), 'w').write('\n'.join(out) + '\n')
print('written', len(out), 'lines')
