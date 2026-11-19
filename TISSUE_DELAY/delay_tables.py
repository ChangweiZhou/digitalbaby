# Tables and the summary question of the delay runs, from delay/<label>.json. Writes DELAY_TABLES.md.
#   python3 -I delay_tables.py
# Levels: lag 0: 0.1103 (the reviewer's level of seed 73129, used for every seed); lags 1 to 4: c0(L) + 0.02 of the window (the rule that
# gives the reviewer's levels 0.090 and 0.076 at lags 1 and 2 on seed 73129; c0(3) is high because the stream's structure makes the byte
# three back partly predictable from the present byte).
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = (73129, 93683, 116089)
CANDS = [('none', 'no delay'), ('rand32', 'random 0–32 on every connection between cells'), ('rand16', 'random 0–16'),
         ('q4d32', '32 on 1/4'), ('dist32', 'proportional to distance, dmax 32')]
BASES = [('B', 'B (rule=timing norm=1 lines_learn=1 eta_s=1 no_make=1)'), ('B1', 'B1 (B with we=1 wi=1/4)')]


def load(l):
    p = os.path.join(HERE, 'delay', f'{l}.json')
    return json.load(open(p)) if os.path.exists(p) else None


def levels(c0):
    return {'0': 0.1103, **{k: c0[k] + 0.02 for k in ('1', '2', '3', '4')}}


def mk(v, L, lev):
    if v is None:
        return '—'
    return f'**{v:.3f}**' if v >= lev[L] else f'{v:.3f}'


def answer(w):
    """For one window: (present byte read by the tool at d = 1, 2, 4; for lags 1..3: above level somewhere (a or b), among near (a or b))."""
    lev = levels(w['c0']); g = {r['d']: r for r in w['rows']}
    s0 = all(d in g and g[d]['a']['0'] is not None and g[d]['a']['0'] >= lev['0'] for d in ('1', '2', '4'))
    dist_rows = [r for r in w['rows'] if r['d'] not in ('pool', 'near')]
    some = {L: {rd: any((r[rd][L] or 0) >= lev[L] for r in dist_rows + [g['pool']]) for rd in ('a', 'b')} for L in ('1', '2', '3')}
    near = {L: {rd: (g['near'][rd][L] or 0) >= lev[L] for rd in ('a', 'b')} for L in ('1', '2', '3')}
    return s0, some, near


out = ['# Delays in the tissue that carries the present byte far: tables (written by delay_tables.py from delay/*.json)\n',
       '16,384 bytes unless said; window: the last 4,096 bytes. (a) the readings tool as it is; (b) no floor, each cell divided by its own '
       'standard deviation. near: the drivers of the output lines and the cells of their neighbourhoods (reach.py\'s set). **Bold**: at or above '
       'the level (lag 0: 0.1103; lags 1 to 4: c0(L) + 0.02 of the window).\n',
       '## Summary: the present byte by the tool at d = 1, 2, 4, and bytes one to three back above their levels somewhere and among near cells\n',
       '| Base | Delay | Seed | S0 (a) d=1/2/4 | present at 1,2,4 by (a) | lag 1: somewhere (a/b) / near (a/b) | lag 2 | lag 3 | divergence 64/512 | connections a cell | CPU s |',
       '|---|---|---|---|---|---|---|---|---|---|---|']
yn = lambda x: 'yes' if x else 'no'
for bl, bn in BASES:
    for cl, cn in CANDS:
        for s in SEEDS:
            r = load(f'{bl}_{cl}_{s}')
            if r is None:
                out.append(f'| {bl} | {cn} | {s} | not run |' + ' |' * 7); continue
            w = r['windows'][-1]; g = {x['d']: x for x in w['rows']}
            s0, some, near = answer(w)
            s0v = ' / '.join(f"{g[d]['a']['0']:.3f}" if d in g and g[d]['a']['0'] is not None else '—' for d in ('1', '2', '4'))
            cell = lambda L: f"{yn(some[L]['a'])}/{yn(some[L]['b'])} / {yn(near[L]['a'])}/{yn(near[L]['b'])}"
            out.append(f"| {bl} | {cn} | {s} | {s0v} | {yn(s0)} | {cell('1')} | {cell('2')} | {cell('3')} | "
                       f"{r['divergence']['64']:.2g} / {r['divergence']['512']:.2g} | {w['connections_a_cell']:.1f} | {r['cpu_s']:.0f} |")
out += ['\n## Every row\n', 'Entries: S at lags 0 / 1 / 2 / 3 / 4. "max out": the largest raw output of the cells at d over the window.\n']
for bl, bn in BASES:
    for cl, cn in CANDS:
        for s in SEEDS:
            r = load(f'{bl}_{cl}_{s}')
            if r is None:
                continue
            for wi, w in enumerate(r['windows']):
                lev = levels(w['c0'])
                out += [f"\n### {bn}, delay {cn}, seed {s}, window {w['window'][0]}–{w['window'][1]} (c0 at lags 0–4: "
                        + ' / '.join(f"{w['c0'][k]:.3f}" for k in ('0', '1', '2', '3', '4')) + f"; connections a cell {w['connections_a_cell']:.1f}; "
                        + f"near cells by distance {w['near_by_distance']})\n",
                        '| d | cells | mean output | max out | (a) lags 0/1/2/3/4 | (b) lags 0/1/2/3/4 |', '|---|---|---|---|---|---|']
                for x in w['rows']:
                    out.append(f"| {x['d']} | {x['cells']} | {x['mean_output']:.4f} | {x['max_output']:.3f} | "
                               + ' / '.join(mk(x['a'][k], k, lev) for k in ('0', '1', '2', '3', '4')) + ' | '
                               + ' / '.join(mk(x['b'][k], k, lev) for k in ('0', '1', '2', '3', '4')) + ' |')
open(os.path.join(HERE, 'DELAY_TABLES.md'), 'w').write('\n'.join(out) + '\n')
print('\n'.join(out[:5 + 30]))
