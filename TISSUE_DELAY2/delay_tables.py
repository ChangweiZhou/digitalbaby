# Tables and the summary question of the delay runs, from delay/<label>.json. Writes DELAY_TABLES.md.
#   python3 -I delay_tables.py
# Levels: lag 0: 0.1103 (the reviewer's level of seed 73129, used for every seed); lags 1 to 4: c0(L) + 0.02 of the window (the rule that
# gives the reviewer's levels 0.090 and 0.076 at lags 1 and 2 on seed 73129; c0(3) is high because the stream's structure makes the byte
# three back partly predictable from the present byte).
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = (73129, 93683, 116089)
CANDS = [('none', 'no delay'), ('rand8', 'random 0–8 (made after the first rows)'), ('rand12', 'random 0–12 (made after the first rows)'),
         ('rand16', 'random 0–16 on every pair of cells'), ('rand32', 'random 0–32'), ('q4d32', '32 on 1/4 of the pairs'),
         ('rand16_64k', 'random 0–16, 65,536 bytes (summary from its last window)'), ('rand32_64k', 'random 0–32, 65,536 bytes (last window)'),
         ('q4d32_64k', '32 on 1/4, 65,536 bytes (last window)')]
BASES = [('N', 'N (rule=timing prune_only=1 norm=1 lines_learn=1 we=3/4 wi=1/4 gain=3 rho=1 gmax=8 mk_long=1 a_make=1/4096 make_period=1024)'),
         ('Nt', 'N with tau=u2-128')]


def load(l):
    p = os.path.join(HERE, 'delay', f'{l}.json')
    return json.load(open(p)) if os.path.exists(p) else None


def levels(c0):
    return {'0': 0.1103, **{k: c0[k] + 0.02 for k in ('1', '2', '3', '4')}}


def mk(v, L, lev):
    if v is None:
        return '—'
    return f'**{v:.3f}**' if v >= lev[L] else f'{v:.3f}'


def answer(w, div):
    """The reviewer's question of 08:12 UTC for one window: the present byte read by the tool at d = 1, 2, 4, 8, 10; bytes one to three back
    above their levels among the near cells (by (a), by (b)); the largest output of the near cells above 1/2; no divergence."""
    lev = levels(w['c0']); g = {r['d']: r for r in w['rows']}
    s0 = {d: (g[d]['a']['0'] if d in g else None) for d in ('1', '2', '4', '8', '10')}
    s0ok = all(v is not None and v >= lev['0'] for v in s0.values())
    near = {L: {rd: (g['near'][rd][L] or 0) >= lev[L] for rd in ('a', 'b')} for L in ('1', '2', '3')}
    mx = g['near']['max_output']
    nodiv = div['64'] == 0 and div['512'] == 0
    allok = s0ok and all(near[L]['a'] or near[L]['b'] for L in ('1', '2', '3')) and mx > 0.5 and nodiv
    return s0, s0ok, near, mx, nodiv, allok


out = ['# Delays in the base of 3b486e0a: tables (written by delay_tables.py from delay/*.json)\n',
       '16,384 bytes unless said; window: the last 4,096 bytes. (a) the readings tool as it is; (b) no floor, each cell divided by its own '
       'standard deviation. near: the drivers of the output lines and the cells of their neighbourhoods (reach.py\'s set). **Bold**: at or above '
       'the level (lag 0: 0.1103; lags 1 to 4: c0(L) + 0.02 of the window). answering: neither silent nor constant (section 7.2).\n',
       '## The question: the present byte by the tool at d = 1, 2, 4, 8, 10; bytes one to three back above their levels among the near cells; '
       'their largest output above 1/2; no divergence\n',
       '| Base | Delay | Seed | S0 (a) at d = 1/2/4/8/10 | all five | near L1 (a/b) | near L2 | near L3 | near largest output | no divergence | all | connections a cell | CPU s |',
       '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
yn = lambda x: 'yes' if x else 'no'
for bl, bn in BASES:
    for cl, cn in CANDS:
        for s in SEEDS:
            r = load(f'{bl}_{cl}_{s}')
            if r is None:
                continue
            w = r['windows'][-1]
            s0, s0ok, near, mx, nodiv, allok = answer(w, r['divergence'])
            s0v = ' / '.join('—' if v is None else f'{v:.3f}' for v in s0.values())
            c = lambda L: f"{yn(near[L]['a'])}/{yn(near[L]['b'])}"
            out.append(f"| {bl} | {cn} | {s} | {s0v} | {yn(s0ok)} | {c('1')} | {c('2')} | {c('3')} | {mx:.3f} | {yn(nodiv)} | **{yn(allok)}** | "
                       f"{w['connections_a_cell']:.1f} | {r['cpu_s']:.0f} |")
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
                        '| d | cells | answering | mean output | max out | (a) lags 0/1/2/3/4 | (b) lags 0/1/2/3/4 |', '|---|---|---|---|---|---|---|']
                for x in w['rows']:
                    out.append(f"| {x['d']} | {x['cells']} | {x['answering']:.3f} | {x['mean_output']:.4f} | {x['max_output']:.3f} | "
                               + ' / '.join(mk(x['a'][k], k, lev) for k in ('0', '1', '2', '3', '4')) + ' | '
                               + ' / '.join(mk(x['b'][k], k, lev) for k in ('0', '1', '2', '3', '4')) + ' |')
open(os.path.join(HERE, 'DELAY_TABLES.md'), 'w').write('\n'.join(out) + '\n')
print('\n'.join(out[:5 + 30]))
