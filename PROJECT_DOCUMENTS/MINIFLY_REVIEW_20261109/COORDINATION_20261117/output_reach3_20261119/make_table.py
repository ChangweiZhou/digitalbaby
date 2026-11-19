# Builds TABLE.md from res/*.json (measure.py).
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = (73129, 93683, 116089); TIS = ('U', 'N0', 'N128', 'N256')
def J(t, s):
    p = os.path.join(HERE, 'res', f'{t}_s{s}.json'); return json.load(open(p)) if os.path.exists(p) else None
cols = [(t, s) for t in TIS for s in SEEDS]
hdr = '| | ' + ' | '.join(f'{t} {s}' for t, s in cols) + ' |'; sep = '|---|' + '---|' * len(cols)
L = ['# The output end on N0, N128 and N256 beside U (measure.py), 65,536 bytes, share right %', '',
     'U: rule=fixed. N0: rule=timing prune_only=1 norm=1 lines_learn=1 we=3/4 wi=1/4 gain=3 rho=1 gmax=8 mk_long=1 a_make=1/4096 make_period=1024. N128, N256: N0 with tau=u2-128, tau=u2-256.', '',
     '## A. Fitted ridge readers, last quarter: all / R1 / R2', '', hdr, sep]
first = next((J(t, s) for t, s in cols if J(t, s)), None)
if first:
    for k in first['A']:
        c = []
        for t, s in cols:
            d = J(t, s); c.append(' / '.join(f"{100 * d['A'][k]['shares'][q]:.1f}" for q in ('all', 'R1', 'R2')) if d else '')
        L.append(f'| {k} | ' + ' | '.join(c) + ' |')
    L += ['', '## B. Online rules on the cells near each driver: last window / last quarter', '', hdr, sep]
    for k in first['B']:
        c = []
        for t, s in cols:
            d = J(t, s); c.append(f"{100 * d['B'][k]['last_window']:.1f} / {100 * d['B'][k]['last_quarter']:.1f}" if d else '')
        L.append(f'| {k} | ' + ' | '.join(c) + ' |')
    L += ['', '## C. End of the first window of 4,096 whose share reached the end mark / ref_1 / ref_2 of that window ("-": never)', '', hdr, sep]
    for k in first['B']:
        c = []
        for t, s in cols:
            d = J(t, s)
            if not d: c.append(''); continue
            f = d['B'][k]['first_window_reaching']; c.append(' / '.join(str(f[q]) if f[q] else '-' for q in ('end mark', 'ref_1', 'ref_2')))
        L.append(f'| {k} | ' + ' | '.join(c) + ' |')
    L += ['', '## Set (a)', '', hdr, sep,
          '| cells / mean output / largest output | ' + ' | '.join((lambda d: f"{d['cells_a']} / {d['mean_output_a']:.2g} / {d['max_output_a']:.2g}" if d else '')(J(t, s)) for t, s in cols) + ' |']
L.append('')
open(os.path.join(HERE, 'TABLE.md'), 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
