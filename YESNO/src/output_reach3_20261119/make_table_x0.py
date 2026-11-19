# Builds TABLE_X0.md from res/x0_*.json (measure_x0.py).
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = (73129, 93683, 116089); TIS = ('N0', 'N64', 'N256'); X0 = ('0.375', '0.5', '0.625', '0.75', '1')
RULES = ('D eta=0.015625', 'B eps=0.0625', 'A eps=0 kappa=0.125', 'A eps=0.0625 kappa=0.125')
def J(t, s):
    p = os.path.join(HERE, 'res', f'x0_{t}_s{s}.json'); return json.load(open(p)) if os.path.exists(p) else None
D = {(t, s): J(t, s) for t in TIS for s in SEEDS}
def cell(vals):
    vals = [v for v in vals if v is not None]
    return '' if not vals else (f'{100 * vals[0]:.1f}' if len(vals) == 1 else f'{100 * min(vals):.1f}–{100 * max(vals):.1f}') + f' ({len(vals)})'
def cross(t, x, r, lvl):
    got = [D[(t, s)]['x0'][x]['B'][r]['first_window_reaching'][lvl] for s in SEEDS if D[(t, s)] and x in D[(t, s)]['x0']]
    hit = [g for g in got if g]; return f"{len(hit)}/{len(got)}" + (f" ({min(hit):,})" if hit else '') if got else ''
L = ['# The output end placed at x0 (measure_x0.py): share right on the last quarter, %, range over the seeds run (number of seeds in brackets)', '',
     'Drivers: 256 points (x0, y random), the nearest free non-feedback cell; inputs: the non-feedback cells within 1/8 of the point. Online inputs as they are.', '',
     '| tissue | x0 | band, fitted | own inputs, fitted | D (supervised) | B (yes/no) | A eps 0 (yes/no) | A eps 1/16 (yes/no) |', '|---|---|---|---|---|---|---|---|']
for t in TIS:
    for x in X0:
        ok = [(s, D[(t, s)]) for s in SEEDS if D[(t, s)] and x in D[(t, s)]['x0']]
        if not ok: continue
        row = [cell([d['x0'][x]['A']['all cells of the band']['shares']['all'] for _, d in ok]),
               cell([d['x0'][x]['A']["each driver's own inputs"]['shares']['all'] for _, d in ok])]
        row += [cell([d['x0'][x]['B'][r]['last_quarter'] for _, d in ok]) for r in RULES]
        L.append(f'| {t} | {x} | ' + ' | '.join(row) + ' |')
L += ['', '## Seeds in which a window of 4,096 reached the level (earliest end of such a window in brackets): end mark / ref_1 / ref_2', '',
      '| tissue | x0 | D | B | A eps 0 | A eps 1/16 |', '|---|---|---|---|---|---|']
for t in TIS:
    for x in X0:
        if not any(D[(t, s)] and x in D[(t, s)]['x0'] for s in SEEDS): continue
        L.append(f'| {t} | {x} | ' + ' | '.join(' / '.join(cross(t, x, r, l) for l in ('end mark', 'ref_1', 'ref_2')) for r in RULES) + ' |')
c = next((d['common'] for d in D.values() if d), None)
if c:
    L += ['', 'References (last quarter, seed range): ' + ', '.join(f"{k} " + cell([D[(TIS[0], s)]['common'][k]['shares']['all'] for s in SEEDS if D[(TIS[0], s)]]) for k in c)]
L.append('')
open(os.path.join(HERE, 'TABLE_X0.md'), 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
