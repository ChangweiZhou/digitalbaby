# Builds TABLE_Q.md from res/Q_*.json and res/QL_*.json (measure_q.py).
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = (93683, 73129, 116089, 146821, 224268, 746958); TIS = ('L8-256', 'L8-128', 'A4-256', 'A4-128')
RULES = ['B eps=0.0625', 'D eta=0.015625', 'Q eps=0.0625 eta=0.25', 'Q eps=0.0625 eta=0.0625', 'Q eps=0.0625 eta=0.015625',
         'Qe eps=0.0625 eta=0.25', 'Qe eps=0.0625 eta=0.0625', 'Qe eps=0.0625 eta=0.015625']
def J(n):
    p = os.path.join(HERE, 'res', n + '.json'); return json.load(open(p)) if os.path.exists(p) else None
def fw(v):
    f = v['first_window_reaching']; return '/'.join(f'{f[k] // 1024}k' if f[k] else '-' for k in ('ref_1', 'ref_2'))
L = ['# Rule Q beside B and D on the long-axon tissues (measure_q.py), 65,536 bytes, birth\'s drivers at x = 1', '',
     'Cell: share right on the last quarter, % (running standardisation | clipped at 8), then the first window of 4,096 reaching ref_1/ref_2 in each form (k = 1,024 bytes; "-" never).',
     'Seeds 146821, 224268, 746958 (added at the request of the reviewer at 13:31) only on L8-128 and A4-256. Levels on the last quarter: ref_1 10.1-10.7, ref_2 12.1-12.7, end mark 6.2-6.3.', '']
for t in TIS:
    L += [f'## {t}', '', '| rule | ' + ' | '.join(f'seed {s}' for s in SEEDS) + ' |', '|---|' + '---|' * len(SEEDS)]
    for r in RULES:
        c = []
        for s in SEEDS:
            d = J(f'Q_{t}_s{s}')
            if not d: c.append(''); continue
            a, b = d['B'][f'run {r}'], d['B'][f'clip {r}']
            c.append(f"{100 * a['last_quarter']:.1f} \\| {100 * b['last_quarter']:.1f} ({fw(a)} \\| {fw(b)})")
        L.append(f'| {r} | ' + ' | '.join(c) + ' |')
    L.append('')
d = J('QL_L8-256_s73129')
if d:
    L += ['## L8-256, seed 73129, 262,144 bytes', '', '| rule | last quarter (run \\| clip) | mean per 32,768 bytes (run) | largest row norm of W at 65,536 / 131,072 / end (run) |', '|---|---|---|---|']
    for r in RULES:
        a, b = d['B'][f'run {r}'], d['B'][f'clip {r}']; w = a['windows']
        blocks = ' '.join(f'{100 * sum(w[i:i + 8]) / len(w[i:i + 8]):.1f}' for i in range(0, len(w), 8))
        nm = r.replace(' eps=0.0625', '').replace('eps=0.0625', '')
        key = lambda c: f"run {r.split()[0]} eta={r.split('eta=')[1]}" if 'eta=' in r else f"run {r.split()[0]} eta=0"
        rn = ' / '.join(f"{d['rownorm'][str(k)].get(key(0), float('nan')):.3g}" if str(k) in d['rownorm'] else '-' for k in sorted(map(int, d['rownorm'])))
        L.append(f"| {r} | {100 * a['last_quarter']:.1f} \\| {100 * b['last_quarter']:.1f} | {blocks} | {rn} |")
    L += ['', 'ref_1 per 32,768 bytes: ' + ' '.join(f"{100 * sum(d['levels']['ref_1'][i:i + 8]) / 8:.1f}" for i in range(0, 64, 8)),
          'ref_2 per 32,768 bytes: ' + ' '.join(f"{100 * sum(d['levels']['ref_2'][i:i + 8]) / 8:.1f}" for i in range(0, 64, 8)), '']
open(os.path.join(HERE, 'TABLE_Q.md'), 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
