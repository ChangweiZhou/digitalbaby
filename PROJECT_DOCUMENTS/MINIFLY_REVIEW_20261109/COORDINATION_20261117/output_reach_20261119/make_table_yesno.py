# Builds TABLE_YESNO.md from yesno/*.json (yesno.py: R1, R2, R3, D; yesno2.py, labels y2_*: A, B).
import json, os
HERE = os.path.dirname(os.path.abspath(__file__)); Y = os.path.join(HERE, 'yesno')
SEEDS = (73129, 93683, 116089); INPUTS = ('oh1', 'oh2', 'oh3', 'U', 'M')
NAMES = dict(oh1='present byte (one-hot)', oh2='present + previous byte', oh3='last three bytes', U='U cells near own driver', M='M cells near own driver')
FAM = ('R1', 'R2', 'R3', 'A', 'B', 'D')
def load(inp, s, nb):
    out = {}
    for lab in (f'{inp}_s{s}', f'y2_{inp}_s{s}'):
        for suf in ('', f'_{nb}'):
            p = os.path.join(Y, lab + suf + '.json')
            if os.path.exists(p):
                d = json.load(open(p))
                if d['bytes'] == nb: out.setdefault('rules', {}).update(d['rules']); out['refs'] = d['references']; out['windows'] = d['windows']
    return out
def passed(v, lvl, wins):
    for x, l, w in zip(v, lvl, wins):
        if x >= l: return w[1]
    return None
L = ['# The output end learning online from yes/no (yesno.py, yesno2.py): share right, %', '',
     'Cell: the last window of 4,096 bytes for the best setting of the family (the setting in brackets: eps, eta or kappa), then the bytes after '
     'which a window first reached the share of the most frequent byte of that window (f) and ref_1 of that window (r1); "-" never.', '']
for nb in (65536, 262144):
    rows = []
    for inp in INPUTS:
        for fam in FAM:
            cells = []
            for s in SEEDS:
                d = load(inp, s, nb)
                ks = [k for k in d.get('rules', {}) if k.split()[0] == fam]
                if not ks: cells.append(None); continue
                k = max(ks, key=lambda k: d['rules'][k][-1]); v = d['rules'][k]; R = d['refs']
                pf = passed(v, R['most frequent byte of the window'], d['windows']); p1 = passed(v, R['ref_1'], d['windows'])
                cells.append(f"{100 * v[-1]:.1f} ({' '.join(k.split()[1:])}); f {pf if pf else '-'}, r1 {p1 if p1 else '-'}")
            if any(cells): rows.append(f"| {NAMES[inp]} | {fam} | " + ' | '.join(c or '' for c in cells) + ' |')
    if not rows: continue
    L += [f'## {nb:,} bytes', '', '| input | rule | ' + ' | '.join(f'seed {s}' for s in SEEDS) + ' |', '|---|---|' + '---|' * 3] + rows
    for rn in ('most frequent byte of the window', 'end mark (byte 10)', 'ref_1', 'ref_2', 'ref_3'):
        c = []
        for s in SEEDS:
            d = load('oh1', s, nb) or load('U', s, nb)
            c.append(f"{100 * d['refs'][rn][-1]:.1f}" if d else '')
        L.append(f'| reference | {rn} | ' + ' | '.join(c) + ' |')
    L.append('')
open(os.path.join(HERE, 'TABLE_YESNO.md'), 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
