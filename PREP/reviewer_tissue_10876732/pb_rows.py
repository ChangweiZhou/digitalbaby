# pb_rows.py <prefix>: one line per tissue of pbstand/<prefix>*.json: near the output lines (far.py's group), the linear and the nearest-centroid
# reader, for the present byte and 1, 2, 3 back: end mark / equals sign / digits above 1/2 (of 10); then the stand.py figures; and per
# tissue kind and draw, on how many seeds the end mark and the equals sign are read above 1/2 and the digits above 1/2 (median, smallest).
import sys, os, json, glob
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
pre = sys.argv[1] if len(sys.argv) > 1 else 'A4'
R = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(HERE, 'pbstand', pre + '*.json')))]
def kind(o): return o['label'].rsplit('_', 2)[0] + ('_own' if o['args'].get('draw') == 'own' else '_rev')
for o in sorted(R, key=lambda o: (kind(o), o['seed'])):
    w = list(o['windows'].values())[0]; e = o['loops_end']['excitatory']; g = o['perbyte']['near output']
    f = lambda rd: ' '.join(f"{g['lags'][l][rd]['end']:.2f}/{g['lags'][l][rd]['eq']:.2f}/{g['lags'][l][rd]['digits_above_half']}" for l in '0123')
    print(f"{kind(o):12s} {o['seed']:>6} near({g['cells']}) linear {f('linear')} | centroid {f('centroid')} | conns {w['conns_per_cell']:.2f} (max {w['conns_max']}) from-line cells {o['cells_from_input_line']} "
          f"| made {o['end']['made']} wd {o['end']['given']} lastq {o['made_last_quarter']}/{o['given_last_quarter']} | div {o['divergence']['512']:.1e} | exc loops {e['loops']}/{e['largest']}")
print('== counts: seeds on which the reader names the end mark / the equals sign above 1/2 of the time; digits above 1/2 (median, smallest), near the output lines')
for k in sorted(set(kind(o) for o in R)):
    rs = [o for o in R if kind(o) == k]
    for rd in ('linear', 'centroid'):
        s = []
        for l in '0123':
            v = [o['perbyte']['near output']['lags'][l][rd] for o in rs]
            d = [x['digits_above_half'] for x in v]
            s.append(f"lag {l}: end {sum(x['end'] > 0.5 for x in v)}/{len(v)} eq {sum(x['eq'] > 0.5 for x in v)}/{len(v)} digits {np.median(d):.0f} (min {min(d)})")
        print(f"   {k:12s} {rd:8s} " + ' | '.join(s))
