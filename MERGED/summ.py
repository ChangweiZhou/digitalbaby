# Compact rows from ntd/<label>.json: python3 -I summ.py <label>... [--end E]
import sys, json, os
HERE = os.path.dirname(os.path.abspath(__file__))
args = [a for a in sys.argv[1:] if not a.startswith('--')]
end = sys.argv[sys.argv.index('--end') + 1] if '--end' in sys.argv else None
for lab in args:
    r = json.load(open(os.path.join(HERE, 'ntd', lab + '.json')))
    for E, w in r['windows'].items():
        if end and E != end:
            continue
        rows = {x['d']: x for x in w['rows']}
        dv = w['divergence']
        f = lambda v: '-' if v is None else f'{v:.3f}'
        print(f"{lab} end {E}: conns {w['conns_mean']:.2f} ({w['conns_min']}-{w['conns_max']}), given up {w['given_up']}, made {w['made']}, >half {w['conns_over_half_of_exc_total']}, "
              f"div {'/'.join(str(dv[k]['cells']) for k in ('1','16','64','256','512'))}, maxd {w['max_distance']}, ms/byte {r.get('ms_per_byte', 0):.1f}")
        for d in [k for k in rows if k != 'pool']:
            x = rows[d]
            if d in ('1', '2', '4') or int(d) >= 8:
                print(f"   d{d:>2s} n{x['cells']:4d} >1/2 {x.get('cells_max_over_half', float('nan')):.2f} max {x['max_output']:.3f} mean {x['mean_output']:.1e} pres {f(x['present'])} prev {f(x['previous'])} | S0 tool {f(x['S_tool']['0'])} nofloor {f(x['S_nofloor']['0'])} | nofloor S1 {f(x['S_nofloor']['1'])} S2 {f(x['S_nofloor']['2'])}")
        print(f"   C0 lag1 {w['c0']['1']:.3f} lag2 {w['c0']['2']:.3f}")
