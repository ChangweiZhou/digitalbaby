# S(group, lag) for lags 0 to 3 by the tool in the groups near18 and nearL, per window, from ntd/<label>.json: python3 -I nearL.py <label>...
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__))
for lab in sys.argv[1:]:
    r = json.load(open(os.path.join(HERE, 'ntd', lab + '.json')))
    for E, w in r['windows'].items():
        rows = {x['d']: x for x in w['rows']}; c0 = w['c0']
        out = []
        for g in ('near18', 'nearL'):
            x = rows[g]
            out.append(f"{g} n{x['cells']} max {x['max_output']:.2f} S0-3 " + "/".join(f"{x['S_tool'][L]:.3f}" for L in '0123') + " nofloor " + "/".join(f"{x['S_nofloor'][L]:.3f}" for L in '0123'))
        print(f"{lab} {E}: " + " | ".join(out) + " | levels " + "/".join(f"{(0.1103 if L == '0' else c0[L] + 0.02):.3f}" for L in '0123'))
