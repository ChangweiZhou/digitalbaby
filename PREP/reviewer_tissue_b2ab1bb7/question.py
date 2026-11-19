# The reviewer's question of 19 November (07:20 UTC), checked per seed for a label prefix:
#   (i)   the tool as it is (a) reads the present byte at d = 1, 2, 4 above 0.1103;
#   (ii)  (a) S(d, 0) > 0.1103 at every d = 8 to 11 that has cells, with a largest output above 1/2 there;
#   (iii) (a) the byte one back above C<L(1) + 0.02 somewhere, and (iv) the byte two back above C<L(2) + 0.02 somewhere (any d or the pool);
#   (v)   two copies do not diverge: the largest output difference after 512 bytes below 1e-6.
# usage: python3 question.py <prefix> [<prefix> ...]
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); SEEDS = (73129, 93683, 116089); LEVEL = 0.1103
for pre in sys.argv[1:]:
    res = []
    for s in SEEDS:
        p = os.path.join(HERE, 'where_tau', f'{pre}_{s}.json')
        if not os.path.exists(p):
            res.append(None); continue
        r = json.load(open(p)); nb = int(r['args'].get('nbytes', 16384))
        clt = json.load(open(os.path.join(HERE, 'where', 'CLT.json' if nb == 16384 else f'CLT_{nb}.json')))[str(s)]
        rows = {x['d']: x for x in r['rows']}
        a = lambda d, L: (rows[d]['a'][L] or {}).get('S') if d in rows else None
        i_ = all((a(d, '0') or 0) > LEVEL for d in ('1', '2', '4'))
        far = [d for d in ('8', '9', '10', '11') if d in rows]
        ii = bool(far) and all((a(d, '0') or 0) > LEVEL and rows[d]['max_output'] > 0.5 for d in far)
        where1 = [d for d in rows if d != 'unreached' and (a(d, '1') or 0) > clt['1'] + 0.02]
        where2 = [d for d in rows if d != 'unreached' and (a(d, '2') or 0) > clt['2'] + 0.02]
        v = r['divergence']['512'] < 1e-6
        res.append(dict(seed=s, i=i_, ii=ii, iii=bool(where1), iv=bool(where2), v=v, S0=[round(a(d, '0') or 0, 3) for d in ('1', '2', '4')],
                        far=[(d, round(a(d, '0') or 0, 3), round(rows[d]['max_output'], 2)) for d in far], lag1=sorted(where1, key=lambda q: (q == 'pool', int(q) if q.isdigit() else 0)),
                        lag2=sorted(where2, key=lambda q: (q == 'pool', int(q) if q.isdigit() else 0)), div512=r['divergence']['512'], conns=round(r['conns_per_cell'], 1)))
    ok = all(x and x['i'] and x['ii'] and x['iii'] and x['iv'] and x['v'] for x in res)
    print(f"== {pre}: {'MEETS' if ok else 'does not meet'} the question on all three seeds")
    for x in res:
        if x is None:
            print("   (missing)"); continue
        print(f"   seed {x['seed']}: (i) {x['i']} {x['S0']} | (ii) {x['ii']} {x['far']} | (iii) {x['iii']} lag1 at {x['lag1']} | (iv) {x['iv']} lag2 at {x['lag2']} | (v) {x['v']} {x['div512']:.1e} | conns {x['conns']}")
