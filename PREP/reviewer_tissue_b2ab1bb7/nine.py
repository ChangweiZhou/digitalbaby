# The nine entries of roadmap step 1 by the tool as it is (lags 0, 1, 2 at d = 1, 2, 4), per seed, against two levels:
#   level A (the reviewer's): lag 0 above 0.1103; lag L >= 1 above C0(L) + 0.02;
#   level B (stricter): lag 0 above 0.1103; lag L >= 1 above C<L(L) + 0.02 (for lag 1 the same as level A).
# Also the far cells (d = 8 to 11) and the cells nearest the output lines (the largest distance with at least 8 cells): S(d, 0..2) by the
# tool, largest output, cells answering. usage: python3 nine.py <prefix> [...]
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); SEEDS = (73129, 93683, 116089)
for pre in sys.argv[1:]:
    print(f"== {pre}")
    for s in SEEDS:
        p = os.path.join(HERE, 'where_tau', f'{pre}_{s}.json')
        if not os.path.exists(p):
            print(f"   seed {s}: missing"); continue
        r = json.load(open(p)); nb = int(r['args'].get('nbytes', 16384))
        clt = json.load(open(os.path.join(HERE, 'where', 'CLT.json' if nb == 16384 else f'CLT_{nb}.json')))[str(s)]
        rows = {x['d']: x for x in r['rows']}
        a = lambda d, L: (rows[d]['a'][L] or {}).get('S') if d in rows else None
        lvA = {'0': 0.1103, '1': r['c0']['1'] + 0.02, '2': r['c0']['2'] + 0.02}
        lvB = {'0': 0.1103, '1': clt['1'] + 0.02, '2': clt['2'] + 0.02}
        ent = [(d, L, a(d, L)) for L in '012' for d in ('1', '2', '4')]
        okA = sum((v or 0) > lvA[L] for d, L, v in ent); okB = sum((v or 0) > lvB[L] for d, L, v in ent)
        failA = [f"S({d},{L})={v:.3f}<{lvA[L]:.3f}" for d, L, v in ent if (v or 0) <= lvA[L]]
        failB = [f"S({d},{L})={v:.3f}<{lvB[L]:.3f}" for d, L, v in ent if (v or 0) <= lvB[L]]
        ds = sorted(int(k) for k in rows if k.isdigit()); out_d = max(d for d in ds if rows[str(d)]['cells'] >= 8)
        far = " ".join(f"d{d}: {a(d,'0'):.3f}/{a(d,'1'):.3f}/{a(d,'2'):.3f} mx{rows[d]['max_output']:.2f} ans{rows[d]['cells_with_output']}/{rows[d]['cells']}" for d in ('8', '9', '10', '11') if d in rows)
        o = str(out_d)
        print(f"   seed {s}: nine entries above level A {okA}/9" + (f" ({'; '.join(failA)})" if failA else "") + f", level B {okB}/9" + (f" ({'; '.join(failB)})" if failB else "")
              + f" | div512 {r['divergence']['512']:.1e} | conns {r['conns_per_cell']:.2f} (max {r.get('conns_max')}) made {r.get('made')} withdrawn {r.get('given')}")
        print(f"      far (S lag 0/1/2, largest output, cells answering): {far}")
        print(f"      nearest the output (d = {o}): {a(o,'0'):.3f}/{a(o,'1'):.3f}/{a(o,'2'):.3f} mx{rows[o]['max_output']:.2f} ans{rows[o]['cells_with_output']}/{rows[o]['cells']}")
