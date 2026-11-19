# nine.py for the files of where_win.py: per window, the nine entries (lags 0, 1, 2 at d = 1, 2, 4, the tool as it is) against level A
# (lag 0 above 0.1103; lag L above C0(L) + 0.02) and level B (lag L above C<L(L) + 0.02), with the window's own C0 and C<L; the far cells
# (d = 8 to 11) and the cells nearest the output. usage: python3 nine_win.py <label> [...]
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
for lab in sys.argv[1:]:
    r = json.load(open(os.path.join(HERE, 'where_tau', f'{lab}.json')))
    print(f"== {lab}: divergence after 512 bytes {r['divergence']['512']:.1e}; tau draw: {r['tau_draw']}")
    for ws, w in sorted(r['windows'].items(), key=lambda kv: int(kv[0])):
        rows = {x['d']: x for x in w['rows']}
        a = lambda d, L: (rows[d]['a'][L] or {}).get('S') if d in rows else None
        lvA = {'0': 0.1103, '1': w['c0']['1'] + 0.02, '2': w['c0']['2'] + 0.02}
        lvB = {'0': 0.1103, '1': w['c_lt']['1'] + 0.02, '2': w['c_lt']['2'] + 0.02}
        ent = [(d, L, a(d, L)) for L in '012' for d in ('1', '2', '4')]
        okA = sum((v or 0) > lvA[L] for d, L, v in ent); okB = sum((v or 0) > lvB[L] for d, L, v in ent)
        fails = [f"S({d},{L})={(v or 0):.3f}" for d, L, v in ent if (v or 0) <= lvB[L]]
        vals = ' / '.join(' '.join(f"{(a(d, L) or 0):.3f}" for d in ('1', '2', '4')) for L in '012')
        far = ' '.join(f"d{d} {a(d,'0'):.3f}/{a(d,'1'):.3f}/{a(d,'2'):.3f} mx{rows[d]['max_output']:.2f} ans{rows[d]['cells_with_output']}/{rows[d]['cells']}" for d in ('8', '9', '10', '11') if d in rows)
        ds = sorted(int(k) for k in rows if k.isdigit()); o = str(max(d for d in ds if rows[str(d)]['cells'] >= 8))
        print(f"   bytes {w['window'][0]}-{w['window'][1]}: {vals} | level A {okA}/9, level B {okB}/9" + (f" (below B: {', '.join(fails)})" if fails else "")
              + f" | conns {w['conns_per_cell']:.2f} max {w['conns_max']} made {w['made']} withdrawn {w['given']}")
        print(f"      far: {far} | nearest the output d={o}: {a(o,'0'):.3f}/{a(o,'1'):.3f}/{a(o,'2'):.3f} ans{rows[o]['cells_with_output']}/{rows[o]['cells']}")
