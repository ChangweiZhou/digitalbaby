# One line a window from ntd/<label>.json: S(d,0) by the tool at d = 1/2/4; for lags 1 and 2 the distances where S(d, lag) by the tool
# exceeds C0(lag) + 0.02 (the level), with the best S there; the same with no floor; largest output at d = 8 and 11; connections a cell
# (mean, largest); made and given up; cells differing after 64 and 512 bytes.   python3 -I trow.py <label>...
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__))
for lab in sys.argv[1:]:
    r = json.load(open(os.path.join(HERE, 'ntd', lab + '.json')))
    for E, w in r['windows'].items():
        rows = {x['d']: x for x in w['rows'] if x['d'] != 'pool'}
        S = lambda d, L, k='S_tool': rows[d][k][L] if d in rows and rows[d][k][L] is not None else float('nan')
        def above(L, k):
            lv = w['c0'][L] + 0.02
            hits = [(d, S(d, L, k)) for d in rows if S(d, L, k) > lv]
            return f"lag{L}>{lv:.3f}: " + (",".join(d for d, _ in hits) + f" (best {max(v for _, v in hits):.3f})" if hits else "none")
        mo = lambda d: f"{rows[d]['max_output']:.2f}" if d in rows else '-'
        dv = w['divergence']
        print(f"{lab} {E}: S0 {S('1','0'):.3f}/{S('2','0'):.3f}/{S('4','0'):.3f} d8/d11 {S('8','0'):.3f}/{S('11','0'):.3f} | tool {above('1','S_tool')}; {above('2','S_tool')} | nofloor {above('1','S_nofloor')}; {above('2','S_nofloor')} | "
              f"max d8/d11 {mo('8')}/{mo('11')} | conns {w['conns_mean']:.2f} (max {w['conns_max']}) made {w['made']} given {w['given_up']} | div64/512 {dv['64']['cells']}/{dv['512']['cells']} | maxd {w['max_distance']} | ms/byte {r['ms_per_byte']:.1f}")
