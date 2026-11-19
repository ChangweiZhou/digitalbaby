# Table of S(group, lag 0..3) by the tool in near18 and nearL, last window, for the long-axon tissues: python3 -I nearLtable.py
import os, json
HERE = os.path.dirname(os.path.abspath(__file__))
print("| tissue | seed | near18 (cells): S lag 0 / 1 / 2 / 3 | nearL (cells): S lag 0 / 1 / 2 / 3 | levels lag 0 / 1 / 2 / 3 | whole tissue S(d,0) d = 1 / 2 / 4 | largest distance | connections a cell (largest) | made / given up | cells differing after 64 / 512 bytes | ms a byte |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for T in ("T5L", "T6L64", "T6L128", "T6L256", "T7L", "T8L"):
    for s in (73129, 93683, 116089):
        f = os.path.join(HERE, 'ntd', f'{T}_s{s}.json')
        if not os.path.exists(f): continue
        r = json.load(open(f)); w = r['windows']['65536']; rows = {x['d']: x for x in w['rows']}; c0 = w['c0']
        g = lambda k: f"({rows[k]['cells']}) " + " / ".join(f"{rows[k]['S_tool'][L]:.3f}" for L in '0123')
        lv = " / ".join(f"{(0.1103 if L == '0' else c0[L] + 0.02):.3f}" for L in '0123')
        S = lambda d: rows[d]['S_tool']['0']
        dv = w['divergence']
        print(f"| {T} | {s} | {g('near18')} | {g('nearL')} | {lv} | {S('1'):.3f} / {S('2'):.3f} / {S('4'):.3f} | {w['max_distance']} | {w['conns_mean']:.2f} ({w['conns_max']}) | {w['made']:,} / {w['given_up']:,} | {dv['64']['cells']} / {dv['512']['cells']} | {r['ms_per_byte']:.1f} |")
