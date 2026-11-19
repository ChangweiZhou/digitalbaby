# The table of T1 to T4 from ntd/<T>_s<seed>.json: python3 -I ttable.py > T_TABLE.md
import os, json
HERE = os.path.dirname(os.path.abspath(__file__))
print("| T | seed | window end | S(d,0) tool d = 1 / 2 / 4 | lag 1 above level: distances (best S) | lag 2 above level: distances (best S) | largest output d = 8 / 11 | connections a cell (largest) | made / given up | cells differing after 64 / 512 bytes | ms a byte |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
def rng(ds):
    ds = sorted(int(d) for d in ds)
    if not ds: return "none"
    out, a, p = [], ds[0], ds[0]
    for d in ds[1:] + [None]:
        if d is not None and d == p + 1: p = d; continue
        out.append(f"{a}" if a == p else f"{a}–{p}");
        if d is not None: a = p = d
    return ",".join(out)
for T in ("T1", "T2", "T3", "T4"):
    for s in (73129, 93683, 116089):
        f = os.path.join(HERE, 'ntd', f'{T}_s{s}.json')
        if not os.path.exists(f): continue
        r = json.load(open(f))
        for E, w in r['windows'].items():
            rows = {x['d']: x for x in w['rows'] if x['d'] != 'pool'}
            S = lambda d, L: rows[d]['S_tool'][L] if d in rows and rows[d]['S_tool'][L] is not None else float('nan')
            def ab(L):
                lv = w['c0'][L] + 0.02; h = [d for d in rows if S(d, L) > lv]
                return f"{rng(h)}" + (f" ({max(S(d, L) for d in h):.3f})" if h else "")
            mo = lambda d: f"{rows[d]['max_output']:.2f}" if d in rows else "–"
            dv = w['divergence']
            print(f"| {T} | {s} | {int(E):,} | {S('1','0'):.3f} / {S('2','0'):.3f} / {S('4','0'):.3f} | {ab('1')} | {ab('2')} | {mo('8')} / {mo('11')} | {w['conns_mean']:.2f} ({w['conns_max']}) | {w['made']:,} / {w['given_up']:,} | {dv['64']['cells']} / {dv['512']['cells']} | {r['ms_per_byte']:.1f} |")
