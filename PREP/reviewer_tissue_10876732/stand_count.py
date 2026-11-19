# stand_count.py: over the 15 seeds of stand/S_*.json, per draw: for each of the nine entries and for lags 0, 1, 2 near the output lines,
# on how many seeds it is above level A and level B, and the smallest value (with its seed); and a check of the reviewer's draw against
# the reviewer's own far.py results in far_rev/ (largest absolute difference over the nine and near lags 0-4).
import os, json, glob
HERE = os.path.dirname(os.path.abspath(__file__))
R = [json.load(open(p)) for p in glob.glob(os.path.join(HERE, 'stand', 'S_*.json'))]
w = lambda o: list(o['windows'].values())[0]
for draw in ('rev', 'own'):
    rs = [o for o in R if (o['args'].get('draw') == 'own') == (draw == 'own')]
    print(f"== {'reviewer' if draw == 'rev' else 'own'} draw, {len(rs)} seeds, 16,384 bytes, window 12288-16383 (level A: lag 0 > 0.1103, lag L > c0(L)+0.02; B: lag L > C<L(L)+0.02)")
    items = [(f'S({d},{L})', lambda o, d=d, L=L: w(o)['nine'][f'{d},{L}'], str(L)) for L in (0, 1, 2) for d in (1, 2, 4)] + \
            [(f'near lag {L}', lambda o, L=L: w(o)['near_tool'][str(L)], str(L)) for L in (0, 1, 2)]
    for name, get, L in items:
        vals = [(get(o), o) for o in rs]
        nA = sum(v > w(o)['level_A'][L] for v, o in vals); nB = sum(v > w(o)['level_B'][L] for v, o in vals)
        v, o = min(vals, key=lambda t: t[0])
        print(f"   {name:11s} above A {nA:2d}/{len(rs)}  above B {nB:2d}/{len(rs)}  smallest {v:.3f} (seed {o['seed']}; levels A {w(o)['level_A'][L]:.3f} B {w(o)['level_B'][L]:.3f})")
print("== check of the reviewer's draw against far_rev/ (the reviewer's far.py)")
for o in sorted([o for o in R if o['args'].get('draw') != 'own'], key=lambda o: o['seed']):
    p = os.path.join(HERE, 'far_rev', f"L8_256_s{o['seed']}.json")
    if not os.path.exists(p):
        print(f"   {o['seed']}: no far.py result"); continue
    r = json.load(open(p))
    dn = max(abs(w(o)['nine'][f'{d},{L}'] - r['tool'][str(d)][str(L)]) for d in (1, 2, 4) for L in (0, 1, 2))
    dr = max(abs(w(o)['near_tool'][str(L)] - r['near_tool'][str(L)]) for L in range(5))
    print(f"   {o['seed']}: largest difference, nine {dn:.1e}, near lags 0-4 {dr:.1e}")
