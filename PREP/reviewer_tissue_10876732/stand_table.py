# stand_table.py: the rows of stand/*.json (the helper's L8-256 runs) as text; usage: stand_table.py [label prefix ...]
import sys, os, json, glob
HERE = os.path.dirname(os.path.abspath(__file__))
f = lambda v: ' nan ' if v is None else f'{v:.3f}'
pre = sys.argv[1:] or ['']
for p in sorted(glob.glob(os.path.join(HERE, 'stand', '*.json'))):
    o = json.load(open(p)); lab = o['label']
    if not any(lab.startswith(x) for x in pre):
        continue
    dv = o['divergence']['512']; lp = o['loops_end']
    print(f"== {lab}: {'own' if o['args'].get('draw') == 'own' else 'reviewer'} draw, {o['nbytes']} bytes, {o['long_cells']} long-axon cells | divergence after 512 bytes {dv:.1e} | "
          f"loops (excitatory senders) {lp['excitatory']['loops']} with {lp['excitatory']['cells_in_loops']} cells, largest {lp['excitatory']['largest']}; (all) {lp['all']['loops']} / {lp['all']['cells_in_loops']} / {lp['all']['largest']} | "
          f"made in the last quarter {o['made_last_quarter']}, withdrawn {o['given_last_quarter']}")
    for ws, w in sorted(o['windows'].items(), key=lambda kv: int(kv[0])):
        n = w['nine']
        nine = ' / '.join(' '.join(f(n[f'{d},{L}']) for d in (1, 2, 4)) for L in (0, 1, 2))
        bA = w['nine_below_A']; bB = w['nine_below_B']
        lv = lambda k, lev: f"S({k})={n[k]:.3f}<={lev[k.split(',')[1]]:.3f}"
        print(f"   bytes {w['window'][0]}-{w['window'][1]}: nine (lag 0 / 1 / 2 at d = 1 2 4) {nine} | above A {9 - len(bA)}/9" + (f" ({'; '.join(lv(k, w['level_A']) for k in bA)})" if bA else '')
              + f", above B {9 - len(bB)}/9" + (f" ({'; '.join(lv(k, w['level_B']) for k in bB)})" if bB else ''))
        nt = w['near_tool']; nn = w['near_nofloor']
        print(f"      near the output lines ({w['near_cells']} cells): tool lags 0-4 " + ' '.join(f(nt[str(L)]) for L in range(5)) + " | no floor " + ' '.join(f(nn[str(L)]) for L in range(5))
              + f" | levels A " + ' '.join(f(w['level_A'][str(L)]) for L in range(5)) + " B " + ' '.join(f(w['level_B'][str(L)]) for L in range(5))
              + f" | below A: {','.join(w['near_below_A']) or 'none'}, below B: {','.join(w['near_below_B']) or 'none'} | largest output {w['near_max_output']:.2f}, share above 1/2 {w['near_share_above_half']:.2f}")
        print(f"      connections a cell {w['conns_per_cell']:.2f} (largest {w['conns_max']}), from long-axon cells {w['share_from_long']:.3f} | made {w['made']}, withdrawn {w['given']}")
