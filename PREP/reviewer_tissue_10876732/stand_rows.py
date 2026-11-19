# stand_rows.py: one line per 16,384-byte tissue of stand/S_*.json: the nine entries above A and B (and those below), the near-output lags
# 0-4 and those below A and B, near cells' largest output and share above 1/2, connections, share from long-axon cells, made and withdrawn
# (all, last quarter), divergence after 512 bytes, loops among excitatory cells (count/cells/largest).
import os, json, glob
HERE = os.path.dirname(os.path.abspath(__file__))
for p in sorted(glob.glob(os.path.join(HERE, 'stand', 'S_*.json')), key=lambda p: (json.load(open(p))['args'].get('draw', 'rev'), int(json.load(open(p))['seed']))):
    o = json.load(open(p)); w = list(o['windows'].values())[0]; e = o['loops_end']['excitatory']
    nt = w['near_tool']
    print(f"{'own' if o['args'].get('draw') == 'own' else 'rev'} {o['seed']:>6}: nine A {9 - len(w['nine_below_A'])}/9 B {9 - len(w['nine_below_B'])}/9"
          + (f" (below: {';'.join(k + '=' + format(w['nine'][k], '.3f') for k in sorted(set(w['nine_below_A']) | set(w['nine_below_B'])))})" if w['nine_below_B'] or w['nine_below_A'] else '')
          + f" | near " + ' '.join(f"{nt[str(L)]:.3f}" for L in range(5)) + f" below A {','.join(w['near_below_A']) or '-'} B {','.join(w['near_below_B']) or '-'}"
          + f" | max {w['near_max_output']:.2f} >1/2 {w['near_share_above_half']:.2f} | conns {w['conns_per_cell']:.2f} (max {w['conns_max']}) long {w['share_from_long']:.3f} | made {w['made']} wd {w['given']} lastq {o['made_last_quarter']}/{o['given_last_quarter']}"
          + f" | div {o['divergence']['512']:.1e} | exc loops {e['loops']}/{e['cells_in_loops']}/{e['largest']}")
