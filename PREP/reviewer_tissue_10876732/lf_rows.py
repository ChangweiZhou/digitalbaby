# lf_rows.py: one block per tissue of stand/<prefix>_*.json with near2.py's groups (a) enlarged, (b) within 1/8, (c) long-axon only.
import sys, json, glob, os
HERE = os.path.dirname(os.path.abspath(__file__))
for p in sorted(glob.glob(os.path.join(HERE, 'stand', (sys.argv[1] if len(sys.argv) > 1 else 'LF') + '_*.json'))):
    o = json.load(open(p)); e = o['loops_end']['excitatory']
    for ws, w in sorted(o['windows'].items(), key=lambda kv: int(kv[0])):
        n = w['nine']; f = lambda d: ' '.join(f"{d[str(L)]:.3f}" for L in range(5))
        print(o['label'], 'bytes', w['window'], 'nine', ' / '.join(' '.join(f"{n[f'{d},{L}']:.3f}" for d in (1, 2, 4)) for L in (0, 1, 2)), '| below A', w['nine_below_A'], 'B', w['nine_below_B'])
        print('   (a)', f(w['near_tool']), 'below A', w['near_below_A'], 'B', w['near_below_B'], '| (b)', w['near_b']['cells'], f(w['near_b']['tool']), 'below A', w['near_b_below_A'], 'B', w['near_b_below_B'],
              '| (c)', w['near_c']['cells'], f(w['near_c']['tool']) if w['near_c'].get('tool') else None)
        print('   levels A', f(w['level_A']), 'B', f(w['level_B']), '| conns %.2f max %d long %.3f | drivers %d from long %d | made %d wd %d | near max %.2f >1/2 %.2f; (b) %.2f' % (
              w['conns_per_cell'], w['conns_max'], w['share_from_long'], w['driver_connections'], w['driver_connections_from_long'], w['made'], w['given'], w['near_max_output'], w['near_share_above_half'], w['near_b']['share_above_half']))
    print('   life: last quarter made %d wd %d | div %.1e | exc loops %s' % (o['made_last_quarter'], o['given_last_quarter'], o['divergence']['512'], (e['loops'], e['largest'])))
