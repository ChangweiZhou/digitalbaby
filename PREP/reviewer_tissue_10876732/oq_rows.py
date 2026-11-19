# oq_rows.py: per life of outq/*.json (the reviewer's outq.py): D 1/64 and Q 1/16 z1, last quarter (per cent), for 'own' and 'own traces 2,4',
# beside ref_1, ref_2, ref_3 of the same quarter; then Q/D.
import os, json, glob
HERE = os.path.dirname(os.path.abspath(__file__))
for p in sorted(glob.glob(os.path.join(HERE, 'outq', '*.json'))):
    o = json.load(open(p)); r = o['last_quarter_refs']; s = o['sets']
    def g(k, nm): return 100 * s[k]['rules'][nm]['last_quarter'] if k in s else float('nan')
    D0, Q0, D2, Q2 = g('own', 'D eta=1/64'), g('own', 'Q 1/16 z1'), g('own traces 2,4', 'D eta=1/64'), g('own traces 2,4', 'Q 1/16 z1')
    print(f"{o['label']:22s} ref_1 {100 * r['ref_1']:5.2f} ref_2 {100 * r['ref_2']:5.2f} ref_3 {100 * r['ref_3']:5.2f} | own: D {D0:5.2f} Q {Q0:5.2f} (Q/D {Q0 / D0:.2f}) | own traces 2,4: D {D2:5.2f} Q {Q2:5.2f} (Q/D {Q2 / D2:.2f})")
