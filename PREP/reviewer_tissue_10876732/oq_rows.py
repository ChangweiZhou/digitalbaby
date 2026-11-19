# oq_rows.py: per life of outq/*.json (the reviewer's outq.py): D 1/64 and Q 1/16 z1, last quarter (per cent), for 'own', 'own traces 2,4' and
# 'own traces 2,4,8' where run, beside ref_1, ref_2, ref_3 of the same quarter; then, per tissue, on how many seeds D and Q with the copies
# (traces 2,4) pass ref_1 and ref_2, and their smallest / median / largest.
import os, json, glob
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
R = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(HERE, 'outq', '*.json')))]
SETS = ('own', 'own traces 2,4', 'own traces 2,4,8')
def g(o, k, nm): return 100 * o['sets'][k]['rules'][nm]['last_quarter'] if k in o['sets'] else None
f = lambda v: '  -  ' if v is None else f'{v:5.2f}'
for o in R:
    r = o['last_quarter_refs']
    print(f"{o['label']:22s} ref_1 {100 * r['ref_1']:5.2f} ref_2 {100 * r['ref_2']:5.2f} ref_3 {100 * r['ref_3']:5.2f} | " + ' | '.join(f"{k}: D {f(g(o, k, 'D eta=1/64'))} Q {f(g(o, k, 'Q 1/16 z1'))}" for k in SETS))
print('== per tissue: seeds on which D and Q with the copies (traces 2,4) pass ref_1 / ref_2; smallest / median / largest')
for tk in sorted(set(o['label'].rsplit('_s', 1)[0] for o in R)):
    rs = [o for o in R if o['label'].rsplit('_s', 1)[0] == tk and 'own traces 2,4' in o['sets']]
    if not rs: continue
    for nm in ('D eta=1/64', 'Q 1/16 z1'):
        v = [(g(o, 'own traces 2,4', nm), 100 * o['last_quarter_refs']['ref_1'], 100 * o['last_quarter_refs']['ref_2']) for o in rs]
        x = [a for a, _, _ in v]
        print(f"   {tk:8s} {nm:11s} n={len(rs):2d}: above ref_1 {sum(a > r1 for a, r1, _ in v)}, above ref_2 {sum(a > r2 for a, _, r2 in v)} | {min(x):.2f} / {np.median(x):.2f} / {max(x):.2f}")
