# yq_rows.py: per setting (M, nbytes) over seeds and generators of res/Y*.json: share answered yes in the last quarter (per cent), its smallest /
# median / largest over the generators per seed and over all; cells speaking and one voice (last quarter); how many runs pass ref_1 / ref_2.
import os, json, glob, re
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
R = [json.load(open(p)) for p in glob.glob(os.path.join(HERE, 'res', 'Y*.json'))]
key = lambda o: re.match(r'(Y[^_]+_n\d+k)', o['label']).group(1)
for k in sorted(set(key(o) for o in R), key=lambda k: (int(k.split('_n')[1][:-1]), int(k[2:].split('_')[0]))):
    rs = [o for o in R if key(o) == k]; y = [100 * o['right'][3] for o in rs]
    print(f"== {k}: {len(rs)} runs | yes, last quarter: {min(y):.2f} / {np.median(y):.2f} / {max(y):.2f} | above ref_1 {sum(100 * o['right'][3] > 100 * o['refs']['ref_1'] for o in rs)}, above ref_2 {sum(o['right'][3] > o['refs']['ref_2'] for o in rs)}"
          f" | speaking {min(o['spoke_mean'][3] for o in rs):.2f}-{max(o['spoke_mean'][3] for o in rs):.2f}, one voice {100 * min(o['one_voice'][3] for o in rs):.1f}-{100 * max(o['one_voice'][3] for o in rs):.1f}")
    for s in sorted(set(o['args']['seed'] for o in rs), key=int):
        ss = sorted([o for o in rs if o['args']['seed'] == s], key=lambda o: o['gen'])
        print(f"   seed {s:>6} ref_1 {100 * ss[0]['refs']['ref_1']:5.2f} ref_2 {100 * ss[0]['refs']['ref_2']:5.2f} | yes by quarter, gen " +
              ' | '.join(f"{o['gen']}: " + ' '.join(f"{100 * v:5.2f}" for v in o['right']) for o in ss))
