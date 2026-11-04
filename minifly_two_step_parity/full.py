from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev

from qualify import ARMS, world
from scipy.stats import t

ROOT = Path(__file__).resolve().parent
if __name__ == '__main__':
    suites = {'distance': list(range(20261103101,20261103113)),
              'item': list(range(20261103201,20261103213))}
    jobs = [(seed,suite,arm) for suite,seeds in suites.items() for seed in seeds for arm in ARMS]
    plan = dict(suites=suites, arms=ARMS, threshold=.85, advantage=.10,
                source_sha256=hashlib.sha256((ROOT/'qualify.py').read_bytes()).hexdigest())
    (ROOT/'FULL_PLAN.json').write_text(json.dumps(plan,indent=2)+'\n')
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(world, *zip(*jobs)))
    summary = {}
    for suite in suites:
        scores = {arm: [r['joint']/r['test'] for r in rows if r['suite']==suite and r['arm']==arm]
                  for arm in ARMS}
        contrasts = {}
        for arm in ('timescale','shuffled'):
            d = [a-b for a,b in zip(scores['aligned'],scores[arm])]
            m = mean(d)
            h = float(t.ppf(.975,len(d)-1))*stdev(d)/len(d)**.5
            contrasts[arm] = [m,m-h,m+h]
        summary[suite] = dict(means={a:mean(v) for a,v in scores.items()}, contrasts=contrasts)
    primary=summary['distance']
    gate = primary['means']['aligned']>=.85 and all(m>=.10 and lo>0 for m,lo,hi in primary['contrasts'].values())
    output=dict(plan=plan, numerical_gate=bool(gate), protocol_status='PROTOTYPE_FULL_ROSTER', summary=summary, rows=rows)
    tmp=ROOT/'FULL_RESULTS.tmp'
    tmp.write_text(json.dumps(output,indent=2)+'\n')
    tmp.replace(ROOT/'FULL_RESULTS.json')
    print(json.dumps(summary,indent=2))
