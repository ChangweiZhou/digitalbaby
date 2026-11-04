import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import mean, stdev
from scipy.stats import t
from qualify import ARMS, execute

ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
    source=hashlib.sha256((ROOT/'qualify.py').read_bytes()).hexdigest()
    engineering=json.loads((ROOT/'ENGINEERING.json').read_text())
    assert engineering['status']=='ENGINEERING_PASS' and engineering['source_sha256']==source
    suites={'clean':list(range(20261103801,20261103813)),
            'continued_noise':list(range(20261103901,20261103913))}
    plan=dict(version='V193',suites=suites,arms=ARMS,source=source,
              reducer=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              checkpoint=128,changed_floor=.85,stable_floor=.90,paired_margin=.10)
    (ROOT/'OFFICIAL_PLAN.json').write_text(json.dumps(plan,indent=2)+'\n')
    summary={}
    rows=[]
    for name,seeds in suites.items():
        jobs=[(seed,arm,name=='continued_noise') for seed in seeds for arm in ARMS]
        with ProcessPoolExecutor(max_workers=4) as pool:
            results=list(pool.map(execute,jobs))
        assert len(results)==48
        for i in range(0,48,4):
            group=results[i:i+4]
            assert all(all(row[key]==group[0][key] for key in ('fork','queries','stream')) for row in group)
        by={arm:[row for row in results if row['arm']==arm] for arm in ARMS}
        curves={a:{k:[mean(r['scores'][k][i] for r in rr) for i in (0,1)]
                   for k in rr[0]['scores']} for a,rr in by.items()}
        d=[a['scores']['change_128'][1]-b['scores']['change_128'][1]
           for a,b in zip(by['operator'],by['fully_frozen'])]
        m=mean(d)
        h=float(t.ppf(.975,11))*stdev(d)/12**.5
        ref=curves['joint']['change_128']
        op=curves['operator']['change_128']
        valid=ref[0]>=plan['stable_floor'] and ref[1]>=plan['changed_floor']
        gate=valid and op[0]>=plan['stable_floor'] and op[1]>=plan['changed_floor'] and m>=plan['paired_margin'] and m-h>0
        summary[name]=dict(reference_valid=valid,numerical_gate=gate,curves=curves,
                           operator_minus_frozen=[m,m-h,m+h],operator_minus_joint_changed=op[1]-ref[1])
        rows.extend(results)
        print(name,json.dumps(summary[name]),flush=True)
    tmp=ROOT/'OFFICIAL_RESULTS.tmp'
    tmp.write_text(json.dumps(dict(plan=plan,primary_gate=summary['clean']['numerical_gate'],summary=summary,rows=rows),indent=2)+'\n')
    tmp.replace(ROOT/'OFFICIAL_RESULTS.json')
