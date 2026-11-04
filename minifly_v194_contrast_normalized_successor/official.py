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
    seeds=list(range(20261103941,20261103953))
    plan=dict(version='V194',seeds=seeds,conditions=['clean','noise'],arms=ARMS,source=source,
              reducer=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),checkpoint=512,
              changed_floor=.85,stable_floor=.90,paired_margin=.10,stable_loss=.03,clean_floor=.95,clean_loss=.02)
    (ROOT/'OFFICIAL_PLAN.json').write_text(json.dumps(plan,indent=2)+'\n')
    jobs=[(seed,arm,noisy) for noisy in (False,True) for seed in seeds for arm in ARMS]
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(execute,jobs))
    assert len(rows)==96
    for i in range(0,len(rows),4):
        group=rows[i:i+4]
        for r in group:
            assert r['final_maps']==group[0]['final_maps'] and r['stream']==group[0]['stream']
            assert r['queries']==group[0]['queries'] and r['fork'][0]==group[0]['fork'][0]
    summary={}
    for noisy in (False,True):
        by={a:[r for r in rows if r['continued_noise']==noisy and r['arm']==a] for a in ARMS}
        curves={a:{k:[mean(r['scores'][k][i] for r in rr) for i in (0,1)]
                   for k in rr[0]['scores']} for a,rr in by.items()}
        def paired(arm,panel):
            d=[a['scores']['change'][panel]-b['scores']['change'][panel] for a,b in zip(by[arm],by['raw'])]
            m=mean(d)
            h=float(t.ppf(.975,11))*stdev(d)/12**.5
            return [m,m-h,m+h]
        summary['noise' if noisy else 'clean']=dict(curves=curves,vs_raw={a:[paired(a,i) for i in (0,1)] for a in ARMS if a!='raw'},
                 graph_512=[mean(r['graph']['change'][i] for r in by['raw']) for i in (0,1)])
    clean,noise=summary['clean'],summary['noise']
    clean_valid=min(clean['curves']['raw']['change'])>=plan['clean_floor']
    headroom=1-noise['curves']['raw']['change'][1]>=plan['paired_margin']
    c=clean['curves']['normalized']['change']
    n=noise['curves']['normalized']['change']
    stable,changed=noise['vs_raw']['normalized']
    checks=dict(clean_reference=clean_valid,headroom=headroom,clean=min(c)>=plan['clean_floor']
        and min(x[0] for x in clean['vs_raw']['normalized'])>=-plan['clean_loss'],
        changed=n[1]>=plan['changed_floor'] and changed[0]>=plan['paired_margin'] and changed[1]>0,
        stable=n[0]>=plan['stable_floor'] and stable[0]>=-plan['stable_loss'])
    out=dict(plan=plan,checks=checks,gate=all(checks.values()),summary=summary,rows=rows)
    tmp=ROOT/'OFFICIAL_RESULTS.tmp'
    tmp.write_text(json.dumps(out,indent=2)+'\n')
    tmp.replace(ROOT/'OFFICIAL_RESULTS.json')
    print(json.dumps(dict(checks=checks,block_512={k:{a:v['change'] for a,v in x['curves'].items()} for k,x in summary.items()},noise_contrasts=noise['vs_raw'],graph_512=noise['graph_512']),indent=2))
