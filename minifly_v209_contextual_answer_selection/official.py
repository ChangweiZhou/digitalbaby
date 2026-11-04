"""V209 frozen contextual answer selection evaluation."""
from concurrent.futures import ProcessPoolExecutor
from itertools import product
import json
from pathlib import Path
from statistics import mean,stdev
from scipy.stats import t
import qualify as q

ROOT=q.ROOT
HASH_SEEDS=(20931,20932,20933)
WORLDS=tuple(range(209101,209113))
MODES=(('identity','true',False,False),('identity','random',False,False),
       ('identity','none',False,False),('context','true',False,False),
       ('context','random',False,False),('context','none',False,False),
       ('context','true',True,False),('context','true',False,True))
KEYS=('identity_true','identity_random','identity_none','context_true','context_random',
      'context_none','context_permuted','context_tie_shuffle')

def worker(task):
    seed,mode_idx=task
    q.HASH_SEED=seed
    selector,feedback,permute,tie_shuffle=MODES[mode_idx]
    rows=[]
    model=q.train(selector,feedback)
    original_store=model.store.copy()
    for world,view in product(WORLDS,q.VIEWS):
        model.store=original_store.copy()
        result=q.evaluate(model,view,permute,world,tie_shuffle)
        rows.append(dict(seed=seed,world=world,view=view,arm=KEYS[mode_idx],**result))
    return rows

def ci(values):
    m=mean(values)
    h=float(t.ppf(.975,len(values)-1))*stdev(values)/len(values)**.5
    return [m,m-h,m+h]

if __name__=='__main__':
    dev=json.loads((ROOT/'ENGINEERING.json').read_text())
    assert dev['source_sha256']==q.sha((ROOT/'qualify.py').read_bytes())
    plan=dict(version='V209',hash_seeds=HASH_SEEDS,worlds=WORLDS,arms=KEYS,views=q.VIEWS,
              source={p.name:q.sha(p.read_bytes()) for p in (Path(__file__),ROOT/'qualify.py')},
              gate=dict(exact=.85,paired_gain=.20,coverage=.90,familiar_loss=.03))
    (ROOT/'OFFICIAL_PLAN.json').write_text(json.dumps(plan,indent=2)+'\n')
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows=[]
        for group in pool.map(worker,product(HASH_SEEDS,range(len(MODES)))):
            rows.extend(group)
            print(group[0]['seed'],group[0]['arm'],len(rows),flush=True)
    assert len(rows)==len(HASH_SEEDS)*len(WORLDS)*len(q.VIEWS)*len(KEYS)
    index={(r['arm'],r['view'],r['world'],r['seed']):r for r in rows}
    assert len(index)==len(rows)
    def series(arm,view,metric):
        return [mean(index[arm,view,w,s][metric]/100 for s in HASH_SEEDS) for w in WORLDS]
    summary={}
    contrasts={}
    for view in q.VIEWS:
        summary[view]={a:{k:mean(series(a,view,k)) for k in ('correct','available','oracle','ties')} for a in KEYS}
        contrasts[view]={a:ci([x-y for x,y in zip(series('context_true',view,'correct'),series(a,view,'correct'))])
                         for a in ('identity_true','context_random','context_permuted','context_tie_shuffle')}
    def pass_panel(view):
        row=summary[view]['context_true']
        d=contrasts[view]['identity_true']
        return row['correct']>=.85 and row['available']>=.90 and d[0]>=.20 and d[1]>0
    guard=contrasts['familiar']['identity_true'][0]>=-.03
    output=dict(primary_gate=guard and pass_panel('joint'),template_gate=guard and pass_panel('new_template'),
                familiar_guard=guard,summary=summary,contrasts=contrasts,rows=rows)
    tmp=ROOT/'OFFICIAL_RESULTS.tmp'
    tmp.write_text(json.dumps(output,indent=2)+'\n')
    tmp.replace(ROOT/'OFFICIAL_RESULTS.json')
    print('gates',output['primary_gate'],output['template_gate'],flush=True)
