"""Frozen delayed write selection evaluation."""
from concurrent.futures import ThreadPoolExecutor
from itertools import product
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev
import tempfile

from scipy.stats import t
from qualify import ARMS, ROOT, base, run

VIEWS=('original','missing','misleading','before','after')
SEEDS=(1,2,3)
BATCHES=12
WORLD=2030412

def sha(data): return hashlib.sha256(data).hexdigest()

def render(folder):
    stories=base.materialize(WORLD,folder,BATCHES)
    canonical=(folder/'original.txt').read_text()
    (folder/'missing.txt').write_text(canonical.replace(' to the ',' toward that '))
    (folder/'misleading.txt').write_text(canonical.replace(' the ',' the quiet '))
    for view in ('before','after'):
        rows=[]
        for index,line in enumerate(canonical.splitlines()):
            if '\t' not in line:
                prefix,location=line.rsplit(' ',1)
                places=[p for p in base.PLACES if p!=location[:-1]]
                distractor=places[index%len(places)]
                line=(prefix+' '+distractor+' and then to the '+location if view=='before'
                      else line[:-1]+', not the '+distractor+'.')
            rows.append(line)
        (folder/f'{view}.txt').write_text('\n'.join(rows)+'\n')
    return stories

def compare(a,b):
    delta=[mean(x['batch'][i] for x in a)-mean(x['batch'][i] for x in b) for i in range(BATCHES)]
    m=mean(delta)
    h=float(t.ppf(.975,BATCHES-1))*stdev(delta)/(BATCHES**.5)
    return [m,m-h,m+h]

if __name__=='__main__':
    dev=json.loads((ROOT/'ENGINEERING.json').read_text())
    assert dev['runner_sha256']==sha((ROOT/'qualify.py').read_bytes())
    with tempfile.TemporaryDirectory() as tmp:
        folder=Path(tmp)
        stories=render(folder)
        targets=[(row[1]+'\n').encode().hex() for story in stories for i,row in enumerate(story,1) if i%3==0]
        assert len(targets)==1200
        plan=dict(world=WORLD,seeds=SEEDS,batches=BATCHES,arms=ARMS,views=VIEWS,
                  sources={str(p.name):sha(p.read_bytes()) for p in (ROOT/'qualify.py',Path(__file__),base.SOURCE,base.TRAIN)},
                  stories={v:sha((folder/f'{v}.txt').read_bytes()) for v in VIEWS},
                  gates=dict(canonical=.95,canonical_drop=.03,recall=.90,accuracy=.85,gain=.10,distractor=.05))
        (ROOT/'OFFICIAL_PLAN.json').write_text(json.dumps(plan,indent=2)+'\n')
        (ROOT/'OFFICIAL_STORIES.json').write_text(json.dumps(stories)+'\n')
        rows=[]
        with ThreadPoolExecutor(max_workers=4) as pool:
            for r in pool.map(run,[(a,'original',folder,folder/f'{a}-{seed}.npz',None,seed,BATCHES)
                                  for a,seed in product(ARMS,SEEDS)]):
                rows.append(r)
            for r in pool.map(run,[(a,v,folder,None,folder/f'{a}-{seed}.npz',seed,BATCHES)
                                  for a,v,seed in product(ARMS,VIEWS[1:],SEEDS)]):
                rows.append(r)
                print(len(rows),r['arm'],r['view'],r['seed'],r['exact_answer_with_own_newline'],flush=True)
    assert len(rows)==75
    for r in rows:
        assert r['frozen']==next(x['frozen'] for x in rows if x['view']=='original' and x['arm']==r['arm'] and x['seed']==r['seed'])
        hit=[int(x==y) for x,y in zip(r['replies'],targets)]
        assert len(hit)==len(r['replies'])==1200
        r['batch']=[mean(hit[i:i+100]) for i in range(0,1200,100)]
    groups={(a,v):[r for r in rows if (r['arm'],r['view'])==(a,v)] for a,v in product(ARMS,VIEWS)}
    summary={}
    for v in VIEWS:
        summary[v]={}
        for a in ARMS:
            rr=groups[a,v]
            assert len(rr)==3
            diag={k:mean(r['gate_audit'][k] for r in rr) for k in rr[0]['gate_audit']}
            summary[v][a]=dict(accuracy=mean(mean(r['batch']) for r in rr),recall=diag['hit']/2400,
                               distractor_trigger_rate=diag['distractor_hits']/2400 if v in ('before','after') else None,
                               eos=mean(mean(bytes.fromhex(b).endswith(b'\n') for b in r['replies']) for r in rr),audit=diag)
    contrasts={v:{a:compare(groups['arrived',v],groups[a,v]) for a in ('immediate','left')} for v in VIEWS}
    checks=dict(canonical=summary['original']['arrived']['accuracy']>=.95 and contrasts['original']['immediate'][0]>=-.03)
    for v in ('missing','misleading'):
        row=summary[v]['arrived']
        checks[v]=row['recall']>=.90 and row['accuracy']>=.85 and all(
            d[0]>=.10 and d[1]>0 for d in contrasts[v].values())
    for v in ('before','after'):
        row=summary[v]['arrived']
        checks[v]=row['recall']>=.90 and row['distractor_trigger_rate']<=.05
    result=dict(gate=all(checks.values()),checks=checks,plan=plan,summary=summary,contrasts=contrasts,rows=rows)
    tmp=ROOT/'OFFICIAL_RESULTS.tmp'
    tmp.write_text(json.dumps(result,indent=2)+'\n')
    tmp.replace(ROOT/'OFFICIAL_RESULTS.json')
