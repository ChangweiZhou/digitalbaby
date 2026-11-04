"""V203 frozen QA1 generalization on fresh stories. QA2 is a separate stage."""
from concurrent.futures import ThreadPoolExecutor
import itertools
import json
from pathlib import Path
from statistics import mean, stdev
import tempfile

from scipy.stats import t

from qualify import ROOT, SOURCE, TRAIN, VIEWS, evaluate, materialize, sha

SEEDS=(1,2,3)
WORLD_SEED=2030411
BATCHES=12

def score(row,stories):
    replies=[bytes.fromhex(b) for b in row['replies']]
    hits=[]; first=[]; eos=[]; answer_word=[]; stale=[]
    i=0
    for story in stories:
        seen={}
        for j,(person,place,_) in enumerate(story,1):
            if j%3:
                seen.setdefault(person,[]).append(place)
                continue
            expected=place.encode()
            answer=replies[i]
            i+=1
            hits.append(int(answer==expected+b'\n'))
            first.append(int(answer[:1]==expected[:1]))
            eos.append(int(answer.endswith(b'\n')))
            word=answer.rstrip(b'\n')
            answer_word.append(int(word==expected))
            stale.append(int(word!=expected and word in [p.encode() for p in seen[person][:-1]]))
    assert i==len(replies)==1200
    row['metrics']=dict(exact=mean(hits),first=mean(first),eos=mean(eos),
                        answer_word=mean(answer_word),stale=mean(stale))
    row['batch_exact']=[mean(hits[j:j+100]) for j in range(0,len(hits),100)]
    assert abs(row['metrics']['exact']-row['exact_answer_with_own_newline'])<=.00050001
    return row

def paired(a,b):
    # Average within each story batch across the three reused hash seeds.
    d=[mean(x['batch_exact'][i] for x in a)-mean(x['batch_exact'][i] for x in b) for i in range(BATCHES)]
    m=mean(d)
    h=float(t.ppf(.975,BATCHES-1))*stdev(d)/(BATCHES**.5)
    return [m,m-h,m+h]

if __name__=='__main__':
    assert sha(TRAIN.read_bytes())=='c64ba9b1a49bbb111ffc4cd2ae4f129d25c012399dd7ade9679eea464b9afb05'
    dev=json.loads((ROOT/'ENGINEERING.json').read_text())
    assert dev['runner_sha256']==sha((ROOT/'qualify.py').read_bytes())
    plan=dict(version='QA1 frozen generalization',seeds=SEEDS,batches=BATCHES,story_seed=WORLD_SEED,
              design={'views':VIEWS,'gold':[True,False],'controls':['learned','oracle','random']},
              original_floor=.95,unseen_floor=.90,max_drop=.05,
              sources={p.name:sha(p.read_bytes()) for p in (Path(__file__),ROOT/'qualify.py',SOURCE,TRAIN)})
    (ROOT/'OFFICIAL_PLAN.json').write_text(json.dumps(plan,indent=2)+'\n')
    with tempfile.TemporaryDirectory() as temp:
        folder=Path(temp)
        stories=materialize(WORLD_SEED,folder,BATCHES)
        plan['stories']={v:sha((folder/f'{v}.txt').read_bytes()) for v in VIEWS}
        (ROOT/'OFFICIAL_PLAN.json').write_text(json.dumps(plan,indent=2)+'\n')
        (ROOT/'OFFICIAL_STORIES.json').write_text(json.dumps(stories)+'\n')
        with ThreadPoolExecutor(max_workers=4) as pool:
            init=[('original',True,s,folder,folder/f'trained-{s}.npz',None,'learned',BATCHES) for s in SEEDS]
            rows=list(pool.map(evaluate,init))
            assert len({r['frozen'] for r in rows})==3
            jobs=[(v,g,s,folder,None,folder/f'trained-{s}.npz',c,BATCHES)
                  for v,g,s,c in itertools.product(VIEWS,(True,False),SEEDS,('learned','oracle','random'))
                  if (v,g,c)!=('original',True,'learned')]
            for row in pool.map(evaluate,jobs):
                rows.append(row)
                print(len(rows),row['view'],row['gold'],row['seed'],row['control'],
                      row['exact_answer_with_own_newline'],flush=True)
    assert len(rows)==54
    for r in rows:
        score(r,stories)
        assert r['frozen']==rows[SEEDS.index(r['seed'])]['frozen']
    groups={(v,g,c):[r for r in rows if (r['view'],r['gold'],r['control'])==(v,g,c)]
            for v,g,c in itertools.product(VIEWS,(True,False),('learned','oracle','random'))}
    summary=[dict(view=v,gold=g,control=c,metrics={k:mean(r['metrics'][k] for r in group) for k in group[0]['metrics']},
                 gate_audit={k:mean(r['gate_audit'][k] for r in group) for k in group[0]['gate_audit']})
             for (v,g,c),group in groups.items()]
    baseline=groups['original',True,'learned']
    contrasts={}
    passes={}
    for view in VIEWS:
        test=groups[view,False,'learned']
        contrasts[view]=dict(no_gold_vs_gold=paired(test,groups[view,True,'learned']),
                             no_gold_vs_original_gold=paired(test,baseline))
        passes[view]=mean(r['metrics']['exact'] for r in test)>=.90 and contrasts[view]['no_gold_vs_original_gold'][0]>=-.05
    original_valid=mean(r['metrics']['exact'] for r in baseline)>=.95
    result=dict(gate=original_valid and all(passes.values()),original_valid=original_valid,passes=passes,
                plan=plan,summary=summary,paired_intervals=contrasts,rows=rows)
    tmp=ROOT/'OFFICIAL_RESULTS.tmp'
    tmp.write_text(json.dumps(result,indent=2)+'\n')
    tmp.replace(ROOT/'OFFICIAL_RESULTS.json')
