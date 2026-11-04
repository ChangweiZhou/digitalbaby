"""Matched V209B study with new story and hash seeds."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from statistics import mean
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parent
V209=ROOT.parent/'minifly-v209'/'qualify.py'
spec=importlib.util.spec_from_file_location('v209',V209)
q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
SEEDS=(7,8,9)
WORLDS=(209201,209202,209203)
VIEWS=('familiar','joint','new_template','distractor')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def data(seed,n,view,path):
    lines=[]
    for story in q.worlds(seed,n):
        for i,row in enumerate(story,1):
            text,target=q.render(row,view)
            lines.append(f'{i} {text}' if target is None else f'{i} {text}\t{target}\t1')
    path.write_text('\n'.join(lines)+'\n')

def claude(train,test,seed,original_split=False,shuffle_ties=False):
    code=(ROOT/'nobuilt.py').read_text().split("\ntr='/tmp/babi")[0]
    if not original_split:
        code=code.replace('((tr[:cut],False),(va[cv:],True))','((tr,False),(va,True))')
    code=code.replace('    res=[]','    res=[]; availability=[]; ties=[]; tie_rng=random.Random(209880)')
    code=code.replace('                out=words[max(sc,key=sc.get)] if sc else None', '''                order=list(sc)
                if os.environ['TIE_SHUFFLE']=='1': tie_rng.shuffle(order)
                out=words[max(order,key=sc.get)] if sc else None''')
    code=code.replace('                if test: res.append(out==a)', '''                if test:
                    res.append(out==a)
                    availability.append(any(words[c]==a for c in cand))
                    ties.append(bool(sc) and sum(v==max(sc.values()) for v in sc.values())>1)''')
    code=code.replace('return round(float(np.mean(res)),3)',"return dict(correct=sum(res),available=sum(availability),ties=int(sum(ties)),total=len(res))")
    code+="\nimport sys\nprint(json.dumps(run(*sys.argv[1:3])))\n"
    env=dict(os.environ,PYTHONHASHSEED=str(seed),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',
             FB='true',EIDF='0',BASE='1',TIE_SHUFFLE=str(int(shuffle_ties)))
    return json.loads(subprocess.check_output([sys.executable,'-B','-c',code,str(train),str(test)],env=env,text=True))

def primary(job):
    folder,seed,world,view=job
    train=folder/'training.txt';test=folder/f'{world}-{view}.txt'
    result=claude(train,test,seed)
    result['tie_randomized']=claude(train,test,seed,shuffle_ties=True)['correct']
    print('Claude',seed,world,view,result['correct'],flush=True)
    return dict(arm='claude',seed=seed,world=world,view=view,**result)

if __name__=='__main__':
    plan=dict(version='V209B',hash_seeds=SEEDS,worlds=WORLDS,training_seed=20901,views=VIEWS,
              source={p.name:sha(p) for p in (Path(__file__),ROOT/'nobuilt.py',V209)})
    with tempfile.TemporaryDirectory() as d:
        folder=Path(d)
        orig=ROOT.parent/'minifly-v202'/'data'/'qa1_single-supporting-fact_train.txt'
        held=folder/'round14-joint.txt'
        generator=ROOT.parent/'minifly-v205'/'mkpara.py'
        subprocess.run([sys.executable,'-B',str(generator),str(orig),'newnames+newlocs',str(held),'0'],check=True)
        data(20901,120,'familiar',folder/'training.txt')
        for world in WORLDS:
            for view in VIEWS:data(world,20,view,folder/f'{world}-{view}.txt')
        plan['data_sha256']={p.name:sha(p) for p in sorted(folder.glob('*.txt'))}
        (ROOT/'PLAN.json').write_text(json.dumps(plan,indent=2)+'\n')
        spot=claude(orig,held,SEEDS[0],original_split=True)
        print('Round 14 seed 7 spot check',spot,flush=True)
        with ThreadPoolExecutor(max_workers=2) as pool:
            rows=list(pool.map(primary,[(folder,s,w,v) for s,w in zip(SEEDS,WORLDS) for v in VIEWS]))
        for seed,world in zip(SEEDS,WORLDS):
            q.HASH_SEED=seed
            for selector in ('identity','context'):
                trained=q.train(selector,'true')
                store=trained.store.copy()
                for view in VIEWS:
                    trained.store=store.copy()
                    result=q.evaluate(trained,view,world_seed=world)
                    trained.store=store.copy()
                    diagnostic=q.evaluate(trained,view,world_seed=world,shuffle_ties=True)
                    rows.append(dict(arm=selector,seed=seed,world=world,view=view,
                                     tie_randomized=diagnostic['correct'],**result))
    assert len(rows)==36 and all(r['total']==100 for r in rows)
    summary={v:{a:{k:mean(r[k] for r in rows if r['view']==v and r['arm']==a)
                   for k in ('correct','available','ties','tie_randomized')}
                for a in ('claude','identity','context')} for v in VIEWS}
    (ROOT/'RESULTS.json').write_text(json.dumps(dict(version='V209B',spot=spot,summary=summary,rows=rows),indent=2)+'\n')
    print(json.dumps(summary))
