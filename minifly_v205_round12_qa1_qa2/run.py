"""V205: replay the Round 12 final roster at one new Python hash seed."""
from concurrent.futures import ThreadPoolExecutor
from itertools import product
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'
Q1=ROOT.parent/'minifly-v202'/'data'
Q2=DATA
SEED=4
def file(task,split):
    return (Q1 if task==1 else Q2)/f'qa{task}_{"single-supporting-fact" if task==1 else "two-supporting-facts"}_{split}.txt'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def call(name,*args,cwd=ROOT):
    subprocess.run([sys.executable,'-B',str(ROOT/name),*map(str,args)],cwd=cwd,check=True)
def run(job):
    task,arm,view,options=job
    env=dict(os.environ,PYTHONHASHSEED=str(SEED),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    env.update(options)
    cmd=[sys.executable,'-B',str(ROOT/('babi21.py' if task==1 else 'qa2.py'))]
    row=json.loads(subprocess.check_output(cmd,env=env,text=True,cwd=ROOT))
    row.update(task=task,arm=arm,view=view,hash_seed=SEED)
    print(task,arm,view,row.get('acc',row.get('exact_answer_with_own_newline')),flush=True)
    return row
if __name__=='__main__':
    expected=('c64ba9b1a49bbb111ffc4cd2ae4f129d25c012399dd7ade9679eea464b9afb05',
              '55acf66cef2f6d798e2aa1d056e1ec8f24e910ea9215b639450574321132959b',
              'a4978af5bc0dc30191810e7c9949aaa6eae939ed0f6beb9fcb5f4330c2a27860',
              'eff93fb9cd81ba02eb765772cf7cc269769ed44ec71cbbbc19cfb16ee5eba0f6')
    inputs=[file(a,b) for a,b in product((1,2),('train','test'))]
    assert tuple(map(sha,inputs))==expected
    fresh=DATA/'codexfresh'
    fresh.mkdir(exist_ok=True)
    call('gen_codex_fresh.py',cwd=fresh)
    plan_path=ROOT.parent/'minifly-qa-generalization'/'OFFICIAL_PLAN.json'
    assert {v:sha(fresh/f'{v}.txt') for v in ('original','verb','prefix')}==json.loads(plan_path.read_text())['stories']
    q1views={'orig':file(1,'test')}
    for view in ('located','hasgone','madeit','reversed','newnames','newlocs','newnames_newlocs','qfound'):
        p=DATA/f'qa1_test_HELDOUT_{view}.txt'
        call('mkpara.py',file(1,'test'),view.replace('_','+'),p,0)
        q1views[view]=p
    for view in ('original','verb','prefix'):
        for prefix in ('codex','nogold'):
            q1views[f'{prefix}_{view}']=fresh/f'{view}.txt'
    q2views={'orig':file(2,'test')}
    for view in ('newnames','newobjs','newnames_newobjs'):
        p=DATA/f'qa2_test_HELDOUT_{view}.txt'
        call('mkpara2.py',file(2,'test'),view.replace('_','+'),p)
        q2views[view]=p
    jobs=[]
    q1common=dict(GORD='3,4,5',QNOSTORE='1',STEPN='2',MATCHN='3',GTH='.15',UNIT='1',
                  SEG='1',WIN='32',HABT='0',FREEZE='1',CREDIT='match',BIN='1',VALID='0',TRAIN=str(file(1,'train')))
    models=dict(R11=dict(STORE='gated'),F=dict(STORE='gword+vocabsent',ENDW='1',QIDF='1'),
                FH=dict(STORE='gword+vocabsent',ENDW='1',QIDF='1',QIDFADAPT='1',QIDFDECAY='.95'))
    for (arm,config),(view,path) in product(models.items(),q1views.items()):
        jobs.append((1,arm,view,dict(q1common,**config,TEST=str(path),NOGOLD=str(int(view.startswith('nogold_'))))))
    common=dict(VALID='0',FREEZE='1',MASKV='1',TRAIN=str(file(2,'train')))
    for view,path in q2views.items():jobs.append((2,'ttb',view,dict(common,ARB='ttb',TEST=str(path))))
    for arm,config in (('fixed',dict(ARB='0')),('direct',dict(MODE='direct')),('logit',dict(ARB='1'))):
        jobs.append((2,arm,'orig',dict(common,**config,TEST=str(file(2,'test')))))
    for a,b in ((1,2),(2,1)):
        jobs.append((2,f'seq{a}{b}','orig',dict(common,ARB='ttb',FULLSEQ='1',TEST=str(file(2,'test')),
                     TRAINSEQ=f'{file(a,"train")},{file(b,"train")}',EVALS=f'{file(1,"test")},{file(2,"test")}')))
    plan=dict(version='V205',hash_seed=SEED,source={n:sha(ROOT/n) for n in ('babi21.py','qa2.py','mkpara.py','mkpara2.py','run.py')},
              data={p.name:sha(p) for p in set(inputs+list(q1views.values())+list(q2views.values()))},
              jobs=jobs)
    (ROOT/'PLAN.json').write_text(json.dumps(plan,indent=2)+'\n')
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(run,jobs))
    assert len(rows)==54
    (ROOT/'RESULTS.json').write_text(json.dumps(dict(version='V205',rows=rows),indent=2)+'\n')
