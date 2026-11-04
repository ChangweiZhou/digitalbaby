"""Engineering-only QA1 generalization and QA2 state-machine checks."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'minifly-v203'/'babi16.py'
TRAIN=ROOT.parent/'minifly-v202'/'data'/'qa1_single-supporting-fact_train.txt'
VIEWS=('original','verb','prefix')
PEOPLE=('Mary','John','Daniel','Sandra')
PLACES=('kitchen','garden','office','hallway','bathroom','bedroom')
TEMPLATES={
    'original':('moved to the','travelled to the','went to the','journeyed to the','went back to the'),
    'verb':('strolled into the','walked into the','headed for the','wandered to the','hurried into the'),
    'prefix':('strolled into that','walked straight into','headed for this','wandered into our','hurried toward that')}

def sha(raw): return hashlib.sha256(raw).hexdigest()

def materialize(seed,folder,batches=1):
    rng=np.random.default_rng(seed)
    common=[]
    for _ in range(20*batches):
        story=[]
        latest={}
        for _ in range(5):
            for _ in range(2):
                p=str(rng.choice(PEOPLE))
                a=str(rng.choice([place for place in PLACES if place!=latest.get(p,(None,))[0]]))
                variant=int(rng.integers(5))
                story.append((p,a,variant))
                latest[p]=(a,len(story))
            p=str(rng.choice(sorted(latest)))
            a,index=latest[p]
            story.append((p,a,index))
        common.append(story)
    def render(view):
        rows=[]
        for story in common:
            for i,(person,place,x) in enumerate(story,1):
                if i%3:
                    rows.append(f'{i} {person} {TEMPLATES[view][x]} {place}.')
                else:
                    rows.append(f'{i} Where is {person}? \t{place}\t{x}')
        return '\n'.join(rows)+'\n'
    training=TRAIN.read_text()
    for view in VIEWS:
        text=render(view)
        for story in text.split('\n1 '):
            assert story not in training
        (folder/f'{view}.txt').write_text(text)
    return common

def program():
    code=SOURCE.read_text()
    code='import hashlib\n'+code
    code=code.replace('    for items,test in phases:', '''    if os.environ.get('QA_LOAD'):
        phases=phases[-1:]
    for items,test in phases:''')
    code=code.replace('        if FROZ[0]: TAGS.clear()', '''        if FROZ[0]:
            TAGS.clear()
            def learned_hash():
                h=hashlib.sha256(P.W.tobytes()+G.tobytes())
                for key,value in sorted(A.items()):
                    h.update(repr(key).encode())
                    h.update(value.tobytes())
                return h.hexdigest()
            if os.environ.get('QA_LOAD'):
                path=os.environ['QA_LOAD']
                raw=open(path,'rb').read()
                if hashlib.sha256(raw).hexdigest()!=open(path+'.sha').read():
                    raise ValueError('Checkpoint digest mismatch')
                with np.load(path,allow_pickle=False) as saved:
                    P.W[:]=saved['pred']; M[:]=saved['memory']; G[:]=saved['gate']
                    meta=json.loads(str(saved['meta']))
                    for (step,context),value in zip(meta['answer_keys'],saved['answers']):
                        A[(step,bytes.fromhex(context))]=value.copy()
                    hist[:]=bytes.fromhex(meta['history']); FOLLOW[0]=meta['follow']
            if os.environ.get('QA_SAVE'):
                path=os.environ['QA_SAVE']
                meta=dict(history=hist.hex(),follow=FOLLOW[0],answer_keys=[(n,b.hex()) for n,b in A])
                np.savez(path,pred=P.W,memory=M,gate=G,meta=json.dumps(meta),answers=np.array(list(A.values())))
                with open(path+'.sha','w') as f: f.write(hashlib.sha256(open(path,'rb').read()).hexdigest())
            frozen=learned_hash()
            initial=M.copy(); initial_hist=hist[:]; initial_follow=FOLLOW[0]
            diag=dict(hit=0,miss=0,false_write=0,non_target=0,follow=0,row_writes=0)
            replies=[]
''')
    code=code.replace("            if kind=='story': lastloc={}; continue",'''            if kind=='story':
                lastloc={}
                if FROZ[0]:
                    M[:]=initial; hist[:]=initial_hist; FOLLOW[0]=initial_follow
                continue''')
    code=code.replace("    for items,test in phases:", """    for items,test in phases:
        if test:
            lengths=[]
            for kind,x in items:
                if kind=='story': lengths.append(0)
                elif kind=='s': lengths[-1]+=len((x+' ').encode())
            story_index=-1
            story_offset=0""")
    code=code.replace("                if FROZ[0]:\n                    M[:]=initial;", """                if FROZ[0]:
                    story_index+=1
                    story_offset=0
                    rng=np.random.default_rng(9000+story_index)
                    selected=set(rng.choice(lengths[story_index],10,replace=False).tolist())
                    M[:]=initial;""")
    code=code.replace('    def feed(b,store=True,learnpred=True):','    def feed(b,store=True,learnpred=True):\n        nonlocal story_offset')
    code=code.replace('        for y in b:', '        for pos,y in enumerate(b):')
    code=code.replace("            if store and STOREMODE in",'''            if FROZ[0] and store:
                location_start=b.rfind(b' ',0,len(b)-2)+1
            if store and STOREMODE in''')
    code=code.replace("                if gw<=GTH and FOLLOW[0]>0:",'''                if FROZ[0]:
                    target=pos==location_start
                    arm=os.environ.get('QA_CONTROL','learned')
                    if arm=='oracle': gw=float(target)
                    if arm=='random': gw=float(story_offset in selected)
                    story_offset+=1
                    field=('hit' if gw>GTH else 'miss') if target else ('false_write' if gw>GTH else 'non_target')
                    diag[field]+=1
                if gw<=GTH and FOLLOW[0]>0:''')
    code=code.replace('                    FOLLOW[0]-=1;', '''                    if FROZ[0]: diag['follow']+=1
                    FOLLOW[0]-=1;''')
    code=code.replace('M[ix]+=(a*(T[None,:]-M[ix])).astype(np.float32)',
                      "M[ix]+=(a*(T[None,:]-M[ix])).astype(np.float32); diag.__setitem__('row_writes',diag['row_writes']+len(ix)) if FROZ[0] else None")
    code=code.replace("                if test:\n                    out=answer();", "                if test:\n                    out=answer(); replies.append(out.hex());")
    code=code.replace("                feed(ans,store=not QNOSTORE)",'''                if FROZ[0] and os.environ['QA_GOLD']=='0':
                    feed(b'\\n',store=False)
                else:
                    feed(ans,store=not QNOSTORE)''')
    code=code.replace('    acc=np.mean','    assert learned_hash()==frozen\n    acc=np.mean')
    code=code.replace("'n':len(res)", "'frozen':frozen,'gate_audit':diag,'replies':replies,'n':len(res)")
    return code

def evaluate(task):
    view,gold,seed,folder,save,load,*options=task
    control,batches=options or ('learned',1)
    env=dict(os.environ,PYTHONHASHSEED=str(seed),QA_GOLD=str(int(gold)),
             QA_SAVE=str(save or ''),QA_LOAD=str(load or ''),QA_CONTROL=control,FREEZE='1',BIN='1',VALID='0',
             STORE='gated',EPOCHS='1',GORD='3,4,5',GTH='.15',QNOSTORE='1',STEPN='2',
             MATCHN='3',UNIT='1',SEG='1',WIN='32',HABT='0',CREDIT='match',RATE='1',
             OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    cmd=[sys.executable,'-B','-c',program(),str(TRAIN),str(folder/f'{view}.txt')]
    row=json.loads(subprocess.check_output(cmd,env=env,text=True,stderr=subprocess.PIPE))
    assert row['n']==100*batches and sum(row['gate_audit'][k] for k in ('hit','miss'))==200*batches
    if control=='random':
        assert sum(row['gate_audit'][k] for k in ('hit','false_write'))==200*batches
    return dict(view=view,gold=gold,seed=seed,control=control,**row)

def qa2_fixture():
    place={}; carried={}; dropped={}
    def move(p,l): place[p]=l
    def pickup(p,o): carried[o]=p; dropped.pop(o,None)
    def drop(p,o):
        assert carried.pop(o)==p
        dropped[o]=place[p]
    def answer(o):
        if o in carried:
            return place[carried[o]]
        return dropped[o]
    move('Mary','garden'); pickup('Mary','apple'); move('Mary','office')
    assert answer('apple')=='office'
    drop('Mary','apple'); move('Mary','kitchen')
    assert answer('apple')=='office'
    move('John','office'); pickup('John','apple'); move('John','hallway')
    assert answer('apple')=='hallway'
    def consume_intermediate(raw):
        if not raw.endswith(b'\n') or raw[:-1].decode() not in place:
            return None
        return place[raw[:-1].decode()]
    assert consume_intermediate(b'John\n')=='hallway'
    assert consume_intermediate(b'John') is None
    assert consume_intermediate(b'Unknown\n') is None

if __name__=='__main__':
    assert sha(TRAIN.read_bytes())=='c64ba9b1a49bbb111ffc4cd2ae4f129d25c012399dd7ade9679eea464b9afb05'
    qa2_fixture()
    with tempfile.TemporaryDirectory() as tmp:
        folder=Path(tmp)
        materialize(301,folder)
        jobs=[(v,g,1,folder,None,None) for v in VIEWS for g in (True,False)]
        with ThreadPoolExecutor(max_workers=3) as pool:
            rows=list(pool.map(evaluate,jobs))
        assert len({r['frozen'] for r in rows})==1
        checkpoint=folder/'trained.npz'
        saved=evaluate(('original',False,1,folder,checkpoint,None))
        restored=evaluate(('original',False,1,folder,None,checkpoint))
        assert saved==restored==rows[1]
        with checkpoint.open('ab') as f:
            f.write(b'corrupt')
        try:
            evaluate(('original',False,1,folder,None,checkpoint))
        except subprocess.CalledProcessError as exc:
            assert 'Checkpoint digest mismatch' in exc.stderr
        else:
            raise AssertionError('Accepted a corrupted checkpoint')
    report=dict(status='QA1_DEVELOPMENT_PASS_QA2_REFERENCE_ONLY',
                source_sha256=sha(SOURCE.read_bytes()),runner_sha256=sha(Path(__file__).read_bytes()),
                qa2='Reference move/pickup/drop/repick semantics and strict generated-intermediate validation',
                restart='Fresh process checkpoint reproduces frozen state, every output byte, and gate diagnostics',
                rows=rows)
    (ROOT/'ENGINEERING.json').write_text(json.dumps(report,indent=2)+'\n')
    print([(r['view'],r['gold'],r['exact_answer_with_own_newline'],r['gate_audit']) for r in rows])
