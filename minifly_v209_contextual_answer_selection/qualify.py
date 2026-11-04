"""V209 engineering checks. Round 14 mechanics reimplemented from the memo."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import random
import re
import subprocess
import sys
from pathlib import Path
import tempfile

ROOT=Path(__file__).resolve().parent
CAPACITY=65536
CANDIDATES=16
HASH_SEED=0
PEOPLE=('Mary','John','Daniel','Sandra')
PLACES=('kitchen','garden','office','hallway','bathroom','bedroom')
NEW_PEOPLE=('Lucy','Paul','Robert','Olivia')
NEW_PLACES=('cellar','garage','studio','attic','porch','library')
VIEWS=('familiar','names','places','joint','new_template','distractor','qfound')

def sha(raw):return hashlib.sha256(raw).hexdigest()
def hashed(key):
    return int.from_bytes(hashlib.blake2s(key.encode(),digest_size=4,person=HASH_SEED.to_bytes(8,"little")).digest(),'little')%CAPACITY

def addresses(text):
    return {hashed(text[i:i+n]) for n in (2,3,4) for i in range(len(text)-n+1)}

def payload(sentence):
    tokens=re.findall(r'[A-Za-z]+',sentence)
    return tuple((w,tuple(tokens[j] if 0<=j<len(tokens) else '^'
                         for j in (i-2,i-1,i+1,i+2)))
                 for i,w in enumerate(tokens[:CANDIDATES]))

def features(candidate,selector):
    word,ctx=candidate
    if selector=='identity':return [hashed('word:'+word)]
    return sorted({hashed(f'neighbor:{i}:{w}') for i,w in enumerate(ctx)}
                  | {hashed(f'pair:{i}:{ctx[i]}:{ctx[i+1]}') for i in (0,2)})

class Learner:
    def __init__(self,selector):
        self.selector=selector
        self.store={}
        self.df=Counter()
        self.questions=0
        self.counts={}
        self.hits=self.trials=0

    def write(self,sentence):
        record=payload(sentence)
        for row in addresses(sentence):
            self.store[row]=record

    def recall(self,query):
        votes=Counter()
        for row in sorted(addresses(query)):
            if row not in self.store:
                continue
            weight=math.log((1+self.questions)/(1+self.df[row]))
            for candidate in self.store[row]:
                votes[candidate]+=weight
        return sorted(votes,key=lambda c:(-votes[c],c))[:CANDIDATES]

    def answer(self,candidates,permute=None):
        if not candidates:return None,False
        base=self.hits/max(1,self.trials)
        contexts=[ctx for _,ctx in candidates]
        if permute is not None:
            random.Random(permute).shuffle(contexts)
        def score(i):
            word,_=candidates[i]
            keys=features((word,contexts[i]),self.selector)
            return sum((self.counts.get(k,(0,0))[0]+2*base)/
                       (self.counts.get(k,(0,0))[1]+2) for k in keys)/len(keys)
        scores=[score(i) for i in range(len(candidates))]
        best=max(scores)
        # Equal scores retain the predeclared recall ordering.
        return candidates[scores.index(best)][0],scores.count(best)>1

    def feedback(self,query,candidates,target):
        self.df.update(addresses(query))
        self.questions+=1
        if target is None:return
        for candidate in candidates:
            good=int(candidate[0]==target)
            self.hits+=good
            self.trials+=1
            for row in features(candidate,self.selector):
                h,n=self.counts.get(row,(0,0))
                self.counts[row]=(h+good,n+1)

    def frozen(self):
        return dict(selector=self.selector,df=sorted(self.df.items()),questions=self.questions,
                    counts=sorted(self.counts.items()),hits=self.hits,trials=self.trials)

def worlds(seed,n):
    rng=random.Random(seed)
    for _ in range(n):
        latest={}
        story=[]
        for step in range(5):
            for j in range(2):
                p=rng.randrange(4); l=rng.randrange(6)
                latest[p]=l
                story.append(('s',p,l,rng.randrange(3),rng.choice([a for a in range(6) if a!=l]),j))
            p=rng.choice(sorted(latest))
            story.append(('q',p,latest[p],0,0,0))
        yield story

def render(row,view):
    typ,p,l,verb,distractor,order=row
    persons=NEW_PEOPLE if view in ('names','joint','new_template','distractor') else PEOPLE
    places=NEW_PLACES if view in ('places','joint','new_template','distractor') else PLACES
    name=persons[p]; place=places[l]
    if typ=='q':
        return (f'Where can {name} be found?' if view=='qfound' else f'Where is {name}?'),place
    if view=='new_template':
        return f'{place} is where {name} ended up.',None
    if view=='qfound':
        return f'{name} can be found in the {place}.',None
    if view=='distractor':
        alt=places[distractor]
        text=f'{name} went to the {place}, not the {alt}.' if order else f'{name} avoided the {alt} and went to the {place}.'
        return text,None
    return f'{name} {("went to","moved to","journeyed to")[verb]} the {place}.',None

def train(selector,feedback):
    m=Learner(selector)
    training=list(worlds(20901,120))
    targets=[render(r,'familiar')[1] for story in training for r in story if r[0]=='q']
    random.Random(20971).shuffle(targets)
    shuffled=iter(targets)
    for story in training:
        for row in story:
            text,label=render(row,'familiar')
            if label is None:
                m.write(text)
            else:
                retrieved=m.recall(text)
                target=label if feedback=='true' else next(shuffled) if feedback=='random' else None
                m.feedback(text,retrieved,target)
    return m

def evaluate(m,view,permute=False,world_seed=20902,shuffle_ties=False):
    correct=available=ties=oracle=total=0
    outputs=[]
    before=m.frozen()
    for story in worlds(world_seed,20):
        last={}
        for row in story:
            text,label=render(row,view)
            if label is None:
                m.write(text)
                last[row[1]]=payload(text)
            else:
                c=m.recall(text)
                if shuffle_ties:
                    random.Random(world_seed+total).shuffle(c)
                answer,tie=m.answer(c,20990+total if permute else None)
                correct+=int(answer==label)
                available+=int(any(w==label for w,_ in c))
                ties+=int(tie)
                oracle+=int(m.answer(last[row[1]])[0]==label)
                total+=1
                outputs.append(answer)
    assert m.frozen()==before and total==100 and len(m.store)<=CAPACITY
    return dict(correct=correct,available=available,ties=ties,oracle=oracle,total=total,outputs=outputs,occupied_rows=len(m.store))

def dump(m,path):
    data=json.dumps(dict(hash_seed=HASH_SEED,model=m.frozen(),store=sorted(m.store.items()))).encode()
    path.write_bytes(data)
    path.with_suffix('.sha').write_text(sha(data))

def load(path):
    data=path.read_bytes()
    if sha(data)!=path.with_suffix('.sha').read_text():
        raise ValueError('Checkpoint digest mismatch')
    saved=json.loads(data)
    assert saved['hash_seed']==HASH_SEED
    p=saved['model'];m=Learner(p['selector'])
    m.df=Counter(dict(p['df']))
    m.questions=p['questions'];m.counts={k:tuple(v) for k,v in p['counts']}
    m.hits=p['hits'];m.trials=p['trials']
    m.store={i:tuple((w,tuple(ctx)) for w,ctx in payload) for i,payload in saved['store']}
    return m

def job(args):
    selector,feedback,view,permute=args
    return dict(selector=selector,feedback=feedback,view=view,permuted=permute,
                **evaluate(train(selector,feedback),view,permute))

if __name__=='__main__':
    if len(sys.argv)>1:
        print(json.dumps(evaluate(load(Path(sys.argv[1])),'joint')))
        sys.exit()
    words=dict(payload('Mary moved to the attic.'))
    assert words['attic']==('to','the','^','^')
    assert features(('attic',words['attic']),'context')==features(('porch',words['attic']),'context')
    m=Learner('context')
    m.write('Mary moved to the attic.')
    a=m.frozen()
    m.answer(m.recall('Where is Mary?'))
    assert m.frozen()==a
    roster=[(s,f,v,False) for s in ('identity','context') for f in ('true','random','none') for v in VIEWS]
    roster += [('context','true',v,True) for v in VIEWS]
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(job,roster))
    assert rows[21+3]==job(roster[21+3])
    with tempfile.TemporaryDirectory() as d:
        path=Path(d)/'trained.json'
        model=train('context','true')
        dump(model,path)
        expected=evaluate(model,'joint')
        cmd=[sys.executable,'-B',__file__,str(path)]
        assert json.loads(subprocess.check_output(cmd))==expected
        path.write_bytes(path.read_bytes()+b'bad')
        rejected=subprocess.run(cmd,capture_output=True,text=True)
        assert rejected.returncode and 'Checkpoint digest mismatch' in rejected.stderr
    out=dict(version='V209',status='DEVELOPMENT_ONLY_ROUND14_REIMPLEMENTATION',
             source_sha256=sha(Path(__file__).read_bytes()),rows=rows,
             cycles=['Candidate-context binding and causal feedback fixtures',
                     '49 matched development runs with 100 questions per view',
                     'Parallel equality, fresh-process checkpoint recovery and corrupt checkpoint rejection'])
    (ROOT/'ENGINEERING.json').write_text(json.dumps(out,indent=2)+'\n')
    for row in rows:
        if row['feedback']=='true':
            print(row['selector'],row['view'],row['permuted'],row['correct'],row['available'],row['oracle'])
