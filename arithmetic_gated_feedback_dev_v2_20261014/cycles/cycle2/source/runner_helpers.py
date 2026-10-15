"""Environment-side archived countermodels and pure output scoring, unchanged."""
import collections
from protocol import probes

class Shortcuts:
    def __init__(self,feedback):
        self.tables={k:collections.defaultdict(collections.Counter) for k in ('exact','canonical','length','suffix','bag')}
        self.glob=collections.Counter()
        for r in feedback:
            answer=r['observed']
            if answer is None and r['reward']==1:answer=r['actual']
            if answer is None:continue
            p=bytes.fromhex(r['prompt']);keys=(r['prompt'],r['group'],len(p),p[-2:],tuple(sorted(p)))
            self.glob[answer]+=1
            for k,x in zip(self.tables,keys):self.tables[k][x][answer]+=1
    def predict(self,r):
        mode=lambda c:min(c,key=lambda a:(-c[a],a)) if c else None
        p=bytes.fromhex(r['prompt']);keys=(r['prompt'],r['group'],len(p),p[-2:],tuple(sorted(p)))
        a={k:mode(self.tables[k].get(x,{})) for k,x in zip(self.tables,keys)};a['global']=mode(self.glob)
        a['length']=a['length'] or a['global'];a['suffix']=a['suffix'] or a['global'];a['bag']=a['bag'] or a['length'];return a

def checkpoint(c,n,w,feedback,taught=False):
    before=c.digest();counter=Shortcuts(feedback);cpu_before=c.resource['production_cpu']
    exposed={r['prompt']:dict(prompt=r['prompt'],reply=r['expected'],operation=r['operation'],group=r['group']) for r in feedback}
    rows=[(label,probes([r for r in exposed.values() if len(r['operation'])==(1 if label=='binary' else 2)],None) if taught else probes(w[label+'_held'],None)) for label in ('binary','mixed')]
    scores={}
    for label,rs in rows:
        answers=c.generate_batch([bytes.fromhex(x['prompt']) for x in rs]);scores[label]=[]
        for x,(answer,status) in zip(rs,answers):
            scores[label].append(dict(x,actual=answer.hex(),status=status,exact=answer.hex()==x['reply'] and status=='EOS',shortcuts=counter.predict(x)))
    assert before==c.digest()
    if taught:
        c.resource['diagnostic_cpu']=c.resource.get('diagnostic_cpu',0.)+c.resource['production_cpu']-cpu_before
        c.resource['production_cpu']=cpu_before
    return dict(n=n,scores=scores,before=before,after=c.digest(),cpu=c.resource['production_cpu'],resource=dict(c.resource))
