"""Stage reservoirs over observed presentations; no model import or task parser."""
import hashlib,json
from random import Random

class Reservoir:
    def __init__(self,seed,cap):
        if cap<1: raise ValueError("positive capacity required")
        self.cap=cap;self.seed=seed;self.draw_rng=Random(seed+9)
        self.rows={};self.count={};self.names={};self.rng={};self.stats={}
    def begin(self,stage,name):
        self.rows[stage]=[];self.count[stage]=0;self.names[stage]=name
        self.rng[stage]=Random(self.seed*1000003+stage*1009)
        self.stats[stage]={"draws":{},"composition":{"n":0,"wrong_full":0,"wrong_final":0,"malformed":0},"operation":{"n":0,"wrong_full":0,"wrong_final":0,"malformed":0}}
    def past(self,stage):return [i for i in sorted(self.rows) if i<stage and self.rows[i]]
    def draw(self,stage,n):
        prior=self.past(stage);out=[]
        for _ in range(n):
            i=self.draw_rng.choice(prior);row=self.draw_rng.choice(self.rows[i]);out.append(row)
            d=self.stats[stage]["draws"];d[str(i)]=d.get(str(i),0)+1
        return [([*p],[*t]) for p,t in out]
    def admit(self,stage,items):
        rows=self.rows[stage];rng=self.rng[stage]
        for p,t in items:
            self.count[stage]+=1;seen=self.count[stage];row=(tuple(p),tuple(t))
            if len(rows)<self.cap:rows.append(row)
            else:
                j=rng.randrange(seen)
                if j<self.cap:rows[j]=row
    def targets(self,stage,observed,answers,malformed):
        for (p,t),ans,bad in zip(observed,answers,malformed):
            v=self.stats[stage]["composition" if p[0]==99 else "operation"]
            v["n"]+=1;v["wrong_full"]+=list(ans)!=list(t);v["wrong_final"]+=list(ans[-3:])!=list(t[-3:]);v["malformed"]+=bad
    def snapshot(self,stage):
        out=[]
        for i,rows in self.rows.items():
            raw=json.dumps(rows,separators=(",", ":")).encode()
            out.append(dict(stage=i,task=self.names[i],presentations=self.count[i],slots=len(rows),distinct_prompts=len({p for p,t in rows}),distinct_records=len(set(rows)),sha256=hashlib.sha256(raw).hexdigest()))
        return dict(stage=stage,reservoirs=out,replay=self.stats[stage])

def answers(outputs,prompts,eos):
    result=[];bad=[]
    for p,o in zip(prompts,outputs):
        iscomp=p[0]==99
        valid=(len(o)==6 and o[2]==59 and o[-1]==eos and all(48<=o[i]<=57 for i in (0,1,3,4))) if iscomp else (len(o)==3 and o[-1]==eos)
        result.append(o if valid else ([48,48,59,48,48,eos] if iscomp else [48,48,eos]));bad.append(not valid)
    return result,bad
