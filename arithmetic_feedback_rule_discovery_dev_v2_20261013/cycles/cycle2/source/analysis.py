"""Independent behavior/cost reducer; never imports a learner."""
import math
from collections import Counter,defaultdict
import numpy as np
from scipy.stats import t

def exact(row):return row['status']=='EOS' and row['payload_hex']==row['answer_hex']
def rate(rows):return sum(exact(r) for r in rows)/len(rows) if rows else 0.
def rates(rows):
    return {op:rate([r for r in rows if r.get('operation')==op]) for op in sorted({r.get('operation') for r in rows})}
def shortcut(training,queries):
    def keys(prompt):return (None,len(prompt),prompt[-2:],tuple(sorted(Counter(prompt).items())))
    tables=[defaultdict(Counter) for _ in range(4)]
    for rec in training:
        p=bytes.fromhex(rec['prompt_hex']);answer=bytes.fromhex(rec['answer_hex'])
        for table,key in zip(tables,keys(p)):table[key][answer]+=1
    def mode(counts):return sorted(counts,key=lambda a:(-counts[a],a))[0] if counts else None
    global_mode=mode(tables[0][None]);scores=[]
    for q in queries:
        p=bytes.fromhex(q['prompt_hex']);values=[]
        for i,key in enumerate(keys(p)):
            counts=tables[i][key]
            prediction=mode(counts)
            if prediction is None:
                prediction=mode(tables[1][len(p)]) if i==3 else global_mode
            values.append(prediction==bytes.fromhex(q['answer_hex']))
        scores.append(values)
    return max(np.mean(scores,axis=0)) if scores else 0.
def common_cpu(a,b):
    horizon=min(a[-1][0],b[-1][0])
    if horizon<=0:raise ValueError('positive common CPU')
    def area(rows):
        last=0.;score=rows[0][1];integral=0.;costs=np.maximum.accumulate([x[0] for x in rows])
        for cost,(_,value) in zip(costs,rows):
            if cost>horizon:break
            integral+=score*(cost-last);last=cost;score=value
        return (integral+score*(horizon-last))/horizon
    return area(a),area(b)
def mean_test(values):
    x=np.asarray(values,float);n=len(x)
    if n<2 or not np.isfinite(x).all():raise ValueError('independent finite paired worlds')
    mean=float(x.mean());sd=float(x.std(ddof=1))
    if sd==0:return min(1.,math.exp(-n*mean*mean/2)) if mean>0 else 1.
    return float(t.sf(mean/(sd/math.sqrt(n)),n-1))
def holm(values,alpha=.05):
    order=np.argsort(values);result=[False]*len(values)
    for rank,i in enumerate(order):
        if values[i]>alpha/(len(values)-rank):break
        result[int(i)]=True
    return result
