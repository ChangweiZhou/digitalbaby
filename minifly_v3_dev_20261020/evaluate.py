"""Read-only exact own-output evaluator, outside the operative learner."""
import time
import numpy as np


def score(model, records):
    rows=[]
    for prompt,want in records:
        got=model.generate(prompt)
        exact=got['payload']==want and got['ended_with_EOS'] and not got['truncated']
        rows.append(dict(prompt=prompt.hex(),want=want.hex(),got=got['payload'].hex(),
                         ended_with_EOS=got['ended_with_EOS'],truncated=got['truncated'],
                         exact=bool(exact),tokens=got['tokens']))
    return dict(n=len(rows),correct=sum(r['exact'] for r in rows),
                accuracy=sum(r['exact'] for r in rows)/len(rows),rows=rows)


def checkpoint(model, sets, phase, count, production_cpu):
    before=model.digest();start=time.process_time()
    scores={k:score(model,v) for k,v in sets.items()}
    assert model.digest()==before,'evaluation modified production learner'
    return dict(phase=phase,count=count,records=model.records,time=model.time,
                production_cpu=production_cpu,evaluation_cpu=time.process_time()-start,
                digest=before,scores=scores,diagnostics=model.diagnostics())


def sample_area(points):
    x=np.log2(np.array([r['count'] for r in points])+1)
    y=np.array([r['scores']['HELD']['accuracy'] for r in points])
    return float(np.trapezoid(y,x)/(x[-1]-x[0]))


def compute_area(points,cstar):
    total=min(cstar,float(points[0]['production_cpu']))*points[0]['scores']['HELD']['accuracy']
    for i,p in enumerate(points):
        lo=float(p['production_cpu']);hi=float(points[i+1]['production_cpu']) if i+1<len(points) else cstar
        total+=max(0.,min(cstar,hi)-min(cstar,lo))*p['scores']['HELD']['accuracy']
    return total/cstar
