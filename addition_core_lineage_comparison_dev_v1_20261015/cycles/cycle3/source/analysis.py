"""Evaluator-only metrics and locked statistical primitives, not learner inputs."""
import math
import collections
import numpy as np
from scipy import stats


def accuracy(rows):
    return sum(r['correct'] for r in rows)/len(rows)


def metrics(history,cstar):
    curve=[m for m in history['metrics'] if m['stage']=='OLD']
    n=np.array([m['n'] for m in curve],float)
    y=np.array([accuracy(m['scores']['held']) for m in curve])
    x=np.log2(n+1)
    js=float(np.trapezoid(y,x)/(x[-1]-x[0]))
    area=0.
    for i,m in enumerate(curve):
        left=min(cstar,m['cpu'])
        right=min(cstar,curve[i+1]['cpu']) if i+1<len(curve) else cstar
        area+=max(0.,right-left)*y[i]
    final=history['metrics'][-1]
    return dict(J_sample=js,J_compute=area/cstar,R_final=accuracy(final['scores']['held']))


def paired_test(delta):
    delta=np.asarray(delta,float)
    if len(delta)<2 or not np.isfinite(delta).all():
        raise ValueError('at least two independent finite world differences')
    mean=float(delta.mean())
    sd=float(delta.std(ddof=1))
    if sd==0:
        return dict(mean=mean,CI95=[mean,mean],p=0. if mean>0 else 1.)
    se=sd/math.sqrt(len(delta))
    width=float(stats.t.ppf(.975,len(delta)-1)*se)
    return dict(mean=mean,CI95=[mean-width,mean+width],p=float(stats.t.sf(mean/se,len(delta)-1)))


def iut(six_pvalues):
    if len(six_pvalues)!=6 or any(not 0<=p<=1 for p in six_pvalues):
        raise ValueError('six required endpoint/causal tests')
    return max(six_pvalues)


def holm(four_pvalues,alpha=.05):
    if len(four_pvalues)!=4:
        raise ValueError('locked four comparison family')
    order=sorted(range(4),key=lambda i:four_pvalues[i])
    result=[False]*4
    for rank,i in enumerate(order):
        if four_pvalues[i]>alpha/(4-rank):
            break
        result[i]=True
    return result


def shortcut_envelope(rows,pairs):
    # Evaluator-only fitted witnesses. Correct held answers are used only to score.
    taught=[(bytes.fromhex(r['prompt']),bytes.fromhex(r['answer'])) for r in rows if r['allowed']]
    if not taught:
        raise ValueError('no delivered training feedback')
    freq=collections.Counter(y for _,y in taught)
    def mode(counts):
        return min(counts,key=lambda y:(-counts[y],y)) if counts else min(freq,key=lambda y:(-freq[y],y))
    majority=mode(freq)
    raw={p:y for p,y in taught}
    def parts(p):return p[:-1].split(b'+')
    def key(p):return tuple(sorted(parts(p)))
    exchange={key(p):y for p,y in taught}
    left,right=collections.defaultdict(collections.Counter),collections.defaultdict(collections.Counter)
    for p,y in taught:
        a,b=parts(p);left[a][y]+=1;right[b][y]+=1
    counts=collections.Counter()
    for a,b in pairs:
        p=f'{a}+{b}='.encode();answer=str(a+b).encode();x,z=parts(p)
        choices=dict(majority=majority,raw=raw.get(p,majority),exchange=exchange.get(key(p),majority),
                     left=mode(left[x]),right=mode(right[z]),marginal_sum=mode(left[x]+right[z]))
        for name,prediction in choices.items():counts[name]+=int(prediction==answer)
    accuracies={k:v/len(pairs) for k,v in counts.items()}
    # Human numeric linear features are a witness ONLY, never the envelope/learner.
    X=np.array([[1.,*map(float,parts(p))] for p,_ in taught])
    y=np.array([float(answer) for _,answer in taught])
    beta=np.linalg.lstsq(X,y,rcond=None)[0]
    witness=sum(int(round(float(np.array([1.,a,b])@beta)))==a+b for a,b in pairs)/len(pairs)
    return dict(envelope=max(accuracies.values()),countermodels=accuracies,
                numerical_linear_witness=witness,witness_is_not_an_operative_model=True)


def scientific_comparison(world_profiles,anchor='N151-C16'):
    # Formal use requires exactly the registered independent16-world roster.
    worlds=sorted({r['world'] for r in world_profiles})
    if worlds!=list(range(202610144001,202610144017)):
        raise ValueError('formal statistics prohibit technical/unregistered worlds')
    versions=('C151-C16','RC151-C16','KEEP-C16-CE','G62-C16-CE')
    lookup={(r['world'],r['version']):r for r in world_profiles}
    results=[]
    for version in versions:
        vectors=[[] for _ in range(6)]
        for world in worlds:
            row=lookup[world,version];base=lookup[world,anchor]
            for i,key in enumerate(('J_sample','J_compute','R_final')):
                vectors[i].append(row['metrics']['W'][key]-base['metrics']['W'][key])
            end=row['metrics']['W']['R_final']
            vectors[3].append(end-row['metrics']['N_OLD']['R_final'])
            vectors[4].append(end-row['metrics']['SH_OLD']['R_final'])
            vectors[5].append(end-row['shortcut_envelope'])
        tests=[paired_test(v) for v in vectors]
        results.append(dict(version=version,tests=tests,p_iut=iut([t['p'] for t in tests])))
    for row,passes in zip(results,holm([r['p_iut'] for r in results])):
        row['limited_relative_advantage']=passes
    return results


def fixture_tests():
    assert paired_test([0.]*16)['p']==1
    assert paired_test([.1]*16)['p']==0
    assert paired_test([-.1]*16)['p']==1
    assert iut([0.,0.,0.,0.,0.,1.])==1
    assert holm([.001,.03,.04,.5])==[True,False,False,False]
    assert holm([0.,0.,0.,0.])==[True]*4
    rows=[dict(correct=False)]
    synthetic={'metrics':[dict(stage='OLD',n=0,cpu=0.,scores={'held':rows}),
                         dict(stage='OLD',n=32,cpu=10.,scores={'held':[dict(correct=True)]}),
                         dict(stage='NEW_REST',n=1,cpu=10.,scores={'held':rows})]}
    assert metrics(synthetic,5.)['J_compute']==0.
    assert metrics(synthetic,20.)['J_compute']==.5
    return dict(accepted=True,fixtures=8,science=False,training_records=0)
