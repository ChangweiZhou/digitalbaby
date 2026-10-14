"""Postprocess only: paired-world mean inference, no model or data imports."""
import json
import math
import statistics
import sys
from scipy.stats import t


def paired(values):
    if len(values)<2 or any(not math.isfinite(v) or not -1<=v<=1 for v in values):
        raise ValueError('at least two finite bounded world differences')
    n=len(values);mean=statistics.mean(values)
    if all(v==values[0] for v in values):
        p=1. if mean<=0 else min(1.,math.exp(-n*mean**2/2))
        radius=math.sqrt(2*math.log(40)/n)
        return dict(mean=mean,p=p,CI95=[max(-1.,mean-radius),min(1.,mean+radius)],method='bounded Hoeffding, constant sample')
    se=statistics.stdev(values)/math.sqrt(n)
    radius=float(t.ppf(.975,n-1))*se
    return dict(mean=mean,p=float(t.sf(mean/se,n-1)),CI95=[mean-radius,mean+radius],method='paired world Student t')


def holm(pvalues):
    ranked=sorted(range(len(pvalues)),key=lambda i:pvalues[i]);passed=[False]*len(pvalues)
    for rank,i in enumerate(ranked):
        if pvalues[i]>.05/(len(pvalues)-rank):break
        passed[i]=True
    return passed


if __name__=='__main__':
    if sys.argv[1:]!=['--fixtures-only']:raise SystemExit('Only statistical qualification fixtures are admitted here')
    assert paired([0.]*24)['p']==1.
    assert paired([.1]*24)['p']>.8
    assert paired([1.]*24)['p']<.00001
    assert paired([-.1]*24)['p']==1.
    r=paired([.1,.2,.3,.4]);assert abs(r['mean']-.25)<1e-14 and r['p']<.02
    assert holm([.001,.1,.005])==[True,False,True]
    try:paired([2.,2.])
    except ValueError:pass
    else:raise AssertionError('unbounded difference admitted')
    import scipy,numpy
    result=dict(accepted=True,scope='statistical fixtures only',checks=7,native_records=0,scientific_jobs=0,scipy=scipy.__version__,numpy=numpy.__version__,python=sys.version)
    print(json.dumps(result))
