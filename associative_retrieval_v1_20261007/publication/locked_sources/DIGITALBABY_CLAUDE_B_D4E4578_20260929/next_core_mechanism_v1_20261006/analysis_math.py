"""Only registered confidence primitives; not an exploratory selection script."""
import math
import numpy as np
from scipy.stats import t,beta
def lower(v,alpha,width=None,structural=False):
    a=np.asarray(v,dtype=float)
    if a.ndim!=1 or len(a)<2 or not np.isfinite(a).all():raise ValueError('invalid sample')
    mu=float(a.mean());sd=float(a.std(ddof=1))
    if sd==0:
        if structural:return {'mean':mu,'lower':mu,'method':'proved_structural'}
        if width is None:return {'mean':mu,'lower':None,'method':'UNRESOLVED_ZERO_VARIANCE'}
        lo=mu-width*math.sqrt(math.log(1/alpha)/(2*len(a)));method='Hoeffding_registered_range'
    else:lo=mu-float(t.ppf(1-alpha,len(a)-1))*sd/math.sqrt(len(a));method='paired_Student_approximate'
    return {'mean':mu,'lower':lo,'sd':sd,'method':method,'alpha':alpha,'n':len(a)}
def harm_upper(h,alpha):
    if any(x not in (0,1) for x in h) or not len(h):raise ValueError('harm indicators')
    k=sum(h);n=len(h);return 1. if k==n else float(beta.ppf(1-alpha,k+1,n-k))
