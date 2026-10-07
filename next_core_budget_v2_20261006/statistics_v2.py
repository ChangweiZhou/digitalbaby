"""Registered world-level bounds; no data-dependent significance rule."""
import math
import numpy as np
from scipy.stats import t,beta
def lower(values,alpha,bounds=None):
    a=np.asarray(values,dtype=float)
    if not 0<alpha<1 or a.ndim!=1 or len(a)<2 or not np.isfinite(a).all():raise ValueError('invalid statistical sample/alpha')
    if bounds is not None and ((a<bounds[0]-1e-12).any() or (a>bounds[1]+1e-12).any()):raise ValueError('registered score range exceeded')
    mu=float(a.mean());sd=float(a.std(ddof=1))
    if np.all(a==a[0]):
        if bounds is None:lo=None;method='UNRESOLVED_ZERO_VARIANCE'
        else:lo=mu-(bounds[1]-bounds[0])*math.sqrt(math.log(1/alpha)/(2*len(a)));method='Hoeffding_registered_range'
    else:lo=mu-float(t.ppf(1-alpha,len(a)-1))*sd/math.sqrt(len(a));method='paired_Student_approximate'
    return {'mean':mu,'lower':lo,'sd':sd,'method':method,'alpha':alpha,'n':len(a)}
def positive(bound):return bound['lower'] is not None and bound['lower']>0
def risk_upper(h,alpha=.03):
    if not h or any(type(v) is not int or v not in (0,1) for v in h):raise ValueError('harm indicators')
    return 1. if sum(h)==len(h) else float(beta.ppf(1-alpha,sum(h)+1,len(h)-sum(h)))
