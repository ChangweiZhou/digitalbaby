"""Prospective confirmation interval, never a DEV significance claim."""
import math
import numpy as np
from scipy.stats import t as student

def lower_bound(differences):
    x=np.asarray(differences,dtype=float)
    if x.ndim!=1 or len(x)<2 or not np.isfinite(x).all() or np.any(np.abs(x)>1):raise ValueError('bounded world differences required')
    mean=float(x.mean());variance=float(x.var(ddof=1));n=len(x)
    if variance<=1e-24:return dict(mean=mean,lower=mean-math.sqrt(2*math.log(20)/n),method='HOEFFDING_ZERO_VARIANCE',n=n)
    return dict(mean=mean,lower=mean-float(student.ppf(.95,n-1))*math.sqrt(variance/n),method='PAIRED_STUDENT_ONE_SIDED_95',n=n)
