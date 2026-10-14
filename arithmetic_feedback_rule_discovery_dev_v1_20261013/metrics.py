"""World-level endpoint and paired common-compute definitions; no learner access."""
import numpy as np
from scipy.stats import t as student_t

def J(summary,history='W'):
 return float(np.mean([list(s['held_retained'].values()) for s in summary[history]]))

def G(a,b):
 # Two scores, each integrated on the same paired endpoint, no accuracy interpolation.
 T=min(a[-1]['cpu'],b[-1]['cpu'])
 if T<=0:raise ValueError('positive common production CPU required')
 def integrate(s):
  costs=np.maximum.accumulate([x['cpu'] for x in s]); value=0.; last=0.; integral=0.
  for cost,row in zip(costs,s):
   if cost>T:break
   integral+=value*(cost-last);last=cost;value=float(np.mean(list(row['held_retained'].values())))
  integral+=value*(T-last)
  return integral/T
 return integrate(a),integrate(b)

def superiority(delta):
 x=np.asarray(delta,dtype=float);n=len(x)
 if n<2 or not np.isfinite(x).all():raise ValueError('paired finite worlds required')
 if np.std(x,ddof=1)==0:
  nonzero=x[x!=0]
  return 2.**(-len(nonzero)) if len(nonzero) and np.all(nonzero>0) else 1.
 return float(student_t.sf(np.mean(x)/(np.std(x,ddof=1)/np.sqrt(n)),n-1))

def holm(pvalues,alpha=.05):
 order=np.argsort(pvalues);rejected=[False]*len(pvalues)
 for rank,i in enumerate(order):
  if pvalues[i]>alpha/(len(pvalues)-rank):break
  rejected[i]=True
 return rejected
