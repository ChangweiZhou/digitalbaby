"""Frozen selection/stats only; never imported by the operative learner."""
import math
import numpy as np
from scipy.stats import t

FACTORIAL=tuple(f"F{e}{a}{c}" for e in (0,1) for a in (0,1) for c in (0,1))
ARMS=("NATIVE",)+FACTORIAL

def confirm_roster(selected):
    if selected not in FACTORIAL:raise ValueError("unknown candidate")
    result={"NATIVE","F000",selected}
    for i,bit in enumerate(selected[1:]):
        if bit=="1":
            bits=list(selected[1:]);bits[i]="0";result.add("F"+"".join(bits))
    return ["NATIVE"]+sorted(result-{"NATIVE"})

def paired_interval(differences):
    x=np.asarray(differences,dtype=float)
    if x.ndim!=1 or len(x)<2 or not np.isfinite(x).all():raise ValueError("finite independent worlds required")
    mean=float(x.mean());sd=float(x.std(ddof=1))
    half=float(t.ppf(.975,len(x)-1))*sd/math.sqrt(len(x))
    return dict(n=len(x),mean=mean,lower=mean-half,upper=mean+half)

def select(scores):
    """Aggregated DEV means. Reuse code fixtures, not science to qualify it."""
    if set(scores)!=set(ARMS):raise ValueError("full fixed DEV roster required")
    root=scores['F000'];native=scores['NATIVE'];eligible=[]
    for arm in FACTORIAL:
        row=scores[arm];refs=[native]+([root] if arm!='F000' else [])
        gain=min(row['G']-native['G'],row['Q']-native['Q'])
        if not all(row[k]>max(r[k] for r in refs) for k in ('G','Q')):continue
        if row['OLD_FINAL']<max(r['OLD_FINAL'] for r in refs):continue
        if not row['G']>max(row['G_N_OLD'],row['G_SHORTCUT']):continue
        if not row['HELD_FINAL']>max(row['HELD_N_OLD_FINAL'],row['HELD_SHORTCUT_FINAL']):continue
        if not all(row[k]>row[k+'_CONTROL'] for k in ('OLD_END','OLD_REST','NEW_END','NEW_REST')):continue
        eligible.append((gain,-arm.count('1'),-FACTORIAL.index(arm),arm))
    return max(eligible)[-1] if eligible else None

def confirm_gates(selected,rows):
    if len(rows)!=16 or len({r['world'] for r in rows})!=16:raise ValueError("exactly16 independent worlds")
    required=['G_NATIVE','Q_NATIVE','G_CAUSAL','HELD_CAUSAL','HELD_SHORTCUT',
              'OLD_END_CAUSAL','OLD_END_MODE','OLD_REST_CAUSAL','OLD_REST_MODE',
              'OLD_FINAL_MODE','NEW_END_MODE','NEW_REST_MODE','OLD_PRE_RETAIN','OLD_FINAL_RETAIN']
    if selected!='F000':required+=['G_ROOT','Q_ROOT']
    results={key:paired_interval([r[key] for r in rows]) for key in required}
    passed=all(v['lower']>=0 if key.endswith('_RETAIN') else v['lower']>0 for key,v in results.items())
    return dict(passed=passed,gates=results)

def holm(pvalues):
    if len(pvalues)>3 or any(not 0<=p<=1 for p in pvalues.values()):raise ValueError("fixed secondary family <=3")
    alive=True;result={}
    for i,(key,p) in enumerate(sorted(pvalues.items(),key=lambda x:(x[1],x[0]))):
        alive=alive and p<=.05/(len(pvalues)-i);result[key]=alive
    return result
