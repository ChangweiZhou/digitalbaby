"""The 116 predeclared paired-world contrasts; no post-hoc endpoint selection."""
import math
import numpy as np
from scipy.stats import t
from core import CONFIGS, BASELINE
from worlds import OPS, CHECKPOINTS

ARMS = CONFIGS[:-1]
CANDIDATES = tuple(x for x in ARMS if x != BASELINE)


def j_curve(values):
    if len(values) != 6:
        raise ValueError('full six-point curve required')
    return float(np.trapz(values, np.log2(CHECKPOINTS))/5.)


def curve(receipt,history='W',op=None):
    rows=receipt['histories'][history]['checkpoints']
    values=[row['scores'][op] if op else np.mean(list(row['scores'].values())) for row in rows]
    return values, [row['cpu'] for row in rows]


def cpu_pair(candidate,base):
    ya,xa=curve(candidate); yb,xb=curve(base)
    xa,xb=np.array(xa),np.array(xb)
    if np.any(np.diff(xa)<=0) or np.any(np.diff(xb)<=0):
        raise ValueError('CPU curve must strictly increase')
    lo,hi=max(xa[0],xb[0]),min(xa[-1],xb[-1])
    if hi<=lo:
        return None
    knots=np.unique(np.r_[lo,hi,xa[(xa>lo)&(xa<hi)],xb[(xb>lo)&(xb<hi)]])
    log=np.log(knots)
    diff=np.interp(log,np.log(xa),ya)-np.interp(log,np.log(xb),yb)
    return float(np.trapz(diff,log)/(np.log(hi)-np.log(lo)))


def retention(receipt):
    hist=receipt['histories']['W']
    deltas=[]
    for ci in range(6):
        end=np.mean(list(hist['checkpoints'][ci]['scores'].values()))
        for block in hist['blocks'][4*ci:4*ci+4]:
            deltas.append(end-np.mean(list(block['scores'].values())))
    return float(np.mean(deltas))


def paired_interval(values,m=116):
    if any(v is None or not math.isfinite(v) for v in values) or len(values)<2:
        return {'mean':None,'lower':None,'upper':None,'status':'UNAVAILABLE'}
    x=np.array(values,float)
    mean=float(x.mean()); sd=float(x.std(ddof=1))
    if sd==0:
        return {'mean':mean,'lower':None,'upper':None,'status':'ZERO_VARIANCE_UNRESOLVED'}
    width=float(t.ppf(1-.05/(2*m),len(x)-1)*sd/np.sqrt(len(x)))
    return {'mean':mean,'lower':mean-width,'upper':mean+width,'status':'ESTIMATED',
            'n_worlds':len(x),'family_size':m}


def contrasts(paired):
    out={}
    for candidate in CANDIDATES:
        out[f'{candidate}:J_samples_vs_B']=[j_curve(curve(w[candidate])[0])-j_curve(curve(w[BASELINE])[0]) for w in paired]
        out[f'{candidate}:J_cpu_vs_B']=[cpu_pair(w[candidate],w[BASELINE]) for w in paired]
    for fi,factor in enumerate(('G','R','C')):
        differences=[]
        for world in paired:
            ds=[]
            for arm in ARMS:
                parts=arm.split('-')
                if parts[fi] != ('COMP','LOCAL','FULL')[fi]:
                    continue
                off=parts.copy();off[fi]=('LINEAR','FIXED','DIAG')[fi]
                ds.append(j_curve(curve(world[arm])[0])-j_curve(curve(world['-'.join(off)])[0]))
            assert len(ds)==4
            differences.append(float(np.mean(ds)))
        out[f'factor:{factor}']=differences
    for candidate in CANDIDATES:
        for op in OPS:
            out[f'{candidate}:J_{op}_vs_B']=[j_curve(curve(w[candidate],op=op)[0])-j_curve(curve(w[BASELINE],op=op)[0]) for w in paired]
    for arm in ARMS:
        for control in ('N_ALL','SHUFFLE'):
            for op in OPS:
                out[f'{arm}:W_minus_{control}:{op}']=[j_curve(curve(w[arm],op=op)[0])-j_curve(curve(w[arm],history=control,op=op)[0]) for w in paired]
    for candidate in CANDIDATES:
        out[f'{candidate}:retention_vs_B']=[retention(w[candidate])-retention(w[BASELINE]) for w in paired]
    assert len(out)==116
    return out


def reduce(paired,technical=False):
    cs=contrasts(paired)
    stats={k:paired_interval(v) for k,v in cs.items()}
    eligible=[]
    for arm in CANDIDATES:
        positive=[f'{arm}:J_samples_vs_B',f'{arm}:J_cpu_vs_B']
        positive += [f'{arm}:W_minus_{c}:{op}' for c in ('N_ALL','SHUFFLE') for op in OPS]
        nonnegative=[f'{arm}:J_{op}_vs_B' for op in OPS]+[f'{arm}:retention_vs_B']
        if all(stats[k]['lower'] is not None and stats[k]['lower']>0 for k in positive) and all(
                stats[k]['lower'] is not None and stats[k]['lower']>=0 for k in nonnegative):
            eligible.append(arm)
    eligible.sort(key=lambda a:(np.median([w[a]['histories']['W']['training_core_cpu'] for w in paired]),
                 paired[0][a]['histories']['W']['storage']['allocated_array_bytes'],a))
    return {'technical':technical, 'n_worlds':len(paired), 'contrasts':stats,
            'selected':None if technical else (eligible[0] if eligible else None),
            'verdict':'TECHNICAL_REDUCER_ONLY' if technical else (
                'ADOPT_BOUNDED_FOUR_OPERATION_CORE' if eligible else 'NO_CONFIRMED_REPLACEMENT'),
            'family_size':116}
