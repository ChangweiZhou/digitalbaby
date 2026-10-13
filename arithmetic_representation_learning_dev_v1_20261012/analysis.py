"""Fixed 41-contrast reducer. Worlds, not responses, are statistical units."""
import math
import numpy as np
from scipy.stats import t
from core import CANDIDATES, CONFIGS, BASELINE
from worlds import OPS, CHECKPOINTS

MATCHED = {'LEARN-HASH':'FROZEN-HASH','LEARN-SHARED':'FROZEN-SHARED'}


def j_curve(values):
    if len(values)!=6:
        raise ValueError('complete six-point curve required')
    return float(np.trapz(values,np.log2(CHECKPOINTS))/5)


def curve(r,history='W',op=None):
    rows=r['histories'][history]['checkpoints']
    return [x['scores'][op] if op else np.mean([x['scores'][o] for o in OPS]) for x in rows], [x['cpu'] for x in rows]


def final_score(r,history='W',op=None):
    a=r['histories'][history]['final']
    scores={o:np.mean([x['correct'] for x in a if x['operation']==o]) for o in OPS}
    return float(scores[op] if op else np.mean(list(scores.values())))


def cpu_pair(a,b):
    ya,xa=curve(a);yb,xb=curve(b)
    xa,xb=np.array(xa),np.array(xb)
    if not np.isfinite(xa).all() or not np.isfinite(xb).all() or np.any(np.diff(xa)<=0) or np.any(np.diff(xb)<=0):
        raise ValueError('invalid CPU checkpoints')
    lo,hi=max(xa[0],xb[0]),min(xa[-1],xb[-1])
    if hi<=lo:return None
    knots=np.unique(np.r_[lo,hi,xa[(xa>lo)&(xa<hi)],xb[(xb>lo)&(xb<hi)]])
    log=np.log(knots)
    diff=np.interp(log,np.log(xa),ya)-np.interp(log,np.log(xb),yb)
    return float(np.trapz(diff,log)/(np.log(hi)-np.log(lo)))


def paired_interval(values):
    if len(values)<2 or any(x is None or not math.isfinite(x) for x in values):
        return {'mean':None,'lower':None,'upper':None,'status':'UNAVAILABLE','family_size':41}
    x=np.array(values,float);mean=float(x.mean());sd=float(x.std(ddof=1))
    if sd==0:return {'mean':mean,'lower':None,'upper':None,'status':'ZERO_VARIANCE_UNRESOLVED','family_size':41}
    half=float(t.ppf(1-.05/(2*41),len(x)-1)*sd/np.sqrt(len(x)))
    return {'mean':mean,'lower':mean-half,'upper':mean+half,'n_worlds':len(x),'family_size':41,'status':'ESTIMATED'}


def contrasts(worlds):
    out={}
    for c in CANDIDATES:
        for ref in (MATCHED[c],BASELINE):
            out[f'{c}:J_samples_vs_{ref}']=[j_curve(curve(w[c])[0])-j_curve(curve(w[ref])[0]) for w in worlds]
            out[f'{c}:J_cpu_vs_{ref}']=[cpu_pair(w[c],w[ref]) for w in worlds]
    out['representation_x_hash:J_samples_interaction']=[
        j_curve(curve(w['LEARN-HASH'])[0])-j_curve(curve(w['FROZEN-HASH'])[0])-
        j_curve(curve(w['LEARN-SHARED'])[0])+j_curve(curve(w['FROZEN-SHARED'])[0]) for w in worlds]
    for c in CANDIDATES:
        for op in OPS:
            for h in ('N_ALL','SHUFFLE'):
                out[f'{c}:final_W_minus_{h}:{op}']=[final_score(w[c],op=op)-final_score(w[c],h,op) for w in worlds]
            out[f'{c}:J_samples_vs_LEGACY-B:{op}']=[j_curve(curve(w[c],op=op)[0])-j_curve(curve(w[BASELINE],op=op)[0]) for w in worlds]
            out[f'{c}:final_accuracy_vs_LEGACY-B:{op}']=[final_score(w[c],op=op)-final_score(w[BASELINE],op=op) for w in worlds]
    assert len(out)==41
    return out


def direct_interference(r,op,kind='held'):
    h=r['histories']['W'];deltas=[];no_intervening=[]
    for ci,cp in enumerate(h['checkpoints']):
        blocks=h['blocks'][4*ci:4*ci+4]
        for bi,b in enumerate(blocks):
            if b['operation']!=op:continue
            if kind=='held':
                end,just=cp['scores'][op],b['scores'][op]
            else:
                # Same taught cue set on both sides: later new observations
                # must not change the denominator of old-cue retention.
                earlier=b['taught'];keys={x['prompt_hex'] for x in earlier if x['operation']==op}
                end_rows=[x for x in cp['taught'] if x['operation']==op and x['prompt_hex'] in keys]
                just_rows=[x for x in earlier if x['operation']==op]
                end=np.mean([x['correct'] for x in end_rows]);just=np.mean([x['correct'] for x in just_rows])
            delta=float(end-just)
            (no_intervening if bi==3 else deltas).append(delta)
    return {'mean':float(np.mean(deltas)) if deltas else None,'interference_points':deltas,
            'last_block_no_intervening_points':no_intervening}


def reduce(worlds,technical=False):
    raw=contrasts(worlds);stats={k:paired_interval(v) for k,v in raw.items()}
    profiles={};eligible=[]
    for c in CANDIDATES:
        mechanism=stats[f'{c}:J_samples_vs_{MATCHED[c]}']['lower']
        op_evidence={op:all(stats[f'{c}:final_W_minus_{h}:{op}']['lower'] is not None and
            stats[f'{c}:final_W_minus_{h}:{op}']['lower']>0 for h in ('N_ALL','SHUFFLE')) for op in OPS}
        positive=[f'{c}:{m}_vs_{ref}' for ref in (MATCHED[c],BASELINE) for m in ('J_samples','J_cpu')]
        positive += [f'{c}:final_W_minus_{h}:{op}' for h in ('N_ALL','SHUFFLE') for op in OPS]
        guards=[f'{c}:{m}:{op}' for m in ('J_samples_vs_LEGACY-B','final_accuracy_vs_LEGACY-B') for op in OPS]
        qualifies=all(stats[k]['lower'] is not None and stats[k]['lower']>0 for k in positive) and all(
            stats[k]['lower'] is not None and stats[k]['lower']>=0 for k in guards)
        profiles[c]={'representation_sample_gain_confirmed':mechanism is not None and mechanism>0,
            'final_E3_operations':op_evidence,'all_four_final_E3':all(op_evidence.values()),
            'replacement_recommended':qualifies if not technical else False}
        if qualifies and not technical:eligible.append(c)
    selected=min(eligible,key=lambda c:(np.median([w[c]['histories']['W']['training_core_cpu'] for w in worlds]),
        worlds[0][c]['histories']['W']['storage']['allocated_array_bytes'],c)) if eligible else None
    descriptions={}
    for c in CONFIGS:
        descriptions[c]={
          'W_macro_curve':np.mean([curve(w[c])[0] for w in worlds],axis=0).tolist(),
          'per_operation':{op:{
            'W_curve':np.mean([curve(w[c],op=op)[0] for w in worlds],axis=0).tolist(),
            'final_by_history':{h:float(np.mean([final_score(w[c],h,op) for w in worlds])) for h in ('W','N_ALL','SHUFFLE')},
            'direct_interference_by_world':[direct_interference(w[c],op) for w in worlds],
            'taught_interference_by_world':[direct_interference(w[c],op,'taught') for w in worlds]
          } for op in OPS},
          'W_training_cpu_by_world':[w[c]['histories']['W']['training_core_cpu'] for w in worlds],
          'resource_by_world':[w[c].get('resource',{}) for w in worlds],
          'whole_job_wall_by_world':[w[c]['wall_seconds'] for w in worlds],
          'storage':worlds[0][c]['histories']['W']['storage']}
    return {'technical':technical,'n_worlds':len(worlds),'family_size':41,
            'verdict':'TECHNICAL_PIPELINE_ONLY' if technical else ('RECOMMEND_NEW_ENGINEERING_BASELINE' if selected else 'NO_CONFIRMED_REPLACEMENT'),
            'selected':selected,'candidate_profiles':profiles,'contrasts':stats,
            'paired_world_differences':raw,'descriptive_results':descriptions,
            'evidence_scope':'Technical fixtures only; no scientific E3 claim' if technical else 'Bounded four-operation family-held-out E3; no E4 or mixed expressions'}
