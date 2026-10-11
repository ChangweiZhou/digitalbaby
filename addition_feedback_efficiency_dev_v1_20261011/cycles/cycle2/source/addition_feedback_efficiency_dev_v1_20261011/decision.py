"""Prospective reducer and synthetic gates; technical outcomes never select."""
import math
from collections import Counter,defaultdict
import numpy as np
from scipy.stats import t
import config
import worlds

def old_weights():
    x=np.log2(np.array(config.CHECKPOINTS)+80);w=np.zeros(6)
    for i in range(5):
        amount=(x[i+1]-x[i])/(2*(x[-1]-x[0]));w[i]+=amount;w[i+1]+=amount
    return w
def new_score(fork):
    values=[fork['new_checkpoints'][str(m)]['delayed_new_held']['accuracy'] for m in config.NEW_CHECKPOINTS]
    return float(np.dot([.25,.5,.25],values)),values
def metrics(receipt):
    out={}
    for h,b in receipt['branches'].items():
        old=[b['forks'][str(n)]['final_old_held']['accuracy'] for n in config.CHECKPOINTS]
        j,new=new_score(b['forks']['320'])
        out[h]={'old_score':float(old_weights()@np.array(old)),'old_final':old[-1],
                'old_curve':old,'J':j,'new_curve':new,'new_final':new[-1]}
    for h,f in receipt['extra_forks'].items():
        j,curve=new_score(f);out[h]={'J':j,'new_curve':curve,'new_final':curve[-1]}
    for name,other in (('new_causal_N','N_NEW'),('new_causal_shuffle','SHUFFLED_NEW')):
        out[name]=out['W']['J']-out[other]['J']
    out['old_causal_N']=out['W']['old_score']-out['N_OLD']['old_score']
    out['old_causal_shuffle']=out['W']['old_score']-out['SHUFFLED_OLD']['old_score']
    rows=[r for b in receipt['branches'].values() for group in [b['old_records']]+[
        f['new_records'] for f in b['forks'].values()] for r in group]
    rows += [r for f in receipt['extra_forks'].values() for r in f['new_records']]
    out['production_cpu_seconds']=math.fsum(r['production_cpu_seconds'] for r in rows)
    snaps=[f['final_checkpoint'] for b in receipt['branches'].values() for f in b['forks'].values()]
    snaps += [f['final_checkpoint'] for f in receipt['extra_forks'].values()]
    out['payload_bytes']=max(s['payload_bytes'] for s in snaps)
    out['peak_rss_bytes']=receipt['peak_rss_bytes']
    out['technical_only']=True
    return out

def countermodels(receipt):
    master=receipt['master'];curves=defaultdict(list)
    for m in config.NEW_CHECKPOINTS:
        new=[r for r in master['new'][:m] if r['target'] is not None]
        all_rows=[r for r in master['old']+master['new'][:m] if r['target'] is not None]
        global_counts=Counter(r['target'] for r in all_rows);new_counts=Counter(r['target'] for r in new)
        mode=lambda counts:min(range(9),key=lambda y:(-counts[y],y))
        emit=lambda v:min(range(9),key=lambda y:((y-v)**2,y))
        means=lambda rows:emit(sum(r['target'] for r in rows)/len(rows)) if rows else 0
        left=defaultdict(Counter);right=defaultdict(Counter)
        for row in new:
            left[row['pair'][0]][row['target']]+=1;right[row['pair'][1]][row['target']]+=1
        table={tuple(r['pair']):r['target'] for r in all_rows}
        scores=defaultdict(list)
        for query in worlds.queries(master,'new',False):
            orbit=defaultdict(list)
            for a,b in query['pairs']:
                if (a,b) in table or (b,a) in table:raise AssertionError('held-pair key exposure')
                guess={
                    'raw_key_global_mode':mode(global_counts),'canonical_key_global_mode':mode(global_counts),
                    'raw_key_new_mode':mode(new_counts),'canonical_key_new_mode':mode(new_counts),
                    'global_target_mode':mode(global_counts),'new_target_mode':mode(new_counts),
                    'global_target_mean':means(all_rows),'new_target_mean':means(new),
                    'fixed_rank':master['new_alphabet'].index(a)+master['new_alphabet'].index(b),
                    'zero_birth':0,'left_only':mode(left[a]),'right_only':mode(right[b])}
                for k,y in guess.items():orbit[k].append(y==query['truth_private'])
            for k,v in orbit.items():scores[k].append(sum(v)/len(v))
        for k,v in scores.items():curves[k].append(sum(v)/len(v))
    return {k:{'curve':v,'J':float(np.dot([.25,.5,.25],v))} for k,v in curves.items()}

def lower_bound(values):
    x=np.asarray(values,dtype=float)
    if x.ndim!=1 or len(x)!=64 or not np.isfinite(x).all() or np.max(np.abs(x))>1:
        raise ValueError('64 bounded independent world differences required')
    mean=float(x.mean());var=float(x.var(ddof=1))
    if var<=1e-24:return {'mean':mean,'lower':mean-math.sqrt(2*math.log(20)/len(x)),'method':'Hoeffding'}
    return {'mean':mean,'lower':mean-float(t.ppf(.95,len(x)-1))*math.sqrt(var/len(x)),'method':'paired Student'}

def mean_metrics(rows,arm):
    if not rows:raise ValueError('empty population')
    keys=('old_score','old_final','J','new_final')
    for row in rows:
        m=row[arm]
        for k in keys:
            if not math.isfinite(m['W'][k]) or not 0<=m['W'][k]<=1:raise ValueError('finite bounded ability')
        for k in ('new_causal_N','new_causal_shuffle','old_causal_N','old_causal_shuffle'):
            if not math.isfinite(m[k]) or abs(m[k])>1:raise ValueError('finite bounded causal effect')
        for k in ('production_cpu_seconds','payload_bytes','peak_rss_bytes'):
            if not math.isfinite(m[k]) or m[k]<=0:raise ValueError('finite positive resource charge')
    out={'W':{k:math.fsum(r[arm]['W'][k] for r in rows)/len(rows) for k in keys}}
    for k in ('new_causal_N','new_causal_shuffle','old_causal_N','old_causal_shuffle','production_cpu_seconds'):
        out[k]=math.fsum(r[arm][k] for r in rows)/len(rows)
    for k in ('payload_bytes','peak_rss_bytes'):out[k]=max(r[arm][k] for r in rows)
    return out


def resource_ok(candidate,base):
    return candidate['production_cpu_seconds']<=1.5*base['production_cpu_seconds'] and candidate['payload_bytes']<=2*base['payload_bytes'] and candidate['peak_rss_bytes']<=512*1024**2


def guard_ok(candidate,base,primary=False):
    valid=candidate['W']['old_score']>=base['W']['old_score']-.03
    if primary:
        valid &= candidate['W']['old_final']>=base['W']['old_final']-.03 and candidate['W']['old_final']>=.85
    else:valid &= candidate['W']['J']>=base['W']['J']-.03
    return bool(valid)


def select(rows):
    if len(rows)!=12 or len({r['world'] for r in rows})!=12:raise ValueError('12 distinct DEV world rows required')
    if any(set(config.ARMS)-set(r) for r in rows):raise ValueError('complete DEV roster required')
    counts=Counter((r['family'],r['noise']) for r in rows)
    if counts!=Counter({(f,n):3 for f in ('PATH_LOOP','STAR_LOOP') for n in ('STANDARD','BLURRY')}):raise ValueError('balanced DEV strata required')
    blurry=[r for r in rows if r['noise']=='BLURRY'];std=[r for r in rows if r['noise']=='STANDARD']
    base_b=mean_metrics(blurry,'COV-R');base_s=mean_metrics(std,'COV-R');base_all=mean_metrics(rows,'COV-R')
    eligible=[]
    for arm in config.ARMS[1:]:
        b=mean_metrics(blurry,arm);s=mean_metrics(std,arm);all_=mean_metrics(rows,arm)
        gain=b['W']['J']-base_b['W']['J']
        valid=(gain>=.03 and guard_ok(b,base_b,True) and guard_ok(s,base_s)
               and resource_ok(all_,base_all) and b['new_causal_N']>0 and b['new_causal_shuffle']>0)
        if valid:eligible.append((arm,gain,all_['production_cpu_seconds'],all_['payload_bytes']))
    if not eligible:return {'verdict':'NO_ADVANCE_RETAIN_COV_R','selected':None}
    best=max(x[1] for x in eligible);ties=[x for x in eligible if best-x[1]<=1e-12]
    winner=min(ties,key=lambda x:(x[2],x[3],config.ARMS.index(x[0])))
    return {'verdict':'CANDIDATE_SELECTED','selected':winner[0]}


def confirm(rows,selected):
    if selected not in config.ARMS[1:] or len(rows)!=128:raise ValueError('one selected candidate and 128 worlds')
    if len({r['world'] for r in rows})!=128:raise ValueError('distinct independent worlds required')
    primary=[r for r in rows if r['ordinary'] is False and r['noise']=='BLURRY']
    std=[r for r in rows if r['ordinary'] is False and r['noise']=='STANDARD']
    ordinary=[r for r in rows if r['ordinary'] is True]
    if len(primary)!=64 or len(std)!=32 or len(ordinary)!=32 or Counter(r['noise'] for r in ordinary)!=Counter({'STANDARD':16,'BLURRY':16}):raise ValueError('confirmation strata')
    if any(r['family']!='TRIANGLE_TAIL' for r in rows):raise ValueError('confirmation geometry')
    gain=lower_bound([r[selected]['W']['J']-r['COV-R']['W']['J'] for r in primary])
    effects={k:lower_bound([r[selected][k] for r in primary]) for k in
             ('new_causal_N','new_causal_shuffle','old_causal_N','old_causal_shuffle')}
    shortcut=math.fsum(r[selected]['W']['J']-max(r['shortcut_J'].values()) for r in primary)/64
    checks={'mean_gain':gain['mean']>=.05,'gain_lower':gain['lower']>0,
            'all_causal_lowers':all(v['lower']>0 for v in effects.values()),
            'primary_protection':guard_ok(mean_metrics(primary,selected),mean_metrics(primary,'COV-R'),True),
            'standard_protection':guard_ok(mean_metrics(std,selected),mean_metrics(std,'COV-R')),
            'ordinary_protection':guard_ok(mean_metrics(ordinary,selected),mean_metrics(ordinary,'COV-R')),
            'resources':resource_ok(mean_metrics(rows,selected),mean_metrics(rows,'COV-R')),
            'shortcut_guard':shortcut>=.05}
    return {'verdict':'ADOPT_BOUNDED_FEEDBACK_EFFICIENT_CORE' if all(checks.values()) else 'RETAIN_COV_R',
            'selected':selected,'gain':gain,'causal':effects,'checks':checks,'shortcut_gap':shortcut}


def synthetic_suite():
    assert abs(old_weights().sum()-1)<1e-14
    assert lower_bound([0.]*64)['lower']<0
    assert lower_bound([.5]*64)['lower']>0
    assert lower_bound([-.5]*64)['lower']<0
    assert lower_bound(np.linspace(.1,.3,64))['method']=='paired Student'
    rejected=0
    for bad in ([0.]*320,[float('nan')]*64,[[0.]*64], [1.1]*64):
        try:lower_bound(bad)
        except ValueError:rejected+=1
        else:raise AssertionError('invalid statistical unit accepted')
    return {'verdict':'PASS','invalid_units_rejected':rejected,'science_endpoints_used':0,'selected_scientific_candidate':None}
