"""Prospective four-way screen, one fresh winner confirmation; no tuning."""
import copy, math
from collections import Counter
import config
from parent_decision import metrics,countermodels,mean_metrics,lower_bound,guard_ok

ANCHORS=config.BASELINES
def resource_ok(candidate,base):
    return (candidate['production_cpu_seconds']<=1.5*base['production_cpu_seconds']
            and candidate['payload_bytes']<=2*base['payload_bytes']
            and candidate['peak_rss_bytes']<=512*1024**2)
def select(rows):
    expected=config.BASELINES+config.CANDIDATES
    if len(rows)!=12 or len({r['world'] for r in rows})!=12:raise ValueError('12 independent DEV worlds')
    if any(set(expected)-set(r) for r in rows):raise ValueError('complete DEV roster')
    if Counter((r['family'],r['noise']) for r in rows)!=Counter({(f,n):3 for f in ('PATH_LOOP','STAR_LOOP') for n in ('STANDARD','BLURRY')}):raise ValueError('balanced strata')
    primary=[r for r in rows if r['noise']=='BLURRY'];standard=[r for r in rows if r['noise']=='STANDARD']
    eligible=[];gates={}
    for arm in config.CANDIDATES:
        b=mean_metrics(primary,arm);s=mean_metrics(standard,arm);all_=mean_metrics(rows,arm)
        checks={}
        for baseline in ANCHORS:
            pb=mean_metrics(primary,baseline);sb=mean_metrics(standard,baseline)
            checks[baseline+'_gain']=b['W']['J']-pb['W']['J']>=.03
            checks[baseline+'_primary_guard']=guard_ok(b,pb,True)
            checks[baseline+'_standard_guard']=guard_ok(s,sb)
        checks['causal']=all(b[k]>0 for k in ('new_causal_N','new_causal_shuffle','old_causal_N','old_causal_shuffle'))
        checks['resources']=resource_ok(all_,mean_metrics(rows,'COV-SUPPORT'))
        checks['paired_payload']=all(r[arm]['payload_bytes']<=2*r['COV-SUPPORT']['payload_bytes'] for r in rows)
        gates[arm]=checks
        if all(checks.values()):eligible.append((arm,b['W']['J']-mean_metrics(primary,'COV-SUPPORT')['W']['J'],all_['production_cpu_seconds']))
    if not eligible:return dict(verdict='NO_ADVANCE_RETAIN_USER_CORE',selected=None,gates=gates)
    gain=max(z[1] for z in eligible);ties=[z for z in eligible if abs(z[1]-gain)<=1e-12]
    best=min(ties,key=lambda z:(z[2],config.CANDIDATES.index(z[0])))
    return dict(verdict='ONE_CANDIDATE_SELECTED',selected=best[0],gates=gates)

def confirm(rows,selected):
    if selected not in config.CANDIDATES or len(rows)!=128 or len({r['world'] for r in rows})!=128:raise ValueError('one selected candidate,128 fresh worlds')
    required=config.BASELINES+(selected,)+config.PEERS[selected]
    if any(set(required)-set(r) for r in rows):raise ValueError('missing locked peers/baselines')
    primary=[r for r in rows if not r['ordinary'] and r['noise']=='BLURRY']
    standard=[r for r in rows if not r['ordinary'] and r['noise']=='STANDARD']
    ordinary=[r for r in rows if r['ordinary']]
    if len(primary)!=64 or len(standard)!=32 or len(ordinary)!=32 or Counter(r['noise'] for r in ordinary)!=Counter(STANDARD=16,BLURRY=16):raise ValueError('confirmation strata')
    if any(r['family']!='TRIANGLE_TAIL' for r in rows):raise ValueError('untouched confirmation geometry')
    checks={};bounds={}
    for baseline in ANCHORS:
        bound=lower_bound([r[selected]['W']['J']-r[baseline]['W']['J'] for r in primary]);bounds[baseline]=bound
        checks[baseline+'_gain']=bound['mean']>=.05 and bound['lower']>0
        for name,pop in (('primary',primary),('standard',standard),('ordinary',ordinary)):
            checks[baseline+'_'+name+'_guard']=guard_ok(mean_metrics(pop,selected),mean_metrics(pop,baseline),name=='primary')
    for key in ('new_causal_N','new_causal_shuffle','old_causal_N','old_causal_shuffle'):
        bound=lower_bound([r[selected][key] for r in primary]);bounds[key]=bound;checks[key]=bound['lower']>0
    for peer in config.PEERS[selected]:
        bound=lower_bound([r[selected]['W']['J']-r[peer]['W']['J'] for r in primary]);bounds[peer]=bound;checks[peer]=bound['lower']>0
    shortcut=math.fsum(r[selected]['W']['J']-max(r['shortcut_J'].values()) for r in primary)/64
    checks['shortcut']=shortcut>=.05
    checks['resources']=resource_ok(mean_metrics(rows,selected),mean_metrics(rows,'COV-SUPPORT'))
    checks['paired_payload']=all(r[selected]['payload_bytes']<=2*r['COV-SUPPORT']['payload_bytes'] for r in rows)
    return dict(verdict='ADOPT_BOUNDED_ADDITION_MECHANISM' if all(checks.values()) else 'RETAIN_USER_CORE',
                selected=selected,checks=checks,bounds=bounds,shortcut_gap=shortcut)

def synthetic_suite():
    def item(j=.2,cpu=1.):
        return dict(W=dict(J=j,old_score=.8,old_final=.95,new_final=j),new_causal_N=.5,new_causal_shuffle=.5,
                    old_causal_N=.5,old_causal_shuffle=.5,production_cpu_seconds=cpu,payload_bytes=15000,peak_rss_bytes=40*1024**2)
    dev=[]
    for f in ('PATH_LOOP','STAR_LOOP'):
        for n in ('STANDARD','BLURRY'):
            for i in range(3):dev.append(dict(world=len(dev),family=f,noise=n,**{a:item() for a in config.ARMS}))
    assert select(dev)['selected'] is None
    for row in dev:
        for arm in tuple(a for p in config.PEERS.values() for a in p):row[arm]=item(.9)
    assert select(dev)['selected'] is None,'peer promoted'
    for row in dev:row['KCI']=item(.6)
    assert select(dev)['selected']=='KCI'
    checks=0
    for selected in config.CANDIDATES:
        rows=[]
        for i in range(128):
            row=dict(world=i,family='TRIANGLE_TAIL',ordinary=i>=96,noise='BLURRY' if i<64 or 96<=i<112 else 'STANDARD',shortcut_J=dict(mode=.1),**{a:item() for a in config.ARMS})
            row[selected]=item(.6);rows.append(row)
        assert confirm(rows,selected)['verdict'].startswith('ADOPT')
        for baseline in ANCHORS:
            bad=copy.deepcopy(rows)
            for row in bad:row[baseline]=item(.6)
            assert confirm(bad,selected)['verdict']=='RETAIN_USER_CORE';checks+=1
        for peer in config.PEERS[selected]:
            bad=copy.deepcopy(rows)
            for row in bad:row[peer]=item(.6)
            assert confirm(bad,selected)['verdict']=='RETAIN_USER_CORE';checks+=1
        for key in ('new_causal_N','new_causal_shuffle','old_causal_N','old_causal_shuffle'):
            bad=copy.deepcopy(rows)
            for row in bad:row[selected][key]=0.
            assert confirm(bad,selected)['verdict']=='RETAIN_USER_CORE';checks+=1
        for mode in ('cost','payload','retention','shortcut'):
            bad=copy.deepcopy(rows)
            for row in bad:
                if mode=='cost':row[selected]['production_cpu_seconds']=2.
                elif mode=='payload':row[selected]['payload_bytes']=40000
                elif mode=='retention':row[selected]['W']['old_final']=.7
                else:row['shortcut_J']['mode']=.6
            assert confirm(bad,selected)['verdict']=='RETAIN_USER_CORE';checks+=1
    assert lower_bound([0.]*64)['lower']<0
    return dict(verdict='PASS_SYNTHETIC_ONLY',failed_gate_cases_rejected=checks,science_worlds=0)
