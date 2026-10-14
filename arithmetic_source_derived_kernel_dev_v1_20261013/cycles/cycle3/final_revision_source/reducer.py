"""Receipt-only sample/compute and fixed future selection reducers."""
import collections
import math


def accuracy(probe):
    groups = collections.defaultdict(list)
    for r in probe['rows']: groups[r['operation']].append(float(r['exact']))
    return {op: sum(v)/len(v) for op,v in groups.items()}


def ability(entry):
    a,b = [accuracy(p) for p in entry['probes']]
    return .5*sum(a.values())/len(a)+.5*sum(b.values())/len(b)


def staircase(points, end, starting=0.):
    if end <= 0: raise ValueError('positive common horizon')
    total = 0.; previous_x = 0.; previous_a = starting
    for x,a in points:
        if x < previous_x or not 0 <= a <= 1: raise ValueError('invalid chronological curve')
        cut = min(x,end)
        total += (cut-previous_x)*previous_a
        if x >= end: return total/end
        previous_x,previous_a = x,a
    return (total+(end-previous_x)*previous_a)/end


def metrics(r,cstar=None):
    out = {}
    for name,b in r['branches'].items():
        prefix = r['branches']['W']['binary_curve'] if name=='N_MIXED' else b['binary_curve']
        curve = prefix+b['mixed_curve']
        end = r['binary_n']+r['mixed_n']
        last = b['mixed_curve'][-1]
        final = {p['label']:accuracy(p) for p in last['probes']}
        shortcuts = {}
        for p in last['probes']:
            byop = collections.defaultdict(lambda:collections.defaultdict(list))
            for row in p['rows']:
                for model,pred in row['shortcuts'].items(): byop[row['operation']][model].append(float(pred==row['expected']))
            shortcuts[p['label']] = {op:max(sum(a)/len(a) for a in v.values()) for op,v in byop.items()}
        out[name] = dict(J_sample=staircase([(c['external_W_coordinates'],ability(c)) for c in curve],end),
                         J_compute=None if cstar is None else staircase([(c['production_cpu'],ability(c)) for c in curve],cstar),
                         final=final, shortcut=shortcuts,
                         production_cpu_end=last['production_cpu'],
                         taught={p['label']:accuracy(p) for p in b['taught_final']})
    return out


def select(world_metrics):
    # Input must contain all fixed arms and same worlds, after a full audit.
    names=('B16','B24','B69','E6','E6_FIX')
    if set(world_metrics)!=set(names): raise ValueError('wrong fixed DEV roster')
    means={arm:{key:sum(m['W'][key] for m in rows)/len(rows) for key in ('J_sample','J_compute','production_cpu_end')} for arm,rows in world_metrics.items()}
    reference=sorted(names[:3],key=lambda a:(-means[a]['J_sample'],-means[a]['J_compute'],means[a]['production_cpu_end'],a))[0]
    eligible=[]
    for arm in names[3:]:
        gain={key:means[arm][key]-max(means[g][key] for g in names[:3]) for key in ('J_sample','J_compute')}
        guards=[]
        for op in ('+','-','*','/'):
            ds=[m['W']['final']['old_held'][op]-max(m['N_BINARY']['final']['old_held'][op],m['SH_BINARY']['final']['old_held'][op],m['W']['shortcut']['old_held'][op]) for m in world_metrics[arm]]
            guards.append(sum(ds)/len(ds))
        for mode in ('prior','mixed'):
            ds=[]
            for m in world_metrics[arm]:
                avg=lambda value:sum(value.values())/len(value)
                w=avg(m['W']['final']['new_held'])
                comparison=max(avg(m['N_BINARY']['final']['new_held']),avg(m['SH_BINARY']['final']['new_held']),avg(m['W']['shortcut']['new_held'])) if mode=='prior' else avg(m['N_MIXED']['final']['new_held'])
                ds.append(w-comparison)
            guards.append(sum(ds)/len(ds))
        if min(gain.values())>0 and min(guards)>0:
            params=43529 if arm=='E6_FIX' else 43913
            eligible.append((min(gain.values()),means[arm]['production_cpu_end'],params,arm))
    eligible.sort(key=lambda v:(-v[0],v[1],v[2],0 if v[3]=='E6_FIX' else 1))
    selected=eligible[0][3] if eligible else None
    return dict(selected=selected,reference=reference,twin=None if selected is None else 'E6_FIX' if selected=='E6' else 'E6',means=means,science_authorized=False)


def holm(pvalues,alpha=.05):
    order=sorted(range(len(pvalues)),key=lambda i:pvalues[i])
    passed=[False]*len(pvalues)
    for rank,i in enumerate(order):
        if pvalues[i] > alpha/(len(pvalues)-rank): break
        passed[i]=True
    return passed
