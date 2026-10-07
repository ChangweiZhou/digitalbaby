"""Reduce final world scores from emitted answers, never evaluator alternatives."""
import numpy as np
from runtime import read_receipt,file_sha,execution_identity,digest,make_world
from job_audit import audit,audit_unavailable
def row_accuracy(p,group,q=False,reuse=False,items=None):
    rs=[r for r in p['rows'] if r['stage']==group and (items is None or r['item'] in items)]
    if items is not None and {r['item'] for r in rs}!=set(items):raise ValueError('missing/duplicate intact-old items')
    if items is not None and len(rs)!=len(items):raise ValueError('duplicate intact-old rows')
    if not rs:raise ValueError('missing scored group')
    if q and reuse:return float(np.mean([r['Q_HALF_correct'] for r in rs]))
    if q:return float(np.mean([r['policies']['Q_HALF']['emitted']==r['target'] for r in rs]))
    return float(np.mean([r['correct'] for r in rs]))
def one(L,R,q=False):
    def probe(d,b,name):
        ps=[p for p in d['branches'][b]['probes'] if p['name']==name]
        if len(ps)!=1:raise ValueError('checkpoint count')
        return ps[0]
    m={'world':L['world']};intact=make_world(L['world'],'lifetime')['intact_old_items']
    for g in ('old','new','revision'):
        items=intact if g=='old' else None
        m[g]=row_accuracy(probe(L,'W','final'),g,q,items=items)
        m['N_'+g]=row_accuracy(probe(L,'N_'+g,'final'),g,q,items=items)
    m['reuse_W']=row_accuracy(probe(R,'W','final'),'heldout',q,True)
    m['reuse_N']=row_accuracy(probe(R,'N_old','final'),'heldout',q,True)
    m['taught_old_end']=row_accuracy(probe(R,'W','old_end'),'old',q,True)
    m['taught_final']=row_accuracy(probe(R,'W','final'),'old',q,True)
    m['new_W']=row_accuracy(probe(R,'W','final'),'new',q,True)
    m['new_N']=row_accuracy(probe(R,'N_old','final'),'new',q,True)
    key='Q_HALF_online_cpu_s' if q else 'online_cpu_s'
    m['tauL']=L['branches']['W'][key];m['tauR']=R['branches']['W'][key]
    return m
def collect(folder,worlds,arms,stage):
    from pathlib import Path
    folder=Path(folder);identity=execution_identity();out={a:[] for a in arms};evidence=[];unavailable=set()
    for world in worlds:
        for arm in arms:
            ds=[];missing=False
            for assay in ('lifetime','reuse'):
                p=folder/'receipts'/f'{world}_{arm}_{assay}.json.gz'
                marker=folder/'unavailable'/p.name
                if marker.exists():
                    d=read_receipt(marker);audit_unavailable(d,identity,folder)
                    if d['stage']!=stage or d['world']!=world or d['arm']!=arm or d['assay']!=assay:raise ValueError('mixed diagnostic identity')
                    if p.exists():raise ValueError('receipt and unavailable both present')
                    evidence.append({'file':'unavailable/'+p.name,'sha256':file_sha(marker)});unavailable.add(arm);missing=True;continue
                d=read_receipt(p)
                audit(d,identity,folder,load_export=False)
                if d['stage']!=stage or d['world']!=world or d['arm']!=arm or d['assay']!=assay:raise ValueError('mixed receipt identity')
                ds.append(d);evidence.append({'file':p.name,'sha256':file_sha(p)})
            if missing:continue
            out[arm].append(one(*ds))
            if arm=='ERROR':out.setdefault('Q_HALF',[]).append(one(*ds,q=True))
    for a in unavailable:out.pop(a,None)
    return out,evidence
