"""Pure bounded-budget exploration. Statistical confirmation is separate."""
import math
import numpy as np
from protocol import delta,check_selection
SCORE_KEYS=('old','new','revision','reuse_W','reuse_N','taught_old_end','taught_final','new_W','new_N','N_old','N_new','N_revision')
def validate_metrics(rows,arms,worlds):
    if set(rows)!=set(arms):raise ValueError('missing/extra report configuration')
    for arm,rs in rows.items():
        if [r['world'] for r in rs]!=list(worlds) or len({r['world'] for r in rs})!=len(worlds):raise ValueError('mixed/duplicate/unpaired world roster')
        for r in rs:
            for k in SCORE_KEYS:
                if type(r[k]) not in (int,float) or not math.isfinite(r[k]) or not 0<=r[k]<=1:raise ValueError('invalid accuracy')
            for k in ('tauL','tauR'):
                if type(r[k]) not in (int,float) or not math.isfinite(r[k]) or r[k]<=0:raise ValueError('invalid online W CPU')
    return rows
def vec(rs,key):
    if key=='reuse':return np.array([r['reuse_W']-r['reuse_N'] for r in rs])
    return np.array([r[key] for r in rs])
def eta(rs,target):return 3600*vec(rs,target)/vec(rs,'tauR' if target=='reuse' else 'tauL')
def protection(rows,arm,parent):
    a,b,z=rows[arm],rows[parent],rows['CENTER'];reasons=[]
    for target in ('old','new','revision'):
        if vec(a,target).mean()<.90:reasons.append('accuracy_'+target)
        if (vec(a,target)-vec(b,target)).mean()<-.03-1e-12:reasons.append('parent_protection_'+target)
        if target!='revision' and (vec(a,target)-vec(z,target)).mean()<-.03-1e-12:reasons.append('CENTER_protection_'+target)
    for key in ('reuse','reuse_W'):
        if (vec(a,key)-vec(b,key)).mean()<-.05-1e-12:reasons.append('reuse_protection_'+key)
    for k,threshold in [('taught_old_end',.8),('taught_final',.7),('new_W',.65),('new_N',.65)]:
        if vec(a,k).mean()<threshold:reasons.append('competence_'+k)
    ratio=float((vec(a,'tauL')+vec(a,'tauR')).mean()/(vec(b,'tauL')+vec(b,'tauR')).mean())
    if ratio>1.5:reasons.append('online_CPU_ratio')
    harms=[int(any(x[t]<y[t]-.05-1e-12 for t in ('old','new','revision'))) for x,y in zip(a,b)]
    return {'passes':not reasons,'reasons':reasons,'CPU_ratio':ratio,'harms':harms}
def choose(rows,p,worlds):
    if len(worlds)!=8:raise ValueError('screen n must be 8')
    validate_metrics(rows,[a for a in p['report_configurations'] if a!='S3_RAND' or a in rows],worlds)
    ranked=[];guard={}
    for a in p['eligible_candidates']:
        parent=p['candidate_parent'][a];g=protection(rows,a,parent);guard[a]=g
        if not g['passes'] or sum(g['harms']):continue
        options=[]
        for target in p['selection']['targets_ordered']:
            gain=float((vec(rows[a],target)-vec(rows[parent],target)).mean());d=delta(a,target)
            eg=float((eta(rows[a],target)-eta(rows[parent],target)).mean())
            if gain+1e-12<d or eg<=0:continue
            if target=='reuse' and (vec(rows[a],'reuse').mean()<=0 or vec(rows[a],'reuse_W').mean()<=.5):continue
            norm=gain/d;R=norm/g['CPU_ratio'];cpu=float((vec(rows[a],'tauL')+vec(rows[a],'tauR')).mean())
            options.append({'configuration':a,'parent':parent,'target':target,'delta':d,'gain':gain,'eta_gain':eg,'R':R,'normalized_gain':norm,'total_online_W_CPU':cpu})
        def key(x):return (-x['R'],-x['normalized_gain'],x['total_online_W_CPU'],x['configuration'],p['selection']['targets_ordered'].index(x['target']))
        if options:ranked.append(sorted(options,key=key)[0])
    ranked.sort(key=lambda x:(-x['R'],-x['normalized_gain'],x['total_online_W_CPU'],x['configuration'],p['selection']['targets_ordered'].index(x['target'])))
    winner=ranked[0] if ranked else None
    out={'verdict':'SELECTED_ONE_CANDIDATE' if winner else 'SCREEN_COMPLETE_NO_CANDIDATE','winner':winner,'ranked':ranked,'guards':guard,'alphas':[.03,.01,.01],'screen_worlds':list(worlds),'confirmation_roster':p['confirmation']['physical_roster_by_winner'][winner['configuration']] if winner else []}
    return check_selection(p,out)
