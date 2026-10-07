"""One selected target, three separate IUT claims, no post-confirmation rescue."""
from protocol import check_selection
from selection import validate_metrics,protection,vec,eta
from statistics_v2 import lower,positive,risk_upper
def confirm(rows,p,selection,worlds):
    check_selection(p,selection)
    if not selection['winner'] or len(worlds)!=48 or set(worlds)&set(selection['screen_worlds']):raise ValueError('confirmation selection/n/exposure')
    arms=list(selection['confirmation_roster'])
    if 'ERROR' in arms:arms+=['Q_HALF']
    validate_metrics(rows,[a for a in arms if a!='S3_RAND' or a in rows],worlds)
    w=selection['winner'];a,parent,target=w['configuration'],w['parent'],w['target'];ar,br=rows[a],rows[parent]
    g=protection(rows,a,parent);hu=risk_upper(g['harms'],.03)
    components={'gain':lower(vec(ar,target)-vec(br,target),.03,(-2,2) if target=='reuse' else (-1,1)),
                'efficiency_gain':lower(eta(ar,target)-eta(br,target),.03)}
    if target!='reuse':components['causal_teaching']=lower(vec(ar,target)-vec(ar,'N_'+target),.03,(-1,1))
    else:
        components['prior_teaching']=lower(vec(ar,'reuse'),.03,(-1,1));components['raw_above_chance']=lower(vec(ar,'reuse_W')-.5,.03,(-.5,.5))
    e3_parts={'prior_teaching':lower(vec(ar,'reuse'),.01,(-1,1)),'raw_above_chance':lower(vec(ar,'reuse_W')-.5,.01,(-.5,.5))}
    competence=all(vec(ar,k).mean()>=t for k,t in [('taught_old_end',.8),('taught_final',.7),('new_W',.65),('new_N',.65)])
    e3=competence and all(positive(b) for b in e3_parts.values())
    adopts=(g['passes'] and hu<=.1 and components['gain']['mean']+1e-12>=w['delta'] and all(positive(b) for b in components.values()) and (target!='reuse' or e3))
    mc={};registered=a in p['mechanism_targets'];available=not registered or all(c in rows for c in p['mechanism_targets'][a]['controls'])
    if registered and available:
        for control in p['mechanism_targets'][a]['controls']:
            mc[control]=lower(vec(ar,target)-vec(rows[control],target),.01,(-2,2) if target=='reuse' else (-1,1))
    mechanism=registered and available and adopts and all(positive(b) for b in mc.values())
    unresolved=any(b['lower'] is None or (b['mean']>0 and not positive(b)) for b in components.values())
    verdict=('CONFIRMED_LIMITED_REUSE_UPGRADE' if target=='reuse' else 'CONFIRMED_E1_ENGINEERING_UPGRADE') if adopts else ('PROMISING_BUT_UNRESOLVED' if unresolved else 'RETAIN_V1')
    return {'verdict':verdict,'candidate':a,'parent':parent,'frozen_target':target,'adoption':{'alpha':.03,'passed':adopts,'components':components,'engineering_protection':g,'harm_upper':hu,'harm_risk_passed':hu<=.1},'E3_existence':{'alpha':.01,'passed':bool(e3),'components':e3_parts,'competence':competence},'mechanism':{'alpha':.01,'registered':registered,'available':available,'status':'AVAILABLE' if available else 'DIAGNOSTIC_NOT_QUALIFIED','passed':bool(mechanism),'components':mc},'n':48,'worlds':list(worlds),'nominal_FWER':.05,'t_coverage':'approximate'}
