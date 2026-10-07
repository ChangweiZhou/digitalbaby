"""Stage rosters and fail-closed statistical selection contract."""
import math
from runtime import load_plan,digest
def screen_worlds(plan):return plan['screen']['proposed_world_ids']
def confirmation_worlds(plan):
    c=plan['confirmation'];return list(range(c['proposed_world_id_start'],c['proposed_world_id_end']+1))
def validate_plan(p):
    if p['screen']['n']!=8 or p['confirmation']['n']!=48 or p['screen']['maximum_finalists']!=1:raise ValueError('registered sample sizes/finalist limit')
    if set(screen_worlds(p))&set(confirmation_worlds(p)):raise ValueError('screen/confirmation overlap')
    if len(set(screen_worlds(p)))!=8 or len(confirmation_worlds(p))!=48:raise ValueError('world roster')
    if len(p['physical_configurations'])!=11 or len(p['report_configurations'])!=12:raise ValueError('configuration count')
    if set(p['report_configurations'])!=set(p['physical_configurations'])|{'Q_HALF'}:raise ValueError('Q learning duplication')
    if set(p['eligible_candidates'])&{'CENTER','S3_RAND','REL_PERM10','FIRST10'}:raise ValueError('control cannot advance')
    s=p['statistics'];alphas=[s[k] for k in ('adoption_joint_alpha','E3_existence_joint_alpha','mechanism_joint_alpha')]
    if alphas!=[.03,.01,.01] or not math.isclose(sum(alphas),.05) or s['unused_alpha_reallocated']:raise ValueError('alpha family')
    if not s['reuse_adoption_requires_E3_existence'] or s['severe_harm_threshold_pp']!=5:raise ValueError('evidence/risk contract')
    r=p['confirmation']['physical_roster_by_winner']
    if set(r)!=set(p['eligible_candidates']):raise ValueError('winner roster coverage')
    for a,rs in r.items():
        expected=['CENTER','ERROR']
        if a not in ('ERROR','Q_HALF'):expected+=[a]
        if a=='S3_CUE':expected+=['S3_RAND']
        if a=='REL10':expected+=['REL_PERM10','FIRST10']
        if rs!=expected:raise ValueError('conditional controls/physical roster')
    if p['latin_in_scope'] or p['screen']['new_mutations_allowed'] or p['confirmation']['allow_sample_extension']:raise ValueError('unregistered expansion')
    if p['session']['stop_dispatch_seconds']!=28800 or not 28800<32400<34200<36000:raise ValueError('session contract')
    return p
def delta(arm,target):return .20 if (arm,target)==('ERROR','revision') else (.05 if target=='reuse' else .02)
def roster(p,stage,selection=None):
    if stage=='screen':return p['physical_configurations']
    if stage=='confirm' and selection and selection['winner']:
        return p['confirmation']['physical_roster_by_winner'][selection['winner']['configuration']]
    raise ValueError('unlocked stage/selection')
def check_selection(p,s):
    w=s.get('winner')
    if not w:
        if s.get('verdict')!='SCREEN_COMPLETE_NO_CANDIDATE':raise ValueError('empty selection state')
        return s
    a=w['configuration'];t=w['target']
    if a not in p['eligible_candidates'] or t not in p['selection']['targets_ordered']:raise ValueError('invalid winner/target')
    if w['parent']!=p['candidate_parent'][a] or w['delta']!=delta(a,t):raise ValueError('changed parent/delta')
    if s['confirmation_roster']!=p['confirmation']['physical_roster_by_winner'][a]:raise ValueError('changed confirmation controls')
    if s['alphas']!=[.03,.01,.01]:raise ValueError('changed alpha')
    return s
def jobs(p,stage,selection=None):
    worlds=screen_worlds(p) if stage=='screen' else confirmation_worlds(p)
    return [{'world':w,'arm':a,'assay':assay,'stage':stage} for w in worlds for a in roster(p,stage,selection) for assay in ('lifetime','reuse')]
def job_key(j):return f"{j['world']}_{j['arm']}_{j['assay']}"
