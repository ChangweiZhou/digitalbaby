"""V43 seed-level reporting. Read-only; no policy or physical learner updates."""
from __future__ import annotations
import csv, hashlib, json, math
from collections import defaultdict
from pathlib import Path

POLICIES = ['SUPER','READ-HYBRID','SILENT-HARD','ANCHOR-ONLY','FIXED-50','GLOBAL-BAYES','LOCAL-BAYES-0','LOCAL-BAYES-E2']
LOCAL = 'LOCAL-BAYES-E2'
COMPARATORS = ['SILENT-HARD','FIXED-50','GLOBAL-BAYES','SHUFFLED-ALPHA','SUPER']
LOADS = [64,128,256,512]
SEEDS = list(range(43101,43117))

def _clean(x):
    if isinstance(x,dict): return {str(k):_clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [_clean(v) for v in x]
    if hasattr(x,'item'): return _clean(x.item())
    if isinstance(x,float) and not math.isfinite(x): return None
    return x

def _write_json(path,value):
    path.write_text(json.dumps(_clean(value),indent=2,sort_keys=True,allow_nan=False)+'\n')

def _csv(path, rows, fields=None):
    rows = [_clean(r) for r in rows]
    if fields is None: fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields or ['status'],extrasaction='ignore'); w.writeheader()
        for row in rows:
            w.writerow({k:json.dumps(v,sort_keys=True) if isinstance(v,(list,dict)) else v for k,v in row.items()})

def _num(x):
    return isinstance(x,(float,int)) and not isinstance(x,bool) and math.isfinite(float(x))

def estimate(values):
    """Seed means and Student t intervals; events are never replicates."""
    import numpy as np
    from scipy.stats import t
    vals=np.asarray([float(v) for v in values if _num(v)],dtype=float)
    n=len(vals)
    if not n:return {'n_seeds':0,'mean':None,'sd':None,'se':None,'lo95':None,'hi95':None,'p_two_sided':None}
    mean=float(vals.mean()); sd=float(vals.std(ddof=1)) if n>1 else None
    if n<2: return {'n_seeds':n,'mean':mean,'sd':sd,'se':None,'lo95':None,'hi95':None,'p_two_sided':None}
    se=sd/math.sqrt(n); width=float(t.ppf(.975,n-1))*se
    p=2*float(t.sf(abs(mean/se),n-1)) if se else (1.0 if mean==0 else 0.0)
    return {'n_seeds':n,'mean':mean,'sd':sd,'se':se,'lo95':mean-width,'hi95':mean+width,'p_two_sided':p}

def holm(ps):
    valid=sorted(((i,float(p)) for i,p in enumerate(ps) if _num(p)),key=lambda z:z[1]); out=[None]*len(ps)
    prior=0.
    # A missing planned test does not shrink the preregistered family.
    for j,(i,p) in enumerate(valid): prior=max(prior,min(1.,(len(ps)-j)*p));out[i]=prior
    return out

def _unwrap(record):
    if isinstance(record,dict) and isinstance(record.get('data'),dict): record=record['data']
    if isinstance(record,dict) and isinstance(record.get('result'),dict): record=record['result']
    return record

def _group(r,group,arm):return r.get('groups',{}).get(group,{}).get(arm,{})

def _value(r,arm,metric):
    m=r.get('metrics_by_arm',{}).get(arm,{})
    if _num(m.get(metric)):return float(m[metric])
    mapping={'whole_bpb':('whole','bpb'),'G_bpb':('route/G','bpb'),'G_permutable_bpb':('G_permutable','bpb'),
             'A_bpb':('route/A','bpb'),'T_bpb':('route/T','bpb'),'target_bpb':('target','bpb'),
             'nontarget_bpb':('nontarget','bpb'),'whole_accuracy':('whole','accuracy')}
    if metric in mapping:
        group,key=mapping[metric]; g=_group(r,group,arm)
        if g.get('n',0)>0:
            if _num(g.get(key)):return float(g[key])
            if key=='bpb' and _num(g.get('loss_sum')):return g['loss_sum']/g['n']
            if key=='accuracy' and _num(g.get('correct')):return g['correct']/g['n']
    return None

def _accuracy(a):
    if _num(a):return float(a)
    if isinstance(a,dict):
        for k in ['accuracy','value','rate']:
            if _num(a.get(k)):return float(a[k])
        if a.get('n',a.get('denominator',0))>0 and _num(a.get('correct',a.get('numerator'))):
            return a.get('correct',a.get('numerator'))/a.get('n',a.get('denominator'))
    return None

def _valid(r):
    a=r.get('audits',{})
    q=a.get('physical_acquisition_current')
    # The final protocol requires the unchanged SLOTS path on current-wave
    # aliases0/1 at EVERY wave checkpoint. Terminal success alone is insufficient.
    if isinstance(q,list):
        by={x.get('checkpoint'):x for x in q}
        return len(q)==4 and all(k in by and by[k].get('population_n',0)>0 and
                   _num(by[k].get('population_accuracy')) and by[k]['population_accuracy']>=.9
                   for k in ['after_A','after_B','after_C','after_D'])
    if isinstance(q,dict):
        by=q.get('by_checkpoint',q.get('checkpoints'))
        if isinstance(by,dict):
            expected=['after_A','after_B','after_C','after_D']
            return all(k in by and _accuracy(by[k]) is not None and _accuracy(by[k])>=.9 and
                       by[k].get('n',by[k].get('denominator',0))>0 for k in expected)
        if 'passed' in q and 'minimum_checkpoint_accuracy' in q and q.get('checkpoint_count')==4:
            return q['passed'] is True and q['minimum_checkpoint_accuracy']>=.9 and q.get('minimum_denominator',0)>0
    return False

def _paired(records,load,arm,metric,valid_only=True):
    used=[]
    for r in records:
        if r.get('load')!=load or (valid_only and not _valid(r)):continue
        x,y=_value(r,LOCAL,metric),_value(r,arm,metric)
        if x is not None and y is not None: used.append({'seed':r['seed'],'difference':x-y})
    return {**estimate([x['difference'] for x in used]),'paired_seed_effects':used}

def _gate(status,description,evidence=None):return {'status':status,'criterion':description,'evidence':evidence or {}}

def _pass(ok):return 'pass' if ok else 'fail'

def _complete(records,load,valid=True):
    found=[r for r in records if r.get('load')==load and (not valid or _valid(r))]
    return {r.get('seed') for r in found}==set(SEEDS) and len(found)==len(SEEDS)

def _condition_stat(stat,predicate):return stat['n_seeds']==16 and stat['mean'] is not None and predicate(stat)

def _qualify(root,plan,records):
    engineering={}
    if (root/'ENGINEERING.json').exists():engineering=json.loads((root/'ENGINEERING.json').read_text())
    full=all(_complete(records,l,False) for l in LOADS)
    valid_primary=all(_complete(records,l) for l in [128,256])
    invalid=[{'seed':r['seed'],'load':r['load'],'mature_acquisition':r.get('audits',{}).get('mature_acquisition_current'),
              'physical_acquisition':r.get('audits',{}).get('physical_acquisition_current')} for r in records if not _valid(r)]
    scientific_finite=all(all(_value(r,p,'whole_bpb') is not None for p in POLICIES) for r in records)
    route_ok=all(sum(r.get('audits',{}).get('route_counts',{}).values())==r.get('byte_events',-1) for r in records)
    # These counters represent successful assertions inside the frozen engine.
    firewall_ok=bool(records) and all(r.get('audits',{}).get('firewall_checks',0)==r.get('byte_events',-1) and
                                    r.get('audits',{}).get('probe_checks',0)>0 for r in records)
    eng_hash=hashlib.sha256(json.dumps({k:v for k,v in engineering.items() if k!='sha256'},sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest() if engineering else None
    eng_checks=engineering.get('checks',{})
    eng_runs=engineering.get('runs',[])
    eng_seeds={r.get('seed') for r in eng_runs}
    engineering_ok=engineering.get('passed') is True and engineering.get('sha256')==eng_hash and all(eng_checks.get('E'+str(i)) is True for i in range(9)) and \
        len(eng_runs)==4 and len(eng_seeds)==2 and not (eng_seeds & set(SEEDS)) and {(r.get('seed'),r.get('load')) for r in eng_runs}=={(s,l) for s in eng_seeds for l in [64,256]} and \
        all(r.get('passed') is True and r.get('full_length') is True and r.get('byte_checks',0)>0 and r.get('terminal_probe_checks',0)>0 for r in eng_runs) and \
        engineering.get('source_sha256')==plan.get('source_sha256') and engineering.get('identity',{}).get('protocol_sha256')==plan.get('protocol_sha256')
    status='not_evaluable' if not full or not engineering else _pass(engineering_ok and valid_primary and scientific_finite and route_ok and firewall_ok)
    gates={'Q0':_gate(status,'E0–E8 engineering pass; unchanged SLOTS accuracy≥90% on all intended current-wave learned aliases at each wave checkpoint; finite outputs and exhaustive routes.',
                        {'engineering_passed':engineering.get('passed'),'engineering_evidence_validated':engineering_ok,'primary_cells_all_valid':valid_primary,'all_64_complete':full,
                         'finite_policy_losses':scientific_finite,'route_partition':route_ok,'science_firewall_and_probe_assertions_present':firewall_ok,'physical_capacity_invalid':invalid})}
    required=all(_complete(records,l) for l in [128,256])
    q1={};q2={};q3={}
    for load in [128,256]:
        rows=[r for r in records if r['load']==load]
        ns=[_group(r,'route/G','SILENT-HARD').get('n',0) for r in rows]
        head=[_value(r,'SILENT-HARD','G_bpb')-_value(r,'EVENT-ORACLE','G_bpb') for r in rows
              if _value(r,'SILENT-HARD','G_bpb') is not None and _value(r,'EVENT-ORACLE','G_bpb') is not None]
        q1[str(load)]={'minimum_G_events':min(ns) if ns else None,'headroom_bits_per_G':estimate(head)}
        anchor=[_value(r,'SILENT-HARD','A_bpb')-_value(r,'SUPER','A_bpb') for r in rows
                if _value(r,'SILENT-HARD','A_bpb') is not None and _value(r,'SUPER','A_bpb') is not None]
        q2[str(load)]={'anchor_slow_minus_fast':estimate(anchor),'worst_seed_disadvantage':max(anchor) if anchor else None}
    q1ok=required and all(v['minimum_G_events'] is not None and v['minimum_G_events']>=500 and
                         v['headroom_bits_per_G']['n_seeds']==16 and v['headroom_bits_per_G']['mean']>=.20 for v in q1.values())
    gates['Q1']=_gate(_pass(q1ok) if required else 'not_evaluable','At loads128 and256 each seed has ≥500 G events; mean hard-minus-event-oracle ≥0.20 bits/G.',q1)
    q2means=[v['anchor_slow_minus_fast']['mean'] for v in q2.values()]
    q2ok=required and all(v['anchor_slow_minus_fast']['n_seeds']==16 and v['worst_seed_disadvantage']<=.05 for v in q2.values()) and all(v is not None for v in q2means) and sum(q2means)/2<=0
    q2['mean_of_32_seed_load_means']=sum(q2means)/2 if all(v is not None for v in q2means) else None
    gates['Q2']=_gate(_pass(q2ok) if required else 'not_evaluable','Across128 and256, mean of32 seed/load anchor slow-minus-fast means ≤0; each seed/load disadvantage ≤+0.05 bits/A.',q2)
    q3loads=[l for l in [128,256] if _complete(records,l)]
    for load in q3loads:
        rows=[r for r in records if r['load']==load]
        dwhole=[_value(r,'SILENT-HARD','whole_bpb')-_value(r,'READ-HYBRID','whole_bpb') for r in rows]
        de1=[_value(r,'SILENT-HARD','E1_all')-_value(r,'READ-HYBRID','E1_all') for r in rows
             if _value(r,'SILENT-HARD','E1_all') is not None and _value(r,'READ-HYBRID','E1_all') is not None]
        q3[str(load)]={'whole_bpb_silent_minus_read':estimate(dwhole),'E1_silent_minus_read':estimate(de1)}
    q3ok=required and bool(q3) and engineering_ok and all(v['whole_bpb_silent_minus_read']['mean']<=1e-12 and
            v['E1_silent_minus_read']['n_seeds']==16 and v['E1_silent_minus_read']['mean']>=-.02 for v in q3.values())
    gates['Q3']=_gate(_pass(q3ok) if required and engineering else 'not_evaluable','SILENT-HARD whole loss does not worsen and E1 falls ≤2pp separately at128 and256; consolidation invariant is certified by shared physical trajectory and engineering.',q3)
    p4=_paired(records,256,'SILENT-HARD','whole_bpb')
    gates['Q4']=_gate(_pass(_condition_stat(p4,lambda x:x['mean']<=-.03 and x['hi95']<0)) if _complete(records,256) else 'not_evaluable',
                      'Load256 LOCAL−SILENT-HARD whole bpb ≤−0.03; paired95% upper bound <0.',p4)
    p5=_paired(records,256,'SILENT-HARD','A_after_D_heldout')
    gates['Q5']=_gate(_pass(_condition_stat(p5,lambda x:x['mean']>=-.02 and x['lo95']>-.05)) if _complete(records,256) else 'not_evaluable',
                      'Load256 durable heldout LOCAL−SILENT-HARD ≥−2pp; paired95% lower bound >−5pp.',p5)
    p6=_paired(records,256,'SHUFFLED-ALPHA','G_permutable_bpb')
    cover=[]
    for r in records:
        if r['load']==256:
            n=_group(r,'route/G',LOCAL).get('n',0); m=_group(r,'G_permutable',LOCAL).get('n',0)
            cover.append({'seed':r['seed'],'G_events':n,'permutable_events':m,'coverage':m/n if n else None})
    p6['coverage_by_seed']=cover
    q6ok=_condition_stat(p6,lambda x:x['mean']<=-.05 and x['hi95']<0) and len(cover)==16 and all(x['coverage'] is not None and x['coverage']>=.8 for x in cover)
    gates['Q6']=_gate(_pass(q6ok) if _complete(records,256) else 'not_evaluable','Load256 matched permutable G loss LOCAL−SHUFFLED ≤−0.05, paired95% upper <0; coverage≥80% in EACH seed.',p6)
    simplicity={};q7_evaluable=required
    for arm in ['FIXED-50','GLOBAL-BAYES']:
        checks={};matches=required
        for load in [128,256]:
            w=_paired(records,load,arm,'whole_bpb');g=_paired(records,load,arm,'G_bpb');h=_paired(records,load,arm,'A_after_D_heldout')
            # Difference is LOCAL minus simpler. Better simpler policies count as matching.
            evaluable=w['n_seeds']==16 and g['n_seeds']==16 and h['n_seeds']==16
            q7_evaluable=q7_evaluable and evaluable
            ok=evaluable and -w['mean']<=.01 and -g['mean']<=.02 and h['mean']<=.02
            checks[str(load)]={'LOCAL_minus_simpler_whole':w,'LOCAL_minus_simpler_G':g,'LOCAL_minus_simpler_heldout':h,'evaluable':evaluable,'matches':ok};matches=matches and ok
        simplicity[arm]={'matches_at_both_primary_loads':matches,'loads':checks}
    q7ok=q7_evaluable and not any(v['matches_at_both_primary_loads'] for v in simplicity.values())
    gates['Q7']=_gate(_pass(q7ok) if q7_evaluable else 'not_evaluable','Complexity earns itself only if neither FIXED-50 nor GLOBAL is within one-sided margins 0.01 whole bpb,0.02 G bpb,2pp heldout at BOTH128 and256. All endpoint pairs required. Means, not equivalence-test claims.',simplicity)
    captures=estimate([_value(r,LOCAL,'oracle_capture') for r in records if r['load']==256 and _valid(r)])
    gates['Q8']=_gate(_pass(captures['n_seeds']==16 and captures['mean']>=.5) if _complete(records,256) else 'not_evaluable',
                      'Load256 mean seed-level LOCAL EVENT-ORACLE headroom capture ≥50%; undefined zero-headroom seeds do not pass.',captures)
    p9=_paired(records,256,'SUPER','whole_bpb')
    gates['Q9']=_gate(_pass(p9['n_seeds']==16 and p9['mean']<=.05 and gates['Q5']['status']=='pass') if _complete(records,256) else 'not_evaluable',
                      'Load256 LOCAL whole bpb ≤SUPER+0.05 and Q5 passes.',p9)
    a128=_paired(records,128,'SILENT-HARD','whole_bpb');a256=p4
    rel=[];by={(r['seed'],r['load']):r for r in records if _valid(r)}
    for seed in SEEDS:
        if (seed,128) in by and (seed,256) in by:
            r1,r2=by[seed,128],by[seed,256]
            vals=[_value(r2,LOCAL,'A_after_D_heldout'),_value(r1,LOCAL,'A_after_D_heldout'),_value(r2,'SILENT-HARD','A_after_D_heldout'),_value(r1,'SILENT-HARD','A_after_D_heldout')]
            if all(v is not None for v in vals):rel.append((vals[0]-vals[1])-(vals[2]-vals[3]))
    degradation=estimate(rel)
    q10_evaluable=required and a128['n_seeds']==16 and a256['n_seeds']==16 and degradation['n_seeds']==16
    q10ok=q10_evaluable and a128['mean']<=0 and a256['mean']<=0 and degradation['mean']>=-.02
    gates['Q10']=_gate(_pass(q10ok) if q10_evaluable else 'not_evaluable','LOCAL whole-loss advantage remains nonpositive at128 and256; relative heldout change is no worse than SILENT-HARD by >2pp.',
                       {'whole_LOCAL_minus_hard_128':a128,'whole_LOCAL_minus_hard_256':a256,'relative_heldout_change_256_minus128':degradation})
    return gates

def _decision(gates):
    p=lambda k:gates[k]['status']=='pass'
    if any(gates[k]['status']=='not_evaluable' for k in ['Q0','Q1']):return 'incomplete_or_not_evaluable'
    if not p('Q0'):return 'physical_or_engineering_validity_not_established'
    if not p('Q1'):return 'F: insufficient arbitration headroom'
    if all(p('Q'+str(i)) for i in range(7)) and p('Q7') and p('Q8'):
        return 'A: local specialist arbitration established; '+('end-to-end target met' if p('Q9') else 'architecture gap remains')
    simple=gates['Q7']['evidence']
    if simple.get('FIXED-50',{}).get('matches_at_both_primary_loads'):return 'C: fixed mixture matches LOCAL within practical margins; prefer FIXED-50'
    if simple.get('GLOBAL-BAYES',{}).get('matches_at_both_primary_loads'):return 'B: global scalar matches LOCAL within practical margins; prefer GLOBAL-BAYES'
    if p('Q4') and not p('Q6'):return 'E: prediction gain without established object/event specificity'
    return 'not_established: inspect failed gates; no qualified local-specialist claim'

def _load_model(records,metric):
    """Seed fixed effects and policy×log2(load), with seed-cluster sandwich covariance."""
    import numpy as np
    from scipy.stats import t,f
    rows=[]
    eligible_loads=[l for l in LOADS if _complete(records,l)]
    for r in records:
        if not _valid(r) or r.get('load') not in eligible_loads:continue
        for arm in POLICIES:
            val=_value(r,arm,metric)
            if val is not None:rows.append((r['seed'],r['load'],arm,val))
    seeds=sorted({z[0] for z in rows});loads=sorted({z[1] for z in rows});arms=[LOCAL]+[a for a in POLICIES if a!=LOCAL]
    if len(seeds)<3 or len(loads)<2:return {'status':'not_evaluable','metric':metric,'reason':'Need≥2 complete, fully valid16-seed loads; invalid-load survivors remain descriptive.'}
    names=['intercept']+['seed_'+str(s) for s in seeds[1:]]+['policy_'+a for a in arms[1:]]+['log2_load_centered_128']+['interaction_'+a for a in arms[1:]]
    X=[];y=[];cl=[]
    for seed,load,arm,val in rows:
        x=math.log2(load/128)
        X.append([1]+[int(seed==s) for s in seeds[1:]]+[int(arm==a) for a in arms[1:]]+[x]+[x*int(arm==a) for a in arms[1:]])
        y.append(val);cl.append(seed)
    X=np.asarray(X,float);y=np.asarray(y);n,p=X.shape
    if np.linalg.matrix_rank(X)<p:return {'status':'not_evaluable','metric':metric,'reason':'Rank deficient seed-blocked design.'}
    bread=np.linalg.inv(X.T@X);beta=bread@X.T@y;res=y-X@beta;meat=np.zeros((p,p))
    for seed in seeds:
        ix=np.asarray(cl)==seed;s=X[ix].T@res[ix];meat+=np.outer(s,s)
    cov=bread@meat@bread*(len(seeds)/(len(seeds)-1))*((n-1)/(n-p));se=np.sqrt(np.maximum(0,np.diag(cov)));crit=float(t.ppf(.975,len(seeds)-1))
    idx=list(range(p-(len(arms)-1),p));b=beta[idx];v=cov[np.ix_(idx,idx)];rank=int(np.linalg.matrix_rank(v))
    planned_df=len(idx)
    statistic=float(b@np.linalg.solve(v,b)/planned_df) if rank==planned_df else None
    coeff=[{'term':name,'estimate':float(beta[i]),'se_cluster_seed':float(se[i]),'lo95':float(beta[i]-crit*se[i]),'hi95':float(beta[i]+crit*se[i])} for i,name in enumerate(names) if not name.startswith('seed_')]
    return {'status':'ok','metric':metric,'n_seed_clusters':len(seeds),'n_seed_policy_load_means':n,'valid_loads':loads,'reference_policy':LOCAL,
            'model':'seed fixed effects + policy + log2(load/128) + policy:log2(load/128)',
            'covariance':'seed-cluster sandwich, finite-cluster correction; t/F reference with n_seeds−1 denominator df; exploratory; only complete valid loads enter model',
            'interaction_test':{'status':'ok' if statistic is not None else 'not_evaluable_singular_cluster_covariance','F':statistic,'df1':planned_df,'covariance_rank':rank,
                                'df2':len(seeds)-1,'p':float(f.sf(statistic,planned_df,len(seeds)-1)) if statistic is not None else None},'coefficients':coeff}

def _l50(loads,values):
    if len(loads)<2 or any(v is None for v in values):return None
    crossings=[]
    for i,v in enumerate(values):
        if v==.5:crossings.append(float(loads[i]))
        if i and (values[i-1]-.5)*(v-.5)<0:
            a,b=math.log2(loads[i-1]),math.log2(loads[i]);crossings.append(2**(a+(.5-values[i-1])*(b-a)/(v-values[i-1])))
    return crossings[0] if len(crossings)==1 else None

def _bootstrap_loads(records,reps=2000):
    import numpy as np
    valid={(r['seed'],r['load']):r for r in records if _valid(r)}
    loads=[l for l in LOADS if _complete(records,l)]
    seeds=[s for s in SEEDS if all((s,l) in valid for l in loads)] if loads else []
    if len(seeds)<2 or len(loads)<2:return {'status':'not_evaluable','reason':'Need≥2 fully valid observed loads and≥2 complete seed blocks.'}
    rng=np.random.default_rng(430043);draws=rng.integers(0,len(seeds),size=(reps,len(seeds)))
    out={'status':'ok','repetitions':reps,'random_seed':430043,'n_seed_blocks':len(seeds),'loads':loads,'interpretation':'Nonparametric seed-block bootstrap frequencies; not Bayesian posterior probabilities; all loads preserved in each resampled seed.','policies':{}}
    for arm in POLICIES:
        vals=np.array([[_value(valid[s,l],arm,'A_after_D_heldout') for l in loads] for s in seeds],float)
        observed=vals.mean(0);boot=vals[draws].mean(1);points=[_l50(loads,row.tolist()) for row in boot];points=[p for p in points if p is not None]
        out['policies'][arm]={'L50_observed':_l50(loads,observed.tolist()),'L50_rule':'Single crossing of 50% in observed valid load range; log2-load interpolation; outside range or multiple crossings => NA.',
                             'L50_bootstrap_identifiable_fraction':len(points)/reps,'L50_conditional_lo95':float(np.quantile(points,.025)) if points else None,
                             'L50_conditional_hi95':float(np.quantile(points,.975)) if points else None}
    for metric in ['whole_bpb','G_bpb','A_after_D_heldout']:
        vals=np.array([[[_value(valid[s,l],arm,metric) for l in loads] for arm in POLICIES] for s in seeds],float)
        if not np.isfinite(vals).all():out[metric]={'status':'not_evaluable','reason':'Missing policy metric.'};continue
        boot=vals[draws].mean(1);sign=-1 if metric.endswith('heldout') else 1;oriented=boot*sign
        means=oriented.mean(2);best=means.min(1)
        out[metric]={'policy_bootstrap_frequency_tied_for_best_mean_over_loads':{arm:float(np.mean(np.isclose(means[:,i],best,atol=1e-12,rtol=0))) for i,arm in enumerate(POLICIES)},
                     'LOCAL_pairwise_dominance_at_every_observed_load':{arm:float(np.mean(np.all(oriented[:,POLICIES.index(LOCAL),:]<=oriented[:,i,:]+1e-12,axis=1))) for i,arm in enumerate(POLICIES) if arm!=LOCAL}}
    return out

def _event_columns(root,r):
    import numpy as np
    needed=['route','pF_true','pS_true','logpF_true','logpS_true','local_w_before','local_E_before','fast_slow_agree','slot_age','support','generation_uid','slot_uid']
    if r.get('authority_events_file'):
        path=root/r['authority_events_file']
        expected=r.get('authority_events_sha256')
        if expected and hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError('Authority event file hash mismatch: '+str(path))
        with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in needed if k in z.files}
    events=r.get('authority_events',[])
    return {k:np.asarray([e.get(k) for e in events]) for k in needed if k in events[0]} if events else {}


def _event_summaries(root,records):
    import numpy as np
    support=[];age=[];slotrows=[];oracles=[]
    for r in records:
        cols=_event_columns(root,r)
        if cols:
            gm=cols['route'].astype(str)=='G'
            g={k:v[gm] for k,v in cols.items()}
            n=len(g['route'])
            if n:
                ages=np.asarray(g['slot_age'],float);cuts=np.quantile(ages,[.25,.5,.75])
                weights=np.asarray(g['local_w_before'],float)
                fast=-np.asarray(g['logpF_true'],float)/math.log(2) if 'logpF_true' in g else -np.log2(np.maximum(np.asarray(g['pF_true'],float),1e-300))
                slow=-np.asarray(g['logpS_true'],float)/math.log(2) if 'logpS_true' in g else -np.log2(np.maximum(np.asarray(g['pS_true'],float),1e-300))
                supports=np.asarray(g['support'],int)
                support_labels=np.where(supports==2,'2',np.where((supports>=3)&(supports<=4),'3–4',np.where(supports>=5,'5+','below2')))
                age_labels=np.asarray(['Q'+str(i+1) for i in np.searchsorted(cuts,ages,side='right')])
                for kind,labels,out in [('support',support_labels,support),('age',age_labels,age)]:
                    for key in np.unique(labels):
                        mask=labels==key
                        out.append({'seed':r['seed'],'load':r['load'],'bucket':str(key),'n_events':int(mask.sum()),
                                    'authority_mass':float(weights[mask].sum()),'mean_authority':float(weights[mask].mean()),
                                    'fast_bpb':float(fast[mask].mean()),'slow_bpb':float(slow[mask].mean()),
                                    'fast_slow_agreement':float(np.asarray(g['fast_slow_agree'],float)[mask].mean()) if 'fast_slow_agree' in g else None,
                                    'E2_evidence_mean':float(np.asarray(g['local_E_before'],float)[mask].mean()) if 'local_E_before' in g else None,
                                    'E2_evidence_p05_p50_p95':np.quantile(np.asarray(g['local_E_before'],float)[mask],[.05,.5,.95]).tolist() if 'local_E_before' in g else None,
                                    'age_quartile_boundaries':cuts.tolist() if kind=='age' else None})
                ids=np.asarray(g.get('generation_uid',g.get('slot_uid'))).astype(str)
                unique,inverse=np.unique(ids,return_inverse=True)
                ns=np.bincount(inverse);fs=np.bincount(inverse,weights=fast);ss=np.bincount(inverse,weights=slow)
                for i,uid in enumerate(unique):slotrows.append({'seed':r['seed'],'load':r['load'],'generation_uid':str(uid),'n':int(ns[i]),
                    'fast_loss':float(fs[i]),'slow_loss':float(ss[i]),'oracle_loss':float(min(fs[i],ss[i])),'chosen_expert':'FAST' if fs[i]<=ss[i] else 'SLOW'})
        n=_group(r,'route/G',LOCAL).get('n',0)
        for arm in POLICIES+['SHUFFLED-ALPHA','EVENT-ORACLE','SLOT-ORACLE']:
            b=_value(r,arm,'G_bpb');hard=_value(r,'SILENT-HARD','G_bpb');oracle=_value(r,'EVENT-ORACLE','G_bpb')
            oracles.append({'seed':r['seed'],'load':r['load'],'policy':arm,'G_events':n,'G_bpb':b,'hard_G_bpb':hard,'event_oracle_G_bpb':oracle,
                            'headroom_bits_per_G':hard-oracle if hard is not None and oracle is not None else None,
                            'capture':(hard-b)/(hard-oracle) if None not in (hard,b,oracle) and hard-oracle>1e-12 else None})
    return support,age,slotrows,oracles


def generate(root,plan,records):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    records=[_unwrap(r) for r in records];records=sorted(records,key=lambda r:(r['seed'],r['load']))
    if len({(r['seed'],r['load']) for r in records})!=len(records):raise ValueError('Duplicate seed/load record: would pseudoreplicate seeds.')
    terminal=[];route=[];curves=[];authority=[]
    metrics=['whole_bpb','G_bpb','target_bpb','nontarget_bpb','whole_accuracy','A_after_D_heldout','E1_all','E2_all','oracle_capture','same_identity_E1_all_delays',
             'A_after_D_seen','after_D_current_seen','after_D_current_heldout','revision_after_D_seen','revision_after_D_heldout']
    metrics.extend('revision_'+str(n)+'_'+kind for n in [1,2,4,8] for kind in ['exact','known','seen','heldout'])
    for r in records:
        for arm,data in r.get('authority_summary',{}).items():authority.append({'seed':r['seed'],'load':r['load'],'policy':arm,**data})
        for arm in POLICIES+['SHUFFLED-ALPHA','EVENT-ORACLE','SLOT-ORACLE']:
            terminal.append({'seed':r['seed'],'load':r['load'],'policy':arm,'physical_capacity_valid':_valid(r),**{m:_value(r,arm,m) for m in metrics}})
        for group in ['route/A','route/T','route/G','route/F']:
            for arm,g in r.get('groups',{}).get(group,{}).items():route.append({'seed':r['seed'],'load':r['load'],'route':group[-1],'policy':arm,**g})
    for load in LOADS:
        for arm in POLICIES+['SHUFFLED-ALPHA']:
            for metric in metrics:
                rs=[r for r in records if r['load']==load];ok=[r for r in rs if _valid(r)]
                curves.append({'load':load,'policy':arm,'metric':metric,'n_physical_cells':len(rs),'n_valid_physical_cells':len(ok),
                               'inference_status':'complete_valid' if _complete(records,load) else 'partial_or_capacity_invalid',
                               **estimate([_value(r,arm,metric) for r in ok])})
    contrasts=[]
    for load in LOADS:
        family=[]
        for comp in COMPARATORS:
            metric='G_permutable_bpb' if comp=='SHUFFLED-ALPHA' else 'whole_bpb'
            stat=_paired(records,load,comp,metric)
            row={'load':load,'contrast':LOCAL+' − '+comp,'metric':metric,'family':'primary_mechanistic_256' if load==256 and comp!='SUPER' else 'exploratory',
                 'load_fully_valid':_complete(records,load),**stat,'holm_p':None}
            contrasts.append(row)
            if comp!='SUPER':family.append(row)
            for secondary in ['G_bpb','A_after_D_heldout']:
                if comp=='SHUFFLED-ALPHA':continue
                contrasts.append({'load':load,'contrast':LOCAL+' − '+comp,'metric':secondary,'family':'secondary_exploratory','load_fully_valid':_complete(records,load),
                                  **_paired(records,load,comp,secondary),'holm_p':None})
        for row,p in zip(family,holm([row['p_two_sided'] for row in family])):row['holm_p']=p;row['holm_family_size']=4
    gates=_qualify(root,plan,records)
    support,age,slotrows,oracles=_event_summaries(root,records)
    shuffled=[{'seed':r['seed'],'load':r['load'],**r.get('audits',{}).get('shuffled',{})} for r in records]
    state=[{'seed':r['seed'],'load':r['load'],'audits':{k:v for k,v in r.get('audits',{}).items() if k not in ['shuffled','revision_resets']},
            'initial_physical_hash':r.get('initial_physical_hash'),'final_physical_hash':r.get('final_physical_hash'),
            'final_physical_components':r.get('final_physical_components'), 'physical_events':r.get('physical_events')} for r in records]
    revision=[{'seed':r['seed'],'load':r['load'],'revision_resets':r.get('audits',{}).get('revision_resets'),
               'ledger_births':r.get('audits',{}).get('ledger_births'),'ledger_evictions':r.get('audits',{}).get('ledger_evictions'),
               'revision_commit_events':[e for e in r.get('ledger_lifecycle',[]) if e.get('event')=='revision_reset'],
               'resident_only_generations':r.get('audits',{}).get('all_slot_ledgers_resident'),
               'E2_captures_equal_promotions':r.get('audits',{}).get('E2_seed_captures_equal_promotions')} for r in records]
    report={'version':'V43','status':'complete' if len(records)==64 and all(_complete(records,l,False) for l in LOADS) else 'partial',
            'completed_physical_trajectories':len(records),'expected_physical_trajectories':64,'independent_seed_roster':SEEDS,
            'physical_capacity_invalid':[{'seed':r['seed'],'load':r['load']} for r in records if not _valid(r)],
            'statistical_unit':'seed; counterfactual policies share one physical history per seed/load; repeated loads are blocked by seed',
            'gate_scope':'Qualification requires all16 valid primary seed cells at128 and256. No qualification from selected surviving seeds. Invalid stress cells excluded from inference and described separately.',
            'gates':gates,'decision':_decision(gates),'load_curves':curves,'planned_contrasts':contrasts,
            'load_models':[_load_model(records,m) for m in ['whole_bpb','G_bpb','A_after_D_heldout']],
            'load_bootstrap':_bootstrap_loads(records),'sample_accounting':{'new_physical_byte_presentations':sum(r.get('byte_events',0) for r in records),
                'target_presentations':sum(r.get('target_presentations',0) for r in records),'unique_complete_records_summed_over_histories':sum(r.get('unique_complete_records',0) for r in records),
                'note':'Counterfactual policies do not multiply physical samples; sum of per-history unique records is not global unique byte-string count.'},
            'limitations':['SHUFFLED-ALPHA is offline G-logloss specificity only; no shuffled accuracy, deployable probe or online-causality claim.',
                          'EVENT/SLOT ORACLE are retrospective diagnostics and use outcomes.',
                          'L50 is interpolated only for an identifiable crossing inside observed valid loads; no extrapolation.',
                          'Q7 uses one-sided practical mean margins, not a statistical equivalence claim; better simple policies count as matches.',
                          'Seed-blocked load fits and bootstrap order frequencies are exploratory; 16 seeds limit inference.',
                          'A successful mechanism test is not natural-language persistent learning or biological equivalence.']}
    _csv(root/'TERMINAL.csv',terminal);_csv(root/'LOAD_CURVES.csv',curves);_csv(root/'ROUTE_SUMMARY.csv',route)
    _csv(root/'AUTHORITY_SUMMARY.csv',authority)
    _csv(root/'AUTHORITY_BY_SUPPORT.csv',support);_csv(root/'AUTHORITY_BY_AGE.csv',age);_csv(root/'SLOT_ORACLE.csv',slotrows)
    _csv(root/'ORACLE_HEADROOM.csv',oracles);_csv(root/'POLICY_CONTRASTS.csv',contrasts)
    _write_json(root/'SHUFFLED_ALPHA_AUDIT.json',shuffled);_write_json(root/'STATE_HASH_AUDIT.json',state);_write_json(root/'REVISION_LEDGER_AUDIT.json',revision)
    _write_json(root/'REPORT.json',report)
    _markdown(root,report)
    if records:_figure(root,curves)
    return report

def _fmt(x,digits=4):return 'NA' if x is None else f'{x:.{digits}f}'

def _markdown(root,report):
    lines=['# V43 fast–slow authority results','',f"**Status: {report['status']} — {report['completed_physical_trajectories']}/64 physical trajectories.**",'',
           f"**Decision:** {report['decision']}",'','All policies were scored on the same physical trajectory within each seed/load. The independent unit is the seed; authority events are not independent replications.','',
           '| Gate | Status | Criterion |','|---|---|---|']
    for k,v in report['gates'].items():lines.append(f"| {k} | {v['status']} | {v['criterion']} |")
    lines+=['','## Load256 planned contrasts','','Effects are LOCAL-BAYES-E2 minus comparator. Lower loss is better; paired intervals use seed-level effects. The first four rows form the Holm-adjusted mechanistic family.','',
            '| Comparator | Endpoint | Seeds | Mean effect | 95% CI | Holm p |','|---|---|---:|---:|---|---:|']
    for comp in COMPARATORS:
        metric='G_permutable_bpb' if comp=='SHUFFLED-ALPHA' else 'whole_bpb'
        row=next(r for r in report['planned_contrasts'] if r['load']==256 and r['contrast']==LOCAL+' − '+comp and r['metric']==metric)
        lines.append(f"| {comp} | {metric} | {row['n_seeds']} | {_fmt(row['mean'])} | [{_fmt(row['lo95'])}, {_fmt(row['hi95'])}] | {_fmt(row['holm_p'])} |")
    lines+=['','## Load curves','','Each entry is the valid-seed mean. Missing or invalid cells are explicit in LOAD_CURVES.csv; they cannot qualify a partially surviving primary load.','',
            '| Load | Policy | Whole bpb | G bpb | Old held-out accuracy | Oracle capture |','|---:|---|---:|---:|---:|---:|']
    for load in LOADS:
        for arm in POLICIES:
            values={m:next(r['mean'] for r in report['load_curves'] if r['load']==load and r['policy']==arm and r['metric']==m) for m in ['whole_bpb','G_bpb','A_after_D_heldout','oracle_capture']}
            lines.append(f"| {load} | {arm} | {_fmt(values['whole_bpb'])} | {_fmt(values['G_bpb'])} | {_fmt(values['A_after_D_heldout'])} | {_fmt(values['oracle_capture'])} |")
    lines+=['','## Statistical interpretation','',
            'The descriptive pressure axis is log2(load/128). Exploratory models include seed fixed effects, policy, load, and policy-by-load interactions; uncertainty uses seed-cluster covariance. Whole loss, G loss and durable held-out success are analyzed separately. The binary/proportion endpoint uses a linear probability contrast model; it is not a calibrated probability extrapolator.',
            '', 'Bootstrap resampling preserves all observed loads within each seed. Dominance entries are bootstrap frequencies, not posterior probabilities. L50 is reported only for a single crossing inside the observed valid range; conditional bootstrap intervals are accompanied by the fraction of resamples in which L50 is identifiable.',
            '', '## Sample accounting','',json.dumps(report['sample_accounting'],indent=2),'','## Interpretation limits','']
    lines+=['- '+x for x in report['limitations']]
    lines+=['','The detailed gate evidence, paired effects, model coefficients and bootstrap results are in REPORT.json. No claim is made from an empty denominator or an unobserved endpoint.','']
    (root/'SUMMARY.md').write_text('\n'.join(lines))

def _figure(root,curves):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,3,figsize=(13.4,4.7),layout='constrained')
    selected=['SUPER','SILENT-HARD','ANCHOR-ONLY','FIXED-50','GLOBAL-BAYES',LOCAL]
    colors={'SUPER':'#555555','SILENT-HARD':'#be4b45','ANCHOR-ONLY':'#8e6b9c','FIXED-50':'#e29b32','GLOBAL-BAYES':'#2d977a',LOCAL:'#215baa'}
    for ax,metric,title in zip(axs,['whole_bpb','G_bpb','A_after_D_heldout'],['Whole-stream loss','Supported similarity loss','Durable old held-out recall']):
        for arm in selected:
            rows=[r for r in curves if r['policy']==arm and r['metric']==metric and r['mean'] is not None]
            x=[r['load'] for r in rows];y=[r['mean'] for r in rows]
            ax.plot(x,y,'o-',label=arm,color=colors[arm],lw=2 if arm==LOCAL else 1.35,ms=4)
            good=[r for r in rows if r['lo95'] is not None]
            if good:ax.fill_between([r['load'] for r in good],[r['lo95'] for r in good],[r['hi95'] for r in good],color=colors[arm],alpha=.08)
        ax.set_xscale('log',base=2);ax.set_xticks(LOADS,labels=[str(x) for x in LOADS]);ax.set_xlabel('Introduced identities (load)');ax.set_title(title,fontsize=11);ax.grid(alpha=.18)
        ax.set_ylabel('Accuracy' if metric.endswith('heldout') else 'Bits per byte')
        if metric.endswith('heldout'):ax.set_ylim(-.02,1.02)
    handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=3,fontsize=8,frameon=False)
    fig.suptitle('V43 · Shared physical histories, counterfactual read authority',fontsize=14)
    fig.savefig(root/'V43_results.png',dpi=180);plt.close(fig)

def self_test():
    checks=[]
    a=estimate([1,2,3,4]);assert a['n_seeds']==4 and a['mean']==2.5 and a['lo95']<2.5<a['hi95'];checks.append('paired_seed_t_interval')
    assert estimate([])['mean'] is None and estimate([2])['lo95'] is None;checks.append('empty_and_one_seed_not_confident')
    assert holm([.01,.04,.02,.9])==[.04,.08,.06,.9];checks.append('holm_four_planned_contrasts')
    assert holm([.01,None,None,None])==[.04,None,None,None];checks.append('missing_test_does_not_shrink_Holm_family')
    assert _l50([64,128,256],[.9,.7,.6]) is None;assert _l50([64,128],[.6,.4])==math.sqrt(64*128);checks.append('L50_no_extrapolation')
    assert _l50([64,128,256],[.6,.4,.7]) is None;checks.append('L50_multiple_crossings_unidentified')
    assert not _valid({'audits':{}});checks.append('missing_acquisition_no_vacuous_validity')
    assert not _valid({'audits':{'physical_acquisition_current':[{'checkpoint':'after_'+c,'population_n':0,'population_accuracy':1.0} for c in 'ABCD']}});checks.append('empty_acquisition_denominator_not_valid')
    import tempfile
    # A complete physical roster with missing policy outcomes must not establish
    # that local complexity is necessary, or qualify scaling on surviving pairs.
    fixture=[]
    for seed in SEEDS:
        for load in [128,256]:
            fixture.append({'seed':seed,'load':load,'byte_events':1,'metrics_by_arm':{a:{'whole_bpb':2.,'G_bpb':2.,'A_after_D_heldout':.9,'E1_all':.9} for a in POLICIES},
                            'audits':{'physical_acquisition_current':[{'checkpoint':'after_'+c,'population_n':2,'population_accuracy':1.} for c in 'ABCD']}})
    with tempfile.TemporaryDirectory() as directory:
        baseline=_qualify(Path(directory),{},fixture);assert baseline['Q7']['status']=='fail'
        del fixture[0]['metrics_by_arm']['GLOBAL-BAYES']['G_bpb']
        missing=_qualify(Path(directory),{},fixture);assert missing['Q7']['status']=='not_evaluable';checks.append('missing_simpler_endpoint_cannot_earn_complexity')
        del fixture[0]['metrics_by_arm'][LOCAL]['whole_bpb']
        missing=_qualify(Path(directory),{},fixture);assert missing['Q10']['status']=='not_evaluable';checks.append('scaling_requires_all_seed_pairs')
    return {'passed':True,'checks':checks}
