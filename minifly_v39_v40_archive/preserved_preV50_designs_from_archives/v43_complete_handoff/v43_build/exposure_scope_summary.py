#!/usr/bin/env python3
"""Audit V43 exposure populations from saved counts/routes, without policy ranking.

Default requires all64 locked scientific cells. For a distinct engineering fixture:
  python3 exposure_scope_summary.py --results PATH --engineering-smoke
This does not run a learner, compute policy effects, or introduce qualification gates.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

SEEDS=list(range(43101,43117))
LOADS=[64,128,256,512]
POLICIES={'SUPER','READ-HYBRID','SILENT-HARD','ANCHOR-ONLY','FIXED-50','GLOBAL-BAYES','LOCAL-BAYES-0','LOCAL-BAYES-E2'}
ROUTES='ATGF'

def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def sha(x):return hashlib.sha256(x if isinstance(x,bytes) else canonical(x)).hexdigest()

def count_only(group):
    """Read counts only, requiring the eight online counterfactuals to agree."""
    counts=[group[a]['n'] for a in sorted(POLICIES) if a in group]
    if len(counts)!=len(POLICIES) or len(set(counts))!=1:raise ValueError('Missing/disagreeing shared physical event counts')
    if not isinstance(counts[0],int) or counts[0]<0:raise ValueError('Invalid count')
    return counts[0]

def checked_cell(root,plan,job):
    filename=f"seed_{job['seed']}_load_{job['load']}.json"
    path=root/'completed'/filename
    env=json.loads(path.read_bytes())
    if env.get('identity')!={'plan_sha256':sha(plan),'job':job}:raise ValueError('Completed identity mismatch: '+filename)
    result=env.get('data')
    if env.get('sha256')!=sha(result):raise ValueError('Completed envelope checksum mismatch: '+filename)
    if (result.get('seed'),result.get('load'))!=(job['seed'],job['load']) or result.get('physical_trajectories')!=1:raise ValueError('Physical cell identity mismatch')
    events=(root/result['authority_events_file']).resolve()
    if not events.is_relative_to(root.resolve()):raise ValueError('Authority-event path escapes result directory')
    if sha(events.read_bytes())!=result.get('authority_events_sha256'):raise ValueError('Authority-event checksum mismatch')
    columns=result.get('authority_events_columns',[])
    with np.load(events,allow_pickle=False) as z:
        if set(z.files)!=set(columns)|{'__row_count__'}:raise ValueError('Authority-event column roster mismatch')
        n=int(z['__row_count__'])
        if n!=result['authority_events_count'] or any(len(z[k])!=n for k in columns):raise ValueError('Authority-event row count mismatch')
        if not {'route','category','seed','load','event_index'}<=set(columns):raise ValueError('Required exposure-scope fields absent')
        if n and (not np.all(z['seed']==job['seed']) or not np.all(z['load']==job['load'])):raise ValueError('Authority-event cell identity mismatch')
        event_index=z['event_index']
        if len(np.unique(event_index))!=n:raise ValueError('Duplicated online authority event index')
        scope={k:z[k].copy() for k in ['route','category','event_index']}
        # Stage fields are read only if actually saved. Current frozen schema
        # has none, so G-by-stage is marked unavailable rather than inferred.
        for key in ['stage','wave','kind']:
            if key in z.files:scope[key]=z[key].copy()
    return result,scope,{'completed_file':str(path.relative_to(root)),'completed_file_sha256':sha(path.read_bytes()),
                         'completed_data_sha256':env['sha256'],'authority_events_file':result['authority_events_file'],
                         'authority_events_sha256':result['authority_events_sha256']}

def probe_routes(probes):
    """Collapse repeated policy rows for a single evaluator query; ignore outcomes."""
    if not probes:return {'status':'unavailable','reason':'No saved probe rows.'}
    required={'checkpoint','kind','key_id','alias','arm','route'}
    if any(not required<=set(p) for p in probes):return {'status':'unavailable','reason':'Saved probe rows lack an explicit query identity or route.'}
    queries={}
    for p in probes:
        key=(p['checkpoint'],p['kind'],p['key_id'],p['alias'])
        if p['arm'] not in POLICIES:continue
        if p['route'] not in ROUTES:raise ValueError('Unknown saved probe route')
        q=queries.setdefault(key,{'route':p['route'],'arms':set()})
        if q['route']!=p['route']:raise ValueError('Policy-dependent probe route')
        if p['arm'] in q['arms']:raise ValueError('Duplicate policy row within probe query')
        q['arms'].add(p['arm'])
    if any(q['arms']!=POLICIES for q in queries.values()):raise ValueError('Missing policy rows for saved probe query')
    grouped=defaultdict(Counter)
    for (checkpoint,kind,key_id,alias),q in queries.items():grouped[(checkpoint,kind,alias)][q['route']]+=1
    rows=[]
    for (checkpoint,kind,alias),counts in sorted(grouped.items()):
        rows.append({'checkpoint':checkpoint,'kind':kind,'alias':alias,'unique_queries':sum(counts.values()),'route_counts':{r:counts[r] for r in ROUTES}})
    durable=[x for x in rows if x['kind']=='untouched_A_heldout' and x['alias']==2]
    return {'status':'available','saved_policy_rows':len(probes),'unique_evaluator_queries':len(queries),
            'all_probe_query_routes':rows,'durable_old_heldout_query_routes':durable,
            'deduplication_key':['checkpoint','kind','key_id','alias'],
            'note':'Eight policy rows describe one physical evaluator query. Delayed queries reuse identities and are not independent replications. No probe outcome or policy score is analyzed.'}

def cell_scope(result,cols,provenance):
    routes=cols['route'].astype(str);categories=cols['category'].astype(str)
    if not set(routes)<=set('ATG') or not set(categories)<={'target','nontarget'}:raise ValueError('Unexpected online authority route/category')
    groups=result['groups'];gmask=routes=='G';ng=int(gmask.sum())
    expected_g=count_only(groups['route/G']) if 'route/G' in groups else 0
    if ng!=expected_g:raise ValueError('G-event count does not reconcile with completed aggregates')
    route_counts={r:int(np.sum(routes==r)) for r in 'ATG'}
    for r,n in route_counts.items():
        if n!=result['audits']['route_counts'][r]:raise ValueError('Authority-route count does not reconcile')
    total=count_only(groups['whole']);target=count_only(groups['target']);ordinary=count_only(groups['nontarget'])
    if total!=result['byte_events'] or target+ordinary!=total:raise ValueError('Stream category count does not reconcile')
    g_target=int(np.sum(gmask & (categories=='target')));g_ordinary=int(np.sum(gmask & (categories=='nontarget')))
    stage_rows=[]
    for name,g in sorted(groups.items()):
        parts=name.split('/')
        if len(parts)==3 and parts[-1] in ('target','nontarget') and parts[0] not in ('route','stage'):
            stage_rows.append({'saved_group_name':name,'wave':parts[0],'record_kind':parts[1],'category':parts[2],'all_route_event_count':count_only(g)})
    if stage_rows and sum(x['all_route_event_count'] for x in stage_rows)!=total:raise ValueError('Saved stage/category groups do not partition all events')
    gstage=[]
    if 'stage' in cols:
        counter=Counter((str(s),str(c)) for s,c in zip(cols['stage'][gmask],categories[gmask]))
        gstage=[{'stage':s,'category':c,'G_events':n} for (s,c),n in sorted(counter.items())]
    return {'seed':result['seed'],'load':result['load'],'provenance':provenance,
            'physical_byte_events':total,'target_byte_events':target,'nontarget_byte_events':ordinary,
            'online_authority_route_counts':route_counts,'online_G_events':ng,'online_G_target_events':g_target,
            'online_G_nontarget_events':g_ordinary,'online_G_target_fraction':g_target/ng if ng else None,
            'online_G_nontarget_fraction':g_ordinary/ng if ng else None,
            'online_G_stage_status':'available_saved_field' if gstage else 'unavailable',
            'online_G_stage_counts':gstage,
            'online_G_stage_note':None if gstage else 'Authority rows do not save a stage label. Saved wave/kind/category aggregates combine all routes; they cannot be presented as a G-specific stage breakdown.',
            'all_route_stage_category_counts':stage_rows,'probe_scope':probe_routes(result.get('probes',[]))}

def atomic_text(path,text):
    fd,name=tempfile.mkstemp(prefix=path.name+'.tmp-',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as f:f.write(text);f.flush();os.fsync(f.fileno())
        os.replace(name,path)
    finally:
        if os.path.exists(name):os.unlink(name)

def generate(root,engineering_smoke=False):
    root=Path(root).resolve();plan=json.loads((root/'PLAN.json').read_bytes());jobs=plan['jobs']
    if engineering_smoke:
        if plan.get('mode')!='engineering_smoke' or any(j['seed'] in SEEDS for j in jobs):raise ValueError('Engineering override is restricted to distinct non-science smoke fixtures')
    elif plan.get('mode')!='science' or {(j['seed'],j['load']) for j in jobs}!={(s,l) for s in SEEDS for l in LOADS} or len(jobs)!=64:
        raise ValueError('Default exposure audit requires the locked64-cell science roster')
    expected={f"seed_{j['seed']}_load_{j['load']}.json" for j in jobs}
    found={p.name for p in (root/'completed').glob('*.json')}
    if found!=expected:raise ValueError(f'All planned cells must be complete before exposure audit: {len(found & expected)}/{len(expected)}; unexpected={sorted(found-expected)}')
    source=root/'elm_v43_fast_slow_authority.py'
    if not source.exists() or sha(source.read_bytes())!=plan.get('source_sha256'):raise ValueError('Saved source does not match locked plan')
    cells=[]
    for job in sorted(jobs,key=lambda j:(j['seed'],j['load'])):
        result,cols,provenance=checked_cell(root,plan,job);cells.append(cell_scope(result,cols,provenance))
    loads=[]
    for load in sorted({c['load'] for c in cells}):
        selected=[c for c in cells if c['load']==load];g=sum(c['online_G_events'] for c in selected);t=sum(c['online_G_target_events'] for c in selected)
        durable=Counter();durable_available=True
        for c in selected:
            ps=c['probe_scope']
            if ps['status']!='available':durable_available=False;continue
            final=[p for p in ps['durable_old_heldout_query_routes'] if p['checkpoint']=='after_D']
            if not final:durable_available=False
            for row in final:durable.update(row['route_counts'])
        loads.append({'load':load,'physical_cells':len(selected),'online_G_events':g,'online_G_target_events':t,'online_G_nontarget_events':g-t,
                      'online_G_target_fraction':t/g if g else None,'seeds_with_G_target_events':sum(c['online_G_target_events']>0 for c in selected),
                      'durable_after_D_routes_status':'available' if durable_available else 'unavailable_or_incomplete',
                      'durable_after_D_unique_queries':sum(durable.values()),'durable_after_D_route_counts':{r:durable[r] for r in ROUTES}})
    output={'analysis':'V43 exposure scope','mode':'engineering_smoke' if engineering_smoke else 'complete_scientific_roster',
            'cells':cells,'by_load_descriptive_counts':loads,'completed_physical_cells':len(cells),
            'provenance':{'plan_sha256':sha(plan),'source_sha256':plan['source_sha256'],'helper_sha256':sha(Path(__file__).read_bytes()),
                          'all_completed_envelopes_and_NPZ_hashes_verified':True},
            'interpretation':['Online G outcomes update arbitration evidence; evaluator probes do not update it.',
                'The target/nontarget category is evaluator metadata. This supplemental description does not supply a deployable feature.',
                'If online G is concentrated on ordinary bytes while durable held-out queries use G at identity targets, their exposure populations differ; this is a scope distinction, not a claim about policy performance or causation.',
                'Prediction diagnostics validated on online G events address that online population. Their scores alone do not establish successful arbitration on a different probe population.',
                'All counts are descriptive. Seeds remain the replication unit; this helper adds no statistical tests, qualification gates, thresholds or learner updates.']}
    lines=['# V43 exposure-scope audit','',f"Mode: **{output['mode']}**. Verified {len(cells)} completed physical cells, their envelopes and authority-table hashes.",'',
           'The table counts online G events by recorded target/ordinary category. Probe routes are deduplicated across the eight read policies. No policy scores, accuracies or rankings are analyzed.','',
           '| Load | Cells | Online G | G target | G ordinary | Seeds with G targets | Old held-out after D: A/T/G/F |',
           '|---:|---:|---:|---:|---:|---:|---|']
    for x in loads:
        route='/'.join(str(x['durable_after_D_route_counts'][r]) for r in ROUTES) if x['durable_after_D_routes_status']=='available' else 'unavailable'
        lines.append(f"| {x['load']} | {x['physical_cells']} | {x['online_G_events']} | {x['online_G_target_events']} | {x['online_G_nontarget_events']} | {x['seeds_with_G_target_events']} | {route} |")
    lines+=['','Per-seed/load counts, all saved probe checkpoint/kind/alias route counts and all-route wave/kind/category presentation counts are in EXPOSURE_SCOPE.json. The frozen authority-event table does not include stage labels, so online G-by-stage counts are unavailable; all-route stage totals are explicitly identified as such.','']
    lines+=['- '+x for x in output['interpretation']]
    lines+=['','The JSON records the exact input and helper hashes. This is a supplemental analysis of saved exposure scope, with no change to the frozen runner, read policies, scientific protocol or qualification results.','']
    atomic_text(root/'EXPOSURE_SCOPE.json',json.dumps(output,sort_keys=True,indent=2,allow_nan=False)+'\n')
    atomic_text(root/'EXPOSURE_SCOPE.md','\n'.join(lines))
    return output

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--engineering-smoke',action='store_true')
    args=parser.parse_args();result=generate(args.results,args.engineering_smoke)
    print(json.dumps({'mode':result['mode'],'verified_cells':result['completed_physical_cells'],'outputs':['EXPOSURE_SCOPE.json','EXPOSURE_SCOPE.md']}))

if __name__=='__main__':main()
