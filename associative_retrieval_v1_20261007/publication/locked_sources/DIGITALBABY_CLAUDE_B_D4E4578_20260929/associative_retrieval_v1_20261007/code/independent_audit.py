"""Independent list/set replay; never imports live association query/update code."""
import math
import runtime
import numpy as np
from assays import make_world, permitted, DT
from spec import ALPHABET, SCALES

def assert_read_activity(captured, rows, n, permutation=None):
    expected=np.zeros((len(rows),n))
    for j,r in enumerate(rows):
        ids=r['private_ids']
        if permutation is not None:ids=np.asarray(permutation)[ids]
        expected[j,ids]=1.
    if len(captured)!=4 or any(not np.array_equal(x,expected) for x in captured):
        raise AssertionError('actual retrieved KC activity differs from associated addresses')

def query(table, ids):
    q=set(ids); candidates=[]
    for i,row in enumerate(table):
        score=len(q & set(row['s'])) / math.sqrt(len(q)*len(row['s']))
        if score>0: candidates.append((i,score,row['birth'],row['p']))
    candidates=sorted(candidates,key=lambda r:(-r[1],r[2],r[0]))[:4]
    total=sum(r[1] for r in candidates)
    return [{'slot':i,'score':s,'weight':s/total,'private_ids':p} for i,s,_,p in candidates]

def update(table, cursor, insertions, s, p):
    j=next((i for i,r in enumerate(table) if r['p']==p),None)
    inserted=j is None
    if inserted:
        j=cursor; insertions+=1; cursor=(cursor+1)%64
        row={'s':s,'p':p,'birth':insertions}
        if j==len(table): table.append(row)
        else: table[j]=row
    else: table[j]['s']=s
    return cursor,insertions,j,inserted

def check_prediction(p, table):
    expected=query(table,p['shared_ids'])
    if p['query']!=expected: raise AssertionError('independent retrieval correspondence mismatch')
    s=np.asarray(p['shared']); v=np.asarray(p['private'])
    u=s/SCALES[0]+v/SCALES[1]
    logits=v/SCALES[1]; pi=np.exp(logits-logits.max()); pi/=pi.sum()
    for name,key in [('LINK','retrieval'),('PERM','permutation_retrieval')]:
        r=np.asarray(p[key]); a=u+.5*(r-r.mean())/SCALES[1]
        if not np.array_equal(a,np.asarray(p['policies'][name]['combined'])):
            raise AssertionError('independent retrieval output formula')
        if p['policies'][name]['emitted']!=ALPHABET[int(np.argmax(a))]: raise AssertionError('output byte')
    if not np.array_equal(u,np.asarray(p['policies']['ERROR']['combined'])): raise AssertionError('ERROR parity')
    if p['policies']['ERROR']['emitted']!=ALPHABET[int(np.argmax(u))] or p['emitted']!=p['policies']['LINK']['emitted']:
        raise AssertionError('policy emission')
    if not table and (any(p['retrieval']) or any(p['permutation_retrieval'])): raise AssertionError('empty-table nonzero read')
    return pi

def audit(d, identity):
    if d['source_identity']!=identity or d['development'] is not True: raise AssertionError('source/development scope')
    w=make_world(d['world'],d['assay'])
    if d['fixture_sha256']!=w['sha256']: raise AssertionError('fixture source')
    limit=d['limit']; flags=[]; records=probes=0; common=None
    for branch,b in d['branches'].items():
        if branch not in w['branches'] or len(b['records'])!=limit: raise AssertionError('branch/records')
        if len(b['births'])!=8 or len({x['fly_id'] for x in b['births']})!=8: raise AssertionError('independent birth')
        table=[]; cursor=insertions=0; chronology=[]
        checkpoints={}
        for e,r in zip(w['events'][:limit],b['records']):
            if r['index']!=e['index'] or r['stage']!=e['stage'] or r['item']!=e['item']: raise AssertionError('record chronology')
            flag=permitted(branch,e['stage']); a=r['write']['association']
            pi=check_prediction(r['prediction'],table)
            if r['learn'] is not flag or r['write']['learn'] is not flag: raise AssertionError('branch clamp')
            if r['write']['outcome']!=e['outcome'] or r['predicted_at']!=r['observed_at'] or r['predicted_at']!=e['at']+12*DT:
                raise AssertionError('actual teacher boundary')
            if a['shared_ids']!=r['prediction']['shared_ids'] or a['private_ids']!=r['prediction']['private_ids'] or a['cue_only'] is not True:
                raise AssertionError('pre-answer association address')
            y=np.array([float(z==e['outcome']) for z in ALPHABET])
            if not np.array_equal(pi,np.asarray(r['write']['private_pre_outcome_probabilities'])) or not np.array_equal(pi-y,np.asarray(r['write']['s'])):
                raise AssertionError('unchanged private-only ERROR teaching')
            calls=r['actual_calls']
            if len(calls)!=8: raise AssertionError('actual native calls')
            for i,c in enumerate(calls):
                ids=r['prediction']['shared_ids'] if i<4 else r['prediction']['private_ids']
                coef=[float(ALPHABET[i]!=e['outcome'])] if i<4 else [0.,float((pi-y)[i-4])]
                if c['store']!=i or c['write'] is not flag or c['coefficients']!=coef or c['address_ids']!=ids:
                    raise AssertionError('actual native teaching/address')
                if not flag and (c['no_write_reference_equal'] is not True or any(c[k]!=0 for k in ('applied_l1','delta_l1','delta_l2','delta_max'))):
                    raise AssertionError('actual no-write state')
            cursor,insertions,j,ins=update(table,cursor,insertions,a['shared_ids'],a['private_ids'])
            if (a['slot'],a['inserted'],a['observations'])!=(j,ins,e['index']+1): raise AssertionError('independent association update')
            chronology.append((a['before'],a['after']))
            checkpoints[e['index']+1]=[dict(x) for x in table]
            records+=1
        if common is None: common=chronology
        elif common!=chronology: raise AssertionError('W/N cue-only association mismatch')
        for pr in b['probes']:
            if pr['state_before']!=pr['state_after']: raise AssertionError('probe mutated continuing state')
            tab=checkpoints[pr['after_record']]
            for row in pr['rows']:
                if row['first_pre_feedback'] is not True: raise AssertionError('probe feedback')
                for p in row['predictions']: check_prediction(p,tab)
                for name,policy in row['policies'].items():
                    if policy['correct']!=int(policy['emitted']==row['target']): raise AssertionError('probe scoring')
                    if d['assay']=='reuse':
                        values=[p['policies'][name]['combined'][1]-p['policies'][name]['combined'][0] for p in row['predictions']]
                        if values!=policy['values'] or policy['emitted']!=(ord('L') if values[0]>=values[1] else ord('R')):
                            raise AssertionError('internal choice rule')
                    elif policy['emitted']!=row['predictions'][0]['policies'][name]['emitted']: raise AssertionError('single-cue readout')
                probes+=1
    if d['complete'] and set(d['branches'])!=set(w['branches']): raise AssertionError('incomplete branch roster')
    return {'verdict':'PASS','records':records,'probe_rows':probes,'lives':len(d['branches']),
            'candidate_native_trajectories':len(d['branches']),'comparison_conditions':3}
