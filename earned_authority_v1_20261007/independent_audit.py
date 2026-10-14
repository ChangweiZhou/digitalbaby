"""Separate fixed branch table and label-free readout recomputation.

Native transition observations come from the inherited instrumentation. This
module independently verifies their actual write flags and no-write equality.
"""
import math
import numpy as np
from authority_policy import SCALES,POLICIES

FLAGS={'W':{'old':True,'new':True,'revision':True},
       'N_old':{'old':False,'new':True,'revision':True},
       'N_new':{'old':True,'new':False,'revision':True},
       'N_revision':{'old':True,'new':True,'revision':False}}
def flag_for(branch,stage):return FLAGS[branch][stage]

def independent_readout(predictions,policy):
    s=[np.asarray(p['shared'])/SCALES[0] for p in predictions]
    v=[np.asarray(p['private'])/SCALES[1] for p in predictions]
    paired=len(s)==2
    ds=np.array([x[1]-x[0] for x in s]) if paired else s[0]
    dp=np.array([x[1]-x[0] for x in v]) if paired else v[0]
    sm=float(np.sort(ds)[-1]-np.sort(ds)[-2]);pm=float(np.sort(dp)[-1]-np.sort(dp)[-2])
    trigger=int(np.argmax(ds))!=int(np.argmax(dp)) and .5*pm<sm<pm
    alpha=.5 if policy=='Q_HALF' or (policy=='Q_EARNED' and trigger) else 1.
    u=[a+alpha*b for a,b in zip(s,v)]
    scores=np.array([x[1]-x[0] for x in u]) if paired else u[0]
    return (b'LR' if paired else b'0123')[int(np.argmax(scores))],alpha,bool(trigger),scores

def audit_readouts(predictions,readouts):
    assert set(readouts)==set(POLICIES)
    for policy,r in readouts.items():
        expected,a,tr,scores=independent_readout(predictions,policy)
        assert r['emitted']==expected and r['alpha']==a and r['trigger'] is tr
        assert np.array_equal(r['combined'],scores)
        ds=np.asarray(r['shared']);dp=np.asarray(r['private'])
        assert r['shared_winner']==int(np.argmax(ds)) and r['private_winner']==int(np.argmax(dp))
        assert r['shared_margin']==float(np.sort(ds)[-1]-np.sort(ds)[-2])
        assert r['private_margin']==float(np.sort(dp)[-1]-np.sort(dp)[-2])
        assert len(r['observations'])==len(predictions)
        assert set(r)=={'alpha','trigger','shared_winner','private_winner','shared_margin','private_margin','rule_version','policy','emitted','shared','private','combined','observations'}
        for pred,obs in zip(predictions,r['observations']):
            assert tuple(obs['shared'])==tuple(pred['shared']) and tuple(obs['private'])==tuple(pred['private'])

def audit_record(row,event,branch):
    expected=flag_for(branch,event['stage'])
    assert row['learn'] is expected and row['write']['learn'] is expected
    assert row['operation_order']==['cue','native_predict','policy_emit','outcome','newline','flush']
    assert row['model_emitted']==row['readouts']['Q_EARNED']['emitted']
    audit_readouts([row['prediction']],row['readouts'])
    calls=row['actual_calls'];assert len(calls)==8 and [x['store'] for x in calls]==list(range(8))
    for call in calls:
        assert call['write'] is expected
        if not expected:
            assert call['no_write_reference_equal'] is True
            assert all(call[k]==0 for k in ('applied_l1','delta_l1','delta_l2','delta_max'))
    assert row['write']['outcome']==event['outcome']

def audit_job(doc,identity):
    from bridge_runtime import make_world,check_record
    assert doc['development'] is True and doc['source_identity']==identity
    world=make_world(doc['world'],doc['assay']);assert world['sha256']==doc['fixture_sha256']
    count=len(world['events']) if doc['complete'] else doc['limits']['records']
    branches=world['branches'] if doc['complete'] else ['W']
    assert set(doc['branches'])==set(branches)
    for branch,bd in doc['branches'].items():
        births=bd['births'];assert len(births)==8 and len({b['fly_id'] for b in births})==8
        assert len({b['canonical_B_sha256'] for b in births})==1
        assert len(bd['records'])==bd['records_completed']==count
        for event,row in zip(world['events'][:count],bd['records']):
            check_record(row,'ERROR',event,branch,8);audit_record(row,event,branch)
        expected_probes=[name for index in range(1,count+1) for name in world['boundaries'].get(str(index),[])] if doc['complete'] else ['short_end']
        assert [p['name'] for p in bd['probes']]==expected_probes
        for probe in bd['probes']:
            if doc['complete']:assert probe['at']==world['clocks'][probe['name']]
            expected_rows=[]
            for group,items in world['sets'].items():
                if probe['name'].startswith('old_') and group in ('new','revision'):continue
                if world['assay']=='reuse':
                    for i in range(len(items)//2):
                        a,n=items[2*i:2*i+2]
                        for order in (0,1):
                            pair=(a,n) if order==0 else (n,a)
                            expected_rows.append((group,i,order,ord('L') if order==0 else ord('R'),[r['cue_hex'] for r in pair]))
                else:
                    for i,r in enumerate(items):expected_rows.append((group,i,None,r['outcome'],[r['cue_hex']]))
            actual=[(r['stage'],r['item'],r.get('order'),r['target'],r['cue_hex']) for r in probe['rows']]
            assert sorted(actual)==sorted(expected_rows)
            assert probe['state_before']==probe['state_after']
            for row in probe['rows']:
                assert row['first_pre_feedback'] is True
                audit_readouts(row['predictions'],row['readouts'])
        assert bd['final_time']==(world['clocks']['final'] if doc['complete'] else world['events'][count-1]['at']+165.)
    return {'verdict':'PASS','lives':len(branches),'records':count*len(branches)}
