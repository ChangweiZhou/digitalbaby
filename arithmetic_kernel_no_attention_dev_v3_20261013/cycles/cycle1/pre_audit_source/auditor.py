"""Independent receipt arithmetic/causality checks; no learner import."""
import math

def audit(r):
    assert r['schema']=='KERNEL_V3_TECHNICAL_JOB' and r['technical'] is True and r['completed'] is True
    assert r['executed_external_records']==3*r['binary_n']+4*r['mixed_n']
    assert set(r['branches'])=={'W','N_BINARY','SH_BINARY','N_MIXED'}
    checks=[]
    for name,b in r['branches'].items():
        for seg in b['segments']:
            assert len(seg['records'])==(r['binary_n'] if seg['label']=='binary' else r['mixed_n'])
            allowed=not((name=='N_BINARY' and seg['label']=='binary') or (name=='N_MIXED' and seg['label']=='mixed'))
            assert seg['allowed'] is allowed
            if not allowed:assert seg['before']==seg['after']
            before,after=seg['counts_before'],seg['counts_after']
            targets=0;payload=0
            for i,x in enumerate(seg['records']):
                assert x['index']==i and x['allowed'] is allowed
                tt=[*bytes.fromhex(x['reply']),256]
                assert [p['target'] for p in x['predictions']]==tt
                assert all(0<=p['choice']<=256 and math.isfinite(p['loss']) for p in x['predictions'])
                targets+=len(tt);payload+=len(bytes.fromhex(x['prompt']))+len(bytes.fromhex(x['reply']))
            assert after['external_records']-before['external_records']==len(seg['records'])
            assert after['external_payload_bytes']-before['external_payload_bytes']==payload
            assert after['actual_target_applications']-before['actual_target_applications']==(targets if allowed else 0)
            if not allowed:
                for k in ('optimizer_steps','replay_records','replay_target_applications'):assert after[k]==before[k]
        for probe in b.get('before_mixed',[])+b['final']:
            assert probe['before']==probe['after']
            for x in probe['rows']:
                assert x['exact'] is (x['actual']==x['expected'] and x['status']=='EOS')
                assert x['status'] in ('EOS','TRUNCATED')
                assert len(bytes.fromhex(x['actual']))<=16
        assert b['state_bytes']['pending_records']==0 and b['state_bytes']['raw_replay_payload']<=4096
        if b['last_update']:assert b['last_update']['applied_RMS']<=.00100001
        checks.append(name+' write/exposure/output/resource audit')
    # Taught and held probes cannot reuse a literal reinforced W prompt/group.
    taught={x['prompt'] for seg in r['branches']['W']['segments'] for x in seg['records']}
    for probe in r['branches']['W']['final']:
        if probe['label'] in ('old_held','new_held'):assert all(x['prompt'] not in taught for x in probe['rows'])
    assert r['total_process_cpu']>0 and r['total_wall']>0
    return {'accepted':True,'scope':'technical receipt checks only','checks':checks+['held literal exclusion','exact ownEOS arithmetic','bounded displacement'],'scientific_jobs':0,'external_records':r['executed_external_records']}
