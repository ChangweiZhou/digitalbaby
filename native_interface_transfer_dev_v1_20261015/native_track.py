"""Track A: one fixed, staged E1 diagnostic. Negative behavior stops escalation."""
import time
import copy
import config
import tasks
import models
import numpy as np
from artifacts import write,now

DT=30/14
PERIOD=165.


def scores_and_emitted(prediction,stage):
    if stage=='A1-LEGACY-C4':
        return list(prediction.combined),int(prediction.emitted),list(models.ALPHABET)
    return list(prediction.logits),int(prediction.emitted),list(range(257))


def scheduled_record(model,item,stage,allowed):
    audit_start=time.process_time()
    pre=[(m.fly.m.fast.copy(),m.fly.m.slow.copy()) for m in model.models] if not allowed else None
    audit_cpu=time.process_time()-audit_start
    start=time.process_time();at=model.last_time
    for i,byte in enumerate(item['cue']):model.feed(byte,at+i*DT)
    prediction=model.predict(at+12*DT)  # executed BEFORE receiving the target
    scores,emitted,universe=scores_and_emitted(prediction,stage)
    target=item['answer'][0]
    observation=model.observe_outcome(target,at+12*DT,learn=allowed)
    model.feed(10,at+13*DT)
    # The complete historical public API cost is reported, including its guards.
    api_cpu=time.process_time()-start
    idle_start=time.process_time();model.flush(at+PERIOD);idle_cpu=time.process_time()-idle_start
    l1=sum(observation['shared_l1'])+sum(observation['private_l1'])
    if not allowed and l1!=0:raise AssertionError('no-write control applied native write')
    audit_start=time.process_time();off_error=0.
    if not allowed:
        for (fast,slow),m in zip(pre,model.models):
            f=m.fly.m
            factor=np.array([np.exp(-PERIOD/f.tg),np.exp(-PERIOD/f.kernel['fast_tau'])])
            off_error=max(off_error,float(np.abs(f.fast-fast*factor).max()),
                          float(np.abs(f.slow-slow*np.exp(-PERIOD/f.kernel['slow_tau'])).max()))
        if off_error>1e-8:raise AssertionError(('forbidden legacy/C16 write',off_error))
    audit_cpu+=time.process_time()-audit_start
    return dict(id=item['id'],allowed=allowed,target=target,predicted=emitted,logits=scores,
                output_universe=universe,raw_alpha_l1=l1,api_cpu=api_cpu,idle_cpu=idle_cpu,
                forbidden_delta=off_error,audit_cpu=audit_cpu)


def sequence_record(model,item,allowed):
    before=model.time
    record=model.train(item['cue'],item['answer'],allowed)
    elapsed=model.time-before
    if elapsed>PERIOD:raise AssertionError('frame exceeds fixed diagnostic cadence')
    model.rest(PERIOD-elapsed)
    return dict(id=item['id'],**record)


def probe(model,stage,fixture):
    before=model.digest() if stage=='A3-SEQUENCE-C16' else model.state_digest()
    start=time.process_time();rows=[]
    for item in fixture:
        if stage=='A3-SEQUENCE-C16':
            emitted,status,trace=model.generate_trace(item['cue'])
            teacher=model.teacher_conditioned_trace(item['cue'],item['answer'])
            conditional=[dict(target=r['target'],predicted=r['predicted'],
                              target_margin=models.margin(r['logits'],r['target']),logits=r['logits']) for r in teacher]
            rows.append(dict(id=item['id'],stratum=item['stratum'],expected=item['answer'].hex(),
                emitted=emitted.hex(),status=status,correct=emitted==item['answer'] and status=='EOS',
                premature_EOS=status=='EOS' and len(emitted)<len(item['answer']),
                trace=trace,teacher_conditioned=conditional,
                conditional_is_autonomous_score=False))
        else:
            clone=model.clone();at=clone.last_time
            for i,byte in enumerate(item['cue']):clone.feed(byte,at+i*DT)
            prediction=clone.predict(at+12*DT)
            scores,emitted,universe=scores_and_emitted(prediction,stage)
            target=item['answer'][0]
            rows.append(dict(id=item['id'],stratum=item['stratum'],expected_target=target,
                emitted_target=emitted,logits=scores,output_universe=universe,
                target_margin=models.margin(scores,universe.index(target)),correct=emitted==target,
                status='EXTERNALLY_SCHEDULED_SINGLE_OUTPUT',autonomous_generation=False))
    after=model.digest() if stage=='A3-SEQUENCE-C16' else model.state_digest()
    if before!=after:raise AssertionError('probe changed continuing native state')
    return dict(rows=rows,probe_cpu=time.process_time()-start,diagnostic_stage=stage)


def gate(stage,result):
    W=result['histories']['W']['probes'][-2]['scores']['rows']
    N=result['histories']['NO_WRITE']['probes'][-2]['scores']['rows']
    clean=[r for r in W if r['stratum']=='clean']
    no_clean=[r for r in N if r['stratum']=='clean']
    exact=all(r['correct'] for r in clean)
    if stage!='A3-SEQUENCE-C16':exact=exact and all(r['target_margin']>0 for r in clean)
    challenge=all(r['correct'] for r in W if r['stratum']=='suffix_collision')
    causal=sum(r['correct'] for r in clean)>sum(r['correct'] for r in no_clean)
    qualifies=exact and causal and (challenge if stage=='A3-SEQUENCE-C16' else True)
    return dict(qualified=qualifies,all_clean_acquired=exact,positive_write_control=causal,
                suffix_challenge_acquired=challenge,
                scope='Exact finite-fixture E1 acquisition, not arithmetic E3 or a population-level reliability claim.')


def run_stage(stage,output):
    folder=models.bootstrap.HERE/'results/track_a'
    if output.exists():raise FileExistsError('bounded native stage already committed; no automatic repetition')
    fixture=tasks.native_fixture()
    sequence=tasks.ordered_stream(fixture,config.TRACK_A_DOSES[-1],config.TRACK_A_SEED)
    receipt=dict(schema='BOUNDED_NATIVE_INTERFACE_V1',stage=stage,science=False,started=now(),
                 planned_record_applications=2*len(sequence),histories={})
    for history in ('W','NO_WRITE'):
        model=models.CenteredCore() if stage=='A1-LEGACY-C4' else models.ScheduledC16() if stage=='A2-SCHEDULED-C16' else models.SequenceC16()
        rows=[];probes=[dict(dose=0,phase='IMMEDIATE',scores=probe(model,stage,fixture))]
        for index,item in enumerate(sequence,1):
            row=sequence_record(model,item,history=='W') if stage=='A3-SEQUENCE-C16' else scheduled_record(model,item,stage,history=='W')
            rows.append(dict(index=index,**row))
            if index in config.TRACK_A_DOSES:
                probes.append(dict(dose=index,phase='IMMEDIATE',scores=probe(model,stage,fixture)))
            if index%12==0:
                write(folder/'ACTIVE.json',dict(utc=now(),stage=stage,history=history,index=index,total=len(sequence)))
        rest_start=time.process_time();model.rest(86400.);rest_cpu=time.process_time()-rest_start
        probes.append(dict(dose=len(sequence),phase='AFTER_24H',scores=probe(model,stage,fixture)))
        receipt['histories'][history]=dict(rows=rows,probes=probes,rest_cpu=rest_cpu,
                    births=model.births if stage!='A3-SEQUENCE-C16' else [s.birth for b in model.banks for s in b],
                    resource=model.resource if stage=='A3-SEQUENCE-C16' else dict(api_cpu=sum(r['api_cpu'] for r in rows),
                             idle_cpu=sum(r['idle_cpu'] for r in rows),records=len(rows)))
    receipt['gate']=gate(stage,receipt);receipt['finished']=now()
    from audits import audit_native
    audit=audit_native(receipt)
    write(output,receipt);write(output.with_name(output.name.replace('.json.gz','.audit.json')),audit)
    (folder/'ACTIVE.json').unlink(missing_ok=True)
    return receipt


def run():
    folder=models.bootstrap.HERE/'results/track_a';decisions=[]
    for stage in config.TRACK_A_STAGES:
        result=run_stage(stage,folder/(stage+'.json.gz'))
        decisions.append(dict(stage=stage,**result['gate']))
        if not result['gate']['qualified']:
            decisions.extend(dict(stage=other,qualified=False,not_run=True,reason='Earlier finite-fixture acquisition stage failed; bounded attempt ends here.')
                             for other in config.TRACK_A_STAGES[len(decisions):])
            break
    summary=dict(utc=now(),science=False,decisions=decisions,
                 language_task=False,automatic_retry=False,formal_launch=False)
    write(folder/'SUMMARY.json',summary)
    return summary
