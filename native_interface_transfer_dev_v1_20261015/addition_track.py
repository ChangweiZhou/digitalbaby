"""Track B: same recurrent references, added NEW-dose observations; no new core."""
import copy
import time
import config
import tasks
import models
from artifacts import write,now


def evaluate(model,groups):
    start=time.process_time();before=model.digest();resources=copy.deepcopy(model.resource)
    result={}
    for name,pairs in groups.items():
        examples=[tasks.render(p) for p in pairs]
        outputs=model.generate_batch([p for p,_ in examples])
        result[name]=[dict(prompt=p.hex(),expected=y.hex(),emitted=z.hex(),status=status,
                           correct=z==y and status=='EOS') for (p,y),(z,status) in zip(examples,outputs)]
    if model.digest()!=before:raise AssertionError('read-only probe changed parameters/optimizer')
    model.resource=resources
    return dict(scores=result,query_and_guard_cpu=time.process_time()-start)


def run_job(world,version,output):
    if world not in config.TRACK_B_WORLDS or version not in config.TRACK_B_MODELS:
        raise ValueError('fixed development roster only')
    if output.exists():raise FileExistsError('never overwrite/repeat committed development job')
    source=tasks.addition_streams(world);start=time.monotonic()
    result=dict(schema='ADDITION_ACQUISITION_TRANSFER_CURVES_V1',science=False,world=world,version=version,
                seed=tasks.model_seed(world),started=now(),histories={})
    for history in config.TRACK_B_HISTORY:
        model=models.AdditionReference(version,result['seed'])
        initial_digest=model.digest();old_end_digest=None
        rows=[];curve=[];guard_cpu=0.;query_cpu=0.
        for phase,sequence,checkpoints in [('OLD',source['shuffled'] if history=='SH_OLD' else source['old'],config.TRACK_B_OLD_DOSES),
                                            ('NEW',source['new'],config.TRACK_B_NEW_DOSES)]:
            q=evaluate(model,source['groups']);query_cpu+=q['query_and_guard_cpu']
            curve.append(dict(phase=phase,dose=0,production_cpu=model.resource['production_cpu'],**q))
            for index,(prompt,answer) in enumerate(sequence,1):
                allowed=not (phase=='OLD' and history=='N_OLD')
                guard=time.process_time();before=model.digest() if not allowed else None;guard_cpu+=time.process_time()-guard
                record=model.train(prompt,answer,allowed)
                guard=time.process_time()
                if before is not None and before!=model.digest():raise AssertionError('forbidden parameter or RMS update')
                guard_cpu+=time.process_time()-guard
                rows.append(dict(phase=phase,index=index,prompt=prompt.hex(),answer=answer.hex(),**record))
                if index in checkpoints:
                    q=evaluate(model,source['groups']);query_cpu+=q['query_and_guard_cpu']
                    curve.append(dict(phase=phase,dose=index,production_cpu=model.resource['production_cpu'],**q))
                if index%128==0:
                    write(output.parent.parent/'active'/f'{world}_{version}.json',dict(utc=now(),history=history,phase=phase,index=index))
            if phase=='OLD':old_end_digest=model.digest()
            model.rest(86400.)
            q=evaluate(model,source['groups']);query_cpu+=q['query_and_guard_cpu']
            curve.append(dict(phase=phase+'_REST',dose=len(sequence),production_cpu=model.resource['production_cpu'],**q))
        result['histories'][history]=dict(rows=rows,curve=curve,resource=model.resource,rest_cpu=model.rest_cpu,
                  forbidden_write_guard_cpu=guard_cpu,query_and_guard_cpu=query_cpu,
                  initial_digest=initial_digest,OLD_learning_end_digest=old_end_digest,
                  final_digest=model.digest(),parameter_count=sum(p.numel() for p in model.p.values()))
    result.update(finished=now(),record_applications=sum(len(h['rows']) for h in result['histories'].values()),
                  elapsed=time.monotonic()-start)
    from audits import audit_addition
    audit_start=time.process_time();audit=audit_addition(result);audit_cpu=time.process_time()-audit_start
    result['independent_audit_cpu']=audit_cpu
    write(output,result);write(output.with_name(output.name.replace('.json.gz','.audit.json')),audit)
    (output.parent.parent/'active'/f'{world}_{version}.json').unlink(missing_ok=True)
    return result
