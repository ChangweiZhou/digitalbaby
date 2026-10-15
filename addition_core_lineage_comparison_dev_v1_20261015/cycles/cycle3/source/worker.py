"""Technical-only world/version jobs. No formal-science launch entry point."""
import argparse
import collections
import copy
import os
import resource
import time
from pathlib import Path
import core
import data
from io_utils import write, now, digest
import checkpoint
import measurement

HISTORIES = ('W','N_OLD','SH_OLD')


def evaluate(model, groups):
    start=time.process_time()
    before = model.digest()
    resource = copy.deepcopy(model.resource)
    result = {}
    for name, pairs in groups.items():
        expected = [data.render(pair) for pair in pairs]
        outputs = model.generate_batch([p for p,_ in expected])
        result[name] = [dict(prompt=p.hex(), expected=y.hex(), emitted=z.hex(),status=status,
                            correct=z==y and status=='EOS')
                        for (p,y),(z,status) in zip(expected,outputs)]
    if model.digest() != before:
        raise AssertionError('evaluation changed persistent learner state')
    model.resource = resource
    model.evaluation_cpu=getattr(model,'evaluation_cpu',0.)+time.process_time()-start
    return result


def run(world,version,cycle,output):
    if cycle not in (2,3) or world not in ([data.TECH_WORLDS[0]] if cycle==2 else data.TECH_WORLDS[1:]):
        raise ValueError('technical manifest mismatch')
    old_n,new_n = (128,32) if cycle==2 else (16384,1024)
    source = data.prepare(world,old_n,new_n)
    target = Path(output)
    if target.exists():
        raise FileExistsError('never overwrite or repeat committed technical job')
    active = target.parent.parent/'active'/f'{world}_{version}.json'
    start = time.monotonic()
    receipt = dict(schema='LINEAGE_ADDITION_TECHNICAL_JOB_V1', science=False,cycle=cycle,world=world,
                   version=version,seed=source['seed'],started=now(),histories={},records=0,
                   source_lock_sha256=digest(target.parent.parent/'SOURCE_LOCK.json'))
    checkpoint_cpu=0.
    def preserve(model,history,stage,index,rows,metrics):
        nonlocal checkpoint_cpu
        before=time.process_time()
        checkpoint.save(target.parent.parent/'checkpoints'/f'{world}_{version}.pickle.gz',
            dict(schema='TRUSTED_TECHNICAL_CHECKPOINT_V1',source_lock_sha256=receipt['source_lock_sha256'],
                 model=model,completed_history_receipt=receipt,history=history,stage=stage,
                 completed_index=index,rows=rows,metrics=metrics,
                 cursor_note='At OLD checkpoint, evaluation is complete and stage rest is pending if index==old_n. REST checkpoints include the completed rest.',
                 automatic_resume_authorized=False))
        checkpoint_cpu+=time.process_time()-before
    for history in HISTORIES:
        model = core.create(version,source['seed'])
        footprint=measurement.state_bytes(model)
        rows,metrics = [],[]
        births = [s.birth for b in model.banks for s in b] if isinstance(model,core.Native) else []
        for stage, sequence in [('OLD',source['shuffled'] if history=='SH_OLD' else source['old']),('NEW',source['new'])]:
            if stage=='OLD':
                metrics.append(dict(stage='OLD',n=0,cpu=0.,scores=evaluate(model,source['groups'])))
            for index,(prompt,answer) in enumerate(sequence,1):
                allowed = history!='N_OLD' or stage!='OLD'
                before = model.digest() if not allowed and isinstance(model,core.Recurrent) else None
                r = model.train(prompt,answer,allowed=allowed)
                if before is not None and before != model.digest():
                    raise AssertionError('forbidden recurrent parameter/moment update')
                rows.append(dict(stage=stage,index=index,prompt=prompt.hex(),answer=answer.hex(),**r))
                receipt['records'] += 1
                if index%32==0 or index==len(sequence):
                    write(active,dict(pid=os.getpid(),utc=now(),cycle=cycle,world=world,version=version,
                                      history=history,stage=stage,index=index,records_inflight=receipt['records'],
                                      elapsed=time.monotonic()-start))
                if stage=='OLD' and index in source['checkpoints']:
                    metrics.append(dict(stage='OLD',n=index,cpu=model.resource['production_cpu'],scores=evaluate(model,source['groups'])))
                    if cycle==3:
                        preserve(model,history,stage,index,rows,metrics)
            model.rest(86400.)
            metrics.append(dict(stage=stage+'_REST',n=len(sequence),cpu=model.resource['production_cpu'],scores=evaluate(model,source['groups'])))
            if cycle==3:
                preserve(model,history,stage+'_REST',len(sequence),rows,metrics)
        query_cpu=getattr(model,'evaluation_cpu',0.)
        erasure_start=time.process_time()
        erasure = evaluate(model.erased(),source['groups']) if isinstance(model,core.Native) else None
        erasure_cpu=time.process_time()-erasure_start
        receipt['histories'][history] = dict(rows=rows,metrics=metrics,resource=model.resource,
                                            final_digest=model.digest(),births=births,erasure=erasure,
                                            footprint=footprint,query_cpu=query_cpu,erasure_cpu=erasure_cpu)
    receipt.update(finished=now(),elapsed=time.monotonic()-start,old_n=old_n,new_n=new_n,
                   checkpoint_cpu=checkpoint_cpu,peak_rss_bytes=measurement.peak_rss_bytes())
    from audit import audit_job
    certificate = audit_job(receipt)
    write(target,receipt)
    write(target.with_name(target.name.replace('.json.gz','.audit.json')),certificate)
    active.unlink(missing_ok=True)
    return certificate


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--world',type=int,required=True)
    p.add_argument('--version',choices=core.VERSIONS,required=True)
    p.add_argument('--cycle',type=int,choices=(2,3),required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    try:
        print(run(args.world,args.version,args.cycle,args.output),flush=True)
    except BaseException as e:
        write(Path(args.output).parent.parent/'FAILURE.json',dict(utc=now(),world=args.world,
              version=args.version,error=repr(e)))
        raise
