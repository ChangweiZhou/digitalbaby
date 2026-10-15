"""Technical-only world/version jobs. No formal-science launch entry point."""
import argparse
import collections
import copy
import os
import time
from pathlib import Path
import core
import data
from io_utils import write, now

HISTORIES = ('W','N_OLD','SH_OLD')


def evaluate(model, groups):
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
                   version=version,seed=source['seed'],started=now(),histories={},records=0)
    for history in HISTORIES:
        model = core.create(version,source['seed'])
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
            model.rest(86400.)
            metrics.append(dict(stage=stage+'_REST',n=len(sequence),cpu=model.resource['production_cpu'],scores=evaluate(model,source['groups'])))
        erasure = evaluate(model.erased(),source['groups']) if isinstance(model,core.Native) else None
        receipt['histories'][history] = dict(rows=rows,metrics=metrics,resource=model.resource,
                                            final_digest=model.digest(),births=births,erasure=erasure)
    receipt.update(finished=now(),elapsed=time.monotonic()-start,old_n=old_n,new_n=new_n)
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
