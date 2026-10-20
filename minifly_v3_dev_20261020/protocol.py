"""Registered engineering life; arithmetic/exposure logic is outside core."""
import bootstrap
from dataclasses import asdict
import argparse
import gzip
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from core import Learner,Limits,hash_arrays,multipliers
from data import split,records,metadata
from evaluate import checkpoint,score,sample_area
from checkpoint import save
from artifacts import HERE,atomic_json,sha,source_digest


def run_job(world,arm,seed,output=None):
    output=Path(output) if output else HERE/'cycles/cycle3/jobs'/f'{world}_{arm}'
    output.mkdir(parents=True,exist_ok=True)
    if (output/'RECEIPT.json').exists():raise RuntimeError('immutable job already exists; refusing replay')
    if (output/'START.json').exists():raise RuntimeError('started job requires explicit recovery, not automatic replay')
    spec=json.loads((HERE/'SPEC.json').read_text())
    limits=Limits(**json.loads((HERE/'PARAMETER_LOCK.json').read_text())['limits'])
    identity=source_digest();sets,pairs=split(world)
    atomic_json(output/'START.json',dict(world=world,arm=arm,seed=seed,source_digest=identity,technical_only=True,
                                       dataset=metadata(world),limits=asdict(limits)))
    t=time.process_time();base=Learner(seed,plastic=arm[0]=='P',gradual=arm[-1]=='G',limits=limits)
    initialization_cpu=time.process_time()-t
    birth_digest=base.digest();birth_hash=base.parameter_hash()
    trajectories={};total_records=0;target_events=0;actual_writes=0;all_eval_cpu=0.
    log_path=output/'TRACE.jsonl.gz'
    with gzip.open(log_path,'wt',compresslevel=1) as trace:
        for history in spec['histories']:
            m=base.training_fork();assert m.digest()==birth_digest
            production_cpu=initialization_cpu;points=[];timeonly=None;latest_sham=None
            anchors={};probe_prompts=[p for p,_ in sets['OLD'][:8]]
            def code_snapshot():
                codes=[]
                for prompt in probe_prompts:
                    v=m.clone();v._prompt(prompt);codes.append(v._frame()['x'])
                return codes
            anchors['birth']=code_snapshot()
            def measure(phase,count):
                nonlocal all_eval_cpu
                p=checkpoint(m,sets,phase,count,production_cpu)
                codes=code_snapshot();drift={}
                for label,reference in anchors.items():
                    drift[label]=float(np.mean([float((a*b).sum()/max(1,((a+b)>0).sum()))
                                                for a,b in zip(codes,reference)]))
                p['KC_jaccard_to_anchor']=drift
                if phase=='OLD' and count==960:anchors['after_warmup']=codes
                points.append(p);all_eval_cpu+=p['evaluation_cpu']
                atomic_json(output/'ACTIVE.json',dict(world=world,arm=arm,history=history,phase=phase,
                    count=count,completed_record_applications=total_records,production_cpu=production_cpu,
                    taught=p['scores']['OLD']['accuracy'],held=p['scores']['HELD']['accuracy'],time=m.time))
                print(json.dumps(dict(job=f'{world}_{arm}',history=history,phase=phase,count=count,
                                      old=p['scores']['OLD']['accuracy'],new=p['scores']['NEW']['accuracy'],
                                      held=p['scores']['HELD']['accuracy'],cpu=round(production_cpu,2))),flush=True)
                if not p['diagnostics']['finite']:raise FloatingPointError('nonfinite engineering state')
            measure('OLD',0)
            for phase in ('OLD','NEW'):
                if phase=='NEW':measure('NEW',0)
                for n,row in enumerate(records(world,history,phase),1):
                    before_time=m.time;v0=m.version
                    t=time.process_time();m.train(row['prompt'],row['answer'],row['allowed']);production_cpu+=time.process_time()-t
                    total_records+=1;target_events+=len(m.last_trace)
                    actual_writes+=sum(x['coefficient'] is not None for x in m.last_trace)
                    av,ar=multipliers(m.records,arm[-1]=='G')
                    trace.write(json.dumps(dict(history=history,phase=phase,phase_record=n,record=m.records,
                        prompt=row['prompt'].hex(),answer=row['answer'].hex(),truth=row['truth'].hex(),
                        allowed=row['allowed'],time_before=before_time,time_after=m.time,
                        version_before=v0,version_after=m.version,value_scale=av,representation_scale=ar,
                        target_events=m.last_trace,local_update=m.last_update),separators=(',',':'))+'\n')
                    if phase=='OLD' and n>7620:
                        if latest_sham is None:latest_sham=[]
                        latest_sham.append((row['prompt'],row['answer']))
                    if n in spec['old_checkpoints' if phase=='OLD' else 'new_checkpoints']:
                        measure(phase,n)
                if phase=='OLD':
                    old_end_cpu=production_cpu
                    if history=='SH_OLD':
                        t=time.process_time();sham_score=score(m,latest_sham);all_eval_cpu+=time.process_time()-t
                    else:sham_score=None
                    t=time.process_time();m.rest(spec['idle_seconds']);production_cpu+=time.process_time()-t
                    measure('OLD_REST',7680)
                    save(m,output/f'{history}_OLD_REST.npz')
                    timeonly=m.clone()
                else:
                    t=time.process_time();m.rest(spec['idle_seconds']);production_cpu+=time.process_time()-t
                    measure('FINAL',8320)
            assert m.records==8320
            # Only simulated time advances in this derived diagnostic view.
            timeonly.rest(m.time-timeonly.time)
            tp=checkpoint(timeonly,sets,'TIME_ONLY_FINAL',0,old_end_cpu)
            all_eval_cpu+=tp['evaluation_cpu']
            final_path=output/f'{history}_FINAL.npz';save(m,final_path)
            old_points=[p for p in points if p['phase']=='OLD']
            trajectories[history]=dict(points=points,timeonly_endpoint=tp,
                sample_area=sample_area(old_points),old_end_production_cpu=old_end_cpu,
                final_parameter_hash=m.parameter_hash(),initial_parameter_hash=birth_hash,
                sham_last_pass_label_score=sham_score,
                final_checkpoint=str(final_path.relative_to(HERE)),final_checkpoint_sha256=sha(final_path),
                final_diagnostics=m.diagnostics())
    if source_digest()!=identity:raise RuntimeError('source changed during engineering job')
    receipt=dict(schema='MINIFLY_V3_ENGINEERING_JOB',world=world,arm=arm,seed=seed,technical_only=True,
        scientific_records=0,source_digest=identity,initial_parameter_hash=birth_hash,
        record_applications=total_records,continuous_histories=3,target_events=target_events,
        actual_native_target_writes=actual_writes,trace=str(log_path.relative_to(HERE)),trace_sha256=sha(log_path),
        initialization_cpu=initialization_cpu,evaluation_cpu=all_eval_cpu,trajectories=trajectories)
    atomic_json(output/'RECEIPT.json',receipt)
    atomic_json(output/'COMPLETE.json',dict(receipt_sha256=sha(output/'RECEIPT.json'),record_applications=total_records,
                                          continuous_histories=3,scientific_records=0))
    return dict(job=f'{world}_{arm}',records=total_records)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--world',type=int,required=True);ap.add_argument('--arm',required=True)
    ap.add_argument('--seed',type=int,required=True);args=ap.parse_args();print(run_job(args.world,args.arm,args.seed))
