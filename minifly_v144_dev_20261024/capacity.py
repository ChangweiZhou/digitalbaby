"""Bounded developmental interface trial; full V144 screen is separate."""
import argparse
import json
import os
import pickle
import resource
import sys
import time
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from v144_audit import HERE,ROOT,Ledger,source_lock,write_json
from reference import Reference,ByteTrace,decode,EOS,seed
import task
import numpy as np
sys.path.insert(0,str(ROOT/'minifly_learned_bank_v1_20261024'))
import lb1


def countermodels(training, held):
    rows=[(tuple(p),task.answer(p)) for p in task.oriented(training)]
    def majority(values):
        if not values:return None
        counts={v:values.count(v) for v in set(values)}
        return min(counts,key=lambda x:(-counts[x],x))
    freq=majority([y for p,y in rows])
    raw=dict(rows);canonical={tuple(sorted(p)):y for p,y in rows}
    predictors={
        'raw_lookup':lambda p:raw.get(tuple(p)),
        'unordered_lookup':lambda p:canonical.get(tuple(sorted(p))),
        'frequency':lambda p:freq,
        'prompt_length':lambda p:majority([y for q,y in rows if len(task.prompt(q))==len(task.prompt(p))]),
        'first_operand':lambda p:majority([y for q,y in rows if q[0]==p[0]]),
        'second_operand':lambda p:majority([y for q,y in rows if q[1]==p[1]])}
    return {name:float(np.mean([np.mean([fn(p)==task.answer(p) for p in task.oriented([o])]) for o in held]))
            for name,fn in predictors.items()}


def teacher_metrics(model,pairs):
    hits=0;total=0;loss=0.
    for pair in pairs:
        prefix=bytearray()
        for token in (*task.answer(pair),EOS):
            p=model.trace.context(task.prompt(pair),prefix);m=model.margins(p)
            hits+=decode(m)==token;total+=1
            y=np.array([(token>>j)&1 for j in range(9)])
            loss+=float(np.sum(np.logaddexp(0,m)-y*m))
            if token!=EOS:prefix.append(token)
    return {'teacher_forced_token_accuracy':hits/total,'teacher_forced_BCE_per_token':loss/total,
            'evaluator_target_queries':total}


def snapshot(model,w,training,label,cpu):
    before=model.state_hash()
    he=task.score(model,task.oriented(w['held']));tr=task.score(model,task.oriented(training))
    teacher=teacher_metrics(model,task.oriented(w['held']))
    assert model.state_hash()==before
    return {'label':label,'records':model.records,'observed_target_tokens':model.target_tokens,
            'cpu_s':cpu,'held':he,'taught':tr,'teacher_forced_diagnostic':teacher,'state_hash':before}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['BP','DFA','FROZEN'],required=True)
    ap.add_argument('--passes',type=int,default=64);ap.add_argument('--hidden',type=int,default=128)
    ap.add_argument('--lr',type=float,default=.03);ap.add_argument('--decay',type=float,default=.001)
    ap.add_argument('--tag',required=True);ap.add_argument('--world',default='V144-EXERCISE-C2')
    args=ap.parse_args()
    if not args.world.startswith('V144-EXERCISE-') or args.passes>64:
        raise RuntimeError('this runner admits cycle fixtures only, not full screen or holdback')
    ledger=Ledger(args.tag,'C_EXERCISE');start=time.process_time()
    trace=ByteTrace(lb1.BC.R_TABLE,lb1.fastmem.PERM,lb1.DELTA,lb1.fastmem.FEPerm)
    w=task.world(args.world);training=w['doses']['40'];pairs=task.oriented(training)
    write_json(HERE/'cycles'/(args.tag+'_MANIFEST.json'),{'world':w,'passes':args.passes,'mode':args.mode,
               'model_seed_name':'weights:'+args.world,'evaluation_is_development':True})
    source_lock(HERE/'cycles'/(args.tag+'_SOURCE_LOCK.json'))
    model=Reference(trace,args.world,args.hidden,args.lr,args.decay,args.mode)
    rng=np.random.default_rng(seed('order:'+args.world));snapshots=[]
    snapshots.append(snapshot(model,w,training,'birth',time.process_time()-start))
    checkpoints={1,4,16,64,args.passes}
    for ep in range(1,args.passes+1):
        for ix in rng.permutation(len(pairs)):
            pair=pairs[ix];ledger.emit('record_start',record=model.records+1)
            before=model.target_tokens;model.train(task.prompt(pair),task.answer(pair))
            ledger.emit('record_complete',record=model.records,target_tokens=model.target_tokens-before)
        if ep in checkpoints:
            snapshots.append(snapshot(model,w,training,'pass'+str(ep),time.process_time()-start))
            write_json(HERE/'results'/(args.tag+'_PROGRESS.json'),snapshots[-1])
        if time.process_time()-start>180:raise RuntimeError('exercise CPU ceiling reached; incomplete')
    out=HERE/'results'/args.tag;out.mkdir(exist_ok=False)
    (out/'CHECKPOINT.pkl').write_bytes(pickle.dumps((model,rng.bit_generator.state),protocol=5))
    result={'id':args.tag,'complete':True,'settings':vars(args),'records':model.records,
            'observed_target_tokens':model.target_tokens,'cpu_s':time.process_time()-start,
            'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'metrics':{'snapshots':snapshots,'countermodels':countermodels(training,w['held'])},
            'stage_C_full_screen':False,'holdback_opened':False}
    write_json(out/'RESULT.json',result);ledger.emit('complete',records=model.records,target_tokens=model.target_tokens)
    print(json.dumps({k:result[k] for k in ['id','records','observed_target_tokens','cpu_s']}))
    print(json.dumps({'held':snapshots[-1]['held']['orbit_exact'],'taught':snapshots[-1]['taught']['orbit_exact'],
                      'own_eos':snapshots[-1]['held']['own_eos']}))


if __name__=='__main__':main()
