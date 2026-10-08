"""Whole-world DEV runner; no scientific IDs, selection, or automatic resume."""
import argparse,gzip,hashlib,json,os,time
from pathlib import Path
import numpy as np
import dependencies,settings,data
from learners import build

SOURCE_NAMES=('dependencies.py','settings.py','representation.py','learners.py','data.py','runner.py','audit.py','technical.py','addition_checkpoint.py')
def source_lock():
    parent=dependencies.inherited_checkpoint.source_identity()
    files={name:hashlib.sha256((dependencies.ROOT/name).read_bytes()).hexdigest() for name in SOURCE_NAMES}
    return dict(schema='ADDITION_DEV_LOCK',parent_identity=parent,files=files,science_authorized=False)

def raw_consume(model,raw):
    return [model.feed(int(byte),model.t+settings.DT) for byte in raw]

def probe(model,world,phase,erase=False):
    before=model.state_digest();c=model.clone();c.plastic=False
    if erase:c.erase()
    rng=np.random.default_rng(world+1000+settings.PHASES.index(phase))
    order=[data.ALL[i] for i in rng.permutation(len(data.ALL))]
    rows=[];cpu=time.process_time()
    for pair in order:
        trace=raw_consume(c,data.expression(pair)+b'\n')
        predictions=[r['prediction'] for r in trace if r['prediction'] is not None]
        if len(predictions)!=1:raise AssertionError('one native output before feedback')
        pred=predictions[0];truth=48+pair[0]+pair[1]
        rows.append(dict(pair=list(pair),truth=truth,emitted=pred['emitted'],probabilities=pred['probabilities'],
            exact=pred['emitted']==truth,error=abs(pred['emitted']-truth),loss_bits=-np.log2(pred['probabilities'][truth-48]),trace=trace))
    elapsed=time.process_time()-cpu
    if model.state_digest()!=before:raise AssertionError('read-only probe changed operative state')
    groups={}
    for name,pairs in (('old',data.OLD),('new',data.NEW),('held',data.HELD),('all',data.ALL)):
        selected=[r for r in rows if tuple(r['pair']) in pairs]
        groups[name]=dict(exact=sum(r['exact'] for r in selected)/len(selected),mae=sum(r['error'] for r in selected)/len(selected),
            bpb=sum(r['loss_bits'] for r in selected)/len(selected),questions=len(selected))
    return dict(rows=rows,groups=groups,cpu_seconds=elapsed,erase=erase,operative_digest=before)

def run(world,quick=False):
    settings.require_dev(world);fixture=data.generate(world,quick)
    result=dict(schema='ADDITION_DEV_RECEIPT',world=world,quick=bool(quick),version=settings.VERSION,source=source_lock(),
        evidence='EXPLORATORY_E3_ATTEMPT',fixture=fixture,arms={},science_worlds=0,training_histories=6)
    start_wall=time.monotonic();start_cpu=time.process_time()
    phases=('old','day1') if quick else settings.PHASES
    for arm in settings.ARMS:
        models={branch:build(arm) for branch in settings.BRANCHES}
        initial=[m.state_digest() for m in models.values()]
        if len(set(initial))!=1:raise AssertionError('histories must have identical initial state')
        out=dict(training={},probes={},states={},birth={},costs={})
        for branch,m in models.items():
            out['birth'][branch]=[s.birth_record for s in m.stores] if arm=='NATIVE' else [m.encoder.birth]
        for phase in phases:
            print(f'DEV {world} {arm} {phase}',flush=True)
            if phase.startswith('day'):
                for m in models.values():m.rest(settings.DAY)
            else:
                out['training'][phase]={}
                for branch,m in models.items():
                    m.plastic=data.write_allowed(branch,phase);rows=[];cpu=time.process_time()
                    records=fixture['records'][phase]
                    for record in records:
                        raw=bytes.fromhex(record['shuffled_raw'] if branch=='SHUFFLED' else record['raw'])
                        rows.append(raw_consume(m,raw))
                    out['training'][phase][branch]=rows
                    out['costs'][phase+'/'+branch]=dict(cpu_seconds=time.process_time()-cpu,raw_bytes=sum(len(x) for x in rows),records=len(records))
            out['probes'][phase]={};out['states'][phase]={}
            for branch,m in models.items():
                out['probes'][phase][branch]=probe(m,world,phase)
                out['states'][phase][branch]=dict(digest=m.state_digest(),model_time=m.t,bytes_seen=m.bytes_seen,mutable_bytes=m.mutable_bytes())
        final=phases[-1]
        out['erased_final']=probe(models['W'],world,final,erase=True)
        result['arms'][arm]=out
    result['wall_seconds']=time.monotonic()-start_wall;result['cpu_seconds']=time.process_time()-start_cpu
    result['training_bytes']=sum(c['raw_bytes'] for a in result['arms'].values() for c in a['costs'].values())
    return result

def save(receipt,path):
    path=Path(path)
    if path.exists():raise ValueError('receipt exists; no trajectory rerun/overwrite')
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.pending')
    with tmp.open('xb') as raw:
        with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as z:z.write(json.dumps(receipt,sort_keys=True,allow_nan=False).encode())
        raw.flush();os.fsync(raw.fileno())
    os.replace(tmp,path)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--world',type=int,required=True);parser.add_argument('--out',required=True);parser.add_argument('--quick',action='store_true');args=parser.parse_args()
    if Path(args.out).exists():raise SystemExit('Existing receipt; not running again')
    from audit import audit
    receipt=run(args.world,args.quick);verdict=audit(receipt);receipt['independent_audit']=verdict;save(receipt,args.out)
    print(json.dumps(dict(output=args.out,audit=verdict,wall_seconds=receipt['wall_seconds'],training_bytes=receipt['training_bytes'])),flush=True)
