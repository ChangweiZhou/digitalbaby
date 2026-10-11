"""Technical whole-dose jobs only. No learned metadata or truth passed to model."""
import argparse,gzip,hashlib,json,os,resource,time,gc
from pathlib import Path
import numpy as np
import dependencies,settings,data
from learners import Learner
ROOT=Path(__file__).resolve().parent
SOURCES=('settings.py','dependencies.py','representation.py','learners.py','data.py','runner.py','audit.py','decision.py','technical.py','cycles.py','ops.py','reduce.py','package.py')
def source_lock():
    return dict(files={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCES},parent=dependencies.inherited_checkpoint.source_identity(),science_authorized=False)
def consume(m,raw,tick):
    rows=[];cpu=0.
    # Audit allocations must not trigger an unrelated whole-heap GC during timed core work.
    # Re-enable at the record boundary; inclusive job CPU still counts all housekeeping.
    enabled=gc.isenabled();start_inclusive=time.process_time();gc.disable()
    try:
        for b in raw:
            ids=m.pending['ids'] if m.pending is not None and b in settings.ALPHABET else None
            tick+=1;start=time.process_time();row=m.feed(int(b),tick*settings.DT);cpu+=time.process_time()-start
            if ids is not None:
                capture=time.process_time()
                row['active_state_after']={k:v[ids].tolist() for k,v in m.arrays().items()}
                row['audit_capture_cpu_seconds']=time.process_time()-capture
            rows.append(row)
    finally:
        if enabled:gc.enable()
    if rows:rows[-1]['record_inclusive_cpu_seconds']=time.process_time()-start_inclusive
    return rows,tick,cpu
def probe(model,world,phase,erase=False):
    before=model.state_digest();c=model.clone();c.plastic=False
    if erase:c.erase()
    rows=[];tick=round(c.t/settings.DT)
    order=np.random.default_rng(world+1000+settings.PHASES.index(phase)).permutation(25)
    for index in order:
        pair=data.ALL[index];trace,tick,_=consume(c,data.expression(pair)+b'\n',tick)
        pred=next(row['prediction'] for row in trace if row['prediction'] is not None)
        rows.append(dict(pair=list(pair),truth=48+sum(pair),exact=pred['emitted']==48+sum(pair),trace=trace,**pred))
    if before!=model.state_digest():raise AssertionError('probe parent changed')
    groups={g:float(np.mean([r['exact'] for r in rows if tuple(r['pair']) in pairs])) for g,pairs in (('old',data.OLD),('new',data.NEW),('held',data.HELD))}
    return dict(rows=rows,groups=groups,parent_digest=before,erase=erase)
def run(world,arm,dose):
    settings.require_world(world)
    if arm not in settings.ARMS or dose not in settings.DOSES:raise ValueError('roster')
    start=time.monotonic();cpu=time.process_time();master=data.generate(world)
    receipt=dict(schema='ADDITION_EFFICIENCY_TECH_JOB',world=world,arm=arm,dose=dose,source=source_lock(),master=master,branches={},technical=True,science_worlds=0)
    for branch in settings.BRANCHES:
        m=Learner(arm);tick=0;out=dict(training={},probes={},states={},costs={},birth=m.encoder.birth,initial_digest=m.state_digest())
        for phase,end,day in (('OLD','OLD_END','DAY1'),('NEW','NEW_END','FINAL')):
            m.plastic=data.permitted(branch,phase);records=[];core_cpu=0.
            for block,entries in enumerate(master['blocks'][phase]):
                if block>=dose//10:
                    dt=len(entries)*6*settings.DT;s=time.process_time();m.rest(dt);core_cpu+=time.process_time()-s;tick+=len(entries)*6
                    continue
                for entry in entries:
                    raw=bytes.fromhex(entry['shuffled_raw'] if branch=='SHUFFLED' else entry['raw'])
                    trace,tick,spent=consume(m,raw,tick);core_cpu+=spent
                    padding=6-len(raw)
                    if padding:
                        s=time.process_time();m.rest(padding*settings.DT);core_cpu+=time.process_time()-s;tick+=padding
                    records.append(dict(block=block,trace=trace,padding_ticks=padding))
            out['training'][phase]=records;out['costs'][phase]=dict(core_cpu_seconds=core_cpu,training_bytes=sum(len(r['trace']) for r in records),records=len(records))
            for checkpoint,isday in ((end,False),(day,True)):
                if isday:
                    s=time.process_time();m.rest(settings.DAY);out['costs'][phase]['core_cpu_seconds']+=time.process_time()-s;tick+=settings.DAY_TICKS
                out['states'][checkpoint]=dict(arrays=m.snapshot(),digest=m.state_digest(),ticks=tick,time=m.t,bytes_seen=m.bytes_seen,payload_bytes=m.payload_bytes())
                out['probes'][checkpoint]=probe(m,world,checkpoint)
        out['erased']=probe(m,world,'FINAL',True)
        receipt['branches'][branch]=out
    receipt.update(training_histories=3,training_bytes=sum(c['training_bytes'] for o in receipt['branches'].values() for c in o['costs'].values()),training_records=sum(c['records'] for o in receipt['branches'].values() for c in o['costs'].values()),run_wall_seconds=time.monotonic()-start,run_cpu_seconds=time.process_time()-cpu,peak_rss=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if os.uname().sysname=='Darwin' else 1024))
    return receipt
def save(obj,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise ValueError('No overwrite / rerun')
    tmp=path.with_name(path.name+'.pending')
    with tmp.open('xb') as f:
        with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as z:z.write(json.dumps(obj,sort_keys=True,allow_nan=False).encode())
        f.flush();os.fsync(f.fileno())
    os.link(tmp,path);tmp.unlink()
    fd=os.open(path.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
def execute(world,arm,dose,out):
    out=Path(out);attempt=out.with_name(out.name+'.attempt.json')
    out.parent.mkdir(parents=True,exist_ok=True)
    with attempt.open('x') as f:json.dump(dict(world=world,arm=arm,dose=dose,source=source_lock(),started=time.time()),f)
    r=run(world,arm,dose);save(r,out.parent/'provisional'/out.name)
    from audit import audit
    s=time.process_time();r['independent_audit']=audit(r);r['audit_cpu_seconds']=time.process_time()-s
    if source_lock()!=r['source']:raise AssertionError('source changed before atomic final commit')
    save(r,out);return r
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--world',type=int,required=True);p.add_argument('--arm',choices=settings.ARMS,required=True);p.add_argument('--dose',type=int,choices=settings.DOSES,required=True);p.add_argument('--out',required=True);a=p.parse_args()
    r=execute(a.world,a.arm,a.dose,a.out);print(json.dumps({k:r[k] for k in ('world','arm','dose','training_histories','training_bytes','run_wall_seconds','independent_audit')}))
