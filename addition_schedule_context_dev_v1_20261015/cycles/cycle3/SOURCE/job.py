"""Fixed-roster real job runner; science admission is closed during qualification."""
import copy
import hashlib
import json
import os
import signal
import time
import traceback
from pathlib import Path
import bootstrap,config,data,models,engine,checkpoint,probes,audits
from artifacts import write,read,now
from identity import source_id
PAUSE_REQUESTED=False


def admit(kind,world,technical):
    allowed=config.TECH_WORLDS if technical else config.SCIENCE_WORLDS if kind=='addition' else config.NATIVE_MAPS
    if world not in allowed:raise ValueError('world outside admitted namespace')
    if not technical:
        p=config.HERE/'ops/SCIENCE_AUTHORIZATION.json'
        if not p.exists():raise PermissionError('formal science has not been authorized')
        a=read(p)
        if a.get('authorized') is not True or a.get('source_id')!=source_id():raise PermissionError('science authorization/source binding failed')
        if not read(config.HERE/'QUALIFICATION.json')['accepted']:raise PermissionError('qualification incomplete')


def journal(path,event):
    with Path(path).open('a') as f:
        f.write(json.dumps(event,sort_keys=True,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())


def journal_cursor(path):
    cursor=0;pending=None
    if not Path(path).exists():return 0
    for line in Path(path).read_text().splitlines():
        row=json.loads(line)
        if row['event']=='START':
            if pending is not None or row['cursor']!=cursor+1:raise ValueError('invalid journal start sequence')
            pending=row['cursor']
        elif row['event']=='COMMIT':
            if pending!=row['cursor']:raise ValueError('invalid journal commit sequence')
            cursor=pending;pending=None
        else:raise ValueError('unknown journal event')
    if pending is not None:raise ValueError('ambiguous partial record; replay forbidden')
    return cursor


def identity(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def prepare(folder,manifest,resume,kind):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    if (folder/'RECEIPT.json.gz').exists():raise FileExistsError('completed job cannot be overwritten or replayed')
    mid=identity(manifest)
    if (folder/'MANIFEST.json').exists():
        if not resume:raise FileExistsError('existing job requires an explicit safe resume')
        if read(folder/'MANIFEST.json')!=manifest:raise ValueError('manifest/source changed')
        if read(folder/'STATUS.json')['state']!='PAUSED':raise ValueError('only an acknowledged safe pause may resume')
        model,meta=checkpoint.load(folder/'checkpoints/CURRENT.json.gz',expected_kind=kind,manifest_id=mid)
        if journal_cursor(folder/'journal.jsonl')!=meta['cursor']:raise ValueError('checkpoint is behind executed records; replay forbidden')
        return mid,model,meta['job_state']
    if resume:raise FileNotFoundError('no job to resume')
    write(folder/'MANIFEST.json',manifest)
    return mid,None,dict(history_index=0,index=0,cursor=0,histories={})


def next_id(histories,source_rows,hi,index):
    if index<len(source_rows[histories[hi]]):return histories[hi]+'/'+source_rows[histories[hi]][index]['id']
    if hi+1<len(histories):return histories[hi+1]+'/'+source_rows[histories[hi+1]][0]['id']
    return None


def persist(folder,model,state,mid,nxt,*,permanent=None):
    start=time.process_time();path=folder/'checkpoints/CURRENT.json.gz'
    checkpoint.save(model,path,cursor=state['cursor'],next_id=nxt,manifest_id=mid,job_state=state)
    if permanent:
        target=path.with_name(permanent+'.json.gz');target.write_bytes(path.read_bytes())
    elapsed=time.process_time()-start
    journal(folder/'operation_costs.jsonl',dict(event='CHECKPOINT',cursor=state['cursor'],cpu=elapsed,utc=now()))
    return elapsed


def should_pause(state,stop_after,deadline):
    return PAUSE_REQUESTED or (stop_after is not None and state['cursor']>=stop_after) or (deadline is not None and time.monotonic()>=deadline)


def execute_record(folder,state,label,fn,technical):
    index=state['cursor']+1
    journal(folder/'journal.jsonl',dict(event='START',cursor=index,id=label,utc=now()))
    if technical:
        from ledger import apply
        row=apply(3,str(folder.name)+'/'+label,fn)
    else:row=fn()
    journal(folder/'journal.jsonl',dict(event='COMMIT',cursor=index,id=label,utc=now()))
    state['cursor']=index
    return row


def run_addition(folder,world,schedule,*,technical=False,resume=False,stop_after=None,deadline=None):
    admit('addition',world,technical)
    if schedule not in config.SCHEDULES:raise ValueError('fixed schedules only')
    if stop_after is not None and not technical:raise ValueError('test-only stop boundary')
    folder=Path(folder);source=data.make_addition(world,technical);source_rows={h:data.ordered(source,schedule,h) for h in config.HISTORIES}
    manifest=dict(schema=config.SCHEMA,kind='addition',world=world,schedule=schedule,technical=technical,
        source_id=source_id(),stream_ids={h:identity(source_rows[h]) for h in config.HISTORIES},expected_records=3*(source['old_n']+source['new_n']))
    mid,restored,state=prepare(folder,manifest,resume,'KEEP');started=time.monotonic()
    try:
        for hi in range(state['history_index'],len(config.HISTORIES)):
            history=config.HISTORIES[hi];rows=source_rows[history]
            model=restored if hi==state['history_index'] and restored is not None else models.Keep(data.seed(world,'model'))
            restored=None
            h=state['histories'].setdefault(history,dict(rows=[],curve=[],old_count=0,new_count=0,seen=[],captures=[],checkpoint_cpu=0.,query_cpu=0.))
            index=state['index'] if hi==state['history_index'] else 0
            state['history_index']=hi;state['index']=index
            plan=data.checkpoint_plan(source,schedule)
            def probe(t,phase='OBSERVATION'):
                if any(q['T']==t and q['phase']==phase for q in h['curve']):return
                q=probes.addition(model,source['groups'],set(h['seen']))
                examples=[data.example(p) for p in source['groups']['held']]
                h['query_cpu']+=q['query_cpu']
                h['curve'].append(dict(T=t,phase=phase,tags=plan.get(t,[]) if phase=='OBSERVATION' else ['AFTER_REST'],
                    old_count=h['old_count'],new_count=h['new_count'],operative_digest=model.digest(),
                    production_cpu=model.resource['production_cpu'],shortcuts=probes.modes(h['rows'],examples),**q))
            if index==0:
                probe(0)
                h['checkpoint_cpu']+=persist(folder,model,state,mid,next_id(config.HISTORIES,source_rows,hi,0),permanent=f'{history}_T0')
            for position in range(index,len(rows)):
                row=rows[position];capture=history=='W' and schedule=='BLOCKED' and row['kind']=='NEW' and row['source_index'] in config.DIAG_NEW
                actual=execute_record(folder,state,history+'/'+row['id'],lambda:engine.apply_addition(model,row,capture=capture),technical)
                h['rows'].append(actual);state['index']=position+1
                h['old_count' if row['kind']=='OLD' else 'new_count']+=1
                if row['prompt'] not in h['seen']:h['seen'].append(row['prompt'])
                if capture:
                    # Proposal already executed exactly once. Save its gradients/deltas for later read-only views.
                    capfile=folder/'proposals'/f'{history}_NEW{row["source_index"]}.json.gz'
                    proposal=models.Keep(0);proposal.set_state(model.last_capture['post'])
                    checkpoint.save(proposal,capfile,cursor=state['index'],next_id='OFFLINE_ONLY',manifest_id=mid,job_state=model.last_capture)
                    h['captures'].append(str(capfile.relative_to(folder)))
                t=position+1
                if t in plan:probe(t)
                if t in ((len(rows),) if technical else (16384,17408)):
                    model.rest(86400.);probe(t,'AFTER_REST')
                if t==len(rows):
                    h['resource']=copy.deepcopy(model.resource);h['final_digest']=model.digest()
                permanent=t in ((2,len(rows)) if technical else (8192,16384,17408)) or capture
                if permanent:
                    h['checkpoint_cpu']+=persist(folder,model,state,mid,next_id(config.HISTORIES,source_rows,hi,t),permanent=f'{history}_T{t}')
                write(folder/'STATUS.json',dict(state='RUNNING',utc=now(),cursor=state['cursor'],history=history,T=t,expected_records=manifest['expected_records'],technical=technical))
                if should_pause(state,stop_after,deadline):
                    persist(folder,model,state,mid,next_id(config.HISTORIES,source_rows,hi,t))
                    write(folder/'STATUS.json',dict(state='PAUSED',utc=now(),cursor=state['cursor'],technical=technical,safe_boundary=True))
                    return dict(state='PAUSED',cursor=state['cursor'])
            state['index']=0;state['history_index']=hi+1
        receipt=dict(manifest=manifest,histories=state['histories'],record_applications=state['cursor'],technical=technical,elapsed=time.monotonic()-started)
        certificate=audits.audit_addition_job(receipt)
        write(folder/'RECEIPT.json.gz',receipt);write(folder/'AUDIT.json',certificate)
        write(folder/'STATUS.json',dict(state='COMPLETE',utc=now(),cursor=state['cursor'],technical=technical,accepted=True))
        return receipt
    except BaseException as e:
        write(folder/'FAILURE.json',dict(utc=now(),error=repr(e),traceback=traceback.format_exc(),cursor=state['cursor'],automatic_retry=False))
        write(folder/'STATUS.json',dict(state='FAILED',utc=now(),cursor=state['cursor'],technical=technical));raise


def run_native(folder,world,variant,*,technical=False,resume=False,stop_after=None,deadline=None):
    admit('native',world,technical)
    if variant not in config.NATIVE_VARIANTS:raise ValueError('fixed native roster')
    if stop_after is not None and not technical:raise ValueError('test-only stop boundary')
    fixture=data.native_fixture(world);n=4 if technical else 768
    stream=data.shuffled_cycles(fixture,n,__import__('random').Random(data.seed(world,'native-order')))
    source_rows={h:[dict(r,id=r['id']) for r in stream] for h in ('W','NO_WRITE')}
    manifest=dict(schema=config.SCHEMA,kind='native',world=world,variant=variant,technical=technical,
       source_id=source_id(),expected_records=2*n,stream_ids=identity([r['id'] for r in stream]))
    folder=Path(folder);mid,restored,state=prepare(folder,manifest,resume,variant);histories=('W','NO_WRITE')
    try:
        for hi in range(state['history_index'],2):
            history=histories[hi];model=restored if hi==state['history_index'] and restored is not None else models.Native(variant);restored=None
            h=state['histories'].setdefault(history,dict(rows=[],curve=[],checkpoint_cpu=0.,query_cpu=0.))
            index=state['index'] if hi==state['history_index'] else 0;state['history_index']=hi;state['index']=index
            def probe(t,phase='OBSERVATION'):
                if any(q['T']==t and q['phase']==phase for q in h['curve']):return
                q=probes.native(model,fixture,cap=2 if technical else 16);h['query_cpu']+=q['query_cpu']
                if t==n and phase=='OBSERVATION':
                    geometry=probes.native_geometry(model,fixture)
                    h['geometry']=geometry;h['query_cpu']+=geometry['query_cpu']
                h['curve'].append(dict(T=t,phase=phase,**q))
            if index==0:
                probe(0);persist(folder,model,state,mid,next_id(histories,source_rows,hi,0),permanent=history+'_BIRTH')
            for position in range(index,n):
                item=stream[position]
                r=execute_record(folder,state,history+'/'+str(position)+':'+item['id'],lambda:engine.apply_native(model,item,history=='W'),technical)
                h['rows'].append(r);state['index']=position+1;t=position+1
                if t in ((n,) if technical else (16,64,256,768)):probe(t)
                if t==n:
                    model.rest(86400.);probe(t,'AFTER_REST')
                    h['resource']=copy.deepcopy(model.resource);h['final_digest']=model.digest()
                    persist(folder,model,state,mid,next_id(histories,source_rows,hi,t),permanent=history+'_FINAL')
                write(folder/'STATUS.json',dict(state='RUNNING',utc=now(),cursor=state['cursor'],history=history,T=t,technical=technical))
                if should_pause(state,stop_after,deadline):
                    persist(folder,model,state,mid,next_id(histories,source_rows,hi,t))
                    write(folder/'STATUS.json',dict(state='PAUSED',utc=now(),cursor=state['cursor'],technical=technical,safe_boundary=True))
                    return dict(state='PAUSED',cursor=state['cursor'])
            state['index']=0;state['history_index']=hi+1
        receipt=dict(manifest=manifest,histories=state['histories'],record_applications=state['cursor'],technical=technical)
        certificate=audits.audit_native_job(receipt)
        write(folder/'RECEIPT.json.gz',receipt);write(folder/'AUDIT.json',certificate)
        write(folder/'STATUS.json',dict(state='COMPLETE',utc=now(),cursor=state['cursor'],technical=technical,accepted=True))
        return receipt
    except BaseException as e:
        write(folder/'FAILURE.json',dict(utc=now(),error=repr(e),traceback=traceback.format_exc(),cursor=state['cursor'],automatic_retry=False))
        write(folder/'STATUS.json',dict(state='FAILED',utc=now(),cursor=state['cursor'],technical=technical));raise


# Per-segment accounting survives an acknowledged pause; it is separate from model training CPU.
def measured(fn):
    from functools import wraps
    @wraps(fn)
    def call(folder,*args,**kwargs):
        cpu=time.process_time();wall=time.monotonic()
        try:return fn(folder,*args,**kwargs)
        finally:
            p=Path(folder)
            if p.exists():journal(p/'segment_costs.jsonl',dict(event='SEGMENT',cpu=time.process_time()-cpu,wall=time.monotonic()-wall,utc=now()))
    return call

run_addition=measured(run_addition)
run_native=measured(run_native)
