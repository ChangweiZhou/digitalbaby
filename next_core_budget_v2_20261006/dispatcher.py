"""Detached-capable, deadline-bounded queue; no automatic scientific restart."""
import argparse,subprocess,signal,time,json,os,resource,stat
MODULE_WALL=time.monotonic()
from pathlib import Path
from runtime import ROOT,load_plan,execution_identity,atomic_json,read_receipt,file_sha
from locks import validate_manifest
from protocol import job_key
from job_audit import audit,audit_unavailable
INTERRUPTED=False
def disk_bytes(root,*,skip_scratch=False,exclude=()):
    """Count each extant regular file once, including temporary writes.

    Atomic writers may replace/unlink an enumerated name before stat. Only
    ENOENT is benign; permission and other I/O errors still stop the run.
    """
    excluded=set(exclude);total=0
    for path in root.rglob('*'):
        if path in excluded or (skip_scratch and 'scratch' in path.parts):continue
        try:info=path.stat()
        except FileNotFoundError:continue
        if stat.S_ISREG(info.st_mode):total+=info.st_size
    return total
def interrupted(sig,frame):
    global INTERRUPTED
    INTERRUPTED=True
def decision(elapsed,limits):
    if elapsed>=limits['worker_exit_deadline_seconds']:return 'WATCHDOG_STOP'
    if elapsed>=limits['safe_pause_request_seconds']:return 'REQUEST_SAFE_PAUSE'
    if elapsed>=limits['stop_dispatch_seconds']:return 'DRAIN'
    return 'DISPATCH'
def poll_owned(v):
    # Exactly one reaper owns each child; wait4 retains CPU accounting even after SIGKILL.
    if v.get('code') is not None:return v['code']
    pid,status,usage=os.wait4(v['proc'].pid,os.WNOHANG)
    if not pid:return None
    code=os.waitstatus_to_exitcode(status);v['code']=code;v['proc'].returncode=code;v['reaped_CPU_s']=usage.ru_utime+usage.ru_stime
    return code
def signal_owned(v,sig):
    if poll_owned(v) is None:
        try:os.kill(v['proc'].pid,sig)
        except ProcessLookupError:pass
        return True
    return False
def ready(j,folder,done):return j['arm']!='S3_RAND' or f"{j['world']}_S3_CUE_{j['assay']}" in done
def dispatch(m,folder,*,resume=False,session_origin=None):
    global INTERRUPTED
    INTERRUPTED=False;validate_manifest(m)
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    if (folder/'FAILURE.json').exists() and m['stage']!='qualification':raise ValueError('scientific failure requires review, no automatic restart')
    ident=execution_identity();limits=m['session'];pending=[];done=[];skips=[]
    started=MODULE_WALL if session_origin is None else session_origin;active={};children=[];failure=None;forced=False;paused=False;unavailable=[];last_resource=0.;last_status=0.;disk=0
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,interrupted)
    mf=folder/'MANIFEST.json.gz'
    from runtime import write_receipt
    if mf.exists() and read_receipt(mf)!=m:raise ValueError('dispatcher manifest changed')
    if not mf.exists():write_receipt(mf,m)
    for j in m['jobs']:
        k=job_key(j);p=folder/'receipts'/f'{k}.json.gz'
        if p.exists():
            audit(read_receipt(p),ident,folder,allow_short=m['stage']=='qualification',load_export=False);done.append(k);skips.append(k)
        else:
            marker=folder/'unavailable'/f'{k}.json.gz'
            if marker.exists():audit_unavailable(read_receipt(marker),ident,folder);unavailable.append(k)
            else:pending.append(j)
    manifest_bytes=mf.read_bytes()
    caffeinate=None
    if Path('/usr/bin/caffeinate').exists():caffeinate=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())])
    launch={'pid':os.getpid(),'stage':m['stage'],'workers':m['workers'],'manifest_sha256':file_sha(mf),'started_unix':time.time(),'caffeinate_pid':None if caffeinate is None else caffeinate.pid}
    atomic_json(folder/'LAUNCH.json',launch)
    old_attempts=[]
    attempt_paths=list((folder/'dispatcher_attempts').glob('*.json')) if m['stage']=='qualification' else list((ROOT/'science').glob('*/dispatcher_attempts/*.json'))
    for p in attempt_paths:old_attempts.append(json.loads(p.read_text()))
    previous_worker=sum(x['worker_s'] for x in old_attempts);previous_cpu=sum(x['process_CPU_s'] for x in old_attempts)
    ledger_worker=previous_worker;ledger_cpu=previous_cpu
    def status(state):
        atomic_json(folder/'STATUS.json',{'stage':m['stage'],'state':state,'pid':os.getpid(),'updated_unix':time.time(),'completed_jobs':len(done),'planned_jobs':len(m['jobs']),'diagnostic_unavailable_jobs':len(unavailable),'completed_lives':sum(load_plan()['assays'][j['assay']]['lives_per_job'] if m['short'] is None else 1 for j in m['jobs'] if job_key(j) in done),'skipped_committed':len(skips),'pending_jobs':len(pending),'active_jobs':list(active),'session_elapsed_s':time.monotonic()-started,'charged_worker_s':ledger_worker+sum(time.monotonic()-v['at'] for v in active.values()),'process_CPU_s_committed_attempts':ledger_cpu,'execution_source_identity':ident})
    try:
        while pending or active:
            elapsed=time.monotonic()-started;mode=decision(elapsed,limits)
            resource_limit=False
            if m['stage']!='qualification':
                p=load_plan();r=p['resource']
                if elapsed-last_resource>=2:
                    disk=disk_bytes(ROOT,skip_scratch=True);last_resource=elapsed
                charged=ledger_worker+sum(time.monotonic()-v['at'] for v in active.values())
                live_cpu=0.
                for key,v in active.items():
                    hp=folder/'active'/f'{key}.json'
                    try:heartbeat_text=hp.read_text()
                    except FileNotFoundError:continue
                    else:
                        h=json.loads(heartbeat_text)
                        if h['pid']==v['proc'].pid:
                            live_cpu+=h['attempt_CPU_s']
                            if h['peak_rss_bytes']>r['worker_RSS_GiB_cap']*1024**3:failure={'error':f'worker RSS exceeds cap: {key}','category':'RESOURCE_INCOMPLETE'}
                resource_limit=charged>=r['science_charged_worker_hours_cap']*3600 or ledger_cpu+live_cpu>=r['science_process_CPU_hours_cap']*3600 or disk>r['all_deliverables_including_archive_GiB_cap']*1024**3
            if resource_limit:failure={'error':'global science resource cap reached','category':'RESOURCE_INCOMPLETE'}
            if mode in ('REQUEST_SAFE_PAUSE','WATCHDOG_STOP') or INTERRUPTED or failure:
                paused=True
                for v in active.values():
                    if not v['signaled']:signal_owned(v,signal.SIGTERM);v['signaled']=True
            if mode=='WATCHDOG_STOP':
                for v in active.values():
                    if signal_owned(v,signal.SIGKILL):forced=True
            for k,v in list(active.items()):
                code=poll_owned(v);reaped_CPU=v.get('reaped_CPU_s',0.)
                if code is None:
                    if time.monotonic()-v['at']>3600 and not failure:failure={'error':f'job worker timeout: {k}','category':'EXECUTION_FAILURE'}
                    continue
                v['log'].close();elapsed_job=time.monotonic()-v['at'];ledger_worker+=elapsed_job
                data=v['log_path'].read_text();lines=data.splitlines()
                result=None
                for line in reversed(lines):
                    try:obj=json.loads(line)
                    except ValueError:continue
                    if isinstance(obj,dict) and 'status' in obj:result=obj;break
                attempt_id=f'{k}_{v["proc"].pid}_{time.time_ns()}'
                cpu=0.
                apath=folder/'attempts'/f'{k}_{v["proc"].pid}.json'
                if apath.exists():cpu=json.loads(apath.read_text())['cpu_s']
                cpu=max(cpu,reaped_CPU);ledger_cpu+=cpu
                atomic_json(folder/'dispatcher_attempts'/f'{attempt_id}.json',{'job':k,'pid':v['proc'].pid,'exit_code':code,'worker_s':elapsed_job,'process_CPU_s':cpu,'result':result})
                p=folder/'receipts'/f'{k}.json.gz'
                if code==0 and p.exists():
                    audit(read_receipt(p),ident,folder,allow_short=m['stage']=='qualification',load_export=False);done.append(k)
                elif code==0 and result and result['status']=='DIAGNOSTIC_NOT_QUALIFIED':
                    audit_unavailable(read_receipt(folder/'unavailable'/f'{k}.json.gz'),ident,folder);unavailable.append(k)
                elif code==0 and result and result['status']=='PAUSED_SAFE_RECORD':pending.append(v['job']);paused=True
                else:failure=failure or {'error':f'worker {k} exited {code}; {data[-2000:]}','category':'EXECUTION_FAILURE'}
                del active[k]
            if mode=='DISPATCH' and not paused and not failure:
                while len(active)<m['workers']:
                    candidates=[j for j in pending if ready(j,folder,done)]
                    if not candidates:break
                    j=candidates[0];pending.remove(j);k=job_key(j)
                    cp=folder/'checkpoints'/f'{k}.npz'
                    if cp.exists() and not resume:raise ValueError('explicit session resume required')
                    cmd=[os.sys.executable,'-u',str(ROOT/'job_runner.py'),'--world',str(j['world']),'--arm',j['arm'],'--assay',j['assay'],'--folder',str(folder),'--stage',m['stage'],'--manifest',str(mf)]
                    if m['short'] is not None:cmd+=['--short',str(m['short'])]
                    if cp.exists():cmd+=['--resume']
                    lp=folder/'logs'/f'{k}_{time.time_ns()}.log';lp.parent.mkdir(exist_ok=True);lf=lp.open('w')
                    proc=subprocess.Popen(cmd,stdout=lf,stderr=subprocess.STDOUT,start_new_session=True)
                    active[k]={'proc':proc,'job':j,'at':time.monotonic(),'log':lf,'log_path':lp,'signaled':False};children.append(proc.pid)
            if elapsed-last_status>=.5:
                status('STOPPING_FAILURE' if failure else ('PAUSING' if paused or mode!='DISPATCH' else 'RUNNING'));last_status=elapsed
            if not active and (paused or mode!='DISPATCH' or failure):break
            if pending and not active and not any(ready(j,folder,done) for j in pending):raise ValueError('unsatisfied yoke dependency')
            time.sleep(.1)
        if forced:failure={'error':'worker failed safe pause deadline; last reliable checkpoint preserved if present','category':'STOP_NEEDS_REVIEW','worker_exit_detail':failure}
        state=('STOP_NEEDS_REVIEW' if forced else 'FAILED') if failure else ('PAUSED_SESSION_LIMIT' if pending else 'STAGE_COMPLETE')
        out={'state':state,'stage':m['stage'],'complete_jobs':len(done),'planned_jobs':len(m['jobs']),'pending_jobs':pending,'child_pids':children,'skipped_committed':skips,'diagnostic_unavailable':unavailable,'session_elapsed_s':time.monotonic()-started,'charged_worker_s':ledger_worker,'process_CPU_s':ledger_cpu,'manifest_unchanged':mf.read_bytes()==manifest_bytes}
        if failure:atomic_json(folder/'FAILURE.json',dict(failure,committed_jobs=len(done)))
        status(state);atomic_json(folder/'SESSION_RESULT.json',out);return out
    finally:
        for v in active.values():
            if v['proc'].poll() is None:v['proc'].terminate()
            try:v['proc'].wait(timeout=max(.1,min(30,limits['supervisor_exit_before_seconds']-(time.monotonic()-started)-2)))
            except subprocess.TimeoutExpired:v['proc'].kill();v['proc'].wait()
            v['log'].close()
        if caffeinate is not None:caffeinate.terminate();caffeinate.wait(timeout=10)
def main():
    a=argparse.ArgumentParser();a.add_argument('--manifest',required=True);a.add_argument('--folder',required=True);a.add_argument('--resume',action='store_true');args=a.parse_args()
    out=dispatch(read_receipt(args.manifest),args.folder,resume=args.resume);print(json.dumps(out),flush=True)
    if out['state'] in ('FAILED','STOP_NEEDS_REVIEW'):raise SystemExit(1)
if __name__=='__main__':main()
