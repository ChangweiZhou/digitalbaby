"""Same science, record-boundary signal pause and independent W export."""
import argparse,os,signal,time,json,resource,contextlib
MODULE_CPU=time.process_time()
MODULE_WALL=time.monotonic()
from pathlib import Path
import runtime
from runtime import Core,make_world,atomic_json,read_receipt,write_receipt,frozen_worker,frozen_checkpoint,execution_identity,require_sources,SCIENTIFIC_ID,file_sha,digest
from protocol import validate_plan,screen_worlds,confirmation_worlds,job_key,roster
from exporter import save_model,load_model
from mechanisms import DiagnosticUnavailable
from job_audit import audit
PAUSE=False
def pause_signal(sig,frame):
    global PAUSE
    PAUSE=True
def valid_job(world,arm,assay,stage,manifest=None,short=None):
    p=validate_plan(runtime.load_plan())
    if assay not in ('lifetime','reuse') or arm not in p['physical_configurations']:raise ValueError('unregistered task/configuration')
    if stage=='qualification':
        if world not in runtime.DEVELOPMENT_IDS:raise ValueError('qualification world')
    else:
        if short is not None or stage not in ('screen','confirm') or manifest is None:raise ValueError('science stage/short/manifest')
        from locks import validate_manifest
        validate_manifest(manifest,science=True)
        if stage!=manifest['stage'] or {'world':world,'arm':arm,'assay':assay,'stage':stage} not in manifest['jobs']:raise ValueError('job outside sealed manifest')
def run(world,arm,assay,folder,*,stage='qualification',short=None,resume=False,manifest=None):
    global PAUSE
    PAUSE=False;valid_job(world,arm,assay,stage,manifest,short)
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,pause_signal)
    started=MODULE_WALL;cpu_start=MODULE_CPU;identity=execution_identity();require_sources(identity)
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    key=f'{world}_{arm}_{assay}';path=folder/'receipts'/f'{key}.json.gz';cp=folder/'checkpoints'/f'{key}.npz'
    if path.exists():
        d=read_receipt(path);audit(d,identity,folder,allow_short=stage=='qualification');return {'status':'SKIPPED_COMMITTED','path':str(path)}
    w=make_world(world,assay);limit=len(w['events']) if short is None else int(short);branches=w['branches'] if short is None else ['W']
    if not 0<limit<=len(w['events']):raise ValueError('record limit')
    lock=folder/'locks'/f'{key}.lock';lock.parent.mkdir(exist_ok=True)
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,json.dumps({'pid':os.getpid(),'source':identity,'key':key}).encode());os.fsync(fd);os.close(fd)
    active=folder/'active'/f'{key}.json'
    def save(c,cur):
        cur['completed_cpu_s']=prior_cpu+time.process_time()-cpu_start;cur['completed_worker_s']=prior_wall+time.monotonic()-started
        frozen_checkpoint.save(c,cp,cur,identity)
    def heartbeat(cur,branch,in_record):
        atomic_json(active,{'pid':os.getpid(),'source':identity,'branch':branch,'next_record':cur['next_record'],'limit':limit,'in_record':in_record,'updated_unix':time.time(),'elapsed_worker_s':prior_wall+time.monotonic()-started,'attempt_CPU_s':time.process_time()-cpu_start,'attempt_worker_s':time.monotonic()-started,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
    try:
        if resume:
            c,cur=frozen_checkpoint.load(cp,arm,identity)
            if any(cur[k]!=v for k,v in {'world':world,'assay':assay,'fixture_sha256':w['sha256'],'limit':limit,'branch_names':branches,'stage':stage}.items()):raise ValueError('checkpoint stage/fixture/limit mismatch')
        else:
            if cp.exists():raise ValueError('explicit resume required')
            c=None;cur={'world':world,'assay':assay,'stage':stage,'fixture_sha256':w['sha256'],'limit':limit,'branch_names':branches,'branch_index':0,'next_record':0,'branches':{},'completed_cpu_s':0.,'completed_worker_s':0.,'W_export':None}
        prior_cpu,prior_wall=cur['completed_cpu_s'],cur['completed_worker_s']
        yoke=None
        if arm=='S3_RAND':
            yoke=read_receipt(folder/'receipts'/f'{world}_S3_CUE_{assay}.json.gz')
            audit(yoke,identity,folder,allow_short=stage=='qualification',load_export=False)
            if yoke['stage']!=stage or yoke['limits']!={'records':limit,'branches':branches}:raise ValueError('wrong yoke stage/limits')
        for bi in range(cur['branch_index'],len(branches)):
            branch=branches[bi]
            if c is None or cur['branch_index']!=bi:c=Core(arm);cur['branch_index']=bi;cur['next_record']=0
            if branch not in cur['branches']:cur['branches'][branch]={'births':c.births,'records':[],'probes':[],'peak_mutable_bytes':frozen_worker.mutable_bytes(c)}
            bd=cur['branches'][branch]
            if yoke is not None:c.yoked_counts=[[a['moves'] for a in r['write']['adaptations']] for r in yoke['branches'][branch]['records']]
            if PAUSE:
                save(c,cur);return {'status':'PAUSED_SAFE_RECORD','path':str(cp),'branch':branch,'next_record':cur['next_record']}
            for i in range(cur['next_record'],limit):
                heartbeat(cur,branch,True)
                bd['records'].append(frozen_worker.record(c,w['events'][i],branch));cur['next_record']=i+1
                if short is None:
                    for name in w['boundaries'].get(str(i+1),[]):
                        c.flush(w['clocks'][name]);bd['probes'].append(frozen_worker.probe(c,w,w['clocks'][name],name,branch))
                bd['peak_mutable_bytes']=max(bd['peak_mutable_bytes'],frozen_worker.mutable_bytes(c))
                if PAUSE or (i+1)%48==0 or i+1==limit:save(c,cur)
                heartbeat(cur,branch,False)
                if PAUSE:return {'status':'PAUSED_SAFE_RECORD','path':str(cp),'branch':branch,'next_record':i+1}
            if short is not None and not any(p['name']=='short_end' for p in bd['probes']):bd['probes'].append(frozen_worker.probe(c,w,c.last_time,'short_end',branch))
            bd.update(records_completed=limit,final_state_digest=c.state_digest(),final_time=c.last_time,online_cpu_s=sum(r['online_cpu_s'] for r in bd['records'])+sum(p['online_api_cpu_s'] for p in bd['probes'] if p['name']=='final'))
            if arm=='ERROR':bd['Q_HALF_online_cpu_s']=bd['online_cpu_s']+sum(p['Q_HALF_additional_readout_cpu_s'] for p in bd['probes'] if p['name']=='final')
            if branch=='W':
                binding={'world':world,'arm':arm,'assay':assay,'stage':stage,'execution_source_identity':identity,'fixture_sha256':w['sha256'],'records':limit,'time':c.last_time}
                model_path=folder/'models'/f'{key}_W.npz';cur['W_export']=save_model(c,model_path,binding)
                loaded=load_model(model_path,binding)
                if loaded.state_digest()!=c.state_digest():raise AssertionError('W export restore parity')
                del loaded
                # Commit W completion/export before moving to N, retaining the safe W cursor.
                save(c,cur)
            cur['branch_index']=bi+1;cur['next_record']=0;c=None
        require_sources(identity)
        d={'schema':'NEXT_CORE_BUDGET_JOB_V2','development':stage=='qualification','stage':stage,'world':world,'arm':arm,'assay':assay,'source_identity':SCIENTIFIC_ID,'execution_source_identity':identity,'fixture_sha256':w['sha256'],'branches':cur['branches'],'limits':{'records':limit,'branches':branches},'cpu_s':time.process_time()-cpu_start+prior_cpu,'worker_s':time.monotonic()-started+prior_wall,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'complete':short is None,'W_export':cur['W_export']}
        audit(d,identity,folder,allow_short=stage=='qualification',load_export=False);write_receipt(path,d)
        return {'status':'COMMITTED','path':str(path),'lives':len(branches),'records':limit*len(branches),'complete':d['complete']}
    except DiagnosticUnavailable as e:
        if arm!='S3_RAND' or str(e)!='legal random swap pool exhausted':raise
        marker={'schema':'DIAGNOSTIC_UNAVAILABLE_V2','status':'DIAGNOSTIC_NOT_QUALIFIED','world':world,'arm':arm,'assay':assay,'stage':stage,'source_identity':SCIENTIFIC_ID,'execution_source_identity':identity,'fixture_sha256':w['sha256'],'reason':str(e),'partial_cursor':cur,'yoke_sha256':file_sha(folder/'receipts'/f'{world}_S3_CUE_{assay}.json.gz')}
        write_receipt(folder/'unavailable'/f'{key}.json.gz',marker)
        return {'status':'DIAGNOSTIC_NOT_QUALIFIED','path':str(folder/'unavailable'/f'{key}.json.gz')}
    finally:
        atomic_json(folder/'attempts'/f'{key}_{os.getpid()}.json',{'job':key,'pid':os.getpid(),'stage':stage,'cpu_s':time.process_time()-cpu_start,'worker_s':time.monotonic()-started,'execution_source_identity':identity})
        lock.unlink(missing_ok=True);active.unlink(missing_ok=True)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--world',required=True,type=int);ap.add_argument('--arm',required=True);ap.add_argument('--assay',choices=('lifetime','reuse'),required=True);ap.add_argument('--folder',required=True);ap.add_argument('--stage',default='qualification',choices=('qualification','screen','confirm'));ap.add_argument('--short',type=int);ap.add_argument('--resume',action='store_true');ap.add_argument('--manifest');a=ap.parse_args()
    manifest=None if not a.manifest else read_receipt(a.manifest)
    try:r=run(a.world,a.arm,a.assay,a.folder,stage=a.stage,short=a.short,resume=a.resume,manifest=manifest)
    except Exception as e:
        atomic_json(Path(a.folder)/'failures'/f'{a.world}_{a.arm}_{a.assay}.json',{'error_type':type(e).__name__,'error':str(e),'pid':os.getpid(),'stage':a.stage});raise
    print(json.dumps(r),flush=True)
if __name__=='__main__':main()
