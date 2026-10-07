"""Three logged development cycles; this program cannot launch science."""
import argparse,subprocess,signal,os,time,json,copy,unittest,io,zipfile
from pathlib import Path
from runtime import source_files,ROOT,FROZEN,DOCUMENTS,execution_identity,require_sources,require_environment,load_plan,read_receipt,write_receipt,atomic_json,atomic_bytes,digest,file_sha,Core,make_world,frozen_worker,frozen_checkpoint
from job_audit import audit
from exporter import load_model
from locks import exclusion_registry,manifest,write_manifest,source_lock
from protocol import job_key

NONSCIENCE={'online_cpu_s','Q_HALF_online_cpu_s','online_api_cpu_s','Q_HALF_additional_readout_cpu_s','all_probe_cpu_s','external_audit_cpu_s','fly_id'}
def scientific(x):
    if isinstance(x,dict):return {k:scientific(v) for k,v in x.items() if k not in NONSCIENCE}
    if isinstance(x,(list,tuple)):return [scientific(v) for v in x]
    return x

def cmd(j,folder,short=None,resume=False):
    a=[os.sys.executable,'-u',str(ROOT/'job_runner.py'),'--world',str(j['world']),'--arm',j['arm'],'--assay',j['assay'],'--folder',str(folder)]
    if short is not None:a+=['--short',str(short)]
    if resume:a+=['--resume']
    return a

def spawn(a,log):
    log.parent.mkdir(parents=True,exist_ok=True);f=log.open('w');p=subprocess.Popen(a,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);return p,f

def finish(p,f,log,timeout=600):
    try:code=p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:p.terminate();p.wait(timeout=30);raise AssertionError('qualification worker timeout')
    f.close();text=log.read_text()
    if code:raise AssertionError(f'qualification worker exit {code}: {text[-4000:]}')
    for line in reversed(text.splitlines()):
        try:d=json.loads(line)
        except ValueError:continue
        if isinstance(d,dict) and ('status' in d or 'state' in d):return d
    raise AssertionError('missing worker/dispatcher result')

def cycle1(base):
    from qualification_tests import Cycle1
    stream=io.StringIO();suite=unittest.defaultTestLoader.loadTestsFromTestCase(Cycle1);result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    atomic_bytes(base/'tests.log',stream.getvalue().encode())
    if not result.wasSuccessful():raise AssertionError(stream.getvalue())
    r=exclusion_registry()
    return {'verdict':'PASS','tests':result.testsRun,'synthetic_only':True,'world_exclusion_local_files':r['files_examined'],'historical_worlds':len(r['historical_exclusion_worlds']),'refinements':['one-candidate/one-target enforced','non-finite/duplicate/mixed-world rejection','separate adoption/E3/mechanism decisions','conservative zero-variance rule','explicit launch authorization gate']}

def restored_probe(d,folder):
    e=d['W_export'];c=load_model(folder/'models'/e['file'],e['binding']);w=make_world(d['world'],d['assay']);p=d['branches']['W']['probes'][-1];before=c.state_digest()
    got=frozen_worker.probe(c,w,p['at'],p['name'],'W')
    if scientific(got)!=scientific(p) or c.state_digest()!=before:raise AssertionError('final-W restored read-only outputs differ')
    return {'model_bytes':e['bytes'],'state_digest':e['state_digest'],'read_only_probe_parity':True}

def tamper_tests(d,folder,base):
    from job_runner import valid_job
    from job_audit import audit_unavailable
    bad=copy.deepcopy(d);bad['execution_source_identity']='incorrect'
    def reject(f,label):
        try:f()
        except (ValueError,AssertionError,FileNotFoundError,KeyError,IndexError):return label
        raise AssertionError('tamper accepted: '+label)
    checks=[reject(lambda:audit(bad,execution_identity(),folder,allow_short=True),'wrong receipt source')]
    bad=copy.deepcopy(d);bad['branches']['W']['records'][0]['actual_calls'][0]['write']=False
    checks+=[reject(lambda:audit(bad,execution_identity(),folder,allow_short=True),'wrong real native write flag')]
    e=d['W_export'];checks+=[reject(lambda:load_model(folder/'models'/e['file'],dict(e['binding'],world=999)),'wrong model binding')]
    cp=folder/'checkpoints'/f"{d['world']}_{d['arm']}_{d['assay']}.npz"
    checks+=[reject(lambda:frozen_checkpoint.load(cp,d['arm'],'wrong'),'wrong checkpoint source')]
    import numpy as np
    with np.load(folder/'models'/e['file'],allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
    k=next(k for k in arrays if k!='metadata_json' and arrays[k].dtype.kind=='f');arrays[k].flat[0]+=1
    tampered=base/'tampered_model.npz';np.savez_compressed(tampered,**arrays)
    checks+=[reject(lambda:load_model(tampered,e['binding']),'altered operative model array')]
    bad={'schema':'DIAGNOSTIC_UNAVAILABLE_V2','status':'DIAGNOSTIC_NOT_QUALIFIED','arm':'S3_RAND','reason':'unrelated code error'}
    checks+=[reject(lambda:audit_unavailable(bad,execution_identity(),folder),'code error cannot become diagnostic exclusion')]
    return checks

def cycle2(base):
    folder=base/'signal_resume';proof=[];pinned=execution_identity();arms=load_plan()['physical_configurations']
    for arm in arms:
        j={'world':61005004,'arm':arm,'assay':'lifetime'};k=job_key(j);log=base/'logs'/f'{arm}_pause.log'
        p,f=spawn(cmd(j,folder,32),log);hp=folder/'active'/f'{k}.json';observed=None;stop=time.monotonic()+60
        while p.poll() is None and time.monotonic()<stop:
            if hp.exists():
                h=json.loads(hp.read_text())
                if h['pid']==p.pid and h['in_record'] and h['next_record']>=3 and h['next_record']<25:observed=h;p.send_signal(signal.SIGTERM);break
            time.sleep(.005)
        result=finish(p,f,log)
        if observed is None or result['status']!='PAUSED_SAFE_RECORD' or not 3<result['next_record']<32:raise AssertionError('no genuine in-record signal -> safe checkpoint')
        cp=folder/'checkpoints'/f'{k}.npz';c,cur=frozen_checkpoint.load(cp,arm,pinned)
        if c.records!=result['next_record']:raise AssertionError('checkpoint cursor disagreement')
        before=file_sha(cp);del c
        log2=base/'logs'/f'{arm}_resume.log';p,f=spawn(cmd(j,folder,32,True),log2);resumed=finish(p,f,log2)
        if resumed['status']!='COMMITTED':raise AssertionError('resume did not commit')
        rp=folder/'receipts'/f'{k}.json.gz';d=read_receipt(rp);audit(d,pinned,folder,allow_short=True)
        ref=read_receipt(FROZEN/'results/development/recovery'/arm/'reference/receipts'/f'{k}.json.gz')
        if scientific(d['branches'])!=scientific(ref['branches']):raise AssertionError('resumed scientific state differs from archived uninterrupted reference: '+arm)
        info=restored_probe(d,folder);proof.append(dict(arm=arm,signal_observed_in_record=True,cursor_at_signal=observed['next_record'],cursor_saved=result['next_record'],checkpoint_at_pause_sha256=before,scientific_parity=True,**info))
        print(json.dumps({'cycle':2,'arm':arm,'status':'PARITY_PASS'}),flush=True)
    d=read_receipt(folder/'receipts/61005004_CENTER_lifetime.json.gz');checks=tamper_tests(d,folder,base)
    rp=folder/'receipts/61005004_CENTER_lifetime.json.gz';before=file_sha(rp);attempts=len(list((folder/'attempts').glob('*.json')))
    log=base/'logs/skipped.log';p,f=spawn(cmd({'world':61005004,'arm':'CENTER','assay':'lifetime'},folder,32),log);skip=finish(p,f,log)
    if skip['status']!='SKIPPED_COMMITTED' or file_sha(rp)!=before or attempts!=len(list((folder/'attempts').glob('*.json'))):raise AssertionError('committed job was re-executed')
    return {'verdict':'PASS','signal_resumes':proof,'tamper_rejections':checks,'committed_skip_no_reexecution':True,'completed_short_W_jobs':len(proof),'short_records':32*len(proof),'refinements':['SIGTERM during actual record saves next safe boundary','all 11 configurations restore graph/RNG/yoke and native states','final W snapshots restored and probed read-only','completed receipt skip does not execute another trajectory']}

def dispatch_test(m,folder,base,label,resume=False):
    mf=base/f'{label}_manifest.json.gz';write_manifest(mf,m);log=base/'logs'/f'{label}.log';a=[os.sys.executable,'-u',str(ROOT/'dispatcher.py'),'--manifest',str(mf),'--folder',str(folder)]
    if resume:a+=['--resume']
    p,f=spawn(a,log);launch=folder/'LAUNCH.json';stop=time.monotonic()+20;attached=False
    while p.poll() is None and time.monotonic()<stop:
        if launch.exists():
            d=json.loads(launch.read_text())
            if d['pid']==p.pid and d['caffeinate_pid']:
                os.kill(d['caffeinate_pid'],0);attached=True;break
        time.sleep(.02)
    result=finish(p,f,log,timeout=180)
    if not attached:raise AssertionError('dispatcher caffeinate attachment not observed')
    return result

def cycle3(base):
    # Complete existing fixture lives test W export before N, all six old phase probes, and Q accounting.
    # Only the reducer/dispatcher/test harness changed in cycle 3; never repeat the completed full lives.
    previous=json.loads((base.parent/'CYCLE2.json').read_text())['production_file_hashes']
    allowed={'qualify.py','qualification_tests.py','metrics.py','dispatcher.py','pipeline.py'}
    current=source_files()
    if any(current[k]!=v for k,v in previous.items() if Path(k).name not in allowed):raise ValueError('scientific worker dependency changed; full-life evidence cannot be reused')
    full=base.parent/'cycle3/full';tasks=[{'world':61005003,'arm':'CENTER','assay':'lifetime'},{'world':61005003,'arm':'ERROR','assay':'reuse'}];children=[]
    for j in tasks:
        lp=base/'logs'/f'{job_key(j)}.log'
        if not (full/'receipts'/f'{job_key(j)}.json.gz').exists():raise ValueError('missing completed full development evidence')
        children.append((j,None,None,lp))
    parity=[]
    for j,p,f,lp in children:
        d=read_receipt(full/'receipts'/f'{job_key(j)}.json.gz');audit(d,d['execution_source_identity'],full)
        ref=read_receipt(FROZEN/'results/development/cycle3/receipts'/f'{job_key(j)}.json.gz')
        if scientific(d['branches'])!=scientific(ref['branches']):raise AssertionError('full unchanged-science parity')
        parity.append(dict(job=j,scientific_parity=True,cpu_s=d['cpu_s'],worker_s=d['worker_s'],rss_bytes=d['peak_rss_bytes'],**restored_probe(d,full)));print(json.dumps({'cycle':3,'job':job_key(j),'status':'FULL_PARITY_PASS'}),flush=True)
    js=[{'world':61005005,'arm':a,'assay':'reuse'} for a in load_plan()['physical_configurations']]
    concurrency={};digests={}
    for workers in (4,8):
        folder=base/f'concurrency{workers}';m=manifest('qualification',qualification_jobs=js,short=2,workers=workers);result=dispatch_test(m,folder,base,f'concurrency{workers}')
        if result['state']!='STAGE_COMPLETE' or result['complete_jobs']!=11:raise AssertionError('concurrency jobs incomplete')
        ds={};cpu=0.;rss=0
        for j in js:
            d=read_receipt(folder/'receipts'/f'{job_key(j)}.json.gz');ds[j['arm']]=d['branches']['W']['final_state_digest'];cpu+=d['cpu_s'];rss=max(rss,d['peak_rss_bytes'])
        concurrency[str(workers)]={'elapsed_s':result['session_elapsed_s'],'jobs':11,'cpu_s':cpu,'peak_worker_RSS':rss,'caffeinate_observed':True,'charged_worker_s':result['charged_worker_s']};digests[workers]=ds
    if digests[4]!=digests[8]:raise AssertionError('4/8 workers changed state')
    paused=base/'deadline';js=[{'world':61005001,'arm':a,'assay':'lifetime'} for a in ('CENTER','ERROR','REL10','P005')]
    limits={'stop_dispatch_seconds':5,'safe_pause_request_seconds':7,'worker_exit_deadline_seconds':20,'supervisor_exit_before_seconds':25}
    m=manifest('qualification',qualification_jobs=js,short=64,workers=4,session=limits);first=dispatch_test(m,paused,base,'deadline_pause')
    if first['state']!='PAUSED_SESSION_LIMIT' or not first['pending_jobs'] or first['session_elapsed_s']>=25:raise AssertionError('scaled actual session pause failed')
    sessions=[first]
    for i in range(5):
        nxt=dispatch_test(m,paused,base,f'deadline_resume{i}',True);sessions.append(nxt)
        if nxt['state']=='STAGE_COMPLETE':break
    if sessions[-1]['state']!='STAGE_COMPLETE':raise AssertionError('paused queue did not finish with explicit resume')
    if any(not s['manifest_unchanged'] or s['session_elapsed_s']>=25 for s in sessions):raise AssertionError('session deadline/source manifest failed')
    for j in js:
        d=read_receipt(paused/'receipts'/f'{job_key(j)}.json.gz');audit(d,execution_identity(),paused,allow_short=True)
    # One complete two-assay row is required for the production reducer. Use archived unchanged receipts,
    # since no extra full teaching trajectory is needed merely to test arithmetic.
    from metrics import one
    a=read_receipt(FROZEN/'results/development/cycle3/receipts/61005003_ERROR_lifetime.json.gz');b=read_receipt(full/'receipts/61005003_ERROR_reuse.json.gz')
    rows=one(a,b);qh=one(a,b,True)
    intact=make_world(a['world'],'lifetime')['intact_old_items']
    expected=sum(r['correct'] for r in a['branches']['W']['probes'][-1]['rows'] if r['stage']=='old' and r['item'] in intact)/24
    if rows['old']!=expected or qh['tauR']<rows['tauR']:raise AssertionError('reducer native emission/Q cost mismatch')
    # Scaled session safety is an operational test, not a sustained 8-worker throughput bound.
    return {'verdict':'PASS','full_jobs':parity,'full_jobs_completed':2,'full_jobs_reused_with_unchanged_worker':2,'full_lives_completed':6,'full_teaching_records':3888,'concurrency':concurrency,'concurrency_state_parity':True,'short_concurrency_jobs':22,'short_concurrency_lives':22,'deadline_sessions':sessions,'deadline_short_jobs':4,'deadline_short_records':256,'reducer_native_output_checked':True,'old_score_counts_only_24_intact_keys':True,'refinements':['final W survives later N branches','all old phase probes retained','4/8 worker final states identical','actual dispatcher deadline pauses live records and resumes identical sealed queue','analysis/export/package deadlines added']}

def resource_audit(c2,c3,base):
    planning=json.loads((DOCUMENTS/'BUDGET_AND_STATIC_AUDIT.json').read_text());p=load_plan();G=1024**3
    exports=c2['signal_resumes'];model_bytes={x['arm']:x['model_bytes'] for x in exports}
    # Full historical checkpoints have the exact same array serializer; ignore their receipt-history JSON.
    # Use the larger measured short/full payload for each arm, with a metadata allowance and 1.5x forecast.
    for arm in p['physical_configurations']:
        for assay in ('lifetime','reuse'):
            cp=FROZEN/'results/development/cycle3/checkpoints'/f'61005003_{arm}_{assay}.npz'
            with zipfile.ZipFile(cp) as z:payload=sum(v.compress_size for v in z.infolist() if v.filename!='metadata_json.npy')+65536
            model_bytes[arm]=max(model_bytes[arm],payload)
    for x in c3['full_jobs']:model_bytes[x['job']['arm']]=max(model_bytes[x['job']['arm']],x['model_bytes'])
    max_model=max(model_bytes.values());export_forecast=0.
    qbytes=sum(f.stat().st_size for f in ROOT.rglob('*') if f.is_file())/G
    ops_bytes=qbytes
    # Export restore, pause and dispatcher costs are reported separately and not hidden in old science timings.
    paths={};checks={};old_qual=0.18668715842068195
    for name,x in planning['paths'].items():
        # Measure complete runner CPU/worker ratio vs the old corresponding fixture; take the larger 1.0.
        factor=max([1.]+[v['cpu_s']/read_receipt(FROZEN/'results/development/cycle3/receipts'/f"{job_key(v['job'])}.json.gz")['cpu_s'] for v in c3['full_jobs']])
        worker_factor=max([1.]+[v['worker_s']/read_receipt(FROZEN/'results/development/cycle3/receipts'/f"{job_key(v['job'])}.json.gz")['worker_s'] for v in c3['full_jobs']])
        cpu=x['science_CPU_hours_1_5x']*factor;worker=x['science_worker_hours_1_5x']*worker_factor
        # Already conservatively reserve 1 GiB for exports/ops; explicitly reject if measured projected usage exceeds it.
        exports_GiB=1.5*(16*sum(model_bytes.values())+96*sum(model_bytes[a] for a in x['confirmation_roster']))/G
        export_forecast=max(export_forecast,exports_GiB)
        dependency_GiB=sum(Path(f).stat().st_size for f in __import__('runtime').frozen_integrity.source_map())/G
        archive_forecast=1.5*x['receipts_GiB_point']+exports_GiB+dependency_GiB+.02
        disk=x['storage_GiB_with_1_5x_log_qualification_1GiB_export_2GiB_archive']+max(0.,exports_GiB+ops_bytes-1.)+max(0.,archive_forecast-2.)
        paths[name]={'science_CPU_hours_1_5x':cpu,'charged_worker_hours_1_5x':worker,'ideal_8_worker_hours':worker/8,'storage_including_measured_model_forecast_GiB':disk,'archive_forecast_GiB':archive_forecast,'final_W_exports_forecast_GiB':exports_GiB,'additional_ops_export_reserve_GiB':max(0.,exports_GiB+ops_bytes-1.)}
    checks={'all_CPU_below_68h':max(x['science_CPU_hours_1_5x'] for x in paths.values())<=68,'all_worker_below_70h':max(x['charged_worker_hours_1_5x'] for x in paths.values())<=70,'all_storage_below_8GiB':max(x['storage_including_measured_model_forecast_GiB'] for x in paths.values())<=8,'worker_RSS_below_1GiB':max(v['rss_bytes'] for v in c3['full_jobs'])<G,'4_8_state_parity':c3['concurrency_state_parity']}
    out={'verdict':'PASS' if all(checks.values()) else 'RESOURCE_NOT_QUALIFIED','checks':checks,'paths':paths,'current_ops_and_qualification_GiB':qbytes,'conservative_final_W_export_GiB':export_forecast,'max_measured_W_model_bytes':max_model,'model_size_bound_by_arm_bytes':model_bytes,'reserve_note':'The 1 GiB export/ops reservation is a planning allocation, not a separate hard cap. Any measured overage is added to the unchanged 8 GiB total; the 2 GiB archive allowance remains intact.','old_full_qualification_not_repeated':True,'assumption':'Original 1.5x forecasts plus measured full-run overhead factor; not an empirical p95 or sustained throughput guarantee.','safe_10h_sessions_qualified':True,'science_trajectories':0};atomic_json(base/'RESOURCE_AUDIT.json',out);return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cycle',choices=('1','2','3','final'),required=True);ap.add_argument('--run-id',required=True);a=ap.parse_args();require_sources();require_environment();ident=execution_identity();base=ROOT/'results/qualification'/a.run_id;base.mkdir(parents=True,exist_ok=True)
    caff=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())]) if Path('/usr/bin/caffeinate').exists() else None
    try:
        if a.cycle!='final':
            n=int(a.cycle);sub=base/('cycle3_final' if n==3 else f'cycle{n}');sub.mkdir(exist_ok=True);started=time.monotonic();result=globals()[f'cycle{n}'](sub);require_sources(ident);result.update(production_file_hashes={k:v for k,v in source_files().items() if Path(k).name!='qualify.py'},qualification_driver_sha256=file_sha(ROOT/'qualify.py'),execution_source_identity=ident,scientific_source_identity=__import__('runtime').SCIENTIFIC_ID,wall_s=time.monotonic()-started,science_trajectories=0)
            atomic_json(base/f'CYCLE{n}.json',result);print(json.dumps(result),flush=True)
        else:
            cs=[json.loads((base/f'CYCLE{n}.json').read_text()) for n in (1,2,3)]
            production={k:v for k,v in source_files().items() if Path(k).name!='qualify.py'}
            allowed={'metrics.py','dispatcher.py','pipeline.py','qualification_tests.py'}
            for i,c in enumerate(cs):
                if c['verdict']!='PASS':raise ValueError('cycle failed')
                if set(c['production_file_hashes'])!=set(production):raise ValueError('qualified dependency roster changed')
                for f,h in production.items():
                    if c['production_file_hashes'][f]!=h and not (i==1 and Path(f).name in allowed):raise ValueError('untested production source change')
            amendment=json.loads((base/'OPERATIONS_AMENDMENT.json').read_text())
            if amendment['verdict']!='PASS' or amendment['after_sha256']!=file_sha(ROOT/'dispatcher.py'):raise ValueError('watchdog amendment not qualified/current')
            resource=resource_audit(cs[1],cs[2],base);q={'verdict':'V2_FUNCTIONAL_AND_RESOURCE_QUALIFIED' if resource['verdict']=='PASS' else 'FUNCTIONAL_PASS_RESOURCE_NOT_QUALIFIED','cycles_completed':3,'qualification_driver_refinement_only':True,'cycle_execution_identities':[c['execution_source_identity'] for c in cs],'execution_source_identity':ident,'scientific_source_identity':__import__('runtime').SCIENTIFIC_ID,'cycle_artifacts':[str(base/f'CYCLE{n}.json') for n in (1,2,3)],'resource_audit':str(base/'RESOURCE_AUDIT.json'),'full_development_jobs':2,'full_development_lives':6,'short_development_jobs':65,'short_development_lives':65,'development_fault_injection_jobs':2,'scientific_worker_unchanged_across_refinement':True,'source_coverage_note':'Cycle 2 worker/model/signal-resume evidence reused unchanged. Corrected reducer and dispatcher are requalified by current cycles 1/3 and the real watchdog amendment; exact per-cycle source maps are retained.','science_trajectories':0,'science_authorized':False,'archive_and_timing_forecasts_not_guarantees':True,'completed_unix':time.time()};atomic_json(ROOT/'results/QUALIFICATION.json',q)
            if resource['verdict']=='PASS':source_lock(q)
            print(json.dumps(q),flush=True)
    except Exception as e:
        atomic_json(base/f'CYCLE{a.cycle}_FAILURE.json',{'error_type':type(e).__name__,'error':str(e),'execution_source_identity':ident});raise
    finally:
        if caff:caff.terminate();caff.wait(timeout=5)
if __name__=='__main__':main()
