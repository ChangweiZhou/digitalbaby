"""Authorized manifest admission, detached dispatch and compact reduction. No kernel edits."""
import argparse,copy,gzip,hashlib,itertools,json,os,secrets,signal,subprocess,sys,threading,time,traceback,zipfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
OPS=Path(__file__).resolve().parent;ROOT=OPS.parent.parent;OUT=ROOT/'results'
sys.path.insert(0,str(ROOT))
import config,runner,worlds,decision
ORIGINAL_LOCK=config.lock;ORIGINAL_ADMIT=config.admit
COUNTS={'committed_full_jobs':0,'committed_external_records':0,'committed_training_bytes':0,'correlated_state_trajectories':0}
LOCK=threading.Lock();STAGE='STARTING';START=0.;INDEX=[]
PEER={'COV-I':'DIAG-I','DIAG-I':'DIAG-R','DIAG-R':'DIAG-I','COV-REPLAY':'COV-EXTRA','COV-EXTRA':'COV-REPLAY'}
def dump(x,p):runner.atomic_json(x,p)
def load(p):return json.loads(Path(p).read_text())
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def identity():
    d=ORIGINAL_LOCK();d[str((OPS/'science.py').relative_to(ROOT.parent))]=digest(OPS/'science.py');return d
def verify():
    if ORIGINAL_LOCK()!=load(ROOT/'QUALIFICATION.json')['source']:raise AssertionError('qualified sources changed')
    if identity()!=load(OUT/'SOURCE_LOCK.json')['files']:raise AssertionError('adapter/source mismatch')
def register_geometry():
    edges=tuple(map(tuple,config.CONTRACT['confirmation_edges']))
    masks=sorted({tuple(sorted(tuple(sorted((p[i],p[j]))) for i,j in edges)) for p in itertools.permutations(range(5))})
    if len(masks)!=60:raise AssertionError('prespecified geometry')
    worlds.FAMILIES['TRIANGLE_TAIL']=masks

def roster(selected):return ['COV-R',selected,PEER[selected]]
def manifest(stage,rows,arms):
    x={'schema':'FEEDBACK_EFFICIENCY_ADMISSION_V1','stage':stage,'worlds':rows,'arms':list(arms),'source':identity()}
    x['sha256']=hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest();return x

def validate_manifest(x,testing=False):
    y=copy.deepcopy(x);h=y.pop('sha256')
    if h!=hashlib.sha256(json.dumps(y,sort_keys=True).encode()).hexdigest() or x['source']!=identity():raise AssertionError('manifest seal')
    stage=x['stage'];rows=x['worlds'];arms=x['arms']
    if len({r['world'] for r in rows})!=len(rows) or any(type(r['world']) is not int for r in rows):raise AssertionError('distinct worlds')
    if len(set(arms))!=len(arms) or any(a not in config.ARMS for a in arms):raise AssertionError('arm roster')
    if stage=='ADAPTER_TEST' and testing:return x
    from collections import Counter
    strata=Counter((r['family'],r['noise'],r['ordinary']) for r in rows)
    if stage=='DEV':
        expected=Counter({(f,n,False):3 for f in ('PATH_LOOP','STAR_LOOP') for n in ('STANDARD','BLURRY')})
        if strata!=expected or arms!=list(config.ARMS):raise AssertionError('DEV design roster')
    elif stage=='CONFIRM':
        selected=load(OUT/'SELECTION_LOCK.json')['selected']
        expected=Counter({('TRIANGLE_TAIL','BLURRY',False):64,('TRIANGLE_TAIL','STANDARD',False):32,
                          ('TRIANGLE_TAIL','BLURRY',True):16,('TRIANGLE_TAIL','STANDARD',True):16})
        if strata!=expected or arms!=roster(selected):raise AssertionError('CONFIRM design roster')
    else:raise AssertionError('scientific stage not authorized')
    return x

def install(m):
    register_geometry()
    tasks={r['world']:r for r in m['worlds']}
    def admit(world,family,noise=None,ordinary=None):
        if type(world) is not int or world not in tasks or tasks[world]['family']!=family:raise ValueError('outside sealed world/family')
        if noise is not None and (noise,ordinary)!=(tasks[world]['noise'],tasks[world]['ordinary']):raise ValueError('outside sealed stratum')
    config.admit=admit;config.lock=identity

def envelope(r,m):
    return {'schema':'FEEDBACK_EFFICIENCY_SCIENCE_V1','stage':m['stage'],'manifest_sha256':m['sha256'],
            'source':identity(),'world':r['world'],'arm':r['arm'],'science_jobs':1,
            'training_records':r['training_records'],'training_bytes':r['training_bytes'],'state_trajectories':r['state_trajectories'],
            'kernel_markers':'Internal technical serialization retained; this outer envelope defines scientific admission/counts.',
            'kernel_receipt':r}

def audit(e,m):
    r=e['kernel_receipt'];tasks={t['world']:t for t in m['worlds']}
    if e['schema']!='FEEDBACK_EFFICIENCY_SCIENCE_V1' or e['stage']!=m['stage'] or e['manifest_sha256']!=m['sha256'] or e['source']!=identity():raise AssertionError('science envelope seal')
    if r['world'] not in tasks or r['arm'] not in m['arms']:raise AssertionError('off roster job')
    if any(r[k]!=tasks[r['world']][k] for k in ('world','family','noise','ordinary')):raise AssertionError('world stratum')
    if e['world']!=r['world'] or e['arm']!=r['arm'] or e['science_jobs']!=1 or e['training_records']!=2560 or e['state_trajectories']!=23 or e['training_bytes']!=r['training_bytes']:raise AssertionError('science exposure/provenance')
    a=runner.audit(r)
    return {'verdict':'PASS','science_jobs':1,'training_records':2560,'training_bytes':a['training_bytes'],
            'state_trajectories':23,'independent_transitions':2560,'new_controls_audited':True}

def commit(e,p,m):
    p.parent.mkdir(parents=True,exist_ok=True)
    fd=os.open(str(p)+'.claim',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600);os.close(fd)
    if p.exists():raise ValueError('job already committed; no repeat')
    temp=Path(str(p)+'.provisional')
    with gzip.open(temp,'wt') as f:json.dump(e,f,sort_keys=True,separators=(',',':'),allow_nan=False)
    cert=audit(runner.read(temp),m);cert['receipt_sha256']=digest(temp)
    # Certificate first, receipt rename is the acceptance point.
    dump(cert,Path(str(p)+'.audit.json'));os.replace(temp,p);Path(str(p)+'.claim').unlink()

def fresh_rows(stage):
    used=set(config.TECH_TASKS)
    if stage=='CONFIRM':used|={r['world'] for r in load(OUT/'dev/MANIFEST.json')['worlds']}
    specs=([(f,n,False,3) for f in ('PATH_LOOP','STAR_LOOP') for n in ('STANDARD','BLURRY')] if stage=='DEV' else
           [('TRIANGLE_TAIL','BLURRY',False,64),('TRIANGLE_TAIL','STANDARD',False,32),
            ('TRIANGLE_TAIL','BLURRY',True,16),('TRIANGLE_TAIL','STANDARD',True,16)])
    rows=[]
    for f,n,o,count in specs:
        for _ in range(count):
            while True:
                w=secrets.randbelow(2**52)+2**52
                if w not in used:break
            used.add(w);rows.append(dict(world=w,family=f,noise=n,ordinary=o))
    return rows

def status(stage=None,**kw):
    global STAGE
    with LOCK:
        if stage:STAGE=stage
        dump({'stage':STAGE,'heartbeat_epoch':time.time(),'supervisor_pid':os.getpid(),'workers':2,**COUNTS,**kw},OUT/'STATUS.json')

def worker(path,world,arm):
    verify();m=validate_manifest(load(path));install(m)
    task=next(t for t in m['worlds'] if t['world']==world)
    if arm not in m['arms']:raise ValueError('off roster arm')
    folder=path.parent;active=folder/'active'/f'{world}_{arm}.json';active.parent.mkdir(exist_ok=True)
    stop=threading.Event()
    def beat():
        while not stop.is_set():
            dump({'world':world,'arm':arm,'pid':os.getpid(),'heartbeat_epoch':time.time(),'stage':'IN_FLIGHT_NOT_COMMITTED'},active);stop.wait(5)
    thread=threading.Thread(target=beat,daemon=True);thread.start()
    try:
        r=runner.run(world,task['family'],task['noise'],task['ordinary'],arm)
        commit(envelope(r,m),folder/'receipts'/f'{world}_{arm}.json.gz',m)
    finally:stop.set();thread.join();active.unlink(missing_ok=True)

def compact(e):
    r=e['kernel_receipt'];mm=decision.metrics(r);mm.pop('technical_only')
    cm=decision.countermodels(r)
    return {'world':r['world'],'arm':r['arm'],'family':r['family'],'noise':r['noise'],'ordinary':r['ordinary'],
            'source':r['source'],'master_sha256':hashlib.sha256(json.dumps(r['master'],sort_keys=True).encode()).hexdigest(),
            'metrics':mm,'shortcut_J':{k:v['J'] for k,v in cm.items() if k not in ('left_only','right_only')},
            'descriptive_one_operand_J':{k:cm[k]['J'] for k in ('left_only','right_only')},'countermodels':cm}

def grouped(cells,m):
    rows=[]
    for task in m['worlds']:
        group=[c for c in cells if c['world']==task['world']]
        if len(group)!=len(m['arms']) or {c['arm'] for c in group}!=set(m['arms']):raise AssertionError('incomplete world row')
        first=group[0]
        if any(any(c[k]!=first[k] for k in ('family','noise','ordinary','source','master_sha256','shortcut_J')) for c in group):raise AssertionError('unpaired world row')
        row=dict(task,shortcut_J=first['shortcut_J'])
        for c in group:row[c['arm']]=c['metrics']
        rows.append(row)
    return rows

def stage_run(m):
    validate_manifest(m);folder=OUT/m['stage'].lower();folder.mkdir(exist_ok=False)
    dump(m,folder/'MANIFEST.json');install(m)
    tasks=[(t['world'],a) for t in m['worlds'] for a in m['arms']]
    status(m['stage'],stage_total_jobs=len(tasks));abort=threading.Event();cells=[];errors=[]
    def job(task):
        if abort.is_set():return None
        if time.monotonic()-START+600>9.5*3600:
            abort.set();raise RuntimeError('PAUSED_SESSION_LIMIT')
        verify();w,a=task;logs=folder/'logs';logs.mkdir(exist_ok=True)
        with (logs/f'{w}_{a}.log').open('x') as f:
            if abort.is_set():return None
            p=subprocess.Popen([sys.executable,str(OPS/'science.py'),'--worker','--manifest',str(folder/'MANIFEST.json'),
                                '--world',str(w),'--arm',a],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=p.wait(timeout=min(600,10*3600-(time.monotonic()-START)))
            except subprocess.TimeoutExpired:
                p.terminate();p.wait(timeout=10);abort.set();raise RuntimeError('worker exceeded hard deadline; partial work retained')
        path=folder/'receipts'/f'{w}_{a}.json.gz'
        if code:abort.set();raise RuntimeError(f'worker exit {code}: {logs}/{w}_{a}.log')
        cert=load(str(path)+'.audit.json')
        if cert['verdict']!='PASS' or cert['receipt_sha256']!=digest(path):raise AssertionError('acceptance certificate')
        # Exactly one bounded post-commit read; none under status lock.
        e=runner.read(path);cell=compact(e)
        dump(cell,folder/'metrics'/f'{w}_{a}.json')
        return cell,cert,path
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(job,t) for t in tasks]
        for f in as_completed(futures):
            try:
                result=f.result()
                if result is None:continue
                cell,cert,path=result;cells.append(cell)
                with LOCK:
                    COUNTS['committed_full_jobs']+=1;COUNTS['committed_external_records']+=2560
                    COUNTS['committed_training_bytes']+=cert['training_bytes'];COUNTS['correlated_state_trajectories']+=23
                    INDEX.append({'world':cell['world'],'arm':cell['arm'],'stage':m['stage'],'receipt':str(path.relative_to(ROOT)),'sha256':cert['receipt_sha256']})
                    dump({'entries':INDEX},OUT/'COMMIT_INDEX.json')
                dump({'stage':m['stage'],'committed_jobs':len(cells),'stage_total_jobs':len(tasks),'heartbeat_epoch':time.time()},folder/'STATUS.json')
                status(m['stage'],stage_total_jobs=len(tasks),stage_committed_jobs=len(cells))
            except BaseException as exc:abort.set();errors.append(exc)
    if errors:raise errors[0]
    if len(cells)!=len(tasks):raise AssertionError('stage stopped before all jobs committed')
    rows=grouped(cells,m)
    dump({'stage':'COMPLETE','committed_jobs':len(cells),'stage_total_jobs':len(tasks)},folder/'STATUS.json')
    with gzip.open(folder/'WORLD_METRICS.json.gz','wt') as f:json.dump(rows,f,sort_keys=True,allow_nan=False)
    return rows

def report(x,name):
    text=['# FEEDBACK-EFFICIENCY V1 — '+name,'','Verdict: '+x['verdict'],'',
          'E3 improvement test for bounded addition from noisy/missing observed answer bytes. Own outputs on never-taught combinations, delayed probes and intervening learning. Role/additivity and ordinal-output priors are engineered; this is not unrestricted arithmetic or AGI.',
          '','```json',json.dumps(x,indent=2),'```','']
    (OUT/(name+'.md')).write_text('\n'.join(text))

def finish(summary):
    verify();status('FINAL_AUDIT',verdict=summary['verdict'])
    if len(INDEX)!=COUNTS['committed_full_jobs'] or len({(x['world'],x['arm']) for x in INDEX})!=len(INDEX):raise AssertionError('final committed roster')
    dump(summary,OUT/'SUMMARY.json');report({**summary,**COUNTS},'REPORT')
    dump({'verdict':'PASS','source_unchanged':True,'all_committed_jobs_independently_audited':True,'counts':dict(COUNTS),
          'bounded_compact_reduction':True,'no_test_feedback':True,'no_automatic_publish':True},OUT/'FINAL_AUDIT.json')
    status('PACKAGING',verdict=summary['verdict'])
    tmp=OUT/'RESULT_BUNDLE.zip.pending';target=OUT/'RESULT_BUNDLE.zip'
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
        for base in (OUT,OPS):
            for p in sorted(base.rglob('*')):
                if p.is_file() and p.name not in ('RESULT_BUNDLE.zip','RESULT_BUNDLE.zip.pending','PACKAGE.json') and 'active' not in p.parts and '__pycache__' not in p.parts:
                    z.write(p,p.relative_to(ROOT))
        for name in ORIGINAL_LOCK():z.write(ROOT.parent/name,'source/'+name)
    with zipfile.ZipFile(tmp) as z:
        if z.testzip() is not None:raise AssertionError('package CRC failed')
    if sum(p.stat().st_size for b in (OUT,OPS) for p in b.rglob('*') if p.is_file())>2*1024**3:raise AssertionError('artifact budget exceeded')
    os.replace(tmp,target)
    dump({'status':'COMPLETE','bytes':target.stat().st_size,'sha256':digest(target),'verdict':summary['verdict']},OUT/'PACKAGE.json')
    status('COMPLETE',verdict=summary['verdict'],package_complete=True)

def certified_disk_counts(base):
    counts={'committed_full_jobs':0,'committed_external_records':0,'committed_training_bytes':0,'correlated_state_trajectories':0}
    entries=[]
    for path in sorted(Path(base).glob('*/receipts/*.json.gz')):
        cert=load(str(path)+'.audit.json')
        if cert['verdict']!='PASS' or cert['receipt_sha256']!=digest(path):raise AssertionError('invalid committed disk certificate')
        world,arm=path.name[:-8].split('_',1)
        counts['committed_full_jobs']+=1;counts['committed_external_records']+=cert['training_records']
        counts['committed_training_bytes']+=cert['training_bytes'];counts['correlated_state_trajectories']+=cert['state_trajectories']
        entries.append({'world':int(world),'arm':arm,'stage':path.parent.parent.name.upper(),'receipt':str(path),'sha256':cert['receipt_sha256']})
    return counts,entries

def supervisor():
    global START
    START=time.monotonic();verify()
    if load(OUT/'AUTHORIZATION.json')['execution_authorized'] is not True or load(OUT/'ADAPTER_QUALIFICATION.json')['verdict']!='PASS':raise AssertionError('authorization/qualification')
    fd=os.open(OUT/'SUPERVISOR.claim',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600);os.close(fd)
    caff=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())])
    if caff.poll() is not None:raise RuntimeError('caffeinate attach failed')
    dump({'supervisor_pid':os.getpid(),'caffeinate_pid':caff.pid,'workers':2,'detached':True,'started_epoch':time.time(),
          'log':str(OPS/'supervisor.log'),'session_hard_limit_hours':10},OUT/'LAUNCH.json')
    stop=threading.Event()
    def beat():
        while not stop.wait(10):status()
    thread=threading.Thread(target=beat,daemon=True);thread.start()
    try:
        dev=manifest('DEV',fresh_rows('DEV'),config.ARMS);rows=stage_run(dev);status('DEV_ANALYSIS')
        selected=decision.select(rows)
        selected.update({'source':identity(),'locked_epoch':time.time(),'confirmation_entropy_created':False})
        dump(selected,OUT/'SELECTION_LOCK.json');report(selected,'DEV_REPORT')
        if selected['selected'] is None:summary=selected
        else:
            cm=manifest('CONFIRM',fresh_rows('CONFIRM'),roster(selected['selected']))
            dump({'selection_sha256':digest(OUT/'SELECTION_LOCK.json'),'generated_epoch':time.time(),'drawn_after_selection':True},OUT/'CONFIRM_ADMISSION.json')
            rows=stage_run(cm);status('CONFIRM_ANALYSIS');summary=decision.confirm(rows,selected['selected']);report(summary,'CONFIRM_REPORT')
        finish(summary)
    except BaseException as exc:
        paused=str(exc)=='PAUSED_SESSION_LIMIT'
        recovered,entries=certified_disk_counts(OUT)
        with LOCK:
            COUNTS.update(recovered);INDEX[:]=entries
            dump({'entries':INDEX},OUT/'COMMIT_INDEX.json')
        dump({'error':str(exc),'traceback':traceback.format_exc(),'counts':dict(COUNTS),'epoch':time.time()},OUT/('SESSION_RESULT.json' if paused else 'FAILURE.json'))
        status('PAUSED_SESSION_LIMIT' if paused else 'FAILED',error=str(exc));raise
    finally:stop.set();thread.join();caff.terminate();caff.wait(timeout=10)

def tests():
    register_geometry()
    # Audit an already accepted technical trajectory: zero new learning records.
    r=runner.read(ROOT/'cycles/cycle3/receipts/941004_COV-R.json.gz');r['source']=identity()
    row={k:r[k] for k in ('world','family','noise','ordinary')}
    m=manifest('ADAPTER_TEST',[row],['COV-R']);validate_manifest(m,True);install(m)
    e=envelope(r,m);audit(e,m);rejected=[]
    for k,v in [('stage','CONFIRM'),('manifest_sha256','bad'),('training_records',0),('arm','COV-I')]:
        bad=copy.deepcopy(e);bad[k]=v
        try:audit(bad,m)
        except (AssertionError,ValueError):rejected.append(k)
        else:raise AssertionError('bad science envelope accepted')
    for args in [(row['world'],'TRIANGLE_TAIL','STANDARD',False),(999999,'PATH_LOOP','STANDARD',False),(row['world'],'PATH_LOOP','BLURRY',False)]:
        try:config.admit(*args)
        except ValueError:rejected.append('admission')
        else:raise AssertionError('off manifest admission')
    try:validate_manifest(m)
    except AssertionError:rejected.append('test_manifest_science_rejection')
    else:raise AssertionError('test admitted to science')
    cell=compact(e);rr=grouped([cell],m);assert rr[0]['COV-R']['W']['J']==cell['metrics']['W']['J']
    assert all(len(set(roster(a)))==3 for a in config.ARMS[1:])
    from pipeline import suite
    pipeline=suite(OPS/'pipeline_test')
    fixture=OPS/'count_test/dev/receipts';fixture.mkdir(parents=True,exist_ok=True)
    path=fixture/'123_COV-R.json.gz';path.write_bytes(b'SYNTHETIC_CERTIFICATE_ONLY_ZERO_TRAINING')
    dump({'verdict':'PASS','receipt_sha256':digest(path),'training_records':2560,'training_bytes':15000,'state_trajectories':23},Path(str(path)+'.audit.json'))
    counts,entries=certified_disk_counts(OPS/'count_test')
    assert counts['committed_full_jobs']==1 and counts['committed_external_records']==2560 and len(entries)==1
    # This is a synthetic counting fixture, not a kernel receipt or science record.

    config.admit=ORIGINAL_ADMIT;config.lock=ORIGINAL_LOCK
    return {'verdict':'PASS','adapter_only':True,'qualified_kernel_unchanged':True,'new_training_records':0,'science_jobs':0,
            'existing_fixture_independent_audit':True,'rejected_cases':rejected,'compact_reducer':True,'failure_disk_count_recovery':True,'peer_rosters':True,'pipeline':pipeline}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--test',action='store_true');p.add_argument('--supervisor',action='store_true');p.add_argument('--worker',action='store_true');p.add_argument('--manifest',type=Path);p.add_argument('--world',type=int);p.add_argument('--arm');a=p.parse_args()
    if a.test:print(json.dumps(tests(),sort_keys=True))
    elif a.worker:worker(a.manifest,a.world,a.arm)
    elif a.supervisor:supervisor()
    else:raise ValueError('explicit mode')
