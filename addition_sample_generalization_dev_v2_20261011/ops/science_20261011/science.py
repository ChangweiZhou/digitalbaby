"""Authorized admission/dispatch/reduction only; qualified learning kernel unchanged."""
import argparse, copy, gzip, hashlib, json, math, os, secrets, signal, subprocess, sys, threading, time, traceback, zipfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

OPS=Path(__file__).resolve().parent
ROOT=OPS.parent.parent
sys.path.insert(0,str(ROOT))
import config, runner, decision
from assets import canonical
OUT=ROOT/'results'
ORIGINAL_LOCK=config.lock
ORIGINAL_ADMIT=config.admit
FILES=('science.py',)

def dump(x,p): runner.atomic_json(x,p)
def load(p): return json.loads(Path(p).read_text())
def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def identity():
    d=ORIGINAL_LOCK()
    for name in FILES: d[str((OPS/name).relative_to(ROOT.parent))]=digest(OPS/name)
    return d

def verify_sources():
    wanted=load(OUT/'SOURCE_LOCK.json')
    if identity()!=wanted['files']: raise AssertionError('frozen kernel/adapter source mismatch')
    if canonical()['inherited_source']!=wanted['inherited_source']: raise AssertionError('inherited source mismatch')

def install(manifest):
    def admit(world,family):
        rows=[r for r in manifest['worlds'] if r['world']==world and r['family']==family]
        if len(rows)!=1: raise ValueError('world/family outside sealed stage')
        if manifest['stage']=='DEV' and family=='ODD_CYCLE': raise ValueError('confirmation geometry not admitted to DEV')
    config.admit=admit
    config.lock=identity

def envelope(receipt,manifest):
    # Retain internal qualified-kernel namespace fields explicitly, never call them science counts.
    return {'schema':'SAMPLE_GEN_SCIENCE_ENVELOPE_V2','stage':manifest['stage'],
            'manifest_sha256':manifest['sha256'],'source':identity(),
            'kernel_envelope_convention':'technical markers are historical internal serialization, not cohort classification',
            'world':receipt['world'],'arm':receipt['arm'],'science_jobs':1,
            'training_records':receipt['training_records'],'training_bytes':receipt['training_bytes'],
            'state_trajectories':receipt['state_trajectories'],'kernel_receipt':receipt}

def audit(e,manifest):
    if e['schema']!='SAMPLE_GEN_SCIENCE_ENVELOPE_V2' or e['stage']!=manifest['stage'] or e['manifest_sha256']!=manifest['sha256'] or e['source']!=identity(): raise AssertionError('science admission/source seal')
    r=e['kernel_receipt']
    rows=[x for x in manifest['worlds'] if x['world']==r['world']]
    if len(rows)!=1 or r['arm'] not in manifest['arms']: raise AssertionError('off-roster science')
    t=rows[0]
    if any(r[k]!=t[k] for k in ('world','family','noise','ordinary')) or e['world']!=r['world'] or e['arm']!=r['arm']: raise AssertionError('task provenance')
    if e['science_jobs']!=1 or e['training_records']!=2400 or e['state_trajectories']!=21 or e['training_bytes']!=r['training_bytes']: raise AssertionError('exposure count')
    a=runner.audit(r)
    for branch in r['branches'].values():
        for f in branch['forks'].values():
            for snap in (f['old_checkpoint'],f['final_checkpoint']):
                if r['arm']!='STABLE-ID' and snap['payload_bytes']>131072: raise AssertionError('candidate memory cap')
    return {'verdict':'PASS','science_jobs':1,'training_records':2400,'training_bytes':a['training_bytes'],'state_trajectories':21,'independent_record_equations':True}

def write_receipt(e,p,manifest):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    fd=os.open(str(p)+'.claim',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600); os.close(fd)
    if p.exists(): raise ValueError('already committed; no repeat')
    provisional=Path(str(p)+'.provisional')
    with gzip.open(provisional,'wt') as f: json.dump(e,f,sort_keys=True,separators=(',',':'),allow_nan=False)
    a=audit(runner.read(provisional),manifest)
    a['receipt_sha256']=digest(provisional)
    os.replace(provisional,p); Path(str(p)+'.claim').unlink()
    dump(a,p.with_name(p.name+'.audit.json'))
    return a

def seal_manifest(stage,rows,arms):
    x={'schema':'SEALED_SCIENCE_ADMISSION_V2','stage':stage,'worlds':rows,'arms':list(arms),'source':identity()}
    x['sha256']=hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
    return x

def read_manifest(p):
    x=load(p); h=x.pop('sha256')
    if hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()!=h or x['source']!=identity(): raise AssertionError('manifest seal')
    x['sha256']=h
    if x['stage'] not in ('DEV','CONFIRM') or len({r['world'] for r in x['worlds']})!=len(x['worlds']): raise AssertionError('manifest roster')
    return x

def fresh_rows(stage):
    # Independent entropy sampled at DEV admission, and only after SELECTION_LOCK for CONFIRM.
    rows=[]; used=set(config.TECH_WORLDS)|{835001}
    if stage=='CONFIRM': used|={r['world'] for r in read_manifest(OUT/'dev/MANIFEST.json')['worlds']}
    strata=([(f,n,False,3) for f in ('PATH_LOOP','STAR_LOOP') for n in ('STANDARD','BLURRY')] if stage=='DEV' else
            [('ODD_CYCLE',n,False,32) for n in ('STANDARD','BLURRY')]+[(f,n,True,8) for f in ('PATH_LOOP','STAR_LOOP') for n in ('STANDARD','BLURRY')])
    for family,noise,ordinary,count in strata:
        for _ in range(count):
            while True:
                w=secrets.randbelow(2**52)+2**52
                if w not in used: break
            used.add(w); rows.append(dict(world=w,family=family,noise=noise,ordinary=ordinary))
    return rows

def receipt_metric(e):
    x=decision.metrics(e['kernel_receipt']); x.pop('technical_only')
    return x

def means(receipts,arms):
    result={}
    for a in arms:
        entries=[receipt_metric(e) for e in receipts if e['arm']==a]
        result[a]={k:float(np.mean([m['W'][k] for m in entries])) for k in ('score','final_old_held','final_new_held')}
        for k in ('causal_N','causal_shuffled'): result[a][k]=float(np.mean([m[k] for m in entries]))
        result[a]['production_cpu_seconds']=sum(float(row['production_cpu_seconds']) for e in receipts if e['arm']==a for b in e['kernel_receipt']['branches'].values() for rows in [b['old_records']]+[f['new_records'] for f in b['forks'].values()] for row in rows)/len(entries)
        result[a]['persistent_bytes']=max(s['payload_bytes'] for e in receipts if e['arm']==a for b in e['kernel_receipt']['branches'].values() for s in [b['birth']]+[f['final_checkpoint'] for f in b['forks'].values()])
    return result

def choose(m):
    eligible=[]
    for a in config.ARMS[2:]:
        x=m[a]; gain=x['score']-m['COMPACT-ID']['score']
        if x['final_old_held']>=.75 and x['final_new_held']>=.60 and x['causal_N']>0 and x['causal_shuffled']>0 and gain>=.03 and x['persistent_bytes']<=131072:
            eligible.append(a)
    if not eligible: return None
    best=max(m[a]['score']-m['COMPACT-ID']['score'] for a in eligible)
    tied=[a for a in eligible if best-(m[a]['score']-m['COMPACT-ID']['score'])<=1e-12]
    return min(tied,key=lambda a:(m[a]['production_cpu_seconds'],m[a]['persistent_bytes'],config.ARMS.index(a)))

def roster(a):
    group=next(g for g in [('COV-R','DIAG-R'),('TOPO-G','TOPO-S','TOPO-R'),('REPLAY-P','REPLAY-U')] if a in g)
    return ['STABLE-ID','COMPACT-ID']+list(group)

def confirm_result(receipts,selected,arms):
    opaque=[e for e in receipts if not e['kernel_receipt']['ordinary']]
    guard=[e for e in receipts if e['kernel_receipt']['ordinary']]
    keyed={(e['world'],e['arm']):receipt_metric(e) for e in opaque}
    primary={}
    for kind in ('gain','causal_N','causal_shuffled'):
        strata=[]
        for noise in ('STANDARD','BLURRY'):
            values=[]
            for e in opaque:
                if e['arm']==selected and e['kernel_receipt']['noise']==noise:
                    m=keyed[(e['world'],selected)]
                    values.append(m['W']['score']-keyed[(e['world'],'COMPACT-ID')]['W']['score'] if kind=='gain' else m[kind])
            strata.append(values)
        primary[kind]=decision.lower_bound(strata)
    pm=means(opaque,arms); gm=means(guard,arms)
    x=pm[selected]; checks={
        'minimum_primary_gain':primary['gain']['mean']>=.05,'primary_lower_positive':primary['gain']['lower']>0,
        'old_causal_lower_positive':primary['causal_N']['lower']>0,'shuffled_causal_lower_positive':primary['causal_shuffled']['lower']>0,
        'final_old_at_least_75':x['final_old_held']>=.75,'final_new_at_least_60':x['final_new_held']>=.60,
        'ordinary_score_protected':gm[selected]['score']>=gm['COMPACT-ID']['score']-.05,
        'ordinary_final_protected':gm[selected]['final_old_held']>=gm['COMPACT-ID']['final_old_held']-.05}
    return {'verdict':'ADOPT_BOUNDED_SAMPLE_EFFICIENT_CORE' if all(checks.values()) else 'RETAIN_REFERENCE',
            'selected_candidate':selected,'checks':checks,'primary':primary,'opaque_means':pm,'ordinary_guard_means':gm}

def worker(manifest_path,world,arm):
    verify_sources(); manifest=read_manifest(manifest_path); install(manifest)
    row=next(x for x in manifest['worlds'] if x['world']==world)
    if arm not in manifest['arms']: raise ValueError('off roster arm')
    stage_dir=Path(manifest_path).parent; active=stage_dir/'active'/f'{world}_{arm}.json'; active.parent.mkdir(exist_ok=True)
    stop=threading.Event()
    def beat():
        while not stop.is_set():
            dump({'pid':os.getpid(),'world':world,'arm':arm,'heartbeat_epoch':time.time(),'stage':'IN_FLIGHT_NOT_COMMITTED'},active); stop.wait(5)
    t=threading.Thread(target=beat,daemon=True); t.start()
    try:
        r=runner.run(row['world'],row['family'],row['noise'],row['ordinary'],arm)
        e=envelope(r,manifest); write_receipt(e,stage_dir/'receipts'/f'{world}_{arm}.json.gz',manifest)
    finally:
        stop.set(); t.join(); active.unlink(missing_ok=True)

def counts():
    files=list(OUT.glob('*/receipts/*.json.gz'))
    bytes_seen=0
    for p in files: bytes_seen+=runner.read(p)['training_bytes']
    return {'committed_full_jobs':len(files),'committed_external_records':len(files)*2400,
            'committed_training_bytes':bytes_seen,'state_trajectories_with_shared_prefix':len(files)*21}

SESSION_START=0
STAGE='STARTING'
STATE_LOCK=threading.Lock()

def status(stage=None,**kw):
    global STAGE
    with STATE_LOCK:
        if stage: STAGE=stage
        x={'stage':STAGE,'heartbeat_epoch':time.time(),'supervisor_pid':os.getpid(),'workers':2,**counts(),**kw}
        dump(x,OUT/'STATUS.json')

def stage_run(manifest):
    name=manifest['stage'].lower(); folder=OUT/name; folder.mkdir(exist_ok=False)
    dump(manifest,folder/'MANIFEST.json'); install(manifest)
    tasks=[(r['world'],a) for r in manifest['worlds'] for a in manifest['arms']]
    status(manifest['stage'],stage_total_jobs=len(tasks))
    completed=[]; processes={}; lock=threading.Lock(); abort=threading.Event()
    def job(task):
        w,a=task
        if abort.is_set(): raise RuntimeError('admission closed after failure')
        if time.monotonic()-SESSION_START+120>9*3600: raise RuntimeError('PAUSED_SESSION_LIMIT')
        verify_sources()
        logs=folder/'logs'; logs.mkdir(exist_ok=True)
        log=logs/f'{w}_{a}.log'
        with log.open('x') as f:
            if abort.is_set(): raise RuntimeError('admission closed after failure')
            p=subprocess.Popen([sys.executable,str(OPS/'science.py'),'--worker','--manifest',str(folder/'MANIFEST.json'),'--world',str(w),'--arm',a],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            with lock: processes[task]=p
            code=p.wait()
            with lock: processes.pop(task,None)
        if code: raise RuntimeError(f'worker exit {code}: {log}')
        receipt=runner.read(folder/'receipts'/f'{w}_{a}.json.gz')
        # Disk artifact was independently audited before commitment by its worker.
        cert=load(folder/'receipts'/f'{w}_{a}.json.gz.audit.json')
        if cert['verdict']!='PASS' or cert['receipt_sha256']!=digest(folder/'receipts'/f'{w}_{a}.json.gz'): raise AssertionError('missing independent acceptance')
        return receipt
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures={pool.submit(job,t):t for t in tasks}
        try:
            for f in as_completed(futures):
                completed.append(f.result()); status(manifest['stage'],stage_total_jobs=len(tasks))
                dump({'stage':manifest['stage'],'committed_jobs':[{'world':e['world'],'arm':e['arm']} for e in completed]},folder/'COMMIT_INDEX.json')
        except BaseException:
            abort.set()
            for f in futures: f.cancel()
            with lock:
                for p in processes.values():
                    if p.poll() is None: p.terminate()
            raise
    if len(completed)!=len(tasks): raise AssertionError('whole stage incomplete')
    analysis=[]
    for e in completed:
        mm=receipt_metric(e)
        thresholds={}
        for limit in (.75,.90):
            reached=[n+80 for n,v in zip(config.CHECKPOINTS,mm['W']['persistent_old_held']) if v>=limit]
            thresholds[str(limit)]={'first_observed_total_samples':min(reached) if reached else None,'right_censored_at_400':not bool(reached)}
        analysis.append({'world':e['world'],'arm':e['arm'],'noise':e['kernel_receipt']['noise'],'ordinary':e['kernel_receipt']['ordinary'],'metrics':mm,'thresholds':thresholds,'countermodels':decision.countermodels(e['kernel_receipt'])})
    with gzip.open(folder/'WORLD_METRICS.json.gz','wt') as f: json.dump(analysis,f,sort_keys=True,allow_nan=False)
    dump({'stage':'COMPLETE','jobs':len(completed)},folder/'STATUS.json')
    return completed

def report(summary,name):
    text=['# SAMPLE-GEN V2 '+name,'','Verdict: '+summary.get('verdict','DEV_SCREEN'),'','This tests E3 within bounded addition with explicit identity/role and ordinal-output priors. It does not establish arbitrary arithmetic or broad intelligence.','', '```json',json.dumps(summary,indent=2), '```','']
    (OUT/(name+'.md')).write_text('\n'.join(text))

def package(summary):
    status('PACKAGING',verdict=summary['verdict'])
    dump({'verdict':'PASS','sources_verified':True,'complete_stages_only':True,'no_test_feedback':True,'independent_equations_per_committed_job':True,**counts()},OUT/'FINAL_AUDIT.json')
    report({**summary,**counts()},'REPORT')
    path=OUT/'RESULT_BUNDLE.zip'; temporary=OUT/'RESULT_BUNDLE.zip.pending'
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED) as z:
        for folder in (OUT,OPS):
            for p in sorted(folder.rglob('*')):
                if p.is_file() and p.name not in ('RESULT_BUNDLE.zip','RESULT_BUNDLE.zip.pending','PACKAGE.json') and 'active' not in p.parts:
                    z.write(p,str(p.relative_to(ROOT)))
        for name in ORIGINAL_LOCK(): z.write(ROOT.parent/name,'source/'+name)
    with zipfile.ZipFile(temporary) as z:
        if z.testzip() is not None: raise AssertionError('package CRC')
    artifact_bytes=sum(p.stat().st_size for base in (OUT,OPS) for p in base.rglob('*') if p.is_file())
    if artifact_bytes>2*1024**3: raise AssertionError('artifact resource cap including receipts and bundle')
    os.replace(temporary,path)
    dump({'status':'COMPLETE','bytes':path.stat().st_size,'sha256':digest(path),'verdict':summary['verdict']},OUT/'PACKAGE.json')
    status('COMPLETE',verdict=summary['verdict'],package_complete=True)

def supervisor():
    global SESSION_START
    SESSION_START=time.monotonic(); verify_sources()
    def terminate(signum,frame): raise RuntimeError('supervisor received termination signal '+str(signum))
    signal.signal(signal.SIGTERM,terminate)
    auth=load(OUT/'AUTHORIZATION.json')
    if not auth['execution_authorized'] or load(OUT/'ADAPTER_QUALIFICATION.json')['verdict']!='PASS': raise AssertionError('admission not authorized/qualified')
    caffeinate=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    dump({'supervisor_pid':os.getpid(),'caffeinate_pid':caffeinate.pid,'workers':2,'log':str(OPS/'supervisor.log'),'started_epoch':time.time(),'detached':True},OUT/'LAUNCH.json')
    stop=threading.Event()
    def beat():
        while not stop.wait(10): status()
    heartbeat=threading.Thread(target=beat,daemon=True); heartbeat.start()
    try:
        dev_manifest=seal_manifest('DEV',fresh_rows('DEV'),config.ARMS)
        receipts=stage_run(dev_manifest)
        status('DEV_ANALYSIS'); m=means(receipts,config.ARMS); selected=choose(m)
        screen={'verdict':'CANDIDATE_SELECTED' if selected else 'NO_CANDIDATE_ADVANCES','selected_candidate':selected,'means':m,'selection_uses_only_DEV':True}
        dump(screen,OUT/'DEV_SUMMARY.json'); report(screen,'DEV_REPORT')
        selection={**screen,'source':identity(),'locked_epoch':time.time(),'confirm_entropy_created':False}
        dump(selection,OUT/'SELECTION_LOCK.json')
        if selected is None: summary=screen
        else:
            # This is the only confirmation admission, after immutable selection.
            confirmation=seal_manifest('CONFIRM',fresh_rows('CONFIRM'),roster(selected))
            dump({'selection_sha256':digest(OUT/'SELECTION_LOCK.json'),'generated_epoch':time.time(),'private_roots_drawn_after_selection':True},OUT/'CONFIRM_ADMISSION.json')
            receipts=stage_run(confirmation)
            status('CONFIRM_ANALYSIS'); summary=confirm_result(receipts,selected,confirmation['arms'])
            dump(summary,OUT/'CONFIRM_SUMMARY.json'); report(summary,'CONFIRM_REPORT')
        verify_sources(); dump(summary,OUT/'SUMMARY.json'); stop.set(); heartbeat.join(); package(summary)
    except BaseException as exc:
        stop.set(); heartbeat.join()
        paused=str(exc)=='PAUSED_SESSION_LIMIT'
        dump({'error':str(exc),'traceback':traceback.format_exc(),'epoch':time.time(),**counts()},OUT/('SESSION_RESULT.json' if paused else 'FAILURE.json'))
        status('PAUSED_SESSION_LIMIT' if paused else 'FAILED',error=str(exc)); raise
    finally:
        caffeinate.terminate()

def tests():
    decision.synthetic_suite()
    m={a:{'score':.1,'final_old_held':.1,'final_new_held':.1,'causal_N':0,'causal_shuffled':0,'production_cpu_seconds':2,'persistent_bytes':100} for a in config.ARMS}
    assert choose(m) is None
    for a in ('COV-R','DIAG-R'): m[a].update(score=.6,final_old_held=.9,final_new_held=.7,causal_N=.3,causal_shuffled=.3)
    m['DIAG-R']['production_cpu_seconds']=1
    assert choose(m)=='DIAG-R'
    m['DIAG-R']['final_new_held']=.59
    assert choose(m)=='COV-R'
    assert len(roster('TOPO-G'))==5 and len(roster('COV-R'))==4
    rows=[{'world':835004,'family':'PATH_LOOP','noise':'STANDARD','ordinary':False}]
    manifest=seal_manifest('DEV',rows,config.ARMS); install(manifest)
    rejected=0
    for args in [(835008,'ODD_CYCLE'),(835001,'PATH_LOOP'),(999,'PATH_LOOP')]:
        try: config.admit(*args)
        except ValueError: rejected+=1
        else: raise AssertionError('offstage admission accepted')
    # Use an already committed technical fixture to test only new envelope/auditor wiring.
    r=runner.read(ROOT/'cycles/cycle3/receipts/835004_COMPACT-ID.json.gz')
    r['source']=identity(); e=envelope(r,manifest); audit(e,manifest)
    for field,value in [('stage','CONFIRM'),('training_records',0),('manifest_sha256','bad'),('arm','COV-R')]:
        bad=copy.deepcopy(e); bad[field]=value
        try: audit(bad,manifest)
        except AssertionError: rejected+=1
        else: raise AssertionError('bad science envelope accepted')
    assert rejected==7
    # Synthetic confirmation data test only: no generator, learner or real endpoint.
    synthetic=[]
    for ordinary,world_count in ((False,64),(True,32)):
        for i in range(world_count):
            for a in roster('COV-R'):
                v=.95 if a=='COV-R' else .20
                branches={}
                for history in config.HISTORIES:
                    old=v if history=='W' else .10
                    branches[history]={'forks':{str(n):{'final_old_held':{'accuracy':old},
                        'final_new_held':{'accuracy':.70},'immediate_old_held':{'accuracy':old},
                        'final_checkpoint':{'payload_bytes':100},'new_records':[]} for n in config.CHECKPOINTS},
                        'old_records':[],'birth':{'payload_bytes':100}}
                synthetic.append({'world':i+(1000 if ordinary else 0),'arm':a,
                    'kernel_receipt':{'ordinary':ordinary,'noise':'STANDARD' if i<world_count//2 else 'BLURRY','branches':branches}})
    answer=confirm_result(synthetic,'COV-R',roster('COV-R'))
    assert answer['verdict']=='ADOPT_BOUNDED_SAMPLE_EFFICIENT_CORE' and all(answer['checks'].values())
    bad=copy.deepcopy(synthetic)
    for e in bad:
        if e['arm']=='COV-R' and e['kernel_receipt']['ordinary']:
            for f in e['kernel_receipt']['branches']['W']['forks'].values(): f['final_old_held']['accuracy']=0
    assert confirm_result(bad,'COV-R',roster('COV-R'))['verdict']=='RETAIN_REFERENCE'
    thin=[e for e in synthetic if e['world']!=0]
    try: confirm_result(thin,'COV-R',roster('COV-R'))
    except ValueError: pass
    else: raise AssertionError('missing primary world silently accepted')
    config.admit=ORIGINAL_ADMIT; config.lock=ORIGINAL_LOCK
    return {'verdict':'PASS','confirmation_positive_negative_and_incomplete':True,'selection_none_and_ties':True,'family_rosters':True,'rejected_admission_envelope_cases':rejected,'existing_fixture_audit':True,'new_native_records':0,'science_jobs':0,'kernel_files_modified':False}

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--test',action='store_true'); p.add_argument('--supervisor',action='store_true'); p.add_argument('--worker',action='store_true'); p.add_argument('--manifest',type=Path); p.add_argument('--world',type=int); p.add_argument('--arm'); a=p.parse_args()
    if a.test: print(json.dumps(tests(),sort_keys=True))
    elif a.worker: worker(a.manifest,a.world,a.arm)
    elif a.supervisor: supervisor()
    else: raise ValueError('explicit mode required')
