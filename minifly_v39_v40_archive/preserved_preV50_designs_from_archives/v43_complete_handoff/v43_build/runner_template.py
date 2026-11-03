#!/usr/bin/env python3
"""V43: fast/slow read authority. One command runs, resumes, and reports."""
from __future__ import annotations
import os
for _thread_key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_thread_key]='1'
import argparse,base64,collections,contextlib,fcntl,hashlib,importlib,json,math,pickle,platform,resource,shutil,signal,struct,subprocess,sys,time,types,zipfile,zlib
from pathlib import Path
BOOT_SOURCE_SHA=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
VERSION='V43-fast-slow-authority-1.0'
PROTOCOL=__PROTOCOL__
ORIGINAL_PROTOCOL=__ORIGINAL_PROTOCOL__
PAYLOADS=__PAYLOADS__
SEEDS=tuple(range(43101,43117))
LOADS=(64,128,256,512)
ENGINEERING_SEEDS=(943001,943002)
STOP=False
MODULES={}

def dependencies(install=False):
    required={'numpy':'numpy','scipy':'scipy','matplotlib':'matplotlib','sklearn':'scikit-learn'}
    missing=[]
    for module,package in required.items():
        try:importlib.import_module(module)
        except ImportError:missing.append(package)
    if missing:
        if not install:raise RuntimeError('Missing dependencies '+', '.join(missing)+'. Run with --install-deps.')
        subprocess.run([sys.executable,'-m','pip','install',*missing],check=True)
    global np
    import numpy as np

def cast(x):
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,Path):return str(x)
    if isinstance(x,(bytes,bytearray)):return bytes(x).hex()
    if isinstance(x,set):return sorted(x)
    raise TypeError(type(x).__name__)

def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False,default=cast).encode()
def sha(x):return hashlib.sha256(x if isinstance(x,bytes) else canonical(x)).hexdigest()
def source_sha():
    current=sha(Path(__file__).read_bytes())
    if current!=BOOT_SOURCE_SHA:raise RuntimeError('Source changed during execution; refusing mixed source identities')
    return BOOT_SOURCE_SHA

def load_modules():
    if MODULES:return MODULES
    from scipy.stats import t
    def load(name,key,extra=None):
        p=PAYLOADS[key];raw=zlib.decompress(base64.b85decode(p['data']));assert sha(raw)==p['sha256']
        text=raw.decode()
        if key=='claude':
            assert text.count('V39 = _load_v39()')==1
            text=text.replace('V39 = _load_v39()','V39 = M')
        mod=types.ModuleType(name);mod.__file__=str(Path(__file__).resolve());sys.modules[name]=mod
        mod.__dict__.update(extra or {});exec(compile(text,'<embedded:'+key+'>','exec'),mod.__dict__)
        mod.np=np;mod.student_t=t;MODULES[key]=mod;return mod
    M=load('v41_legacy','legacy');C=load('v41_claude','claude',{'M':M})
    H=load('v41_hybrid','hybrid',{'M':M,'C':C,'np':np})
    O=load('v43_offline_controls','offline_controls',{'np':np})
    load('v43_engine','engine',{'M':M,'C':C,'H':H,'np':np,'O':O})
    load('v43_reporting','reporting',{'np':np})
    load('v43_diagnostics','diagnostics',{'np':np})
    return MODULES

def atomic(path,data):
    source_sha()
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp-'+str(os.getpid()))
    with tmp.open('wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)

def write_json(path,x):atomic(path,json.dumps(x,indent=2,sort_keys=True,allow_nan=False,default=cast).encode()+b'\n')
def peak_rss():return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024))

def request_stop(sig,frame):
    global STOP
    STOP=True

@contextlib.contextmanager
def writer_lock(path,wait=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+') as f:
        try:fcntl.flock(f.fileno(),fcntl.LOCK_EX if wait else fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another writer holds '+str(path))
        try:yield
        finally:fcntl.flock(f.fileno(),fcntl.LOCK_UN)

def pack_state(state,identity):
    payload=zlib.compress(pickle.dumps(state,protocol=5),1)
    header=canonical({'identity':identity,'sha256':sha(payload),'format':'V43-pickle5-zlib'})
    return struct.pack('>Q',len(header))+header+payload

def unpack_state(raw,identity):
    if len(raw)<8:raise ValueError('Truncated checkpoint')
    n=struct.unpack('>Q',raw[:8])[0]
    if n>len(raw)-8:raise ValueError('Truncated checkpoint header')
    h=json.loads(raw[8:8+n]);payload=raw[8+n:]
    if h.get('identity')!=identity:raise ValueError('Checkpoint identity mismatch')
    if h.get('sha256')!=sha(payload):raise ValueError('Checkpoint checksum mismatch')
    return pickle.loads(zlib.decompress(payload))

def save_state(path,state,identity):
    data=pack_state(state,identity)
    if path.exists():atomic(path.with_suffix(path.suffix+'.previous'),path.read_bytes())
    atomic(path,data)

def cfg_for(load,smoke=False):
    return {'load':int(load),'keys_per_wave':int(load)//4,'repeats':2 if smoke else 8,'trace_cap':512,'slot_cap':384,'smoke':bool(smoke)}

def make_plan(smoke=False):
    seeds=(943901,) if smoke else SEEDS;loads=(16,) if smoke else LOADS
    return {'version':VERSION,'source_sha256':source_sha(),'protocol_sha256':sha(PROTOCOL.encode()),
            'mode':'engineering_smoke' if smoke else 'science','seeds':list(seeds),'loads':list(loads),
            'jobs':[{'seed':s,'load':l,'config':cfg_for(l,smoke)} for s in seeds for l in loads],
            'expected_trajectories':len(seeds)*len(loads),'policies':['SUPER','READ-HYBRID','SILENT-HARD','ANCHOR-ONLY','FIXED-50','GLOBAL-BAYES','LOCAL-BAYES-0','LOCAL-BAYES-E2'],
            'offline_controls':['SHUFFLED-ALPHA','EVENT-ORACLE','SLOT-ORACLE'],
            'payload_hashes':{k:p['sha256'] for k,p in PAYLOADS.items()}}

def job_name(job):return f"seed_{job['seed']}_load_{job['load']}"
def identity_for(plan,job):return {'plan_sha256':sha(plan),'job':job}

def write_events(path,rows):
    keys=sorted({k for row in rows for k in row});arrays={}
    for key in keys:
        values=[row.get(key) for row in rows];good=[v for v in values if v is not None]
        if all(isinstance(v,(bool,np.bool_)) for v in good) and good:
            arrays[key]=np.array([-1 if v is None else int(v) for v in values],dtype=np.int8)
        elif all(isinstance(v,(int,np.integer)) for v in good) and good:
            arrays[key]=np.array([-1 if v is None else int(v) for v in values],dtype=np.int64)
        elif all(isinstance(v,(float,int,np.number)) for v in good):
            arrays[key]=np.array([np.nan if v is None else float(v) for v in values],dtype=np.float64)
        elif all(isinstance(v,str) for v in good):arrays[key]=np.array(['' if v is None else v for v in values],dtype=str)
        else:arrays[key]=np.array([json.dumps(v,sort_keys=True,default=cast,allow_nan=False) for v in values],dtype=str)
    arrays['__row_count__']=np.array(len(rows),dtype=np.int64)
    import io
    b=io.BytesIO();np.savez_compressed(b,**arrays);atomic(path,b.getvalue())
    with np.load(path,allow_pickle=False) as saved:
        assert int(saved['__row_count__'])==len(rows)
        assert set(saved.files)==set(keys)|{'__row_count__'}
        assert all(len(saved[k])==len(rows) for k in keys)
    return {'sha256':sha(b.getvalue()),'rows':len(rows),'columns':keys}

def completed_read(root,plan,job):
    path=root/'completed'/(job_name(job)+'.json')
    if not path.exists():return None
    env=json.loads(path.read_bytes())
    if env['identity']!=identity_for(plan,job):raise ValueError('Completed identity mismatch: '+path.name)
    if env['sha256']!=sha(env['data']):raise ValueError('Completed checksum mismatch: '+path.name)
    result=env['data']
    if result.get('seed')!=job['seed'] or result.get('load')!=job['load'] or result.get('physical_trajectories')!=1:raise ValueError('Completed result cell mismatch')
    events=root/result['authority_events_file']
    if not events.exists() or sha(events.read_bytes())!=result['authority_events_sha256']:raise ValueError('Authority-event checksum mismatch: '+str(events))
    with np.load(events,allow_pickle=False) as saved:
        n=int(saved['__row_count__'])
        if n!=result['authority_events_count'] or set(saved.files)!=set(result['authority_events_columns'])|{'__row_count__'}:raise ValueError('Authority table schema mismatch')
        if any(len(saved[k])!=n for k in result['authority_events_columns']):raise ValueError('Authority table column-length mismatch')
    return result

def worker(root,plan,job,checkpoint_records=8,stop_after_checkpoints=0,parent_pid=None):
    name=job_name(job);identity=identity_for(plan,job)
    with writer_lock(root/'locks'/(name+'.lock'),wait=True):
        existing=completed_read(root,plan,job)
        if existing is not None:
            print(json.dumps({'job':name,'status':'reused','new_physical_events':0}),flush=True);return 0
        E=MODULES['engine'];cp=root/'checkpoints'/(name+'.checkpoint')
        state=unpack_state(cp.read_bytes(),identity) if cp.exists() else E.init_state(job['seed'],job['config'])
        start=int(state['cursor']);blocks=0;start_time=time.monotonic()
        print(json.dumps({'job':name,'status':'resumed' if cp.exists() else 'started','cursor':start}),flush=True)
        while not state['done'] and not STOP and (parent_pid is None or os.getppid()==parent_pid):
            E.step(state,job['config'],max_units=checkpoint_records);save_state(cp,state,identity);blocks+=1
            progress={'job':name,'cursor':int(state['cursor']),'records':len(state['stream']['rows']),
                      'physical_byte_events':int(state.get('byte_events',state['cursor']*14)),
                      'done':bool(state['done']),'peak_rss_bytes':peak_rss(),'elapsed_seconds_this_invocation':time.monotonic()-start_time}
            write_json(root/'progress'/(name+'.json'),progress)
            if blocks%8==0 or state['done']:print(json.dumps(progress),flush=True)
            if stop_after_checkpoints and blocks>=stop_after_checkpoints:break
        if not state['done']:
            print(json.dumps({'job':name,'status':'paused','cursor':state['cursor']}),flush=True);return 75
        result=E.summarize(state,job['config']);events=result.pop('authority_events',state.get('authority_events',[]))
        ep=root/'authority_events'/(name+'.npz');meta=write_events(ep,events)
        result.update(authority_events_file=str(ep.relative_to(root)),authority_events_sha256=meta['sha256'],authority_events_count=meta['rows'],authority_events_columns=meta['columns'])
        write_json(root/'completed'/(name+'.json'),{'identity':identity,'sha256':sha(result),'data':result})
        write_json(root/'execution'/(name+'_'+str(time.time_ns())+'.json'),{'job':name,'start_record':start,'end_record':state['cursor'],'new_physical_events':14*(state['cursor']-start),'elapsed_seconds':time.monotonic()-start_time,'peak_rss_bytes':peak_rss()})
        print(json.dumps({'job':name,'status':'completed','new_physical_events':14*(state['cursor']-start),'peak_rss_bytes':peak_rss()}),flush=True)
        return 0

def basic_tests():
    E=MODULES['engine'];R=MODULES['reporting'];O=MODULES['offline_controls'];D=MODULES['diagnostics']
    checks={'engine':E.self_test(),'reporting':R.self_test(),'offline_controls':O.self_test(),'diagnostics':D.self_test()}
    obj={'array':np.arange(9),'rng':np.random.default_rng(94),'cursor':8};ident={'test':VERSION}
    raw=pack_state(obj,ident);restored=unpack_state(raw,ident)
    assert np.array_equal(restored['array'],obj['array']) and restored['rng'].bit_generator.state==obj['rng'].bit_generator.state
    for bad,ident2 in [(raw[:-1]+bytes([raw[-1]^1]),ident),(raw,{'test':'wrong'})]:
        try:unpack_state(bad,ident2)
        except (ValueError,zlib.error):pass
        else:raise AssertionError('Invalid checkpoint accepted')
    if not all(checks[k].get('passed',False) for k in ('engine','reporting','offline_controls','diagnostics')):raise AssertionError('Module self-test reported failure')
    checks['runner_checkpoint_roundtrip_and_rejections']=True
    return checks

def engineering_cell(root,seed,load,parent_pid=None):
    identity={'source_sha256':source_sha(),'protocol_sha256':sha(PROTOCOL.encode())}
    p=root/'engineering'/f'{seed}_{load}.json'
    with writer_lock(root/'locks'/f'engineering_{seed}_{load}.lock',wait=True):
        if p.exists():
            env=json.loads(p.read_text())
            if env['identity']!=identity or env['sha256']!=sha(env['data']):raise ValueError('Engineering cell cache mismatch')
            r=env['data']
        else:
            cp=root/'engineering'/f'{seed}_{load}.checkpoint';eid={'engineering':identity,'seed':seed,'load':load}
            def save_engineering(st):
                save_state(cp,st,eid);observed=st['observed']
                progress={'seed':seed,'load':load,'cursor':observed['cursor'],'records':len(observed['stream']['rows']),'peak_rss_bytes':peak_rss()}
                write_json(root/'engineering'/f'{seed}_{load}.progress.json',progress)
                if observed['cursor']%64==0:print(json.dumps(progress),flush=True)
            r=MODULES['engine'].engineering(seed,load,checkpoint_load=lambda:unpack_state(cp.read_bytes(),eid) if cp.exists() else None,checkpoint_save=save_engineering,should_stop=lambda:STOP or (parent_pid is not None and os.getppid()!=parent_pid))
            write_json(p,{'identity':identity,'sha256':sha(r),'data':r})
        if not r.get('passed') or not r.get('full_length'):raise AssertionError('Incomplete engineering cell')
        return r

def engineering(root,jobs=4):
    root.mkdir(parents=True,exist_ok=True);path=root/'ENGINEERING.json';identity={'source_sha256':source_sha(),'protocol_sha256':sha(PROTOCOL.encode())}
    expected={(s,l) for s in ENGINEERING_SEEDS for l in (64,256)}
    if path.exists():
        saved=json.loads(path.read_text())
        if saved.get('sha256')!=sha({k:v for k,v in saved.items() if k!='sha256'}):raise ValueError('Engineering cache checksum mismatch')
        if saved['identity']!=identity or not saved.get('passed'):raise ValueError('Engineering cache mismatch or failure')
        if {(r['seed'],r['load']) for r in saved['runs']}!=expected or len(saved['runs'])!=4:raise ValueError('Engineering roster mismatch')
        if not all(r.get('passed') and r.get('full_length') and r.get('byte_checks',0)>0 and r.get('terminal_probe_checks',0)>0 for r in saved['runs']):raise ValueError('Engineering evidence incomplete')
        return saved
    tests=basic_tests();pending=list(sorted(expected));active={};runs=[]
    try:
        while pending or active:
            while pending and len(active)<jobs and not STOP:
                seed,load=pending.pop(0);p=root/'engineering'/f'{seed}_{load}.json'
                if p.exists():runs.append(engineering_cell(root,seed,load));continue
                log=root/'engineering'/f'{seed}_{load}.log';log.parent.mkdir(exist_ok=True);f=log.open('ab')
                source_sha();proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--out-dir',str(root),'--engineering-cell',f'{seed}:{load}','--parent-pid',str(os.getpid())],stdout=f,stderr=subprocess.STDOUT)
                active[proc.pid]=(proc,f,{'seed':seed,'load':load})
                print(json.dumps({'engineering_seed':seed,'load':load,'status':'started'}),flush=True)
            if STOP:stop_workers(active);raise InterruptedError('Engineering stopped')
            for pid,(proc,f,job) in list(active.items()):
                code=proc.poll()
                if code is not None:
                    f.close();del active[pid]
                    if code:raise RuntimeError('Engineering worker failed: '+str(job)+' exit '+str(code))
                    runs.append(engineering_cell(root,job['seed'],job['load']))
                    print(json.dumps({'engineering_seed':job['seed'],'load':job['load'],'status':'passed'}),flush=True)
            if active:time.sleep(.5)
    finally:stop_workers(active)
    assert {(r['seed'],r['load']) for r in runs}==expected and all(r['full_length'] and r['passed'] for r in runs)
    engine_names=set(tests['engine']['checks']);offline_ok=tests['offline_controls']['passed']
    evidence={
        'E0':all(r['byte_checks']>0 and r['terminal_probe_checks']>0 for r in runs),
        'E1':all(r['firewall_checks']==r['byte_checks'] for r in runs),
        'E2':any('hash' in n.lower() or 'guard' in n.lower() for n in engine_names),
        'E3':all(r['probe_checks']>0 for r in runs),
        'E4':all('exclusive_route_'+x in engine_names for x in 'ATGF'),
        'E5':'pre_E2_promotion_capture' in engine_names and {'G_prediction_uses_prior_evidence','G_evidence_updates_only_after_outcome'} <= engine_names,
        'E6':'reconsolidation_fresh_monotone_generation' in engine_names and 'eviction_removes_ledger' in engine_names,
        'E7':'exact_revision_reset' in engine_names,
        'E8':offline_ok}
    if not all(evidence.values()):raise AssertionError('Engineering evidence map incomplete: '+str(evidence))
    result={'identity':identity,'source_sha256':source_sha(),'passed':True,'checks':evidence,'self_tests':tests,'runs':sorted(runs,key=lambda r:(r['seed'],r['load']))}
    result['sha256']=sha(result);write_json(path,result);return result

def package(root):
    def keep(p):
        if not p.is_file() or 'locks' in p.parts or '.tmp-' in p.name or p.name=='ARTIFACT_HASHES.json':return False
        if 'engineering' in p.parts and '.checkpoint' in p.name:
            return not (p.parent/(p.name.split('.checkpoint')[0]+'.json')).exists()
        if 'checkpoints' in p.parts:
            name=p.name.split('.checkpoint')[0]
            return not (root/'completed'/(name+'.json')).exists()
        return True
    paths=[p for p in root.rglob('*') if keep(p)]
    hashes={str(p.relative_to(root)):sha(p.read_bytes()) for p in sorted(paths)}
    write_json(root/'ARTIFACT_HASHES.json',hashes);paths.append(root/'ARTIFACT_HASHES.json')
    out=root.with_name(root.name+'_results.zip');tmp=out.with_name(out.name+'.tmp')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(paths):z.write(p,str(p.relative_to(root)))
    os.replace(tmp,out);return out

def combined_events(root,records):
    target=root/'AUTHORITY_EVENTS.npz';index=root/'AUTHORITY_EVENTS_INDEX.json'
    inputs=[{'file':r['authority_events_file'],'sha256':r['authority_events_sha256'],'rows':r['authority_events_count']} for r in records]
    if index.exists():
        prior=json.loads(index.read_text())
        if prior['inputs']==inputs:
            if not target.exists() or sha(target.read_bytes())!=prior['combined_sha256']:raise ValueError('Combined event archive checksum mismatch')
            return prior
    columns=sorted({k for r in records for k in r['authority_events_columns']});chunks={k:[] for k in columns}
    for r in records:
        with np.load(root/r['authority_events_file'],allow_pickle=False) as z:
            n=int(z['__row_count__'])
            for k in columns:chunks[k].append(z[k] if k in z.files else np.full(n,np.nan))
    arrays={k:np.concatenate(v) for k,v in chunks.items()};total=sum(x['rows'] for x in inputs)
    arrays['__row_count__']=np.array(total,dtype=np.int64)
    import io
    data=io.BytesIO();np.savez_compressed(data,**arrays);atomic(target,data.getvalue())
    result={'inputs':inputs,'total_rows':total,'columns':columns,'combined_sha256':sha(data.getvalue()),'rows_are':'A/T/G observed online authority events only'}
    write_json(index,result);return result

def report(root,plan,diagnostics=True):
    records=[r for job in plan['jobs'] if (r:=completed_read(root,plan,job)) is not None]
    r=MODULES['reporting'].generate(root,plan,records)
    if len(records)==len(plan['jobs']):combined_events(root,records)
    if diagnostics and len(records)==len(plan['jobs']):
        if plan['mode']=='science':MODULES['diagnostics'].run(root,None,should_stop=lambda:STOP)
        else:write_json(root/'PREDICTABILITY_STATUS.json',{'status':'engineering_smoke_not_science','not_run':True})
    return r,package(root)

def stop_workers(active):
    for proc,f,job in active.values():
        if proc.poll() is None:proc.terminate()
    deadline=time.monotonic()+30
    while any(proc.poll() is None for proc,f,job in active.values()) and time.monotonic()<deadline:time.sleep(.1)
    for proc,f,job in active.values():
        if proc.poll() is None:proc.kill()
        proc.wait();f.close()
    active.clear()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir','--out',default='v43_results');parser.add_argument('--install-deps',action='store_true')
    parser.add_argument('--self-test',action='store_true');parser.add_argument('--engineering',action='store_true');parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--report-only',action='store_true');parser.add_argument('--jobs',type=int,default=4)
    parser.add_argument('--checkpoint-records',type=int,default=8);parser.add_argument('--stop-after-checkpoints',type=int,default=0)
    parser.add_argument('--worker',default=None,help=argparse.SUPPRESS)
    parser.add_argument('--parent-pid',type=int,default=None,help=argparse.SUPPRESS)
    parser.add_argument('--engineering-cell',default=None,help=argparse.SUPPRESS)
    args=parser.parse_args();dependencies(args.install_deps);load_modules()
    signal.signal(signal.SIGINT,request_stop);signal.signal(signal.SIGTERM,request_stop)
    if args.jobs<1 or args.checkpoint_records<1:raise ValueError('Positive jobs/checkpoint-records required')
    if args.self_test:print(json.dumps(basic_tests(),indent=2,default=cast));return 0
    root=Path(args.out_dir).resolve();root.mkdir(parents=True,exist_ok=True)
    if args.engineering_cell:
        seed,load=map(int,args.engineering_cell.split(':'))
        if seed not in ENGINEERING_SEEDS or load not in (64,256):raise ValueError('Not a planned engineering cell')
        print(json.dumps(engineering_cell(root,seed,load,args.parent_pid),default=cast),flush=True);return 0
    if args.worker:
        plan=json.loads((root/'PLAN.json').read_text());s,l=map(int,args.worker.split(':'))
        expected=make_plan(plan['mode']=='engineering_smoke')
        if plan!=expected:raise ValueError('Worker plan/source mismatch')
        job=next(j for j in plan['jobs'] if j['seed']==s and j['load']==l)
        return worker(root,plan,job,args.checkpoint_records,args.stop_after_checkpoints,args.parent_pid)
    with writer_lock(root/'locks'/'runner.lock'):
        caffeinate=None;active={}
        if sys.platform=='darwin' and shutil.which('caffeinate'):caffeinate=subprocess.Popen(['caffeinate','-dims','-w',str(os.getpid())])
        try:
            if args.engineering:engineering(root,args.jobs);return 0
            plan=make_plan(args.smoke)
            if (root/'PLAN.json').exists():
                if json.loads((root/'PLAN.json').read_text())!=plan:raise ValueError('Existing plan/source mismatch')
            else:write_json(root/'PLAN.json',plan)
            atomic(root/Path(__file__).name,Path(__file__).read_bytes());atomic(root/'PROTOCOL.md',PROTOCOL.encode());atomic(root/'ORIGINAL_DESIGN.md',ORIGINAL_PROTOCOL.encode())
            write_json(root/'MANIFEST.json',{'source_sha256':source_sha(),'protocol_sha256':sha(PROTOCOL.encode()),'plan_sha256':sha(plan),'seeds':plan['seeds'],'loads':plan['loads'],'expected_trajectories':plan['expected_trajectories']})
            if not args.smoke:engineering(root,args.jobs)
            else:write_json(root/'ENGINEERING.json',{'passed':True,'source_sha256':source_sha(),'mode':'smoke_only','self_tests':basic_tests()})
            invocation={'argv':sys.argv[1:],'python':sys.version,'platform':platform.platform(),'dependencies':{n:importlib.import_module(n).__version__ for n in ('numpy','scipy','matplotlib','sklearn')},'jobs':args.jobs,'source_sha256':source_sha()}
            inv=str(time.time_ns());write_json(root/'invocations'/(inv+'.json'),invocation)
            before={job_name(j) for j in plan['jobs'] if completed_read(root,plan,j) is not None}
            if not args.report_only:
                pending=[j for j in plan['jobs'] if job_name(j) not in before];active={};failures=[]
                while pending or active:
                    while pending and len(active)<args.jobs and not STOP:
                        job=pending.pop(0);name=job_name(job);log=root/'logs'/(name+'.log');log.parent.mkdir(exist_ok=True)
                        f=log.open('ab');cmd=[sys.executable,str(Path(__file__).resolve()),'--out-dir',str(root),'--worker',f"{job['seed']}:{job['load']}",'--checkpoint-records',str(args.checkpoint_records),'--parent-pid',str(os.getpid())]
                        if args.stop_after_checkpoints:cmd+=['--stop-after-checkpoints',str(args.stop_after_checkpoints)]
                        source_sha();proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT);active[proc.pid]=(proc,f,job)
                        print(json.dumps({'job':name,'status':'launched','pid':proc.pid}),flush=True)
                    if STOP:
                        pending.clear()
                        for proc,f,job in active.values():
                            if proc.poll() is None:proc.terminate()
                    for pid,(proc,f,job) in list(active.items()):
                        code=proc.poll()
                        if code is not None:
                            f.close();del active[pid]
                            print(json.dumps({'job':job_name(job),'exit_code':code}),flush=True)
                            if code not in (0,75):failures.append({'job':job_name(job),'exit_code':code});pending.clear()
                    if active:time.sleep(.5)
                    if failures:
                        stop_workers(active);break
                if failures:write_json(root/'FAILURES.json',failures);raise RuntimeError('Worker failed; inspect job log: '+str(failures))
            after={job_name(j) for j in plan['jobs'] if completed_read(root,plan,j) is not None}
            write_json(root/'execution'/('invocation_'+inv+'.json'),{'completed_jobs':len(after),'expected_jobs':len(plan['jobs']),'reused_jobs':len(before),'new_completed_jobs':len(after-before),'peak_parent_rss_bytes':peak_rss(),'stop_requested':STOP})
            result,z=report(root,plan,diagnostics=not STOP)
            print(json.dumps({'completed_trajectories':len(after),'expected_trajectories':len(plan['jobs']),'reused_trajectories':len(before),'new_completed_trajectories':len(after-before),'results_zip':str(z)}),flush=True)
            return 0 if len(after)==len(plan['jobs']) else 75
        finally:
            stop_workers(active)
            if caffeinate is not None:caffeinate.terminate()

if __name__=='__main__':
    try:raise SystemExit(main())
    except (KeyboardInterrupt,InterruptedError):raise SystemExit(75)
