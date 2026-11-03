#!/usr/bin/env python3
"""V42: fixed-history read-authority audit. One file, no new training.

Run/resume: python3 elm_v42_read_authority.py --install-deps
The embedded V41 reference data are re-evaluated; the V41 learner is not rerun.
"""
from pathlib import Path
import argparse, base64, gc, hashlib, json, os, platform, resource, signal
import subprocess, sys, tempfile, time, types, zipfile, zlib

VERSION='V42-fixed-history-authority'
PROTOCOL=__PROTOCOL__
PAYLOADS=__PAYLOADS__
REFERENCE=__REFERENCE__
STOP=False
MODULES={}

def stop_handler(sig,frame):
    global STOP
    STOP=True

def canonical(x):
    def cast(v):
        if hasattr(v,'item'):return v.item()
        if isinstance(v,Path):return str(v)
        raise TypeError(type(v).__name__)
    return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False,default=cast).encode()

def digest(x):return hashlib.sha256(x if isinstance(x,bytes) else canonical(x)).hexdigest()

def atomic(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.tmp-{os.getpid()}')
    with tmp.open('wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def write_json(path,x):atomic(path,json.dumps(json.loads(canonical(x)),indent=2).encode()+b'\n')

def memory_peak_bytes():
    n=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(n if sys.platform=='darwin' else n*1024)

def dependencies(install=False):
    try:
        import numpy
        import scipy.stats
        import matplotlib
    except ImportError:
        if not install:raise SystemExit('Run with --install-deps to install NumPy, SciPy and Matplotlib.')
        subprocess.run([sys.executable,'-m','pip','install','numpy','scipy','matplotlib'],check=True)
    global np,student_t
    import numpy as np
    from scipy.stats import t as student_t

def load_modules():
    if MODULES:return MODULES
    for name,p in PAYLOADS.items():
        raw=zlib.decompress(base64.b85decode(p['data']))
        if digest(raw)!=p['sha256']:raise ValueError('Embedded code hash mismatch')
        m=types.ModuleType('v42_'+name);m.__file__=__file__
        m.__dict__.update(np=np,student_t=student_t)
        sys.modules[m.__name__]=m;exec(compile(raw,'<v42:'+name+'>','exec'),m.__dict__)
        MODULES[name]=m
    return MODULES

def reference_data():
    raw=zlib.decompress(base64.b85decode(REFERENCE['data']))
    if digest(raw)!=REFERENCE['sha256']:raise ValueError('Embedded reference hash mismatch')
    return json.loads(raw)

def envelope(identity,data):
    data=json.loads(canonical(data))
    return {'identity':identity,'sha256':digest(data),'data':data}

def unpack_envelope(value,identity):
    if value.get('identity')!=identity:raise ValueError('Saved record belongs to a different source, input or plan')
    if value.get('sha256')!=digest(value.get('data')):raise ValueError('Saved record checksum mismatch')
    return value['data']

def save_checkpoint(path,identity,state):
    encoded=zlib.compress(canonical(envelope(identity,state)),3)
    if path.exists():atomic(path.with_suffix('.previous'),path.read_bytes())
    atomic(path,encoded)

def load_checkpoint(path,identity):
    if not path.exists():return None
    try:value=json.loads(zlib.decompress(path.read_bytes()))
    except (ValueError,zlib.error) as exc:raise ValueError('Corrupt checkpoint; retain it and explicitly restore a verified previous generation') from exc
    return unpack_envelope(value,identity)

def validate_reference(ref):
    # The synthetic engineering fixture has its own evaluator assertions.
    if ref.get('seed')==942001:return {'synthetic_engineering':True}
    old=ref['original_completed_envelope']
    if old.get('sha256')!=digest(old.get('data')):raise ValueError('Original V41 completed-record digest mismatch')
    raw=base64.b64decode(ref['raw_stream_base64'],validate=True)
    if len(raw)!=15344:raise ValueError('Unexpected historical stream length')
    if digest(raw)!=ref['provenance']['stream_sha256']:raise ValueError('Historical stream digest mismatch')
    audit=ref['audits'];historical=old['data']['result']
    if not (audit['learner_hash_before']==audit['learner_hash_after']==historical['learner_full_state_hash']):
        raise ValueError('Terminal learner correspondence or read invariance failed')
    if audit['full_state_hash_before']!=audit['full_state_hash_after']:
        raise ValueError('Reference preparation changed terminal state')
    if audit['observe_calls']!=0 or audit['step_calls']!=0:
        raise ValueError('Reference preparation contains unexpected learning')
    alias={bytes.fromhex(c['context_hex']) for c in ref['alias2_contexts']}
    if len(alias)!=128:raise ValueError('Unexpected alias-context roster')
    history=bytearray();hits=0
    for y in raw:
        hits+=int(bytes(history) in alias)
        history.append(y)
        if len(history)>12:del history[0]
    if hits:raise ValueError('Held-out contexts appeared in training; historical stratum inference invalid')
    return {'original_record_digest':True,'alias2_absence_verified':True,'raw_bytes_scanned':len(raw),'alias2_contexts':len(alias),
            'terminal_state_correspondence_and_invariance':True,'preparation_model_reads':audit['model_read_calls'],'new_training_updates':0}

def identity(plan,seed):return {'plan':digest(plan),'seed':int(seed)}

def completed_records(root,plan):
    records=[]
    for seed in plan['seeds']:
        p=root/'completed'/f'seed_{seed}.json'
        if p.exists():records.append(unpack_envelope(json.loads(p.read_text()),identity(plan,seed)))
    return records

def package(root,plan):
    out=root.parent/(root.name+'_results.zip')
    done=len(completed_records(root,plan))==len(plan['seeds'])
    files=[p for p in root.rglob('*') if p.is_file() and not p.name.startswith('.') and '.tmp-' not in p.name
           and p.name!='ARTIFACT_HASHES.json' and (not done or 'checkpoints' not in p.parts)]
    hashes={str(p.relative_to(root)):digest(p.read_bytes()) for p in files}
    write_json(root/'ARTIFACT_HASHES.json',hashes);files.append(root/'ARTIFACT_HASHES.json')
    tmp=out.with_suffix('.zip.tmp')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=7) as z:
        for p in sorted(files):z.write(p,str(p.relative_to(root)))
    os.replace(tmp,out);return out

def self_test():
    e=MODULES['evaluator'];checks={'evaluator':e.self_test(),'reporting':MODULES['reporting'].self_test()}
    ref=e.smoke_reference();a=e.init_state(ref)
    e.step(a,ref,max_units=7)
    ident={'test':'resume'}
    with tempfile.TemporaryDirectory() as d:
        cp=Path(d)/'state.checkpoint';save_checkpoint(cp,ident,a);b=load_checkpoint(cp,ident)
        while not e.step(a,ref,max_units=11):pass
        while not e.step(b,ref,max_units=3):pass
        assert canonical(e.summarize(a,ref))==canonical(e.summarize(b,ref))
        try:load_checkpoint(cp,{'test':'changed'})
        except ValueError:pass
        else:raise AssertionError('Mismatched plan accepted')
        cp.write_bytes(cp.read_bytes()[:-3]+b'bad')
        try:load_checkpoint(cp,ident)
        except ValueError:pass
        else:raise AssertionError('Corrupted checkpoint accepted')
    checks.update(checkpoint_resume_equal=True,changed_identity_rejected=True,corruption_rejected=True)
    return checks

def main():
    signal.signal(signal.SIGINT,stop_handler);signal.signal(signal.SIGTERM,stop_handler)
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,default=Path('v42_results'))
    ap.add_argument('--install-deps',action='store_true');ap.add_argument('--self-test',action='store_true')
    ap.add_argument('--smoke',action='store_true');ap.add_argument('--report-only',action='store_true')
    ap.add_argument('--checkpoint-records',type=int,default=16)
    ap.add_argument('--stop-after-checkpoints',type=int,default=0)
    args=ap.parse_args()
    if args.checkpoint_records<1 or args.stop_after_checkpoints<0:ap.error('Checkpoint records must be positive; stop count nonnegative')
    dependencies(args.install_deps);mods=load_modules();e=mods['evaluator'];reporter=mods['reporting']
    if args.self_test:print(json.dumps(self_test(),indent=2));return
    inputs=reference_data();refs=[e.smoke_reference()] if args.smoke else inputs['records']
    seeds=[int(r['seed']) for r in refs]
    source=Path(__file__).resolve()
    plan={'version':VERSION,'source_sha256':digest(source.read_bytes()),'protocol_sha256':digest(PROTOCOL.encode()),
          'reference_sha256':REFERENCE['sha256'],'mode':'smoke' if args.smoke else 'fixed_history_audit',
          'seeds':seeds,'expected_jobs':len(seeds),'new_training_updates':0,
          'policies':['READ-HYBRID','SLOT-ONLY','EXACT-ONLY','ANCHOR-ONLY','SUPER']}
    root=args.out.resolve();root.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(root/'.execution.lock').open('a+')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('Another writer is using this output directory')
    caffeinate=None;new_units=0;new_jobs=0;reused=0;blocks=0
    try:
        if (root/'PLAN.json').exists() and json.loads((root/'PLAN.json').read_text())!=plan:
            raise ValueError('Source, input or plan differs from saved run; use the original file or a new output directory')
        write_json(root/'PLAN.json',plan);atomic(root/'DESIGN.md',PROTOCOL.encode());atomic(root/source.name,source.read_bytes())
        write_json(root/'REFERENCE_INPUTS.json',{'schema_version':inputs['schema_version'],'records':refs,'provenance':inputs.get('provenance',{})})
        env={'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'sequential':True}
        if not (root/'ENVIRONMENT.json').exists():write_json(root/'ENVIRONMENT.json',env)
        stamp=str(time.time_ns())
        write_json(root/'invocations'/(stamp+'.json'),{'argv':sys.argv,'environment':env,'peak_rss_bytes':memory_peak_bytes()})
        if sys.platform=='darwin':
            try:caffeinate=subprocess.Popen(['caffeinate','-i','-w',str(os.getpid())])
            except FileNotFoundError:pass
        if not args.report_only:
            for ref in refs:
                seed=int(ref['seed']);ident=identity(plan,seed);rp=root/'completed'/f'seed_{seed}.json'
                if rp.exists():
                    unpack_envelope(json.loads(rp.read_text()),ident);reused+=1
                    print(json.dumps({'seed':seed,'status':'reused'}),flush=True);continue
                if STOP:break
                verify=validate_reference(ref);write_json(root/'input_audits'/f'seed_{seed}.json',verify)
                cp=root/'checkpoints'/f'seed_{seed}.checkpoint'
                state=load_checkpoint(cp,ident);resumed=state is not None
                if state is None:state=e.init_state(ref);save_checkpoint(cp,ident,state)
                print(json.dumps({'seed':seed,'status':'resumed' if resumed else 'started','cursor':state.get('cursor',0)}),flush=True)
                while not STOP:
                    before=int(state['processed_units']);done=bool(e.step(state,ref,max_units=args.checkpoint_records))
                    new_units+=int(state['processed_units'])-before;blocks+=1
                    save_checkpoint(cp,ident,state)
                    write_json(root/'progress'/f'seed_{seed}.json',{'seed':seed,'status':'completed' if done else 'running',
                        'cursor':state['cursor'],'processed_units':state['processed_units'],'total_units':state['total_units'],
                        'resumed':resumed,'peak_rss_bytes':memory_peak_bytes()})
                    if done:
                        result=e.summarize(state,ref);write_json(rp,envelope(ident,result));new_jobs+=1
                        print(json.dumps({'seed':seed,'status':'completed','evaluation_units':state['processed_units'],'new_training_updates':0}),flush=True)
                        break
                    if args.stop_after_checkpoints and blocks>=args.stop_after_checkpoints:break
                if not state['done']:
                    write_json(root/'progress'/f'seed_{seed}.json',{'seed':seed,'status':'paused','cursor':state['cursor'],
                        'processed_units':state['processed_units'],'total_units':state['total_units'],'resumed':resumed,'peak_rss_bytes':memory_peak_bytes()})
                finished=state['done'];del state;gc.collect()
                if not finished or STOP:break
        records=completed_records(root,plan)
        reporter.generate(root,plan,records)
        run={'completed_jobs':len(records),'expected_jobs':len(seeds),'reused_jobs':reused,'new_completed_jobs':new_jobs,
             'new_evaluation_units':new_units,'new_training_updates':0,'stop_requested':STOP,'peak_rss_bytes':memory_peak_bytes()}
        write_json(root/'execution'/(stamp+'.json'),run)
        archive=package(root,plan);print(json.dumps(dict(run,results_zip=str(archive))),flush=True)
    finally:
        if caffeinate is not None:caffeinate.terminate()
        lock.close()

if __name__=='__main__':main()
