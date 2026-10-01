"""Prospective lock and single-worker fail-closed launcher with checkpoint pauses."""
from __future__ import annotations
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
from assay import ROOT,ARMS,source_hashes,runtime_manifest
from audit_receipts import load,validate
FINAL_WORLDS=list(range(300001,300033))
EXPECTED_PARAMS=dict(timing_lr=.05,timing_tau=10.,homeo_beta=1/16,
    eligibility_tau=10.,recent_tau=10.,j_lr=.25,j_gain=.4237781016501581,
    j_bound=16.,j_tau=86400.,revision=3)
CAPS=dict(workers=1,worker_hours=8.,per_life_seconds=300.,peak_rss_bytes=800_000_000,
          results_bytes=200_000_000,active_session_hours=12.)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def atomic(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n');os.replace(temp,path)


def make_lock():
    destination=ROOT/'SOURCE_LOCK.json'
    if destination.exists():raise FileExistsError('existing source lock is immutable')
    final=ROOT/'results/final'
    if final.exists() and list(final.glob('*/*.json.gz')):raise AssertionError('final outcomes already exist')
    calibration=ROOT/'results/cycle1/FE0/290000.json.gz'
    pilot=ROOT/'results/cycle3/FINAL_METRICS.json'
    if not pilot.exists():raise AssertionError('third pilot and analysis must finish before lock')
    report=json.loads(pilot.read_text())
    if not report['audit']['pass_all']:raise AssertionError('third pilot failed audit')
    tracked={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'SPEC_LOCK.md',ROOT/'design/CYCLE3_AUDIT.md',
              ROOT/'results/CYCLE3_TESTS.txt',ROOT/'results/cycle3/FINAL_METRICS.json']}
    d=dict(schema='RESPONSE-SOURCE-LOCK-v1',created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
           source_hashes=source_hashes(),documents=tracked,worlds=FINAL_WORLDS,arms=list(ARMS),bouts=6,
           params=EXPECTED_PARAMS,resource_caps=CAPS,runtime=runtime_manifest(),
           expected_receipt_runtime=load(ROOT/'results/cycle3/FE0/290002.json.gz')['runtime'],
           replay_jobs=[[300001,a] for a in ARMS],
           calibration=dict(rule='RMS of per-row channel-centered cycle1 FE0 W pre-teacher raw values',
                            source_receipt_sha256=sha(calibration),gain=EXPECTED_PARAMS['j_gain']),
           primary=[['T','T_OFF'],['H','FE0'],['J','J_ADD'],['J','J_SHUFFLE']])
    data=json.dumps(d,indent=2,sort_keys=True,allow_nan=False)+'\n'
    with open(destination,'x') as f:f.write(data)
    return dict(path=str(destination),sha256=sha(destination),worlds=len(FINAL_WORLDS),jobs=len(FINAL_WORLDS)*len(ARMS))


def verify_lock():
    p=ROOT/'SOURCE_LOCK.json';d=json.loads(p.read_text())
    if d['source_hashes']!=source_hashes():raise AssertionError('locked executable source drift')
    for name,h in d['documents'].items():
        if sha(ROOT/name)!=h:raise AssertionError('locked document drift: '+name)
    assert d['worlds']==FINAL_WORLDS and d['arms']==list(ARMS) and d['bouts']==6
    assert d['params']==EXPECTED_PARAMS and d['resource_caps']==CAPS
    current=runtime_manifest()
    for k in ('python','numpy','scipy','numba','pandas','thread_environment'):
        if current[k]!=d['runtime'][k]:raise AssertionError('locked runtime drift: '+k)
    expected=d['expected_receipt_runtime']['imported_repo_modules']
    for name,h in expected.items():
        if sha(ROOT.parent/name)!=h:raise AssertionError('transitive runtime source drift: '+name)
    return d,sha(p)


def authorize_final(arm,world,bouts,params):
    d,h=verify_lock()
    if arm not in d['arms'] or world not in d['worlds'] or bouts!=d['bouts'] or params!=d['params']:
        raise AssertionError('job not in prospective lock')
    return h


def rss(pid):
    try:
        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
            if line.startswith('VmRSS:'):return int(line.split()[1])*1024
    except FileNotFoundError:pass
    return 0


def stop_process(process):
    process.terminate()
    try:process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        process.kill();process.wait(timeout=20)


def drive(*,resume=False,publication=True):
    lock,locksha=verify_lock();dest=ROOT/'results/final';dest.mkdir(parents=True,exist_ok=True)
    ledger=dest/'RUN_LEDGER.json';status=dest/'RUN_STATUS.json'
    if ledger.exists():
        if not resume:raise AssertionError('existing ledger requires --resume')
        history=json.loads(ledger.read_text());assert history['lock_sha256']==locksha
    else:history=dict(lock_sha256=locksha,jobs=[],worker_seconds=0.,started_unix=time.time(),active_job=None)
    if history.get('active_job'):
        interrupted=history['active_job'];charged=max(0.,time.time()-interrupted['started_unix'])
        history['worker_seconds']+=charged;history['active_job']=None
        history['jobs'].append({**interrupted,'seconds':charged,'reason':'unobserved interruption conservatively charged','exit_code':None})
    atomic(ledger,history)
    done=[];replayed=[]
    def wall_cap():
        if time.time()-history['started_unix']>CAPS['active_session_hours']*3600:
            atomic(status,dict(state='budget_stop',reason='elapsed wall cap including publication and resumes'))
            raise RuntimeError('elapsed wall cap including publication and resumes')
    def checked(target,world,arm):
        d=load(target)
        validate(d,expected_world=world,expected_arm=arm,expected_bouts=6,
                 expected_sources=lock['source_hashes'],expected_params=lock['params'],
                 expected_runtime=lock['expected_receipt_runtime'],lock_sha256=locksha)
        return d
    def job(world,arm,replay=False):
        verify_lock();wall_cap()
        target=(dest/'replays'/arm/f'{world}.json.gz') if replay else (dest/arm/f'{world}.json.gz')
        if target.exists():checked(target,world,arm);return target
        if history['worker_seconds']>=CAPS['worker_hours']*3600:raise RuntimeError('worker-hours hard cap')
        outdir=dest/'operations';outdir.mkdir(exist_ok=True)
        tag=f'{world}-{arm}'+('-replay' if replay else '')
        stdout=outdir/f'{tag}.stdout.txt';stderr=outdir/f'{tag}.stderr.txt'
        command=[sys.executable,str(ROOT/'src/assay.py'),'--arm',arm,'--world',str(world),'--bouts','6',
                 '--kind','final','--params',json.dumps(EXPECTED_PARAMS,separators=(',',':'))]
        if replay:command+=['--destination',str(target)]
        began=time.monotonic();reason=None
        history['active_job']=dict(world=world,arm=arm,replay=replay,started_unix=time.time())
        atomic(ledger,history)
        with open(stdout,'w') as out,open(stderr,'w') as err:
            p=subprocess.Popen(command,stdout=out,stderr=err)
            while p.poll() is None:
                elapsed=time.monotonic()-began
                size=sum(f.stat().st_size for f in dest.rglob('*') if f.is_file())
                if elapsed>CAPS['per_life_seconds']:reason='per-life time cap'
                elif rss(p.pid)>CAPS['peak_rss_bytes']:reason='worker RSS cap'
                elif history['worker_seconds']+elapsed>CAPS['worker_hours']*3600:reason='worker-hours cap'
                elif time.time()-history['started_unix']>CAPS['active_session_hours']*3600:reason='elapsed wall cap'
                elif size>CAPS['results_bytes']:reason='results disk cap'
                atomic(status,dict(state='running',world=world,arm=arm,replay=replay,completed=len(done),replayed=len(replayed),expected=224,
                                   elapsed_current_seconds=elapsed,worker_seconds=history['worker_seconds']+elapsed))
                if reason:stop_process(p);break
                time.sleep(2)
        elapsed=time.monotonic()-began
        entry=dict(world=world,arm=arm,replay=replay,exit_code=p.returncode,seconds=elapsed,reason=reason)
        history['jobs'].append(entry);history['worker_seconds']+=elapsed;history['active_job']=None;atomic(ledger,history)
        if reason or p.returncode!=0:
            atomic(status,dict(state='technical_stop',job=entry,completed=len(done),expected=224))
            raise RuntimeError('job stopped: '+json.dumps(entry))
        checked(target,world,arm)
        print(json.dumps(dict(completed=len(done)+int(not replay),world=world,arm=arm,replay=replay,seconds=elapsed)),flush=True)
        return target
    for world in FINAL_WORLDS:
        for arm in ARMS:
            job(world,arm);done.append([world,arm])
        if world==300001:
            for arm in ARMS:
                path=job(world,arm,replay=True)
                a=load(dest/arm/f'{world}.json.gz');b=load(path)
                if a['resource']['process_id']==b['resource']['process_id']:raise AssertionError('replay is not a distinct process')
                a.pop('resource');b.pop('resource')
                if a!=b:raise AssertionError('fresh-process replay differs beyond resource fields')
                replayed.append([world,arm])
            atomic(dest/'REPLAY_AUDIT.json',dict(pass_all=True,jobs=replayed,excluded_fields=['resource']))
        if publication:
            spool=ROOT/'scratch/publication';spool.mkdir(parents=True,exist_ok=True)
            request=spool/f'{world}.request.json';ack=spool/f'{world}.ack.json'
            atomic(request,dict(world=world,completed=len(done),branch='response-mechanisms-20261001',lock_sha256=locksha))
            print(json.dumps(dict(publication_request=str(request),completed=len(done))),flush=True)
            while not ack.exists():wall_cap();time.sleep(2)
            acknowledged=json.loads(ack.read_text())
            commit=acknowledged.get('commit_sha','')
            if (acknowledged.get('ok') is not True or acknowledged.get('world')!=world or
                acknowledged.get('lock_sha256')!=locksha or acknowledged.get('remote_verified') is not True or
                len(commit)!=40 or any(c not in '0123456789abcdef' for c in commit)):
                raise RuntimeError('publication ACK not bound to world, lock and verified commit')
            atomic(dest/'publication'/f'{world}.json',acknowledged)
    atomic(status,dict(state='complete_pending_final_audit',completed=len(done),replayed=len(replayed),expected=224,lock_sha256=locksha))
    return dict(completed=len(done),worker_hours=history['worker_seconds']/3600,replayed=replayed)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('lock','verify','run'))
    p.add_argument('--resume',action='store_true');p.add_argument('--no-publication-pauses',action='store_true')
    a=p.parse_args()
    result=make_lock() if a.action=='lock' else {'lock_sha256':verify_lock()[1]} if a.action=='verify' else drive(resume=a.resume,publication=not a.no_publication_pauses)
    print(json.dumps(result),flush=True)
