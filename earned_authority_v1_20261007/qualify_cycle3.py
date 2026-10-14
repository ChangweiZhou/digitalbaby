"""Full-life technical qualification, 8-worker parity and lossless resume."""
import concurrent.futures,copy,gzip,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'cycles/cycle3';OUT.mkdir(parents=True,exist_ok=True)
PYTHON=sys.executable
import bridge_runtime as rt
from local_integrity import locked_identity
from independent_audit import audit_job
from run_job import DEVELOPMENT_WORLDS
IDENT=locked_identity()

def command(world,assay,folder,short=None,resume=False,stop=None):
    c=[PYTHON,str(ROOT/'run_job.py'),'--world',str(world),'--assay',assay,'--folder',str(folder)]
    if short:c+=['--short',str(short)]
    if resume:c+=['--resume']
    if stop:c+=['--stop-at',str(stop)]
    return c

def launch(world,assay,folder,short=None,resume=False,stop=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    start=time.monotonic();p=subprocess.run(command(world,assay,folder,short,resume,stop),capture_output=True,text=True)
    log=folder/f'launch_{"resume" if resume else "fresh"}.log';log.write_text(p.stdout+'\n'+p.stderr)
    if p.returncode:raise RuntimeError(f'qualification process exited {p.returncode}: {log}\n{p.stderr[-1600:]}')
    result=json.loads(p.stdout.strip().splitlines()[-1]);result['process_wall_s']=time.monotonic()-start
    return result

def receipt(folder,world,assay):return rt.read_receipt(Path(folder)/'receipts'/f'{world}_ERROR_{assay}.json.gz')

def fingerprint(doc):
    omit={'cpu_s','worker_s','peak_rss_bytes','online_cpu_s','external_audit_cpu_s','policy_cpu_s','online_api_cpu_s','births'}
    def clean(x):
        if isinstance(x,dict):return {k:clean(v) for k,v in x.items() if k not in omit}
        if isinstance(x,list):return [clean(v) for v in x]
        return x
    return rt.digest(clean(doc))

def main():
    began=time.monotonic();results={};checks=[]
    # The eight parallel jobs each have a separately executed sequential E0 twin.
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures={pool.submit(launch,w,'reuse',OUT/'parallel8'/str(w),4):w for w in DEVELOPMENT_WORLDS}
        parallel={futures[f]:f.result() for f in concurrent.futures.as_completed(futures)}
    for w in DEVELOPMENT_WORLDS:
        launch(w,'reuse',OUT/'sequential'/str(w),4)
        a=receipt(OUT/'parallel8'/str(w),w,'reuse');b=receipt(OUT/'sequential'/str(w),w,'reuse')
        audit_job(a,IDENT);audit_job(b,IDENT);assert fingerprint(a)==fingerprint(b)
    checks.append('eight-worker vs sequential operative digest, writes, outputs and fixtures: 8/8 exact')
    results['parallel8']=parallel
    # Real SIGTERM received during a record: finish it, seal, exit; then resume.
    w=DEVELOPMENT_WORLDS[6];paused=OUT/'signal_resume';paused.mkdir(exist_ok=True)
    logpath=paused/'signal_launch.log'
    with logpath.open('w') as log:
        p=subprocess.Popen(command(w,'reuse',paused,8),stdout=log,stderr=log)
        active=paused/'active'/f'{w}_ERROR_reuse.json'
        deadline=time.monotonic()+60
        while not active.exists() and p.poll() is None and time.monotonic()<deadline:time.sleep(.01)
        if not active.exists():raise AssertionError('missing live worker heartbeat before SIGTERM')
        p.send_signal(signal.SIGTERM);p.wait(timeout=60)
        assert p.returncode==0
    initial=json.loads(logpath.read_text().strip().splitlines()[-1]);assert initial['status']=='PAUSED_TECHNICAL_CHECKPOINT'
    cp=Path(initial['checkpoint']);assert cp.exists()
    # Wrong identity and an altered container must be rejected, without learning.
    try:rt.checkpoint.load(cp,'ERROR','wrong-identity')
    except ValueError:pass
    else:raise AssertionError('wrong checkpoint identity accepted')
    import numpy as np
    with np.load(cp,allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
    arrays['metadata_json'][0]^=1
    corrupt=paused/'corrupt.npz';np.savez_compressed(corrupt,**arrays)
    try:rt.checkpoint.load(corrupt,'ERROR',IDENT)
    except (ValueError,json.JSONDecodeError):pass
    else:raise AssertionError('tampered checkpoint accepted')
    resumed=launch(w,'reuse',paused,8,resume=True)
    reference=OUT/'uninterrupted';launch(w,'reuse',reference,8)
    a=receipt(paused,w,'reuse');b=receipt(reference,w,'reuse');assert fingerprint(a)==fingerprint(b)
    checks.append('real SIGTERM -> sealed record boundary -> resume matches uninterrupted job exactly')
    checks.append('wrong checkpoint identity and corrupted metadata rejected')
    # Invalid requests and duplicate claim do not leave work behind.
    from run_job import run
    bad=OUT/'bad_limit'
    try:run(w,'reuse',bad,short=0)
    except ValueError:pass
    else:raise AssertionError('invalid limit accepted')
    assert not list(bad.rglob('*.lock'))
    claim=OUT/'duplicate_claim';(claim/'locks').mkdir(parents=True,exist_ok=True)
    (claim/'locks'/f'{w}_ERROR_reuse.lock').write_text(str(os.getpid()))
    try:run(w,'reuse',claim,short=2)
    except FileExistsError:pass
    else:raise AssertionError('duplicate job claim accepted')
    assert not list(claim.rglob('*.json.gz'))
    checks.append('invalid-limit cleanup and exclusive job claim passed')
    # Three complete worlds, all causal branches; no held-out efficacy selection.
    assignments=list(zip(DEVELOPMENT_WORLDS[:3],('lifetime','reuse','latin')))
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures={pool.submit(launch,w,assay,OUT/'full'/assay):(w,assay) for w,assay in assignments}
        full={}
        for f in concurrent.futures.as_completed(futures):
            w,assay=futures[f];result=f.result();doc=receipt(OUT/'full'/assay,w,assay);audit=audit_job(doc,IDENT)
            full[assay]={'world':w,'result':result,'audit':audit,'job_cpu_s':doc['cpu_s'],
                         'peak_rss_bytes':doc['peak_rss_bytes'],'receipt_bytes':Path(result['receipt']).stat().st_size,
                         'W_online_cpu_s':doc['branches']['W']['online_cpu_s'],
                         'W_policy_cpu_s':doc['branches']['W']['policy_cpu_s']}
            print(json.dumps({'full_technical_assay':assay,**full[assay]}),flush=True)
    checks.append('full lifetime/reuse/Latin fixtures, every branch/write/probe, 9 lives: PASS')
    # Receipt corruption tests on complete data. They do not rerun any records.
    tamper_rejected=0
    d=receipt(OUT/'full/lifetime',DEVELOPMENT_WORLDS[0],'lifetime')
    for mutation in ('branch','call','birth','teacher','probe_drop','probe_clock','fixture','readout','output_before_teacher'):
        x=copy.deepcopy(d)
        if mutation=='branch':x['branches']['N_old']['records'][0]['learn']=True
        if mutation=='call':x['branches']['N_old']['records'][0]['actual_calls'][0]['write']=True
        if mutation=='birth':x['branches']['W']['births'][1]['fly_id']=x['branches']['W']['births'][0]['fly_id']
        if mutation=='teacher':x['branches']['W']['records'][0]['write']['outcome']^=1
        if mutation=='probe_drop':x['branches']['W']['probes'][-1]['rows'].pop()
        if mutation=='probe_clock':x['branches']['W']['probes'][-1]['at']+=1
        if mutation=='fixture':x['fixture_sha256']='not-the-input'
        if mutation=='readout':x['branches']['W']['probes'][-1]['rows'][0]['readouts']['Q_EARNED']['alpha']=7
        if mutation=='output_before_teacher':x['branches']['W']['records'][0]['operation_order'][2:4]=['outcome','policy_emit']
        try:audit_job(x,IDENT)
        except (AssertionError,ValueError):tamper_rejected+=1
        else:raise AssertionError('tamper accepted: '+mutation)
    checks.append(f'complete-receipt independent tamper suite: {tamper_rejected}/9 rejected')
    # Idempotent reporting checks a committed job and must not re-execute it.
    rp=Path(full['reuse']['result']['receipt']);stamp=rp.stat().st_mtime_ns
    same=launch(DEVELOPMENT_WORLDS[1],'reuse',OUT/'full/reuse')
    assert same['status']=='SKIPPED_COMMITTED' and rp.stat().st_mtime_ns==stamp
    checks.append('committed-job redispatch reads receipt, no rerun or file mutation')
    wall=sum(v['result']['process_wall_s'] for v in full.values())*64/8*4
    disk=sum(v['receipt_bytes'] for v in full.values())*64
    rss=max(v['peak_rss_bytes'] for v in full.values())*8
    ratio=max((v['W_online_cpu_s']+v['W_policy_cpu_s']['Q_EARNED'])/(v['W_online_cpu_s']+v['W_policy_cpu_s']['ERROR']) for v in full.values())
    resource_pass=wall<8*3600 and disk<4*1024**3 and rss<8*1024**3
    result={'cycle':3,'verdict':'PASS' if resource_pass else 'RESOURCE_FAIL','source_identity':IDENT,
            'checks':checks,'checks_passed':len(checks),'full_assays':full,'full_technical_jobs':3,'full_technical_lives':9,
            'full_technical_teacher_records':5040,'short_completed_jobs':18,'short_completed_lives':18,
            'short_teacher_records':80,'SIGTERM_pause_record':initial['next_record'],'formal_science_jobs':0,
            'forecast':{'formal_jobs':192,'formal_lives':576,'worlds':64,'eight_workers_wall_s_with_4x_allowance':wall,
                        'receipts_bytes':disk,'aggregate_8worker_peak_rss_bytes':rss,'max_online_cpu_ratio_vs_ERROR':ratio,
                        'not_a_hard_runtime_guarantee':True},'resource_gate_pass':resource_pass,'wall_s':time.monotonic()-began}
    rt.atomic_json(OUT/'TEST_RESULT.json',result);print(json.dumps(result),flush=True)
    if not resource_pass:raise SystemExit(1)

if __name__=='__main__':
    try:main()
    except Exception as e:
        rt.atomic_json(OUT/'FAILURE.json',{'error':repr(e),'formal_science_jobs':0});raise
