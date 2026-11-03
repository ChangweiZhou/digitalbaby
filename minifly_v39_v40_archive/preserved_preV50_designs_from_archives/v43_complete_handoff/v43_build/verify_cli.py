#!/usr/bin/env python3
"""Actual V43 CLI lifecycle checks on engineering smoke only."""
from pathlib import Path
import argparse, fcntl, hashlib, json, os, shutil, signal, subprocess, sys, time, zipfile

HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE = HERE.parent / 'elm_v43_fast_slow_authority.py'
CHECKS = []
PROCESSES = []

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path, value): Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
def check(name, **details):
    row = {'name': name, 'passed': True, **details}; CHECKS.append(row)
    print(json.dumps(row), flush=True)

def launch(source, out, name, extra=()):
    log = WORK / (name + '.log'); handle = log.open('wb')
    cmd = [sys.executable, str(source), '--smoke', '--jobs', '1', '--checkpoint-records', '1', '--out-dir', str(out), *extra]
    proc = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
    PROCESSES.append((proc, handle))
    return proc, handle, log

def finish(job, expected=0, timeout=90):
    proc, handle, log = job
    code = proc.wait(timeout=timeout); handle.close()
    if code != expected: raise AssertionError(f'{log.name}: exit {code}, expected {expected}:\n{log.read_text()[-5000:]}')
    return log.read_text()

def run(source, out, name, extra=(), expected=0):
    return finish(launch(source, out, name, extra), expected)

def wait_progress(job, out, min_cursor=2, timeout=50):
    proc, _, log = job; start = time.monotonic()
    path = out / 'progress' / 'seed_943901_load_16.json'
    while time.monotonic() - start < timeout:
        if proc.poll() is not None: raise AssertionError(f'Exited before interruption point: {log.read_text()}')
        if path.exists():
            try:
                row = json.loads(path.read_text())
                if row['cursor'] >= min_cursor and not row['done']: return row
            except json.JSONDecodeError: pass
        time.sleep(.02)
    raise TimeoutError('No durable smoke progress')

def worker_pids(log):
    rows=[]
    for line in log.read_text().splitlines():
        try: row=json.loads(line)
        except json.JSONDecodeError: continue
        if row.get('status')=='launched': rows.append(row['pid'])
    return rows

def live(pid):
    try: os.kill(pid,0)
    except ProcessLookupError: return False
    path=Path('/proc')/str(pid)/'stat'
    if path.exists():
        try: return path.read_text().rsplit(')',1)[1].strip().split()[0] != 'Z'
        except FileNotFoundError: return False
    return True

def wait_dead(pids, timeout=40):
    start=time.monotonic()
    while any(live(pid) for pid in pids):
        if time.monotonic()-start>timeout: raise AssertionError(f'Live orphan workers: {[p for p in pids if live(p)]}')
        time.sleep(.05)

def worker_lock_released(out):
    path=out/'locks'/'seed_943901_load_16.lock'
    with path.open('a+') as handle:
        fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        fcntl.flock(handle.fileno(),fcntl.LOCK_UN)
    return True

def clone_inputs(src, dst, completed=True):
    dst.mkdir(parents=True,exist_ok=True)
    for name in ('PLAN.json','MANIFEST.json'):
        if (src/name).exists():shutil.copy2(src/name,dst/name)
    for name in (('completed','authority_events') if completed else ('checkpoints','progress')):
        if (src/name).exists():shutil.copytree(src/name,dst/name)
    return dst

def outputs(out):
    p=out/'completed'/'seed_943901_load_16.json'
    env=json.loads(p.read_text()); event=out/env['data']['authority_events_file']
    return p,event,env

def same_science(out, baseline):
    p,e,r=outputs(out);bp,be,br=outputs(baseline)
    assert p.read_bytes()==bp.read_bytes(), 'Scientific completed result differs'
    assert e.read_bytes()==be.read_bytes(), 'Authority NPZ differs'
    return {'completed_sha256':sha(p),'authority_events_sha256':sha(e),'physical_events':r['data']['byte_events']}

def reject(source,out,name,needle,extra=(),preserve=None):
    before=sha(preserve) if preserve else None
    job=launch(source,out,name,extra)
    proc,handle,log=job;code=proc.wait(timeout=60);handle.close()
    assert code not in (0,75), f'Invalid state accepted in {name}'
    text=log.read_text(); assert needle.lower() in text.lower(), text[-3000:]
    if preserve:assert sha(preserve)==before, f'Corrupt/changed artifact overwritten: {preserve}'
    assert not any('status' in line and 'launched' in line for line in text.splitlines()) or name=='corrupt_checkpoint', 'Learning launched before rejecting invalid committed inputs'
    return {'exit_code':code,'diagnostic':needle,'preserved_input':bool(preserve)}

def archive_check(out, require_checkpoint):
    zpath=out.with_name(out.name+'_results.zip')
    with zipfile.ZipFile(zpath) as z:
        assert z.testzip() is None
        names=z.namelist();assert len(names)==len(set(names))
        hashes=json.loads(z.read('ARTIFACT_HASHES.json'))
        assert set(names)==set(hashes)|{'ARTIFACT_HASHES.json'}
        assert all(hashlib.sha256(z.read(n)).hexdigest()==h for n,h in hashes.items())
        cps=[n for n in names if n.startswith('checkpoints/') and '.checkpoint' in n]
        assert bool(cps)==require_checkpoint,(cps,require_checkpoint)
    return {'zip_sha256':sha(zpath),'members':len(names),'resumable_checkpoints':len(cps),'CRC_passed':True,'exact_manifest_coverage':True}

def main():
    global WORK
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,default=DEFAULT_SOURCE);ap.add_argument('--baseline',type=Path,default=HERE/'cli_smoke');args=ap.parse_args()
    source=args.source.resolve();baseline=args.baseline.resolve();sourcehash=sha(source)
    WORK=HERE/('cli_verification_'+sourcehash[:12]);attempt=1
    while WORK.exists():
        attempt+=1;WORK=HERE/('cli_verification_'+sourcehash[:12]+'_attempt'+str(attempt))
    WORK.mkdir()
    if not (baseline/'PLAN.json').exists() or json.loads((baseline/'PLAN.json').read_text())['source_sha256']!=sourcehash:
        baseline=WORK/'uninterrupted';run(source,baseline,'uninterrupted')
    assert outputs(baseline)[2]['identity']['job']['seed']==943901
    assert json.loads((baseline/'PLAN.json').read_text())['source_sha256']==sourcehash
    check('uninterrupted_baseline_verified',source_sha256=sourcehash,baseline=str(baseline),**same_science(baseline,baseline))

    # Real SIGINT after a committed block, then resume the exact script.
    interrupted=WORK/'sigint';job=launch(source,interrupted,'sigint')
    at=wait_progress(job,interrupted);pids=worker_pids(job[2]);assert pids
    job[0].send_signal(signal.SIGINT);finish(job,75);wait_dead(pids)
    cp=interrupted/'checkpoints'/'seed_943901_load_16.checkpoint'
    partial=clone_inputs(interrupted,WORK/'paused_snapshot',completed=False)
    partial_zip=archive_check(interrupted,True)
    run(source,interrupted,'sigint_resume')
    check('actual_SIGINT_and_resume_equal_uninterrupted',signal_cursor=at['cursor'],orphan_workers_after_stop=0,partial_archive=partial_zip,**same_science(interrupted,baseline))

    # Reuse partial checkpoint to avoid repeating the same engineering prefix.
    killed=clone_inputs(partial,WORK/'sigkill',completed=False)
    start=json.loads((killed/'progress'/'seed_943901_load_16.json').read_text())['cursor']
    job=launch(source,killed,'sigkill');at=wait_progress(job,killed,start+2);pids=worker_pids(job[2]);assert pids
    job[0].kill();finish(job,-signal.SIGKILL);wait_dead(pids)
    for pid in pids:assert not live(pid)
    child_log=(killed/'logs'/'seed_943901_load_16.log').read_text();assert worker_lock_released(killed)
    graceful_orphan_message='"status": "paused"' in child_log
    run(source,killed,'sigkill_resume')
    check('actual_parent_SIGKILL_orphans_stop_and_resume_equal',signal_cursor=at['cursor'],worker_pids=pids,orphan_workers_after_stop=0,worker_lock_released=True,graceful_orphan_message=graceful_orphan_message,exit_mechanism='worker stopped; platform cascade and orphan callback not distinguished',**same_science(killed,baseline))

    before_logs={p.name:sha(p) for p in (killed/'logs').glob('*')};before=outputs(killed)[0].read_bytes()
    text=run(source,killed,'completed_noop')
    assert '"new_completed_trajectories": 0' in text and '"reused_trajectories": 1' in text
    assert outputs(killed)[0].read_bytes()==before
    assert {p.name:sha(p) for p in (killed/'logs').glob('*')}==before_logs
    check('completed_rerun_zero_learning',new_worker_launches=0,new_physical_events=0)

    changed=clone_inputs(baseline,WORK/'changed_plan');p=changed/'PLAN.json';r=json.loads(p.read_text());r['unregistered_change']=True;write(p,r)
    check('changed_plan_rejected',**reject(source,changed,'changed_plan','Existing plan/source mismatch',preserve=p))
    bad=clone_inputs(baseline,WORK/'corrupt_completed');p=bad/'completed'/'seed_943901_load_16.json';r=json.loads(p.read_text());r['data']['seed']+=1;write(p,r)
    check('corrupted_completed_record_rejected',**reject(source,bad,'corrupt_completed','Completed checksum mismatch',preserve=p))
    bad=clone_inputs(baseline,WORK/'corrupt_events');_,p,_=outputs(bad);data=p.read_bytes();p.write_bytes(data[:-1]+bytes([data[-1]^1]))
    check('corrupted_authority_NPZ_rejected',**reject(source,bad,'corrupt_events','Authority-event checksum mismatch',preserve=p))
    bad=clone_inputs(partial,WORK/'corrupt_checkpoint',completed=False);p=bad/'checkpoints'/'seed_943901_load_16.checkpoint';data=p.read_bytes();p.write_bytes(data[:-1]+bytes([data[-1]^1]));before=sha(p)
    job=launch(source,bad,'corrupt_checkpoint');code=job[0].wait(timeout=60);job[1].close();assert code not in(0,75) and sha(p)==before
    assert 'Checkpoint checksum mismatch' in (bad/'logs'/'seed_943901_load_16.log').read_text()
    assert not (bad/'completed').exists();wait_dead(worker_pids(job[2]))
    check('corrupted_checkpoint_rejected_and_worker_reaped',exit_code=code,preserved_checkpoint=True,completed_records=0)

    bad=WORK/'corrupt_engineering';bad.mkdir();p=bad/'ENGINEERING.json';write(p,{'passed':True,'sha256':'wrong'})
    check('corrupted_top_engineering_cache_rejected',**reject(source,bad,'corrupt_engineering','Engineering cache checksum mismatch',extra=('--engineering',),preserve=p))

    locked=clone_inputs(baseline,WORK/'concurrent_writer');p=locked/'locks'/'runner.lock';p.parent.mkdir();p.touch()
    with p.open('a+') as handle:
        fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        check('concurrent_root_writer_rejected',**reject(source,locked,'concurrent_writer','Another writer holds'))
        fcntl.flock(handle.fileno(),fcntl.LOCK_UN)

    # Change only a disposable byte-identical copy after it is running.
    copy=WORK/'source_copy.py';shutil.copy2(source,copy);guard=WORK/'source_guard';job=launch(copy,guard,'source_guard')
    at=wait_progress(job,guard);pids=worker_pids(job[2]);copy.write_bytes(copy.read_bytes()+b'\n# deliberate verification-only source edit\n')
    code=job[0].wait(timeout=60);job[1].close();assert code not in(0,75);wait_dead(pids)
    logs=job[2].read_text()+(guard/'logs'/'seed_943901_load_16.log').read_text()
    assert 'Source changed during execution' in logs
    assert not (guard/'completed').exists()
    check('live_source_mutation_rejected_without_completion',exit_code=code,initial_cursor=at['cursor'],orphan_workers_after_stop=0)

    check('complete_results_archive_integrity',**archive_check(killed,False))
    assert sha(source)==sourcehash
    report={'passed':True,'source_sha256':sourcehash,'scientific_trajectories_run':0,'smoke_seed':943901,'checks':CHECKS,'n_checks':len(CHECKS),'work_directory':str(WORK)}
    write(HERE/'cli_verification.json',report);write(WORK/'VERIFICATION.json',report)
    print(json.dumps({'passed':True,'n_checks':len(CHECKS),'output':str(HERE/'cli_verification.json')}),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        for proc,handle in PROCESSES:
            if proc.poll() is None:
                try:os.killpg(proc.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                try:proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:os.killpg(proc.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
            if not handle.closed:handle.close()
        if 'WORK' in globals():write(WORK/'FAILED.json',{'passed':False,'checks_completed':CHECKS,'error':repr(exc)})
        raise
