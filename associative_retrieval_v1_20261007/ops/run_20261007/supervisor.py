"""Detached, bounded, non-retrying screen/confirm/audit/package pipeline."""
import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path
import bootstrap_ops as boot
from contracts import (WORKERS, DISK_CAP, DISPATCH_SECONDS, OUTER_SECONDS,
                       require_sources, roster, alive)
from io_utils import atomic_json, read_receipt, digest
from assays import make_world
from science_audit import audit_science
from analysis import analyze_stage

RESULTS = boot.ROOT/'results'
SCIENCE = RESULTS/'science'
CHILDREN = {}

def counts():
    jobs = lives = branch_lives = 0
    for stage in ('screen', 'confirm'):
        folder = SCIENCE/stage
        for p in (folder/'receipts').glob('*.json.gz'):
            assay = p.name[:-8].rsplit('_', 1)[1]
            jobs += 1
            lives += 4 if assay == 'lifetime' else 2
        branch_lives += len(list((folder/'branches').glob('*.json.gz')))
    return {'completed_full_jobs': jobs, 'completed_native_lives_in_full_jobs': lives,
            'individually_committed_native_lives': branch_lives,
            'maximum_full_jobs': 144, 'maximum_native_lives': 432}

def status(stage, **extras):
    d = {'schema': 'LINK_LIVE_STATUS_V1', 'stage': stage, 'pid': os.getpid(),
         'updated_unix': time.time(), 'workers': WORKERS, **counts(), **extras}
    atomic_json(RESULTS/'STATUS.json', d)
    atomic_json(boot.ROOT/'STATUS.json', d)
    return d

def bytes_used():
    total = 0
    for p in boot.ROOT.rglob('*'):
        if '.pending-' in p.name:
            continue
        try:
            if p.is_file():
                total += p.stat().st_size
        except FileNotFoundError:
            # A committed life may concurrently unlink its active checkpoint.
            # This is not a corrupt receipt or a scientific failure.
            continue
    return total

def stop_children():
    for stage in ('screen', 'confirm'):
        folder = SCIENCE/stage
        if folder.exists():
            atomic_json(folder/'STOP_REQUEST.json', {'reason': 'SUPERVISOR_FAILURE',
                        'whole_record_boundary_only': True, 'requested_unix': time.time()})
    until = time.monotonic()+90
    while any(p.poll() is None for p in CHILDREN.values()) and time.monotonic() < until:
        time.sleep(1)
    forced = []
    for pid, p in CHILDREN.items():
        if p.poll() is None:
            forced.append(pid)
            p.terminate()
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait(timeout=10)
    return {'children_stopped': list(CHILDREN), 'forced_incomplete_workers': forced}

def create_manifest(stage):
    folder = SCIENCE/stage
    folder.mkdir(parents=True, exist_ok=True)
    lock = require_sources()
    rows = [{**j, 'fixture_sha256': make_world(j['world'], j['assay'])['sha256']} for j in roster(stage)]
    d = {'schema': 'LINK_FIXED_MANIFEST_V1', 'stage': stage,
         'source_identity': lock['native_identity'], 'operations_identity': lock['operations_identity'],
         'jobs': rows, 'conditions': ['ERROR', 'LINK', 'PERM'], 'workers': WORKERS}
    path = folder/'MANIFEST.json'
    if path.exists():
        if json.loads(path.read_text()) != d:
            raise ValueError('frozen stage manifest differs')
    else:
        atomic_json(path, d)
    return folder, d

def run_stage(stage, session_start, resume=False):
    folder, manifest = create_manifest(stage)
    for name in ('receipts', 'branches', 'checkpoints', 'active', 'logs', 'job_audits', 'failures'):
        (folder/name).mkdir(exist_ok=True)
    stop = folder/'STOP_REQUEST.json'
    if stop.exists():
        if not resume:
            raise ValueError('paused dispatch requires explicit resume authorization')
        stop.unlink()
    pending = list(manifest['jobs'])
    active = {}
    failure = None
    paused = False
    last_status = 0.
    deadline = session_start+DISPATCH_SECONDS
    atomic_json(folder/'LAUNCH.json', {'stage': stage, 'pid': os.getpid(), 'started_unix': time.time(),
                                    'workers': WORKERS, 'deadline_unix': deadline,
                                    'manifest_digest': digest(manifest)})
    def request_stop(reason):
        if not stop.exists():
            atomic_json(stop, {'reason': reason, 'requested_unix': time.time(),
                               'whole_record_boundary_only': True, 'automatic_resume': False})
    while pending or active:
        if time.time() >= deadline:
            paused = True
            request_stop('PAUSED_SESSION_LIMIT')
        if bytes_used() > DISK_CAP:
            failure = failure or {'error': 'fixed 8 GiB experiment storage cap exceeded', 'stage': stage}
            request_stop('FAILURE')
        if failure:
            request_stop('FAILURE')
        while pending and len(active) < WORKERS and not failure and not paused:
            j = pending.pop(0)
            rp = folder/'receipts'/f'{j["job"]}.json.gz'
            if rp.exists():
                audit_science(read_receipt(rp), require_sources())
                print(f'SKIP committed {j["job"]}', flush=True)
                continue
            argv = [sys.executable, '-u', str(boot.OPS/'science_worker.py'), '--world', str(j['world']),
                    '--assay', j['assay'], '--stage', stage, '--folder', str(folder),
                    '--stop-file', str(stop), '--deadline', str(deadline)]
            if resume:
                argv.append('--resume')
            log_path = folder/'logs'/f'{j["job"]}.log'
            log = open(log_path, 'a')
            p = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
            CHILDREN[p.pid] = p
            active[p.pid] = {'process': p, 'job': j, 'log': log, 'log_path': log_path,
                             'started_unix': time.time()}
            print(f'START {stage} {j["job"]} pid={p.pid}', flush=True)
        for pid, x in list(active.items()):
            rc = x['process'].poll()
            if rc is None:
                continue
            x['log'].close()
            active.pop(pid)
            CHILDREN.pop(pid, None)
            j = x['job']
            if rc == 0:
                if not (folder/'receipts'/f'{j["job"]}.json.gz').exists():
                    failure = {'error': 'worker exited zero without committed full job', 'job': j, 'stage': stage}
                else:
                    print(f'COMMIT {stage} {j["job"]}', flush=True)
            elif rc == 75:
                if not paused and not failure:
                    failure = {'error': 'unexpected safe-pause exit without dispatcher request', 'job': j, 'stage': stage}
            else:
                fp = folder/'failures'/f'{j["job"]}.json'
                failure = failure or (json.loads(fp.read_text()) if fp.exists() else
                           {'error': f'worker exited {rc}', 'job': j, 'log': str(x['log_path']), 'stage': stage})
                print(f'FAIL {stage} {j["job"]}: {failure["error"]}', flush=True)
        if time.time()-last_status >= 5:
            s = status('STOPPING_FAILURE' if failure else 'PAUSING_SESSION_LIMIT' if paused else stage.upper(),
                       current_science_stage=stage, active_workers=len(active), queued_jobs=len(pending),
                       active_pids=list(active), session_started_unix=session_start,
                       session_deadline_unix=session_start+OUTER_SECONDS,
                       screen_planned_jobs=16, screen_planned_lives=48,
                       confirmation_planned_jobs=128, confirmation_planned_lives=384,
                       live_stage_status=str(folder/'STATUS.json'))
            atomic_json(folder/'STATUS.json', s)
            last_status = time.time()
        if time.time() >= session_start+OUTER_SECONDS-600 and active:
            # Cannot describe a hung job as a safe pause. Preserve its checkpoint,
            # stop it explicitly and disclose the incomplete current record.
            for x in active.values():
                x['process'].terminate()
            failure = {'error': 'workers could not reach safe record boundary before 10-hour session limit',
                       'stage': stage, 'processes': list(active), 'incomplete_current_records': True}
            request_stop('UNEXPECTED_BOUNDARY_TIMEOUT')
        if failure or paused:
            pending.clear()
        if active:
            time.sleep(1)
    result = {'stage': stage, 'status': 'FAILED' if failure else 'PAUSED_SESSION_LIMIT' if paused else 'COMPLETE',
              **counts(), 'ended_unix': time.time(), 'error': failure}
    atomic_json(folder/'SESSION_RESULT.json', result)
    if failure:
        atomic_json(folder/'FAILURE.json', failure)
        raise RuntimeError(json.dumps(failure, ensure_ascii=False))
    if paused:
        status('PAUSED_SESSION_LIMIT', current_science_stage=stage, active_workers=0,
               explicit_authorization_required_to_resume=True, session_started_unix=session_start)
        return None
    if len(list((folder/'receipts').glob('*.json.gz'))) != len(roster(stage)):
        raise AssertionError('exact full-job stage count')
    status('ANALYSIS', current_science_stage=stage, active_workers=0)
    result = analyze_stage(folder, stage)
    print(f'ANALYSIS {stage} {result["verdict"]}', flush=True)
    return result

def finalize(screen, confirm):
    lock = require_sources()
    status('FINAL_AUDIT', active_workers=0)
    audits = []
    for stage in ('screen', 'confirm') if confirm is not None else ('screen',):
        for j in roster(stage):
            d = read_receipt(SCIENCE/stage/'receipts'/f'{j["job"]}.json.gz')
            audits.append(audit_science(d, lock))
    physical_jobs = 16+(128 if confirm is not None else 0)
    lives = 48+(384 if confirm is not None else 0)
    if len(audits) != physical_jobs or sum(a['lives'] for a in audits) != lives:
        raise AssertionError('final exact jobs/lives mismatch')
    final = confirm if confirm is not None else screen
    summary = {'schema': 'LINK_FINAL_RESULT_V1', 'verdict': final['verdict'],
               'adopted': final.get('adopted', False), 'science_accepted': True,
               'completed_full_jobs': physical_jobs, 'completed_native_lives': lives,
               'completed_worlds': 8+(64 if confirm is not None else 0),
               'complete_exposure_records': sum(a['records'] for a in audits),
               'screen': {k: v for k, v in screen.items() if k != 'per_world'},
               'confirm': None if confirm is None else {k: v for k, v in confirm.items() if k != 'per_world'},
               'source_identity': lock['native_identity'], 'operations_identity': lock['operations_identity'],
               'no_extra_designs': True, 'publication': 'not requested'}
    atomic_json(RESULTS/'SUMMARY.json', summary)
    final_audit = {'verdict': 'ACCEPTED', 'science_result_verdict': final['verdict'],
                   'source_lock_verified': True, 'completed_full_jobs': physical_jobs,
                   'completed_native_lives': lives, 'complete_exposure_records': summary['complete_exposure_records'],
                   'independent_job_audits': audits, 'negative_result_is_valid_completion': True,
                   'causal_branch_map': 'W/N_old/N_new/N_revision actual transitions independently replayed',
                   'no_automatic_restart': True, 'counts': counts()}
    atomic_json(RESULTS/'FINAL_AUDIT.json', final_audit)
    report_name = 'CONFIRM_REPORT.md' if confirm is not None else 'SCREEN_REPORT.md'
    report = (boot.ROOT/report_name).read_text()
    report += (f'\n已提交{physical_jobs}个完整世界/任务、{lives}条原生生命史、'
               f'{summary["complete_exposure_records"]:,}条完整记录。独立最终审计ACCEPTED。\n\n'
               '未增加任务、剂量、样本或替代候选。当前实验结束；后续机制清单另行处理。\n')
    (boot.ROOT/'REPORT.md').write_text(report)
    status('PACKAGING', active_workers=0, science_accepted=True, verdict=final['verdict'])
    bundle = boot.ROOT/'RESULT_BUNDLE.zip'
    if bundle.exists():
        raise ValueError('bundle already exists; do not silently replace final package')
    temp = boot.ROOT/'RESULT_BUNDLE.zip.pending-package'
    files = []
    for p in boot.ROOT.rglob('*'):
        if not p.is_file() or p in (temp, bundle):
            continue
        rel = p.relative_to(boot.ROOT)
        if ('__pycache__' in rel.parts or 'numba' in rel.parts or p.name.endswith('.lock')
                or '.pending-' in p.name or 'checkpoints' in rel.parts or 'qualification' in rel.parts
                or rel.parts[0] == 'technical'):
            continue
        files.append(p)
    with zipfile.ZipFile(temp, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as z:
        for p in sorted(files):
            z.write(p, str(p.relative_to(boot.ROOT)))
        z.writestr('PACKAGE_README.txt', 'Final SUMMARY.json and FINAL_AUDIT.json establish accepted science. Live STATUS.json is also available outside this archive. Parent dependencies remain source-locked in SOURCE_LOCK.json; this is a results bundle, not a portable science rerun kit.\n')
    with zipfile.ZipFile(temp) as z:
        bad = z.testzip()
        if bad is not None or 'results/FINAL_AUDIT.json' not in z.namelist():
            raise ValueError('package integrity/required artifact failure')
    os.replace(temp, bundle)
    sha = hashlib.sha256()
    with open(bundle, 'rb') as f:
        for chunk in iter(lambda: f.read(1024**2), b''):
            sha.update(chunk)
    atomic_json(RESULTS/'PACKAGE.json', {'verdict': 'COMPLETE', 'path': str(bundle),
                                       'bytes': bundle.stat().st_size, 'sha256': sha.hexdigest(),
                                       'entries': len(files)+1, 'results_bundle_not_portable_runtime': True})
    status('COMPLETE', active_workers=0, science_accepted=True, verdict=final['verdict'],
           report=str(boot.ROOT/'REPORT.md'), bundle=str(bundle), package_complete=True)
    print(f'COMPLETE {final["verdict"]} jobs={physical_jobs} lives={lives}', flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    with open(boot.OPS/'supervisor.lock', 'a+') as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (RESULTS/'FAILURE.json').exists():
            raise ValueError('terminal failure cannot restart automatically')
        lock = require_sources()
        qual = json.loads((boot.OPS/'QUALIFICATION.json').read_text())
        auth = json.loads((boot.OPS/'AUTHORIZATION.json').read_text())
        if qual['verdict'] != 'PASS' or qual['operations_identity'] != lock['operations_identity'] or auth.get('science_authorized') is not True:
            raise ValueError('qualification/authorization required')
        if (RESULTS/'PACKAGE.json').exists():
            print('SKIP accepted complete package', flush=True)
            return 0
        old = json.loads((RESULTS/'STATUS.json').read_text()) if (RESULTS/'STATUS.json').exists() else {}
        if old.get('stage') == 'PAUSED_SESSION_LIMIT' and not args.resume:
            raise ValueError('session paused; explicit resume required')
        session_start = time.time()
        caffeine = subprocess.Popen(['/usr/bin/caffeinate', '-i', '-w', str(os.getpid())])
        launch = {'pid': os.getpid(), 'caffeinate_pid': caffeine.pid, 'workers': WORKERS,
                  'started_unix': session_start, 'python': sys.executable,
                  'supervisor_log': str(boot.OPS/'supervisor.log'), 'detached': True,
                  'source_identity': lock['native_identity'], 'operations_identity': lock['operations_identity'],
                  'science_changed': False, 'session_limit_hours': 10, 'automatic_retry': False}
        atomic_json(RESULTS/'LAUNCH.json', launch)
        try:
            print('START authorized LINK pipeline, 8 workers', flush=True)
            screen = run_stage('screen', session_start, args.resume)
            if screen is None:
                return 75
            confirm = None
            if screen['advance']:
                if time.time() >= session_start+DISPATCH_SECONDS:
                    status('PAUSED_SESSION_LIMIT', current_science_stage='confirm', active_workers=0,
                           explicit_authorization_required_to_resume=True)
                    return 75
                confirm = run_stage('confirm', session_start, args.resume)
                if confirm is None:
                    return 75
            finalize(screen, confirm)
            return 0
        except Exception as e:
            stopped = stop_children()
            failure = {'error': repr(e), 'traceback': traceback.format_exc(),
                       'pid': os.getpid(), 'at_unix': time.time(), **counts(),
                       'restart_authorized': False, **stopped}
            atomic_json(RESULTS/'FAILURE.json', failure)
            status('FAILED', active_workers=0, error=repr(e), restart_authorized=False)
            print(failure['traceback'], flush=True)
            return 1
        finally:
            if caffeine.poll() is None:
                caffeine.terminate()

if __name__ == '__main__':
    raise SystemExit(main())
