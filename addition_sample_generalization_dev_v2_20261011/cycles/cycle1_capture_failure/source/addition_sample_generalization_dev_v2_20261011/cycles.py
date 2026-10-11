"""Finite three-cycle qualification. Does not select or authorize science."""
import argparse
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import time
import traceback
import zipfile
from pathlib import Path
import config
from runner import atomic_json, audit, commit, read, run
from technical import suite, tamper_suite
from operations import suite as operational_suite
from decision import metrics


def execute(cycle):
    if cycle not in config.TECH_PLAN:
        raise ValueError('cycle')
    folder = config.ROOT/'cycles'/('cycle'+str(cycle))
    folder.mkdir(parents=True, exist_ok=False)
    source = config.lock()
    atomic_json({'stage': 'TECHNICAL_QUALIFICATION', 'cycle': cycle, 'science_worlds': 0,
                 'committed_technical_jobs': 0}, folder/'STATUS.json')
    snapshot = folder/'source'
    snapshot.mkdir()
    for name in source:
        src = config.ROOT.parent/name
        dst = snapshot/name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    atomic_json({'files': source, 'science_authorized': False}, folder/'SOURCE_LOCK.json')
    caff = subprocess.Popen(['caffeinate', '-i', '-w', str(os.getpid())],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    jobs = []
    started = time.monotonic()
    try:
        local = suite(folder/'technical', cycle)
        atomic_json(local, folder/'LOCAL_AUDIT.json')
        print(json.dumps({'cycle': cycle, 'stage': 'LOCAL_AUDIT_PASS'}), flush=True)
        op = operational_suite(folder/'operations',
                               checkpoint_folder=folder/'technical' if cycle >= 2 else None,
                               expected=local['arms'])
        atomic_json(op, folder/'OPERATIONS_AUDIT.json')
        for world, family, noise, ordinary in config.TECH_PLAN[cycle]:
            for arm in config.ARMS:
                result = run(world, family, noise, ordinary, arm)
                path = folder/'receipts'/(str(world)+'_'+arm+'.json.gz')
                audit_result = commit(result, path)
                if cycle >= 2:
                    tamper = tamper_suite(result)
                    atomic_json(tamper, folder/'tamper'/(str(world)+'_'+arm+'.json'))
                entry = {'world': world, 'arm': arm, 'receipt': str(path.relative_to(config.ROOT)),
                         'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                         'audit': audit_result, 'metrics': metrics(result),
                         'wall_seconds': result['wall_seconds'],
                         'inclusive_cpu_seconds': result['inclusive_cpu_seconds'],
                         'receipt_bytes': path.stat().st_size}
                jobs.append(entry)
                atomic_json({'stage': 'TECHNICAL_QUALIFICATION', 'cycle': cycle,
                             'committed_technical_jobs': len(jobs), 'science_worlds': 0,
                             'last_job': [world, arm], 'supervisor_pid': os.getpid(),
                             'caffeinate_pid': caff.pid}, folder/'STATUS.json')
                print(json.dumps({'cycle': cycle, 'committed_jobs': len(jobs),
                                  'world': world, 'arm': arm,
                                  'seconds': result['wall_seconds']}), flush=True)
        if config.lock() != source:
            raise AssertionError('cycle sources changed')
        summary = {'verdict': 'PASS_TECHNICAL_CYCLE', 'cycle': cycle,
                   'technical_jobs': len(jobs), 'external_training_records': len(jobs)*2400,
                   'state_trajectories_with_shared_prefix': len(jobs)*21,
                   'science_worlds': 0, 'DEV_worlds': 0, 'CONF_worlds': 0,
                   'selected_candidate': None, 'jobs': jobs,
                   'cycle_wall_seconds': time.monotonic()-started}
        atomic_json(summary, folder/'SUMMARY.json')
        atomic_json({'stage': 'TECHNICAL_CYCLE_COMPLETE', 'cycle': cycle,
                     'committed_technical_jobs': len(jobs), 'science_worlds': 0}, folder/'STATUS.json')
        return summary
    except BaseException as error:
        atomic_json({'stage': 'TECHNICAL_FAILURE', 'error': repr(error),
                     'traceback': traceback.format_exc(), 'committed_technical_jobs': len(jobs),
                     'science_worlds': 0}, folder/'FAILURE.json')
        raise
    finally:
        caff.terminate()
        caff.wait(timeout=10)


def finish():
    summaries = [json.loads((config.ROOT/'cycles'/('cycle'+str(c))/'SUMMARY.json').read_text())
                 for c in (1, 2, 3)]
    if any(s['verdict'] != 'PASS_TECHNICAL_CYCLE' for s in summaries):
        raise AssertionError('three complete cycles required')
    source = config.lock()
    jobs = [j for s in summaries for j in s['jobs']]
    worst_cpu = max(j['inclusive_cpu_seconds'] for j in jobs)
    worst_wall = max(j['wall_seconds'] for j in jobs)
    # Conservative serial projection, incl. measured offline auditor/tamper
    # expense and cold startup margin; two workers need not be assumed faster.
    cycle_per_job = max(s['cycle_wall_seconds']/s['technical_jobs'] for s in summaries)
    projected_seconds = 588*max(worst_wall, cycle_per_job)*2 + 300
    receipt_max = max(j['receipt_bytes'] for j in jobs)
    current_bytes = sum(p.stat().st_size for p in config.ROOT.rglob('*') if p.is_file())
    projected_bytes = 588*receipt_max*3 + current_bytes*2 + 20*1024*1024
    payload_max = max(json.loads((config.ROOT/'cycles'/('cycle'+str(c))/'LOCAL_AUDIT.json').read_text())
                      ['arms'][a]['payload_bytes'] for c in (1, 2, 3) for a in config.ARMS if a != 'STABLE-ID')
    resource_pass = projected_seconds <= 8*3600 and projected_bytes <= 2*1024**3 and payload_max <= 128*1024
    qualification = {'status': 'TECHNICAL_QUALIFIED_SCIENCE_NOT_AUTHORIZED' if resource_pass else 'RESOURCE_QUALIFICATION_HOLD',
                     'all_nine_arms': True, 'cycles': 3, 'technical_jobs': len(jobs),
                     'technical_external_records': 2400*len(jobs),
                     'technical_state_trajectories_with_shared_prefix': 21*len(jobs),
                     'DEV_worlds': 0, 'CONF_worlds': 0, 'selected_candidate': None,
                     'source': source, 'worst_measured_job_seconds': worst_wall,
                     'worst_inclusive_job_cpu_seconds': worst_cpu,
                     'conservative_maximum_588_job_seconds': projected_seconds,
                     'conservative_maximum_artifact_bytes': projected_bytes,
                     'candidate_payload_max_bytes': payload_max,
                     'resource_pass': resource_pass, 'science_authorized': False}
    atomic_json(qualification, config.ROOT/'QUALIFICATION.json')
    atomic_json({'files': source, 'science_authorized': False}, config.ROOT/'SOURCE_LOCK_FINAL.json')
    atomic_json({'verdict': 'PASS_THREE_TECHNICAL_CYCLES', 'technical_jobs': len(jobs),
                 'resource_gate': qualification['status'], 'sources_verified': True,
                 'science_worlds': 0, 'no_ODD_CYCLE_learning': True}, config.ROOT/'FINAL_AUDIT.json')
    atomic_json({'stage': qualification['status'], 'technical_jobs': len(jobs),
                 'science_jobs': 0}, config.ROOT/'STATUS.json')
    with zipfile.ZipFile(config.ROOT/'RESULT_BUNDLE.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for path in sorted(config.ROOT.rglob('*')):
            if path.is_file() and not any(part in ('scratch', '__pycache__') for part in path.parts) and path.name != 'RESULT_BUNDLE.zip':
                z.write(path, path.relative_to(config.ROOT))
    atomic_json({'status': 'COMPLETE_TECHNICAL_BUNDLE',
                 'zip_bytes': (config.ROOT/'RESULT_BUNDLE.zip').stat().st_size,
                 'sha256': hashlib.sha256((config.ROOT/'RESULT_BUNDLE.zip').read_bytes()).hexdigest(),
                 'science_jobs': 0}, config.ROOT/'PACKAGE.json')
    print(json.dumps(qualification), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--cycle', type=int)
    p.add_argument('--finish', action='store_true')
    a = p.parse_args()
    if a.finish:
        finish()
    elif a.cycle is not None:
        execute(a.cycle)
    else:
        raise ValueError('explicit technical cycle required')
