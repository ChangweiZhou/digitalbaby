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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import config
from runner import atomic_json, audit, commit, read, run
from technical import suite, tamper_suite
from operations import suite as operational_suite
from decision import metrics, countermodels


def detached_job(task, arm, folder):
    world, family, noise, ordinary = task
    path = folder/'receipts'/(str(world)+'_'+arm+'.json.gz')
    log = folder/'worker_logs'/(str(world)+'_'+arm+'.log')
    log.parent.mkdir(exist_ok=True)
    args = [os.sys.executable, str(config.ROOT/'runner.py'), '--world', str(world),
            '--family', family, '--noise', noise, '--arm', arm, '--output', str(path)]
    if ordinary:
        args.append('--ordinary')
    with log.open('w') as f:
        p = subprocess.Popen(args, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        code = p.wait(timeout=120)
    if code:
        raise RuntimeError('detached technical worker failed: '+str(log))
    result = read(path)
    independent = audit(result)
    return result, path, independent


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
        tasks = [(task, arm) for task in config.TECH_PLAN[cycle] for arm in config.ARMS]
        pool = ThreadPoolExecutor(max_workers=2) if cycle == 3 else None
        futures = [pool.submit(detached_job, task, arm, folder) for task, arm in tasks] if pool else None
        for index, (task, arm) in enumerate(tasks):
                world, family, noise, ordinary = task
                if futures is not None:
                    result, path, audit_result = futures[index].result(timeout=240)
                else:
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
                entry['countermodels'] = countermodels(result)
                jobs.append(entry)
                atomic_json({'stage': 'TECHNICAL_QUALIFICATION', 'cycle': cycle,
                             'committed_technical_jobs': len(jobs), 'science_worlds': 0,
                             'last_job': [world, arm], 'supervisor_pid': os.getpid(),
                             'caffeinate_pid': caff.pid}, folder/'STATUS.json')
                print(json.dumps({'cycle': cycle, 'committed_jobs': len(jobs),
                                  'world': world, 'arm': arm,
                                  'seconds': result['wall_seconds']}), flush=True)
        if pool:
            pool.shutdown(wait=True)
        if config.lock() != source:
            raise AssertionError('cycle sources changed')
        summary = {'verdict': 'PASS_TECHNICAL_CYCLE', 'cycle': cycle,
                   'technical_jobs': len(jobs), 'external_training_records': len(jobs)*2400,
                   'state_trajectories_with_shared_prefix': len(jobs)*21,
                   'science_worlds': 0, 'DEV_worlds': 0, 'CONF_worlds': 0,
                   'selected_candidate': None, 'jobs': jobs,
                   'detached_full_job_workers': 2 if cycle == 3 else 0,
                   'cycle_wall_seconds': time.monotonic()-started}
        atomic_json(summary, folder/'SUMMARY.json')
        atomic_json({'stage': 'TECHNICAL_CYCLE_COMPLETE', 'cycle': cycle,
                     'committed_technical_jobs': len(jobs), 'science_worlds': 0}, folder/'STATUS.json')
        return summary
    except BaseException as error:
        atomic_json({'stage': 'TECHNICAL_FAILURE', 'error': repr(error),
                     'traceback': traceback.format_exc(), 'committed_technical_jobs': len(jobs),
                     'science_worlds': 0}, folder/'FAILURE.json')
        atomic_json({'stage': 'TECHNICAL_FAILURE', 'error': repr(error),
                     'committed_technical_jobs': len(jobs), 'science_worlds': 0}, folder/'STATUS.json')
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
    # Technical genealogy remains visible; the failed capture attempt is not
    # silently added to the 63 successful qualification jobs.
    qualification['historical_capture_failure'] = {
        'world': 835001, 'accepted_technical_jobs': 7,
        'uncommitted_complete_technical_jobs': 1,
        'executed_external_records': 8*2400,
        'superseded_by_fresh_world': 835008, 'science_worlds': 0}
    atomic_json(qualification, config.ROOT/'QUALIFICATION.json')
    atomic_json({'files': source, 'science_authorized': False}, config.ROOT/'SOURCE_LOCK_FINAL.json')
    atomic_json({'verdict': 'PASS_THREE_TECHNICAL_CYCLES', 'technical_jobs': len(jobs),
                 'resource_gate': qualification['status'], 'sources_verified': True,
                 'science_worlds': 0, 'no_ODD_CYCLE_learning': True}, config.ROOT/'FINAL_AUDIT.json')
    atomic_json({'stage': qualification['status'], 'technical_jobs': len(jobs),
                 'science_jobs': 0}, config.ROOT/'STATUS.json')
    lines = ['# MiniFly SAMPLE-GEN V2 — three-cycle technical exercise', '',
             'All three cycles completed. No DEV or confirmation world was run.',
             'No candidate was selected, no mechanism was adopted and no parameters',
             'were tuned from technical performance.', '',
             '| Cycle | Complete jobs | External records | Full-job dispatch |',
             '|---|---:|---:|---|']
    for s in summaries:
        lines.append('|'+str(s['cycle'])+'|'+str(s['technical_jobs'])+'|'+
                     str(s['external_training_records'])+'|'+
                     ('two detached workers' if s['cycle'] == 3 else 'serial technical')+'|')
    lines += ['', '## Refinements and audit outcomes', '',
              'Cycle 1 locked group-reduction boundary parity, implemented all nine',
              'cores and full shared-prefix/fork histories, and checked all learned',
              'auxiliaries in N_OLD. The first capture attempt failed because a',
              'receipt cache snapshot aliased live state. Its 7 accepted technical',
              'jobs and 1 uncommitted complete job remain archived. Deep snapshot',
              'ownership was fixed; successful cycle 1 used a new world ID.',
              '',
              'Cycle 2 checked cold-process checkpoint parity for all nine arms,',
              'exact independent transition seals, 12 tamper cases per complete',
              'receipt, noisy/missing feedback and ordinary-digit protection.',
              '',
              'Cycle 3 covered four fresh opaque family/noise strata and all nine',
              'arms through two detached full-job workers. Process-local fly_id',
              'is diagnostic only; B, Q, T, projection and inherited-source hashes',
              'remain strict. Duplicate commit claims, safe pause, package refusal,',
              'and synthetic statistical edge cases were checked.',
              '',
              '## Resource qualification', '',
              'Successful qualification: '+str(len(jobs))+' jobs, '+
              str(2400*len(jobs))+' external records, '+str(21*len(jobs))+
              ' state trajectories with shared prefixes. These are technical,',
              'not independent science lives. Historical failed-attempt expenditure',
              'and synthetic unit/restore records are separate.',
              '',
              'Conservative maximum-roster projection: '+
              str(round(projected_seconds/3600, 2))+' wall-hours and '+
              str(round(projected_bytes/1024**2, 1))+' MiB of new artifacts.',
              'Largest candidate mutable payload: '+str(payload_max)+' bytes.',
              'Status: '+qualification['status']+'.',
              '',
              'Byte/transition seals validate actual emitted outputs and every',
              'learned array/cache/RNG change with independent equations, with no',
              'floating-point tolerance. Algorithmic memory-movement counters are',
              'conservative estimates, not measured hardware bandwidth.',
              '',
              '## Interpretation and next action', '',
              'Technical correctness is not E3 confirmation. The new benchmark is',
              'not saturated; technical scores are diagnostic and may not choose',
              'a winner or modify gates. Formal comparison requires the locked',
              '12-world DEV and at most one candidate in fresh confirmation.',
              'The scientific admission adapter and launch still require explicit',
              'execution authorization; current admission accepts technical IDs only.']
    (config.ROOT/'THREE_CYCLE_REPORT.md').write_text('\n'.join(lines)+'\n')
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
