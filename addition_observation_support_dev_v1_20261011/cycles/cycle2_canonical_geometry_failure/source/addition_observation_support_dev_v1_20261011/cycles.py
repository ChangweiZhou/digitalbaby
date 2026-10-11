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
    if caff.poll() is not None:
        raise RuntimeError('caffeinate did not attach')
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
                if cycle == 2:
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
                   'technical_jobs': len(jobs), 'external_training_records': len(jobs)*2560,
                   'state_trajectories_with_shared_prefix': len(jobs)*23,
                   'science_worlds': 0, 'DEV_worlds': 0, 'CONF_worlds': 0,
                   'caffeinate_attached_to_supervisor': True,
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
    summaries=[json.loads((config.ROOT/'cycles'/('cycle'+str(c))/'SUMMARY.json').read_text()) for c in (1,2,3)]
    if any(s['verdict']!='PASS_TECHNICAL_CYCLE' for s in summaries):raise AssertionError('three complete cycles')
    source=config.lock()
    if summaries[-1]['technical_jobs']!=12:raise AssertionError('last cycle roster')
    if json.loads((config.ROOT/'cycles/cycle3/SOURCE_LOCK.json').read_text())['files']!=source:
        raise AssertionError('final sources changed since qualification')
    from pipeline import suite as pipeline_suite
    pipeline=pipeline_suite(config.ROOT/'finish_pipeline')
    jobs=[j for s in summaries for j in s['jobs']]
    worst=max(j['wall_seconds'] for j in jobs)
    cyclecost=max(s['cycle_wall_seconds']/s['technical_jobs'] for s in summaries)
    projected=420*max(worst,cyclecost)*2+300
    artifact=max(j['receipt_bytes'] for j in jobs)*420*3+20*1024**2
    from collections import defaultdict
    groups=defaultdict(dict)
    for j in jobs:groups[j['world']][j['arm']]=j['metrics']
    cost={}
    for arm in config.ARMS[1:]:
        base_cpu=sum(g['COV-R']['production_cpu_seconds'] for g in groups.values())
        arm_cpu=sum(g[arm]['production_cpu_seconds'] for g in groups.values())
        payload=max(g[arm]['payload_bytes']/g['COV-R']['payload_bytes'] for g in groups.values())
        rss=max(g[arm]['peak_rss_bytes'] for g in groups.values())
        cost[arm]={'production_cpu_ratio':arm_cpu/base_cpu,'max_paired_payload_ratio':payload,
                   'peak_rss_bytes':rss,'pass':arm_cpu/base_cpu<=1.5 and payload<=2 and rss<=512*1024**2}
    resource=projected<=8*3600 and artifact<=2*1024**3 and all(x['pass'] for x in cost.values())
    qualification={'status':'TECHNICAL_QUALIFIED_SCIENCE_NOT_AUTHORIZED' if resource else 'RESOURCE_QUALIFICATION_HOLD',
        'all_three_arms':True,'cycles':3,'technical_jobs':len(jobs),
        'technical_external_records':2560*len(jobs),'correlated_state_trajectories':23*len(jobs),
        'science_jobs':0,'selected_candidate':None,'science_authorized':False,
        'conservative_maximum_420_job_wall_hours':projected/3600,
        'conservative_science_artifact_bytes':artifact,'resource_pass':resource,
        'source':source,'pipeline':pipeline,'measured_paired_resources':cost}
    atomic_json(qualification,config.ROOT/'QUALIFICATION.json')
    atomic_json({'files':source,'science_authorized':False},config.ROOT/'SOURCE_LOCK_FINAL.json')
    atomic_json({'verdict':'PASS_THREE_TECHNICAL_CYCLES','jobs':len(jobs),'records':len(jobs)*2560,
                 'source_parity':True,'science_jobs':0,'resource_status':qualification['status']},config.ROOT/'FINAL_AUDIT.json')
    atomic_json({'stage':qualification['status'],'technical_jobs':len(jobs),'science_jobs':0},config.ROOT/'STATUS.json')
    lines=['# U042 observation support: three-cycle technical report','',
           'Three design–audit–test–revision cycles completed. Technical correctness is not efficacy evidence.',
           'No DEV or confirmation science ran. No candidate was selected.','',
           '| Cycle | Full technical jobs | External records | Dispatch |',
           '|---|---:|---:|---|']
    for s in summaries:
        lines.append(f"| {s['cycle']} | {s['technical_jobs']} | {s['external_training_records']} | "+('two detached workers' if s['cycle']==3 else 'serial technical')+' |')
    lines+=['','## Scope and refinements','',
       'Cycle 1: three complete kernels and independent transition equations; COV-R parity with accepted reference; all 23 correlated trajectories per job. Local tests cover forbidden/missing feedback, cache ownership, bounded support FIFO and strict past majority, output ties and admission rejection.',
       '', 'Cycle 2: noisy STAR_LOOP and ordinary-digit guards, cold-process restore for all three arms, tamper rejection including N_NEW covariance/cache/history/RNG, omitted controls/dose checkpoints, payload and estimated operation counts. Every accepted job reconstructs 2560 records independently.',
       '', 'Cycle 3: four opaque family/noise strata, all three arms, two detached workers; supervisor-attached caffeinate; unchanged sources; no duplicate jobs. Final synthetic pipeline verifies concurrent cached status, failure/partial package refusal, selection and confirmation gates including zero-variance bounds.',
       '', 'Learned state includes weights, confidence matrix, support cache, gate history and learning RNG. N_NEW forbids all of these. Support and RATE gates use past observed evidence; the same fractional gamma updates weights and P. No replay or extra updates exist.',
       '', '## Cost and readiness','',
       f"Accepted: {len(jobs)} technical jobs / {2560*len(jobs)} external records / {23*len(jobs)} correlated state trajectories. Unit tests, synthetic fixtures and checkpoint continuations are separate technical expenditure.",
       f"Conservative full conditional science projection: {projected/3600:.2f} wall-hours; projected receipts/artifacts {artifact/1024**2:.1f} MiB. Status: {qualification['status']}.",
       '', 'The bound includes worst observed technical job/cycle overhead, a factor of two and five-minute startup allowance. It is a projection, not a promise; the authorized science dispatcher must enforce eight-hour admission and a safe ten-hour whole-job limit.',
       '', 'Mutable payload and estimated operation/traffic counters are independently reconstructed. CPU is measured production time; bandwidth counters are algorithmic estimates. Engineered COV kernels retain accepted reference equations; this is not an assertion that unmodified Full151 acquires new arithmetic ability.',
       '', '## Execution boundary','',
       'Technical admission accepts reserved PATH_LOOP/STAR_LOOP worlds only and rejects science worlds and TRIANGLE_TAIL learner runs. Confirmation reducer tests use synthetic metrics only. Formal DEV/confirmation requires a separately authorized admission adapter and launch. No automatic retries, extensions, parameters selected from technical outcomes or GitHub upload.']
    (config.ROOT/'THREE_CYCLE_REPORT.md').write_text('\n'.join(lines)+'\n')
    with zipfile.ZipFile(config.ROOT/'RESULT_BUNDLE.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(config.ROOT.rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts and p.name not in ('RESULT_BUNDLE.zip','PACKAGE.json'):
                z.write(p,p.relative_to(config.ROOT))
    with zipfile.ZipFile(config.ROOT/'RESULT_BUNDLE.zip') as z:
        if z.testzip() is not None:raise AssertionError('package corruption')
    atomic_json({'status':'COMPLETE_TECHNICAL_BUNDLE','sha256':hashlib.sha256((config.ROOT/'RESULT_BUNDLE.zip').read_bytes()).hexdigest(),
                 'science_jobs':0},config.ROOT/'PACKAGE.json')
    print(json.dumps(qualification),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cycle',type=int);p.add_argument('--finish',action='store_true');a=p.parse_args()
    if a.finish:finish()
    elif a.cycle is not None:execute(a.cycle)
    else:raise ValueError('explicit technical exercise only')
