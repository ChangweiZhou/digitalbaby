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
        code = p.wait(timeout=600)
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
                    result, path, audit_result = futures[index].result(timeout=1200)
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


def checkpoints(receipt):
    for branch in receipt['branches'].values():
        yield branch['birth']['state']
        for f in branch['forks'].values():
            yield f['old_checkpoint']['state']
            for k in f['new_checkpoints'].values():yield k['checkpoint']['state']
            yield f['final_checkpoint']['state']
    for f in receipt['extra_forks'].values():
        yield f['old_checkpoint']['state']
        for k in f['new_checkpoints'].values():yield k['checkpoint']['state']
        yield f['final_checkpoint']['state']

def finish():
    from collections import defaultdict
    import math
    summaries=[json.loads((config.ROOT/'cycles'/('cycle'+str(c))/'SUMMARY.json').read_text()) for c in (1,2,3)]
    if [s['technical_jobs'] for s in summaries]!=[5,10,20] or any(s['verdict']!='PASS_TECHNICAL_CYCLE' for s in summaries):raise AssertionError('three complete fixed-roster cycles')
    source=config.lock()
    if json.loads((config.ROOT/'cycles/cycle3/SOURCE_LOCK.json').read_text())['files']!=source:raise AssertionError('final qualified sources changed')
    from pipeline import suite as pipeline_suite
    pipeline=pipeline_suite(config.ROOT/'finish_pipeline')
    jobs=[j for s in summaries for j in s['jobs']]
    if len({(j['world'],j['arm']) for j in jobs})!=35:raise AssertionError('duplicate technical jobs')
    groups=defaultdict(dict);accepted={}
    for j in jobs:
        path=config.ROOT/j['receipt']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=j['sha256'] or j['audit']['verdict']!='PASS':raise AssertionError('committed certificate')
        groups[j['world']][j['arm']]=j['metrics'];accepted[j['world'],j['arm']]=j
    if any(set(g)!=set(config.ARMS) for g in groups.values()):raise AssertionError('full paired roster')
    costs={}
    for arm in config.RESTORE_ARMS:
        base=sum(g['COV-SUPPORT']['production_cpu_seconds'] for g in groups.values())
        cpu=sum(g[arm]['production_cpu_seconds'] for g in groups.values())/base
        payload=max(g[arm]['payload_bytes']/g['COV-SUPPORT']['payload_bytes'] for g in groups.values())
        rss=max(g[arm]['peak_rss_bytes'] for g in groups.values())
        costs[arm]={'production_CPU_vs_SUPPORT':cpu,'max_paired_payload_vs_SUPPORT':payload,'peak_RSS_bytes':rss,
            'pass':cpu<=1.5 and payload<=2 and rss<=512*1024**2}
    matched=0;restoration_events=0;nominal_trace=0.;max_trace_error=0.;fallbacks=0
    fields=('cache','history','rng','buffer','qualified','restorations','direction_rng','updates')
    for world in groups:
        receipts={a:read(config.ROOT/accepted[world,a]['receipt']) for a in config.ARMS}
        states={a:list(checkpoints(receipts[a])) for a in config.RESTORE_ARMS}
        if len({len(v) for v in states.values()})!=1:raise AssertionError('paired state roster')
        for rows in zip(*states.values()):
            for key in fields:
                if any(row[key]!=rows[0][key] for row in rows):raise AssertionError('exact shared auxiliary state '+key)
            matched+=1
        def records(r):
            for branch in r['branches'].values():
                yield from branch['old_records']
                for f in branch['forks'].values():yield from f['new_records']
            for f in r['extra_forks'].values():yield from f['new_records']
        for rows in zip(*(records(receipts[a]) for a in config.RESTORE_ARMS)):
            if len({(r['raw'],r['allowed']) for r in rows})!=1:raise AssertionError('paired byte ledger')
            events=[next((e['update']['restoration'] for e in r['events'] if e['update'] and e['update']['performed']),None) for r in rows]
            selected=('qualified_added','qualified_count','performed','restoration_count','delta')
            if any(e is None for e in events):
                if not all(e is None for e in events):raise AssertionError('write permission parity')
                continue
            if any(tuple(e[k] for k in selected)!=tuple(events[0][k] for k in selected) for e in events):raise AssertionError('exact trigger and nominal dose')
            if events[0]['performed']:
                restoration_events+=1;nominal_trace+=events[0]['delta'];fallbacks+=int(events[0]['fallback'])
                for e in events:max_trace_error=max(max_trace_error,abs(e['trace_added']-.25))
                if max_trace_error>2e-12:raise AssertionError('realized trace rounding budget')
    # Production, auditing, IO and bounded qualification overhead all enter the conservative estimate.
    worst=max(j['wall_seconds'] for j in jobs)
    cyclecost=max(s['cycle_wall_seconds']/s['technical_jobs'] for s in summaries)
    per=max(worst,cyclecost)*2
    total=700*per+300;dev=60*per+300
    artifact=max(j['receipt_bytes'] for j in jobs)*700*3+20*1024**2
    resource=dev<=8*3600 and per+300<=8*3600 and artifact<=2*1024**3 and all(c['pass'] for c in costs.values())
    status='TECHNICAL_QUALIFIED_SCIENCE_NOT_AUTHORIZED' if resource else 'RESOURCE_QUALIFICATION_HOLD'
    tamper=sum(len(json.loads(p.read_text())['rejected']) for p in (config.ROOT/'cycles/cycle2/tamper').glob('*.json'))
    failures=[]
    for p in sorted((config.ROOT/'cycles').glob('*failure*/FAILURE.json')):
        x=json.loads(p.read_text());failures.append({'record':str(p.relative_to(config.ROOT)),'error':x['error'],'committed_jobs':x['committed_technical_jobs']})
    qualification={'status':status,'technical_jobs':35,'external_records':89600,'correlated_state_trajectories':805,
      'DEV_worlds':0,'CONF_worlds':0,'science_jobs':0,'science_authorized':False,'selected_candidate':None,
      'resources':costs,'resource_pass':resource,'max_conditional_hours_conservative':total/3600,
      'DEV_hours_conservative':dev/3600,'projected_8h_sessions':math.ceil(total/(8*3600)),
      'per_full_job_seconds_conservative':per,'artifact_bytes_conservative':artifact,
      'paired_auxiliary_snapshots':matched,'restoration_ledger_groups':restoration_events,
      'nominal_trace_over_committed_correlated_trajectories':nominal_trace,'max_trace_rounding_error':max_trace_error,
      'DIR_fallback_ledger_events':fallbacks,'pipeline':pipeline,'sources':source}
    atomic_json(qualification,config.ROOT/'QUALIFICATION.json')
    atomic_json({'files':source,'science_authorized':False},config.ROOT/'SOURCE_LOCK_FINAL.json')
    audit={'verdict':'PASS_THREE_TECHNICAL_CYCLES','technical_jobs':35,'technical_records':89600,'science_jobs':0,
      'accepted_certificates':35,'tamper_rejections':tamper,'exact_source_parity':True,
      'paired_auxiliary_snapshots':matched,'matching_restoration_events':restoration_events,
      'historical_technical_failures':failures,'resource_status':status}
    atomic_json(audit,config.ROOT/'FINAL_AUDIT.json')
    atomic_json({'stage':status,'technical_jobs':35,'science_jobs':0},config.ROOT/'STATUS.json')
    lines=['# U034三次设计审核试跑修订报告','',
      '目标保持为更少带噪或缺失答案样本下的未见加法自身输出泛化。这里只完成技术资格化；没有运行DEV或确认，没有选择或采用新核心。','',
      '| 循环 | 完整技术任务 | 外部记录 | 执行 |','| --- | ---: | ---: | --- |']
    for s in summaries:lines.append(f"| {s['cycle']} | {s['technical_jobs']} | {s['external_training_records']} | "+('两个脱离终端worker' if s['cycle']==3 else '串行技术任务')+' |')
    lines+=['','五配置完整保留：COV-R、COV-SUPPORT、SUPPORT-DIR、SUPPORT-RAND、SUPPORT-ISO。只有DIR可晋级。两基线240次逐状态数值一致、独立方向/更新参考、同剂量与时刻、全部辅助状态禁写和冷恢复均通过。公开合成资格序列与正式图无学习几何检查不是效能数据。','',
      f'独立审核接受35份完整任务证书，拒绝{tamper}个篡改案例；最终按接受收据验证{matched}组辅助检查点和{restoration_events}组恢复事件。三恢复配置的资格窗口、计数、触发时刻、RNG及名义trace剂量匹配；最大实际trace舍入误差{max_trace_error:.3g}。这些是控制资格，不是DIR改善E3的结果。','',
      '零残差固定回退、至少两个有序键、次数上限64、禁写/缺失/查询完全惰性、原多数仍保留冲突答案、两种角色不合并均有边界测试。恢复窗口不会查询输出答案；重复缓存观察仅构造方向，不作为新的独立样本或重复权重/P证据更新。','',
      '## 实测资源','']
    for arm,c in costs.items():lines.append(f"- {arm}：生产CPU {c['production_CPU_vs_SUPPORT']:.3f}倍SUPPORT；有效载荷最高{c['max_paired_payload_vs_SUPPORT']:.3f}倍；RSS {c['peak_RSS_bytes']/1024**2:.1f}MiB；{'通过' if c['pass'] else '未通过'}。")
    lines+=['',f'60任务DEV保守投影{dev/3600:.2f}小时，条件性700任务全程序{total/3600:.2f}小时。此投影取最差任务或循环单位开销、两倍保守因子与五分钟余量；不是保证。超过单次8小时准入时按整任务边界暂停，并需明确授权续会话，10小时为硬上限。估计归档及审核空间{artifact/1024**2:.1f}MiB。',
      '',f'资格状态：{status}。已提交技术外部记录89600、805条相关状态轨迹；公开单元序列与恢复测试另计，不混入科学样本。技术错误如有记录在FINAL_AUDIT.json和各循环目录，不能转写为科学阴性。',
      '','下一步仅在独立授权后运行12全新世界DEV；满足固定门后一次128全新世界确认。采用须相对COV-SUPPORT与COV-R同时通过5pp主增益及因果、保持、保护和成本门。未晋级仍保留用户默认SUPPORT，不通过换剂量、补样本或晋级诊断对照救当前配方。',
      '', '设计见[EXPERIMENT.md](../addition_uncertainty_restoration_v1_design_20261011/EXPERIMENT.md)；实际源锁和资格记录为SOURCE_LOCK_FINAL.json、QUALIFICATION.json、FINAL_AUDIT.json。本次不上传GitHub，不启动正式后台实验。']
    (config.ROOT/'THREE_CYCLE_REPORT.md').write_text('\n'.join(lines)+'\n')
    archive=config.ROOT/'RESULT_BUNDLE.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(config.ROOT.rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts and p.name not in ('RESULT_BUNDLE.zip','PACKAGE.json'):z.write(p,p.relative_to(config.ROOT))
        for p in sorted(config.DESIGN.glob('*')):
            if p.is_file():z.write(p,'design/'+p.name)
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise AssertionError('zip integrity')
    atomic_json({'archive':archive.name,'bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'science_jobs':0},config.ROOT/'PACKAGE.json')
    print(json.dumps(qualification),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--cycle',type=int);parser.add_argument('--finish',action='store_true');args=parser.parse_args()
    if args.finish:finish()
    elif args.cycle:execute(args.cycle)
    else:raise ValueError('choose cycle or finish')
