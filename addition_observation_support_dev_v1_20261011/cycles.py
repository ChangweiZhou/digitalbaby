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
    from collections import defaultdict
    summaries=[json.loads((config.ROOT/'cycles'/('cycle'+str(c))/'SUMMARY.json').read_text()) for c in (1,2,3)]
    if [s['technical_jobs'] for s in summaries]!=[3,6,12] or any(s['verdict']!='PASS_TECHNICAL_CYCLE' for s in summaries):
        raise AssertionError('three complete fixed-roster cycles required')
    source=config.lock()
    if json.loads((config.ROOT/'cycles/cycle3/SOURCE_LOCK.json').read_text())['files']!=source:
        raise AssertionError('final qualified source changed')
    from pipeline import suite as pipeline_suite
    pipeline=pipeline_suite(config.ROOT/'finish_pipeline')
    jobs=[j for s in summaries for j in s['jobs']]
    if len({(j['world'],j['arm']) for j in jobs})!=21:raise AssertionError('duplicate technical jobs')
    groups=defaultdict(dict);certificates=0
    for j in jobs:
        path=config.ROOT/j['receipt']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=j['sha256'] or j['audit']['verdict']!='PASS':
            raise AssertionError('accepted receipt/certificate integrity')
        groups[j['world']][j['arm']]=j['metrics'];certificates+=1
    if any(set(g)!=set(config.ARMS) for g in groups.values()):raise AssertionError('paired full roster')
    costs={}
    for arm in config.ARMS[1:]:
        base_cpu=sum(g['COV-R']['production_cpu_seconds'] for g in groups.values())
        arm_cpu=sum(g[arm]['production_cpu_seconds'] for g in groups.values())
        payload=max(g[arm]['payload_bytes']/g['COV-R']['payload_bytes'] for g in groups.values())
        rss=max(g[arm]['peak_rss_bytes'] for g in groups.values())
        costs[arm]={'production_cpu_ratio':arm_cpu/base_cpu,'max_paired_payload_ratio':payload,
                   'peak_rss_bytes':rss,'pass':arm_cpu/base_cpu<=1.5 and payload<=2 and rss<=512*1024**2}
    # Final fair-control check uses accepted snapshots, never replays a training record.
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
    auxiliary_checks=0
    for world in groups:
        def job(arm):return next(j for j in jobs if (j['world'],j['arm'])==(world,arm))
        a=list(checkpoints(read(config.ROOT/job('COV-SUPPORT')['receipt'])))
        b=list(checkpoints(read(config.ROOT/job('COV-RATE')['receipt'])))
        if len(a)!=len(b):raise AssertionError('paired snapshot counts')
        for sa,sb in zip(a,b):
            if any(sa[k]!=sb[k] for k in ('cache','history','rng','updates','extra_updates')):
                raise AssertionError('paired actual-history support/RATE auxiliary mismatch')
            auxiliary_checks+=1
    worst=max(j['wall_seconds'] for j in jobs)
    cyclecost=max(s['cycle_wall_seconds']/s['technical_jobs'] for s in summaries)
    projected=config.CONTRACT['max_jobs']*max(worst,cyclecost)*2+300
    artifact=max(j['receipt_bytes'] for j in jobs)*config.CONTRACT['max_jobs']*3+20*1024**2
    resource=projected<=8*3600 and artifact<=2*1024**3 and all(x['pass'] for x in costs.values())
    status='TECHNICAL_QUALIFIED_SCIENCE_NOT_AUTHORIZED' if resource else 'RESOURCE_QUALIFICATION_HOLD'
    qualification={'status':status,'all_three_arms':True,'cycles':3,'technical_jobs':21,
        'technical_external_records':21*2560,'correlated_state_trajectories':21*23,
        'science_jobs':0,'DEV_worlds':0,'CONF_worlds':0,'selected_candidate':None,'science_authorized':False,
        'conservative_maximum_420_job_wall_hours':projected/3600,
        'conservative_science_artifact_bytes':artifact,'resource_pass':resource,
        'measured_paired_resources':costs,'source':source,'pipeline':pipeline,
        'paired_auxiliary_snapshot_checks':auxiliary_checks,'exact_rate_matched':False}
    atomic_json(qualification,config.ROOT/'QUALIFICATION.json')
    atomic_json({'files':source,'science_authorized':False},config.ROOT/'SOURCE_LOCK_FINAL.json')
    tamper=sum(json.loads(p.read_text())['rejected'].__len__() for p in (config.ROOT/'cycles/cycle2/tamper').glob('*.json'))
    failures=[]
    for p in sorted((config.ROOT/'cycles').glob('*failure/FAILURE.json')):
        value=json.loads(p.read_text());failures.append({'record':str(p.relative_to(config.ROOT)),
            'error':value['error'],'committed_jobs':value['committed_technical_jobs'],'science_jobs':0})
    audit={'verdict':'PASS_THREE_TECHNICAL_CYCLES','jobs':21,'records':21*2560,
        'accepted_receipt_certificates_verified':certificates,'tamper_cases_rejected':tamper,
        'source_parity':True,'paired_auxiliary_snapshot_checks':auxiliary_checks,
        'historical_preflight_failures':failures,'resource_status':status,'science_jobs':0}
    atomic_json(audit,config.ROOT/'FINAL_AUDIT.json')
    atomic_json({'stage':status,'technical_jobs':21,'science_jobs':0},config.ROOT/'STATUS.json')
    lines=['# U042三次设计审核试跑修订报告','',
        '目标保持为：给定相同带噪外部样本，让自身输出在未教加法组合上获得更多能力。三次循环完成工程资格化；没有运行正式DEV或确认，没有选择或采用新核心。','',
        '| 循环 | 完整技术任务 | 外部记录 | 执行方式 |','|---|---:|---:|---|']
    for s in summaries:
        lines.append(f"| {s['cycle']} | {s['technical_jobs']} | {s['external_training_records']} | "+('两个脱离终端worker' if s['cycle']==3 else '串行技术任务')+' |')
    lines += ['', '## 固定设计与审核结果','',
        'COV-R、COV-SUPPORT及诊断COV-RATE共三配置。31维输入、加性读出、反馈比例、320条OLD历史、NEW20/40/80、5pp升级门槛及保持/成本保护均未改变。只重建门控和辅助状态，没有调整窗口8、键数32、历史32、严格多数至少2票或冲突因子1/4。',
        '', '循环1验证独立更新方程、与已采用COV-R的240次数值/输出一致性、全部因果分支、严格过去多数、错误多数后窗口更新、FIFO、两方向不共享及w/P使用相同写入因子。',
        '', '循环2增加允许但缺失目标时的完整惰性、查询不读取答案表、几何可解性、冷进程恢复和收据篡改测试。几何预检两次失败：JSON列表需要转换为可哈希边，随后(2,0)需转为环境侧规范化无向边；两次均在完整任务提交前停止，已保留source/traceback，学习键仍为有序字节。它们是技术接口修订，不是科学阴性结果。',
        '', f'循环2拒绝{tamper}个篡改案例，涵盖禁写权限、P、缓存、H、RNG、当前支持决定、随机门、输出、源文件、记录计数及时间。循环3覆盖PATH_LOOP/STAR_LOOP × STANDARD/BLURRY，两个脱离终端worker，caffeinate跟随supervisor。',
        '', f'最终按已接受检查点做{auxiliary_checks}次配对辅助状态比较，COV-SUPPORT/COV-RATE的缓存、H、RNG及更新计数完全一致；没有重跑记录或轨迹。其w/P因门控不同可以不同。全部21份提交收据和审核证书核对通过。',
        '', '## 不应扩大解释','',
        '正式TRIANGLE_TAIL前20条NEW记录的支持门必为1，最早第21条才有直接降低NEW反馈强度的机会。公开几何槽位的边界测试通过，但未运行正式图学习任务。20样本能力若变化，需要结合OLD形成的状态解释。随机门只跟随过去频率，不能宣称逐世界精确等预算。',
        '', '公开合成序列中的窗口多数能够修订，不能保证真实噪声下改善泛化或避免错误多数。当前结果是技术资格化，不是U042有效性证据。固定候选需在全新DEV和一次确认中接受失败；不能靠增加样本、变更目标或调参数救当前配方。',
        '', '## 资源与后续执行','']
    for arm,c in costs.items():
        lines.append(f"- {arm}：生产CPU为参考{c['production_cpu_ratio']:.3f}倍；最大配对可变载荷{c['max_paired_payload_ratio']:.3f}倍；RSS峰值{c['peak_rss_bytes']/1024**2:.1f}MiB；资源检查{'通过' if c['pass'] else '未通过'}。")
    lines += ['',f'条件性420任务全程序保守投影{projected/3600:.2f}小时，收据/审核等预估{artifact/1024**2:.1f}MiB。使用最差完整任务/循环开销、二倍保守因子和五分钟启动余量；不是保证。状态：{status}。',
        '', '完整技术任务合计53760外部记录、483条共享前缀相关轨迹；单元测试、恢复续接及失败预检的合成工作另计，不混入正式样本。循环报告不把技术任务的J值当作正式效能检验。',
        '', '已测试缓存式状态监视、并发提交、不完整/失败打包拒绝、安全会话准入、负面选择与确认判定及归档完整性。三个核仍是工程加性学习器，不是未修改Full151的算术证据。',
        '', '下一步只能在独立授权后启动固定三配置的12新世界DEV；最多一个候选按原门槛推进128新世界确认，条件性最多420任务/1075200记录。初始两个后台worker、supervisor附caffeinate、8小时准入目标及10小时会话上限。无自动重启、追加样本、Github上传或正式实验。',
        '', '实验设计见相邻addition_observation_support_v1_design_20261011/EXPERIMENT.md；最终资格化与源锁见QUALIFICATION.json、SOURCE_LOCK_FINAL.json、FINAL_AUDIT.json。']
    (config.ROOT/'THREE_CYCLE_REPORT.md').write_text('\n'.join(lines)+'\n')
    archive=config.ROOT/'RESULT_BUNDLE.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(config.ROOT.rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts and p.name not in ('RESULT_BUNDLE.zip','PACKAGE.json'):
                z.write(p,p.relative_to(config.ROOT))
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise AssertionError('archive integrity')
    atomic_json({'status':'COMPLETE_TECHNICAL_BUNDLE','sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
        'archive_bytes':archive.stat().st_size,'science_jobs':0,'resource_status':status},config.ROOT/'PACKAGE.json')
    print(json.dumps(qualification),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cycle',type=int);p.add_argument('--finish',action='store_true');a=p.parse_args()
    if a.finish:finish()
    elif a.cycle is not None:execute(a.cycle)
    else:raise ValueError('explicit technical exercise only')
