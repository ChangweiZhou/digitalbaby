"""Three design/audit/technical test/revise cycles, not efficacy science."""
import argparse, copy, hashlib, json, os, shutil, subprocess, time, traceback, zipfile
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import config
from runner import atomic_json,run,commit,read,audit
from technical import suite,tamper_suite
from operations import suite as operations_suite
from decision import metrics,countermodels,synthetic_suite

def one(task,arm,folder):
    world,family,noise,ordinary=task
    path=folder/'receipts'/f'{world}_{arm}.json.gz';log=folder/'logs'/f'{world}_{arm}.log'
    log.parent.mkdir(exist_ok=True)
    command=[os.sys.executable,str(config.ROOT/'runner.py'),'--world',str(world),'--family',family,'--noise',noise,'--arm',arm,'--output',str(path)]
    if ordinary:command.append('--ordinary')
    with log.open('w') as stream:
        process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        code=process.wait(timeout=600)
    if code:raise RuntimeError(f'technical worker failed {log}')
    receipt=read(path)
    # Worker already independently audited the serialized receipt before atomic acceptance.
    certificate=json.loads(log.read_text().strip().splitlines()[-1])
    if certificate['verdict']!='PASS':raise AssertionError('unaccepted receipt')
    m=metrics(receipt)
    # Do not report technical efficacy or use it to select/tune recipes.
    resources={k:m[k] for k in ('production_cpu_seconds','payload_bytes','peak_rss_bytes','external_updates','extra_updates')}
    events={'old_revisions':0,'new_revisions':0,'diagonal_revisions':0,'off_diagonal_revisions':0,
            'old_first_revision':None,'new_first_revision':None,'majority_events':0,'couplings':0,'replays':0,'gate_activated':0,'distinct_replayed_occurrences':0}
    selected=set()
    def scan(rows,phase):
        for index,row in enumerate(rows,1):
            for e in row['events']:
                u=e['update']
                if not u or not u['performed'] or not u['mechanism']:continue
                v=u['mechanism']
                if arm.startswith('RETRO'):
                    events['majority_events']+=int(v['mode'] is not None)
                    n=len(v['changes']);events[phase+'_revisions']+=n
                    if n and events[phase+'_first_revision'] is None:events[phase+'_first_revision']=index
                    raw=bytes.fromhex(row['raw']);events['diagonal_revisions' if raw[0]==raw[2] else 'off_diagonal_revisions']+=n
                elif arm.startswith('ROLE'):events['couplings']+=int(v['eta']>0)*len(v['coupled'])
                elif arm.startswith('COVER'):
                    events['replays']+=int(u['extra_performed'])
                    if v['selected_arrival'] is not None:selected.add(v['selected_arrival'])
                elif arm.startswith('KCI'):events['gate_activated']+=int(any(g<1 for g in v['g']))
    # Reachability on one operative W chronology, not totals of correlated forks.
    branch=receipt['branches']['W'];scan(branch['old_records'],'old');scan(branch['forks']['320']['new_records'],'new')
    events['distinct_replayed_occurrences']=len(selected)
    filebytes=path.stat().st_size
    return dict(world=world,family=family,noise=noise,ordinary=ordinary,arm=arm,receipt=str(path.relative_to(config.ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),receipt_bytes=filebytes,wall_seconds=receipt['wall_seconds'],resources=resources,events=events,audit=certificate)

def execute(cycle):
    folder=config.ROOT/'cycles'/f'cycle{cycle}';folder.mkdir(parents=True,exist_ok=False)
    source=config.lock();source_dir=folder/'source'
    for name in source:
        src=config.ROOT.parent/name;dst=source_dir/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    atomic_json(dict(files=source,science_authorized=False),folder/'SOURCE_LOCK.json')
    (folder/'DESIGN.md').write_text(f'# Technical cycle {cycle}\n\nGoal unchanged: fewer noisy/missing external samples for unseen-addition own-output generalization; OLD protected.\n\nThis cycle uses public reserved fixtures and all 13 locked operations/peers; no scientific selection.\n')
    started=time.monotonic();jobs=[];caff=None
    try:
        caff=subprocess.Popen(['caffeinate','-i','-w',str(os.getpid())],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        if caff.poll() is not None:raise RuntimeError('caffeinate failed')
        local=suite(folder/'technical',cycle);atomic_json(local,folder/'LOCAL_AUDIT.json')
        stat=synthetic_suite();atomic_json(stat,folder/'DECISION_AUDIT.json')
        op=operations_suite(folder/'operations',checkpoint_folder=folder/'technical',expected=local['arms'])
        atomic_json(op,folder/'OPERATIONS_AUDIT.json')
        tasks=[(t,a) for t in config.TECH_PLAN[cycle] for a in config.ARMS]
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(one,t,a,folder) for t,a in tasks]
            for future in as_completed(futures):
                jobs.append(future.result());atomic_json(dict(stage='TECHNICAL_RUNNING',cycle=cycle,committed_jobs=len(jobs),planned_jobs=len(tasks),science_worlds=0),folder/'STATUS.json')
                print(json.dumps(dict(cycle=cycle,committed=len(jobs),planned=len(tasks),arm=jobs[-1]['arm'])),flush=True)
        if source!=config.lock():raise AssertionError('source changed during cycle')
        tamper=[]
        # Each candidate's real auxiliary event and all common ledger/probe cases are tampered.
        for arm in config.BASELINES+config.CANDIDATES:
            job=next(j for j in jobs if j['arm']==arm)
            check=tamper_suite(read(config.ROOT/job['receipt']));tamper.append(dict(arm=arm,**check))
        atomic_json(tamper,folder/'TAMPER_AUDIT.json')
        result=dict(verdict='PASS_TECHNICAL_CYCLE',cycle=cycle,technical_jobs=len(jobs),external_records=len(jobs)*2560,correlated_state_trajectories=len(jobs)*23,science_worlds=0,selected_candidate=None,workers=2,caffeinate=True,wall_seconds=time.monotonic()-started,jobs=jobs)
        atomic_json(result,folder/'SUMMARY.json');atomic_json(dict(stage='TECHNICAL_CYCLE_COMPLETE',committed_jobs=len(jobs),science_worlds=0),folder/'STATUS.json')
        (folder/'AUDIT.md').write_text('All 13 operator/peer transition paths and complete 2,560-record causal branch jobs were independently reconstructed before receipt acceptance. Missing/forbidden state, clone/erase, cold process recovery, numeric bounds, prospective decision failures and receipt tampering were checked. Technical output scores did not select a candidate.\n')
        (folder/'REVISION.md').write_text('Cycle completed. Preserve these source snapshots. Subsequent changes require the next cycle to qualify them; no previous efficacy contract or result is changed. Resources and actual trigger reachability are carried forward.\n')
        return result
    except BaseException as error:
        failure=dict(stage='TECHNICAL_FAILURE',error=repr(error),traceback=traceback.format_exc(),committed_jobs=len(jobs),science_worlds=0)
        atomic_json(failure,folder/'FAILURE.json');raise
    finally:
        if caff is not None:caff.terminate();caff.wait(timeout=10)

def finish():
    summaries=[json.loads((config.ROOT/'cycles'/f'cycle{i}'/'SUMMARY.json').read_text()) for i in (1,2,3)]
    if [s['technical_jobs'] for s in summaries]!=[13,26,78] or any(s['verdict']!='PASS_TECHNICAL_CYCLE' for s in summaries):raise AssertionError('three complete cycles')
    source=config.lock()
    if json.loads((config.ROOT/'cycles/cycle3/SOURCE_LOCK.json').read_text())['files']!=source:raise AssertionError('qualified final sources changed')
    jobs=[j for s in summaries for j in s['jobs']]
    groups={}
    for j in jobs:
        p=config.ROOT/j['receipt']
        if hashlib.sha256(p.read_bytes()).hexdigest()!=j['sha256'] or j['audit']['verdict']!='PASS':raise AssertionError('committed certificate')
        groups.setdefault(j['world'],{})[j['arm']]=j
    if any(set(g)!=set(config.ARMS) for g in groups.values()):raise AssertionError('paired roster')
    costs={}
    for arm in config.CANDIDATES+tuple(a for p in config.PEERS.values() for a in p):
        production=sum(g[arm]['resources']['production_cpu_seconds'] for g in groups.values())
        baseline=sum(g['COV-SUPPORT']['resources']['production_cpu_seconds'] for g in groups.values())
        payload=max(g[arm]['resources']['payload_bytes']/g['COV-SUPPORT']['resources']['payload_bytes'] for g in groups.values())
        rss=max(g[arm]['resources']['peak_rss_bytes'] for g in groups.values())
        costs[arm]=dict(production_CPU_ratio=production/baseline,max_paired_payload_ratio=payload,peak_RSS_bytes=rss,resource_gate_pass=production/baseline<=1.5 and payload<=2 and rss<=512*1024**2)
    measured=max(max(j['wall_seconds'] for j in jobs),max(s['wall_seconds']/s['technical_jobs'] for s in summaries))
    conservative=2*measured+2.
    projected=712*conservative+300
    eligibility={a:all(costs[v]['resource_gate_pass'] for v in (a,)+config.PEERS[a]) for a in config.CANDIDATES}
    from pipeline import suite as finish_suite
    pipe=finish_suite(config.ROOT/'finish_pipeline')
    qualification=dict(status='TECHNICAL_COMPLETE_SCIENCE_NOT_AUTHORIZED',science_worlds=0,selected_candidate=None,technical_jobs=117,external_records=299520,correlated_state_trajectories=2691,resources=costs,resource_eligible_candidate_families=eligibility,conservative_seconds_per_job=conservative,DEV_hours=72*conservative/3600+300/3600,max_conditional_hours=projected/3600,workers=2,source=source,pipeline=pipe)
    atomic_json(qualification,config.ROOT/'QUALIFICATION.json')
    atomic_json(dict(verdict='PASS_TECHNICAL_SCOPE_ONLY',accepted_job_certificates=117,independent_record_transitions=299520,source_identity_verified=True,science_worlds=0,resource_eligibility=eligibility),config.ROOT/'FINAL_AUDIT.json')
    atomic_json(dict(files=source,science_authorized=False),config.ROOT/'SOURCE_LOCK_FINAL.json')
    lines=['# 四机制三次设计—审核—试跑—修订','', '目标保持：用更少带噪或缺失答案样本，提高未见加法组合的自身输出泛化，并保护旧能力。这里只进行了技术试跑，没有DEV/确认效能结果。','', '| 周期 | 完整技术任务 | 外部记录 | 状态 |','|---|---:|---:|---|']
    for s in summaries:lines.append(f"| {s['cycle']} | {s['technical_jobs']} | {s['external_records']} | PASS |")
    lines+=['','共117个完整技术任务、299520条外部记录；每任务23条相关状态轨迹，不能当作独立生命或世界。两基线、四候选和七个诊断对照全部经过同一字节与分支框架。独立审计重建每条更新；缺失/禁写、实际输出先于答案、完整辅助状态、冷恢复和篡改拒绝均通过。','', '候选：U047覆盖重放、M034旧观察重加权、M077/M105数据门控角色耦合、U055局部写入抑制工程代理。没有合并机制，也没有把U055当作真实递归神经回路或Full151本身的验证。','','| 配方 | CPU / SUPPORT | 最大载荷 / SUPPORT | 成本门 |','|---|---:|---:|---|']
    for arm,cost in costs.items():lines.append(f"| {arm} | {cost['production_CPU_ratio']:.3f} | {cost['max_paired_payload_ratio']:.3f} | {'PASS' if cost['resource_gate_pass'] else 'FAIL'} |")
    lines+=['',f"两worker下，DEV保守估计{qualification['DEV_hours']:.2f}小时；最大条件程序{qualification['max_conditional_hours']:.2f}小时。8小时准入、10小时硬限仍有效；不是保证。",'', '成本未过门的配方不是科学阴性；不通过改门槛或替换参数宣称通过。候选名单是否按既定规则缩减，需在正式DEV前明确记录。技术行为分数没有用于选择或调参。','', '设计：[EXPERIMENT.md](../addition_multimechanism_v1_design_20261011/EXPERIMENT.md)。资格、源锁、逐周期记录均保存在本目录。没有启动正式后台实验或上传GitHub。']
    (config.ROOT/'THREE_CYCLE_REPORT.md').write_text('\n'.join(lines)+'\n')
    with zipfile.ZipFile(config.ROOT/'QUALIFICATION_BUNDLE.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in ('THREE_CYCLE_REPORT.md','QUALIFICATION.json','FINAL_AUDIT.json','SOURCE_LOCK_FINAL.json'):
            z.write(config.ROOT/name,name)
        for p in config.ROOT.glob('*.py'):z.write(p,p.name)
        for p in config.DESIGN.glob('*'):
            if p.is_file():z.write(p,'design/'+p.name)
    atomic_json(dict(status='COMPLETE',science_worlds=0,bundle='QUALIFICATION_BUNDLE.zip'),config.ROOT/'PACKAGE.json')
    return qualification

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--cycle',type=int,choices=(1,2,3));parser.add_argument('--finish',action='store_true');a=parser.parse_args()
    print(json.dumps(finish() if a.finish else execute(a.cycle),sort_keys=True),flush=True)
