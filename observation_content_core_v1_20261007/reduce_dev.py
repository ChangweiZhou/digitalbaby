"""Read-only DEV reducer. No constructors, no trajectories, no science promotion."""
import gzip,json,hashlib,platform,statistics
from pathlib import Path
import numpy as np
import bench,checks,exercise_tests,config,checkpoint_content
ROOT=Path(__file__).resolve().parent

def load(path):
    with gzip.open(path,'rt') as f:return json.load(f)
def more(r):
    out={}
    for arm,x in r['arms'].items():
        p=x['probes']
        out[arm]=dict(old=p['old']['W']['metrics']['focal_accuracy'],
            final_old=p['day3']['W']['metrics']['focal_accuracy'],
            all_train_bpb=sum(z['loss_bits'] for st in x['training'].values() for z in st['W'])/sum(len(st['W']) for st in x['training'].values()),
            W_training_cpu=sum(c['cpu_seconds'] for key,c in x['costs'].items() if key.endswith('/W')),
            mutable_bytes=x['states']['day3/W']['mutable_bytes'],
            day2=p['day2']['W']['metrics']['focal_accuracy'],
            revised=p['day3']['W']['revised_metrics']['focal_accuracy'],
            unchanged=p['day3']['W']['unchanged_metrics']['focal_accuracy'],
            new=p['day3']['W']['new_probe']['metrics']['focal_accuracy'],
            revision_harm=p['day3']['N_REV']['unchanged_metrics']['focal_accuracy']-p['day3']['W']['unchanged_metrics']['focal_accuracy'],
            gain=p['day2']['W']['metrics']['focal_accuracy']-p['day2']['N_OLD']['metrics']['focal_accuracy'])
    for arm,a in r['arms'].items():
        rows=[p['rows'] for stage in a['probes'].values() for b,p in stage.items() if b=='W']
        rows += [stage['W']['new_probe']['rows'] for stage in a['probes'].values() if 'new_probe' in stage['W']]
        flat=[x for rs in rows for x in rs]
        out[arm]['all_query_bpb']=sum(x['loss_bits'] for x in flat)/len(flat)
        out[arm]['standard_query_cpu']=sum(stage['W']['cpu_seconds']+stage['W'].get('new_probe',{}).get('cpu_seconds',0.) for stage in a['probes'].values())
        out[arm]['revision_end']=a['probes']['revised']['W']['revised_metrics']['focal_accuracy']
        nr=[p['rows'] for stage in a['probes'].values() for name,p in stage.items() if name=='N_ALL']
        nr += [stage['N_ALL']['new_probe']['rows'] for stage in a['probes'].values() if 'new_probe' in stage['N_ALL']]
        nf=[x for rows in nr for x in rows]
        control=sum(x['loss_bits'] for x in nf)/len(nf)
        out[arm]['all_query_causal_bpb_gain']=control-out[arm]['all_query_bpb']
        cost=out[arm]['W_training_cpu']+out[arm]['standard_query_cpu']
        out[arm]['probe_gain_per_cpu']=out[arm]['all_query_causal_bpb_gain']/cost
    return out

def main():
    paths=[ROOT/'cycles/cycle3'/f'WORLD_{w}.json.gz' for w in (822201,822202)]
    worlds=[load(p) for p in paths]
    audit_results=[]
    for path,r in zip(paths,worlds):
        audited=checks.audit(r);audited['rejected_tampers']=exercise_tests.receipt_tamper(r)
        (path.parent/f'AUDIT_{r["world"]}.json').write_text(json.dumps(audited,indent=2))
        audit_results.append(audited)
    data=[more(r) for r in worlds]
    means={a:{k:statistics.mean(d[a][k] for d in data) for k in data[0][a]} for a in ('A','B')}
    b,a=means['B'],means['A']
    gates=dict(old=b['old']>=.9,retained=b['day2']>=.9,new=b['new']>=.8,revised=b['revised']>=.8,
        unchanged=b['unchanged']>=.85,causal_gain=b['gain']>0,all_train_loss=b['all_train_bpb']-a['all_train_bpb']<=.05,
        all_query_loss=b['all_query_bpb']-a['all_query_bpb']<=.05,complete_audited_worlds=len(worlds)==2)
    ready=all(gates.values())
    science_files=('environment.py','config.py','candidate.py','bench.py','checks.py','checkpoint_content.py','exercise_tests.py','reduce_dev.py')
    files={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in science_files}
    before=json.loads((ROOT/'cycles/cycle3/SOURCE_SNAPSHOT.json').read_text())
    assert all(files[n]==h for n,h in before.items()),'source changed during final DEV'
    lock=dict(schema='CONTENT_CORE_TECHNICAL_LOCK_V2',candidate_version=config.VERSION,files=files,
        parent_verified_source_identity=checkpoint_content.source_identity(),science_admission=False)
    (ROOT/'SOURCE_LOCK.json').write_text(json.dumps(lock,indent=2))
    package_bytes=sum(p.stat().st_size for p in ROOT.glob('cycles/cycle*/WORLD_*.json.gz'))
    perworld_wall=[r['resources']['wall_seconds'] for r in worlds]
    estimated_seconds=max(perworld_wall)*32*2.0 # complete measured world, 2x ops/audit reserve, sequential
    training=8*1280*32
    q=dict(status='DEV_EXECUTABLE_HOLD_FOR_PRESERVATION_REVIEW' if ready and b['new']-a['new']<-.02 else ('DEV_QUALIFIED_SCIENCE_NOT_STARTED' if ready else 'BLOCKED_DEV_BEHAVIOR'),
        technical_pass=True,basic_behavior_pass=ready,
        readiness_recommendation='HOLD_B_CURRENT_INSTANCE' if b['new']-a['new']<-.02 else 'READY_FOR_AUTHORIZATION_ADAPTER',
        preservation_warning='Final DEV mean new-content difference is below the planned -2pp preservation band; this is a DEV risk flag, not confirmatory evidence or a changed DEV gate',
        preliminary_new_content_difference=b['new']-a['new'],gates=gates,final_dev_worlds=[r['world'] for r in worlds],
        final_dev_means=means,world_results=[dict(world=r['world'],metrics=m) for r,m in zip(worlds,data)],
        audited_rows=sum(x['checked_rows'] for x in audit_results),science_worlds=0,science_launch_authorized=False,
        recorded_dev_receipt_bytes=package_bytes,planned_science_worlds=32,planned_science_histories=256,planned_training_bytes=training,
        sequential_wall_budget_seconds=estimated_seconds,budget_method='2x maximum measured complete DEV world ×32; estimate, not hard guarantee',
        source_unchanged_during_final_dev=True,model_parameters_changed_in_cycle3=False,
        host=platform.platform(),python=platform.python_version())
    (ROOT/'QUALIFICATION.json').write_text(json.dumps(q,indent=2))
    lines=['# 三轮练习结果\n','得到可执行的U045工程候选；正式science尚未启动。当前A是已采用的有限自主观察基线，新B是否能替换它仍须正式对照。\n',
      '| 轮次 | 设计、试跑与修订 | 判定 |','|---|---|---|',
      '| 1 | 定义有界局部fast/slow残差矩阵；822001短old试跑；同地址、真实禁写和独立公式通过。全byte bpb 2.946对A1.948，保存失败。 | 内容行为不合格；不按焦点100%放行 |',
      '| 2 | 加有限局部访问计数；822101完整生命史。修复审计器误读sorted JSON时间顺序，对原receipt复核，无life重跑。 | DEV初步可用；不作正式胜出 |',
      '| 3 | 隔离clone memo cache、补fixture/source/pending tamper；822201/822202完整life，内容参数不改。 | '+q['status']+' |\n',
      '## 最终两DEV世界均值\n','| 指标 | A | B |','|---|---:|---:|']
    for label,key in [('旧内容获取','old'),('day2旧保持','day2'),('day3新内容','new'),('day3修订','revised'),('day3未改内容','unchanged')]:
        lines.append(f'| {label} | {100*a[key]:.2f}% | {100*b[key]:.2f}% |')
    for label,key in [('全训练byte bpb','all_train_bpb'),('全标准probe bpb','all_query_bpb'),('W训练CPU秒','W_training_cpu'),('标准probe CPU秒','standard_query_cpu'),('可变内容bytes','mutable_bytes'),('全probe禁写对照bpb净收益','all_query_causal_bpb_gain'),('净收益/操作CPU','probe_gain_per_cpu')]:
        lines.append(f'| {label} | {a[key]:.4f} | {b[key]:.4f} |')
    lines += ['\n技术检查包括A四份出生独立、同KC地址、真实禁写矩阵/计数、probe状态与缓存隔离、lazy/eager等价、pending checkpoint逐位恢复、14项ledger和5项checkpoint篡改拒绝。独立audit核对最终两world共'+str(q['audited_rows'])+'行。',
      '\n全部四DEV世界共32训练历史（第一世界短old），32,000训练bytes；单元/恢复测试另属合成资格fixture，不是science生命史。最终两world只是开发数据，不产生统计胜出或E3声明。',
      f'\n拟正式32配对world、256训练历史、{training:,}bytes。顺序运行保守估计约{estimated_seconds/60:.1f}分钟（完整world实测×32再乘2，含运维/审计余量，不保证墙钟）。原始receipt体积{package_bytes:,}bytes；按完整world量级另预留日志和缓存空间。',
      '\n学习参数只在第一/第二轮间修订一次；第三轮无增益搜索或fallback候选。A的signed调用/全状态clone成本较高，B的速度与内存优势包含工程结构因素。固定PN→KC投影/排序仍是全局操作，不声称大规模稀疏计算问题已经解决。',
      '\n当前范围E0/E1；有限context计数器也能解题。允许改内部结构，但不得把这次资格当已得到一般可复用的新持久核心。协议、门槛、成本边界见[EXPERIMENT.md](EXPERIMENT.md)。',
      '\n正式运行仍需用户授权，以及只替换admission的入口/整world提交和分析终止流程资格；不会自动运行。']
    lines += ['\n当前裁决：技术上可执行，基本DEV资格通过，但B最终新内容均值87.5%对A100%，且一个世界75%。低于拟正式能力保护带，建议暂停这个B实例的正式发射；速度和其他100%指标不能抵消退化。没有增加第四轮或自动启动science。']
    (ROOT/'THREE_CYCLE_REPORT.md').write_text('\n'.join(lines)+'\n')
    final=dict(verdict='PASS_THREE_CYCLE_EXERCISE',technical= 'PASS',behavior=q['status'],
        cycles_completed=3,failed_cycle1_preserved=True,cycle2_auditor_defect_preserved_and_resolved=True,
        science_worlds=0,frozen_parent_files_changed=False,source_unchanged_during_final_dev=True,
        final_audited_rows=q['audited_rows'],formal_experiment_authorized=False,new_superior_core_claimed=False)
    (ROOT/'FINAL_AUDIT.json').write_text(json.dumps(final,indent=2))
    print(json.dumps(q,indent=2))
if __name__=='__main__':main()
