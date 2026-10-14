"""Assemble the completed qualification record; no trajectory execution."""
import hashlib,json,shutil,zipfile
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parent
import bridge_runtime as rt
import local_integrity as li

def main():
    ident=li.locked_identity();c3=json.loads((ROOT/'cycles/cycle3/TEST_RESULT.json').read_text())
    assert c3['verdict']=='PASS' and c3['source_identity']==ident
    guard=False
    with patch.object(li,'identity',return_value='stale-sources'):
        try:li.locked_identity()
        except ValueError:guard=True
    assert guard
    rt.atomic_json(ROOT/'cycles/cycle3/LOCK_GUARD_TEST.json',{'stale_lock_rejected_before_birth':True,'new_learning_records':0})
    # ERROR emission already occurs in native predict. Additional rescore CPU is
    # diagnostic and must not artificially inflate the reference budget.
    full=c3['full_assays'];ratio=max(1+v['W_policy_cpu_s']['Q_EARNED']/v['W_online_cpu_s'] for v in full.values())
    cpbytes={assay:(ROOT/'cycles/cycle3/full'/assay/'checkpoints'/f"{v['world']}_ERROR_{assay}.npz").stat().st_size for assay,v in full.items()}
    receipts=c3['forecast']['receipts_bytes'];checkpoints=sum(cpbytes.values())*64
    staged=2*(receipts+checkpoints)+8*max(cpbytes.values())
    q={'schema':'EARNED_AUTHORITY_QUALIFICATION_V1','source_identity':ident,'technical_verdict':'PASS',
       'scientific_verdict':'NOT_DEMONSTRATED','formal_launch_recommendation':'DO_NOT_AUTOMATICALLY_LAUNCH',
       'reason':'Fresh Latin development world showed a >5pp taught-old loss; no above-chance held-out result. This is a pilot warning, not a population failure proof. Full Latin raw diagnostic coverage is missing.',
       'production_dispatcher_released':False,'formal_science_jobs':0,
       'unit_and_native_test_methods':14,'full_technical_jobs':3,'full_technical_lives':9,'full_teacher_records':5040,
       'short_completed_jobs':18,'short_completed_lives':18,'short_teacher_records':80,
       'eight_worker_parity_worlds':8,'SIGTERM_checkpoint_record':c3['SIGTERM_pause_record'],
       'independent_full_receipt_tampers_rejected':9,'source_lock_guard_pass':guard,
       'resource_forecast':{'formal_worlds':64,'formal_jobs':192,'formal_lives':576,'formal_teacher_records':322560,
          'eight_worker_ideal_wall_hours':c3['forecast']['eight_workers_wall_s_with_4x_allowance']/4/3600,
          'eight_worker_wall_hours_with_4x_allowance':c3['forecast']['eight_workers_wall_s_with_4x_allowance']/3600,
          'receipts_bytes':receipts,'retained_checkpoints_bytes':checkpoints,'staging_and_package_allowance_bytes':staged,
          'aggregate_peak_rss_bytes':c3['forecast']['aggregate_8worker_peak_rss_bytes'],
          'corrected_online_cpu_ratio_vs_native_ERROR':ratio,'not_a_hard_runtime_guarantee':True},
       'accounting_revision':'Native ERROR output is already in online_api CPU. ERROR rescore CPU is diagnostic, excluded from the reference denominator. No records rerun for this correction.'}
    assert staged<4*1024**3 and ratio<=1.05
    rt.atomic_json(ROOT/'QUALIFICATION.json',q)
    # Final audit checks final file identity and the complete artifact inventory,
    # not another validation run of the historical science experiment.
    rt.atomic_json(ROOT/'FINAL_AUDIT.json',{'verdict':'ACCEPTED_THREE_CYCLE_TECHNICAL_EXERCISE',
        'source_identity':ident,'source_map_matches_lock':li.source_map()==json.loads((ROOT/'SOURCE_LOCK.json').read_text())['files'],
        'all_cycles_completed':True,'frozen_upstream_sources_unchanged':True,'formal_science_jobs':0,
        'candidate_adopted':False,'new_learning_rule':False,'no_threshold_tournament':True,
        'limitations':['No 64-world raw Latin localization','Binary outputs essentially Q_HALF','Only three full DEV worlds, one per assay','Formal dispatcher not released'],
        'scientific_status':'No demonstrated persistent-core improvement; do not equate technical pass with adoption.'})
    md=f'''# 三次循环完成报告 — earned authority V1

三次「设计—审核—试跑—修订」已完成。技术资格通过；**没有启动正式实验，也没有证明候选改善了核心**。不建议据此自动启动 64-world 确认，更不应该把置信差值当成可靠性的证明。

| 循环 | 发现与修订 | 实际验证 |
|---|---|---|
| 1 | 二选一不能逐选项使用四通道的 gate；改为整个 query 共用一个权重，保留源代码浮点运算 | 7 项单元测试；76,128 条旧收据记录的 ERROR/Q_HALF 精确读出一致性 |
| 2 | shared 已占优势时减半 private 是数学上无效的；改为固定半权重能翻转选择的严格 margin window | 7 项原生集成测试；12 种阶段/禁写组合、独立存储体、答案前输出、ERROR 状态一致性、非法输入拒绝 |
| 3 | 用完整生命周期与实际中断检验，而非只看短 smoke；修正成本分母，不把 diagnostic rescore 计入 ERROR 生产成本 | 3 个完整 DEV jobs、9 lives、5,040 条教学记录；8-worker 状态逐一等于串行；真实 SIGTERM 在第 {c3['SIGTERM_pause_record']} 条记录后安全封存并无损恢复；9/9 篡改被拒绝 |

另有 18 个完成的短资格 jobs（18 条短 lives，共 80 条记录），用于并发和恢复等价性；它们不是正式科学样本。暂停片段在恢复后继续，未重复执行已封存的记录。没有新旧正式轨迹重跑。

## 对候选的判断

原稿两处缺陷已关闭：无关的二选一通道不会决定权重；新 gate 不再是空操作。但有一个边界必须写明：除了精确平局/边界，新规则的二选一输出与 Q_HALF 相同。它不是一种新的二选一泛化机制。可能收益仅是四输出任务中的选择性保留。

旧 48-world 收据的离线重算中，候选继承 Q_HALF 的 binary W−N +8.68 pp，且 lifetime W 的旧/新/修订均值与 ERROR 相同。这是事后描述，不是新确认。

第三轮实际 Latin DEV world 62010003：旧题 ERROR 9/12（75%），候选 8/12（66.7%）；新题均为 12/16（75%）；未教题 ERROR 0/4，候选 1/4（25%），候选 W−N=0。这只有一个开发世界，不能否定整个家族；但已经暴露了旧记忆损伤，也没有展示所要求的 Latin 正向 reuse。没有再按这些答案调规则或补选新参数。

因此，本轮结论是 **技术验证完成，候选科学效果仍未通过验收**。若用户今后仍选择运行这个负面结果也有价值的假设检验，应沿用固定协议；不能将这个 pilot 改写成“已解决整合”。当前交付不包含正式生产 dispatcher。

## 预算和资源

固定前瞻规模：64 个 fresh paired world vectors × 3 个已有 assay = **192 jobs、576 lives、322,560 次教学记录**。ERROR/Q_HALF/Q_EARNED 共用一份 ERROR 学习轨迹，成本不乘三。没有新题型、alphabet、autonomy 或 NLP 扩展。

本机 3 个完整资格 job 用时约：binary {full['reuse']['result']['process_wall_s']:.1f}s，Latin {full['latin']['result']['process_wall_s']:.1f}s，lifetime {full['lifetime']['result']['process_wall_s']:.1f}s。全规模 8 workers 理想外推 {q['resource_forecast']['eight_worker_ideal_wall_hours']:.2f}h；留四倍运行余量约 **{q['resource_forecast']['eight_worker_wall_hours_with_4x_allowance']:.2f}h**，不是硬保证。

正式收据约 {receipts/1e6:.0f}MB，保留的 checkpoint 约 {checkpoints/1e6:.0f}MB；加封存/打包及临时文件的预留约 **{staged/1024**3:.2f}GiB**，8-worker RSS 外推约 {q['resource_forecast']['aggregate_peak_rss_bytes']/1024**3:.2f}GiB。候选额外读出 CPU 的实测增幅约 {(ratio-1)*100:.3f}%；真正科学成本仍包括资格审计和全部因果分支。

## 审核保留下来的限制

- GitHub 仅可取得一个 R_center 正式 Latin world 和一个 ERROR 技术 pilot 的逐题分值；完整 64-world Latin 向量没有公开。不能虚构为完整定位。
- +4.86 pp 是此前 causal W−N 对比的改善；原始准确率改善只有 +3.47 pp。
- fixed rule 没有学习“何时应信任哪条路”。gap 不是已校准的概率。
- 接口仍是四可见字节地址、四输出通道或给定二选一、训练时到来的答案字节。不宣称无教学学习、开放输出或智能。
- 联合 adoption 仍要求两类 E3、旧/新/修订保护及效率同时合格；一个子任务成功不能替代全部目标。负面结论保留旧核心。

详细文件：`EXPERIMENT.md`、`PROTOCOL_LOCK.json`、`QUALIFICATION.json`、`SOURCE_LOCK.json`、`FINAL_AUDIT.json`，以及每轮目录中的审核、日志和收据。
'''
    (ROOT/'THREE_CYCLE_REPORT.md').write_text(md)
    third='''# Cycle 3 — full-life qualification, audit and final revision

Three complete DEV jobs exercise the lifetime, binary and Latin fixtures and all nine lives. Eight short parallel jobs exactly match their eight serial twins. Real SIGTERM occurs after record 2, safely seals the record, and the resumed job matches its uninterrupted technical twin. Checkpoint corruption and wrong identity fail. Complete-receipt tampering in branch/write/birth/teacher/probe count/clock/fixture/readout/output-order all fails; completed redispatch performs no learning. A stale source lock is rejected before birth.

Final accounting revision: source ERROR already emits during native prediction. Its extra diagnostic rescoring must not be included in the production reference denominator. Recompute candidate CPU/native CPU from saved timers. Include retained checkpoints, temporary writes and packaging in disk forecasts. No trajectories were rerun for either correction.

Technical result PASS; scientific result NOT_DEMONSTRATED. The Latin pilot exposes an 8.33pp old-cue loss and no above-chance held-out result. No candidate weights were tuned on this outcome. Reject an automatic formal-launch recommendation; retain the frozen prospective hypothesis and its limits, rather than declaring a breakthrough or opening another parameter tournament.
'''
    (ROOT/'cycles/cycle3/DESIGN_AUDIT_REVISION.md').write_text(third)
    doc=ROOT.parents[1]/'PROJECT_DOCUMENTS/MINIFLY_EARNED_AUTHORITY_V1_20261007';doc.mkdir(parents=True,exist_ok=True)
    for name in ('EXPERIMENT.md','THREE_CYCLE_REPORT.md','PROTOCOL_LOCK.json','QUALIFICATION.json','SOURCE_LOCK.json','FINAL_AUDIT.json'):
        shutil.copy2(ROOT/name,doc/name)
    # Review bundle preserves all new code, reports, receipts and checkpoints;
    # imported scientific assets are described by the lock, not misrepresented
    # as a self-contained deployment bundle.
    bundle=ROOT/'TECHNICAL_REVIEW_BUNDLE.zip'
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(ROOT.rglob('*')):
            if p.is_file() and p!=bundle and 'scratch' not in p.parts and '__pycache__' not in p.parts and p.name!='PACKAGE.json':z.write(p,str(p.relative_to(ROOT)))
    rt.atomic_json(ROOT/'PACKAGE.json',{'type':'technical_review_not_standalone_cloud_runner','path':str(bundle),
        'bytes':bundle.stat().st_size,'sha256':hashlib.sha256(bundle.read_bytes()).hexdigest(),'formal_science_jobs':0,
        'external_immutable_dependencies':'SOURCE_LOCK.json exact local paths/hashes; not copied into this review bundle'})
    print(json.dumps(q,indent=2))

if __name__=='__main__':main()
