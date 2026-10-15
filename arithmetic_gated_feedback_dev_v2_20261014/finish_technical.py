"""Operational three-cycle closeout; inspect accepted artifacts, never train."""
import hashlib
import json
import shutil
import statistics
import time
import zipfile
from pathlib import Path
import numpy as np
from protocol import ROOT, D, LOCAL_FILES, identity, atomic
from analysis import load, reduce_stage, write_report


def main():
    assert not (ROOT / 'QUALIFICATION.json').exists(), 'closeout already exists; do not repeat reductions'
    c3 = ROOT / 'cycles/cycle3'
    assert json.loads((c3 / 'STATUS.json').read_text())['status'] == 'EXECUTION_ACCEPTED'
    current = identity()
    assert current == json.loads((ROOT / 'SOURCE_LOCK.json').read_text())['source']
    all_rows, shape_checks = [], 0
    for stage in ('cycle2', 'cycle3'):
        folder = ROOT / 'cycles' / stage
        rows = load(folder)
        wanted = {(w, m) for w in D['technical'][stage]['worlds'] for m in ('KEEP', 'G62')}
        assert {(r['world'], r['model']) for r in rows} == wanted and len(rows) == len(wanted)
        lock = json.loads((folder / 'SOURCE_LOCK.json').read_text())['source']
        assert all(r['source'] == lock for r in rows)
        for f in LOCAL_FILES:
            assert hashlib.sha256((folder / 'source' / f).read_bytes()).hexdigest() == lock[f]
        for r in rows:
            width = 64 if r['model'] == 'KEEP' else 62
            shapes = {'E': (259, 24), 'V': (width + 1, 257), 'critic': (width + 1,)}
            for gate in ('z', 'r', 'g') if r['model'] == 'KEEP' else ('i', 'f', 'o', 'g'):
                shapes.update({'W' + gate: (width, 24), 'U' + gate: (width, width), 'b' + gate: (width,)})
            if r['model'] == 'KEEP':
                shapes['C'] = (64, 64)
            for relative in r['archives'].values():
                with np.load(ROOT / relative, allow_pickle=False) as arrays:
                    for name, shape in shapes.items():
                        assert arrays['p_' + name].shape == arrays['m_' + name].shape == shape
                        shape_checks += 2
        all_rows.extend(dict(world=r['world'], model=r['model'], executed_records=r['executed_records']) for r in rows)
        del rows, r
    assert len(all_rows) == 6 and sum(r['executed_records'] for r in all_rows) == 450560
    tamper = json.loads((ROOT / 'cycles/cycle2/TAMPER_TESTS.json').read_text())
    assert tamper['accepted'] and tamper['rejected'] == 21
    tests = {}
    for p in sorted((ROOT / 'cycles/cycle1').glob('TESTS_*.json')):
        r = json.loads(p.read_text())
        assert r['accepted']
        for t in r['tests']:
            assert t['accepted']
            tests[t['name']] = t
    assert len(tests) == 9
    result = reduce_stage(c3)
    atomic(c3 / 'SUMMARY.json', result)
    write_report(c3 / 'REPORT.md', result)
    atomic(ROOT / 'CSTAR_LOCK.json', dict(cstar=result['cstar'], rule=D['metrics']['Cstar_rule'],
        worlds=result['worlds'], source=current, frozen_before_DEV=True))
    verdict = ('TECHNICALLY_QUALIFIED_PENDING_DEV' if result['all20_descriptive_means_positive']
               else 'TECHNICALLY_QUALIFIED_EFFECT_INSUFFICIENT')
    review = dict(accepted=True, stage='cycle3', design='unchanged fixed full-dose worlds and candidate',
        audit='original independent per-job certificates, immutable source snapshots and additional exact archive shapes',
        tests=dict(native_jobs=4, records=442368, related_histories=28, archives=64),
        revise=['Freeze measured full-dose KEEP_OLD Cstar for any later authorized DEV',
                'Keep technical qualification separate from mechanism efficacy; do not launch or adopt from two technical worlds',
                'Retain full-dose failures and phase diagnostics without dose/gate/model changes'],
        scientific_code_changed=False, descriptive_effects_all_positive=result['all20_descriptive_means_positive'],
        source=current)
    atomic(c3 / 'REVIEW.json', review)
    qualification = dict(accepted=True, implemented=True, cycles_completed=3, verdict=verdict,
        native_technical_jobs=6, native_record_applications=450560, related_terminal_histories=42,
        native_state_archives=96, generic_mapping_presentations=8192, generic_mapping_cases=8,
        generic_tests=tests, tamper_rejections=21, exact_shape_checks=shape_checks,
        science_jobs=0, DEV_authorized=False, CONFIRM_authorized=False, GitHub_authorized=False,
        cstar=result['cstar'], efficacy_adopted=False,
        complete_hardware_memory_traffic_measured=False,
        resource_scope='Actual production CPU, peak RSS, model/RMS bytes and explicit SAFE parameter destination-copy bytes; copy counter is not total physical memory traffic',
        scope='Technical implementation qualification only, not a new-core or arithmetic-mastery claim')
    atomic(ROOT / 'QUALIFICATION.json', qualification)
    atomic(ROOT / 'IMPLEMENTATION.json', dict(implemented=True, source=current, qualification='QUALIFICATION.json',
        design='unchanged historical DESIGN.json', science_jobs=0, inactive_ELM_code_imported=False,
        attention=False, transformers=False, arithmetic_rules_to_learner=False,
        formal_science_launch_adapter_implemented=False))
    atomic(ROOT / 'SUMMARY.json', dict(qualification={k: v for k, v in qualification.items() if k != 'generic_tests'}, cycle3=result))
    source_changed = [f for f in LOCAL_FILES if json.loads((ROOT / 'cycles/cycle2/SOURCE_LOCK.json').read_text())['source'][f] != current[f]]
    assert set(source_changed) == {'analysis.py', 'IMPLEMENTATION_NOTES.md'}
    atomic(ROOT / 'FINAL_AUDIT.json', dict(accepted=True, source=current, cycles_completed=3,
        closeout_source=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cycle2_to_cycle3_changes=source_changed, frozen_model_optimizer_runner_auditor_unchanged=True,
        jobs=6, records=450560, related_histories=42, native_state_archives=96,
        original_certificates_accepted=True, tamper_cases_rejected=21, exact_archive_shape_checks=shape_checks,
        scientific_trajectories_replayed=0, DEV_jobs=0, CONFIRM_jobs=0,
        scope='Original accepted certificates, source/count/roster/archive-shape closeout and one saved-output full-dose reduction'))
    lines = ['# GATED-FEEDBACK V2：三轮实现、审核、试跑与修订', '',
        f'结论：**{verdict}**。实现和技术资格完成；没有启动 DEV/CONFIRM，没有采纳新核心。', '',
        '固定目标保持不变：从真实反馈学习四则及两步混合运算，以更少外部样本和实际计算提高未见题整串输出与自身 EOS 的准确率。学习器只接收原始字节和允许的反馈，不接收算术表示、规则或中间结果。', '',
        '| 循环 | 实际工作 | 审核与修订 |', '|---|---|---|',
        '| 1 | 9 类通用检查；52 个目标梯度检查；两模型各 4,096 次任意字节映射训练，均完整回答 8/8 | KEEP 旧源更新对齐、LSTM 独立实现、非零头批量/串行输出、真实回退和全拒绝原子性；去掉 G62 多余上下文缓存并计重算成本 |',
        '| 2 | 1 个短剂量配对世界；2 作业、8,192 记录、32 个状态档案 | 每条反馈、禁写、分叉、查询和档案审计；21 个篡改拒绝；澄清短剂量时钟不能充当正式 Cstar |',
        '| 3 | 2 个固定完整剂量配对世界；4 作业、442,368 记录、64 个状态档案 | 独立原始证书、固定源、档案尺寸核对；锁定实测 Cstar；保留所有失败，不按效果调整剂量或门槛 |', '',
        '合计：**6 个完整技术作业、450,560 次外部记录应用、42 条相关末态历史、96 个完整原生状态档案**。另有通用映射 8,192 次呈现及独立单元夹具，均不属于正式科学样本。', '',
        'KEEP 为独立重写并与旧源数值对齐的 GRU；G62 为标准 LSTM。参数分别 44,170 / 44,046；参数与 RMS 状态分别 706,720 / 704,736 字节。124/128 只描述循环/解码工作状态，不包括训练图、回退副本或进程 RSS。没有 ELM、注意力或 Transformer。', '',
        '第三轮的具体效果如下。两个世界只能作描述，不能作显著性或采纳依据。', '',
        (c3 / 'REPORT.md').read_text().split('\n', 1)[1], '',
        '## 下一步界限', '',
        '本轮授权在三轮技术练习处结束。若必要效果未全部为正，应先讨论这些保存结果，不机械扩展正式样本。若未来获准 DEV，仍只能采用原定唯一候选和全部 20 项必要比较，不能从当前世界挑换模型或只报告正面指标。', '',
        '计时包含 SAFE 的实际重编码、候选评估、复制和恢复。内存报告包含参数/RMS 字节、进程峰值 RSS 和显式参数复制字节；复制计数不是完整物理内存流量，不能据此宣称测全了带宽成本。', '',
        'QUALIFICATION.json、FINAL_AUDIT.json 和各轮 REVIEW.json 区分“代码可用”与“学会算术”。旧核心保持原状。DEV/CONFIRM 的科学启动适配器未开放；当前工作尚未上传 GitHub。']
    (ROOT / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    atomic(ROOT / 'STATUS.json', dict(status='TECHNICAL_AUDIT_ACCEPTED_PACKAGING', verdict=verdict,
        technical_cycles_completed=3, committed_jobs=6, committed_records=450560, science_jobs=0, heartbeat=time.time()))
    package = ROOT / 'RESULT_BUNDLE.zip'
    temp = package.with_suffix('.pending')
    with zipfile.ZipFile(temp, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for p in sorted(ROOT.rglob('*')):
            if not p.is_file() or p in (package, temp) or '__pycache__' in p.parts or p.name == 'PACKAGE.json':
                continue
            z.write(p, str(p.relative_to(ROOT.parent)))
        dependency = ROOT.parent / 'arithmetic_end_to_end_design_v1_20261013/validate_design.py'
        z.write(dependency, str(dependency.relative_to(ROOT.parent)))
    temp.replace(package)
    with zipfile.ZipFile(package) as z:
        assert z.testzip() is None
    atomic(ROOT / 'PACKAGE.json', dict(accepted=True, path=str(package), bytes=package.stat().st_size,
        sha256=hashlib.sha256(package.read_bytes()).hexdigest(), no_native_replay=True))
    atomic(ROOT / 'STATUS.json', dict(status='THREE_CYCLES_COMPLETED', verdict=verdict,
        technical_cycles_completed=3, committed_jobs=6, committed_records=450560, related_histories=42,
        science_jobs=0, package=str(package), heartbeat=time.time()))
    print(json.dumps(dict(verdict=verdict, jobs=6, records=450560, package=str(package))), flush=True)


if __name__ == '__main__':
    main()
