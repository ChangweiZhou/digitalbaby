"""Saved-output technical reduction. No learner import or training launch."""
import collections
import gzip
import json
import statistics
from pathlib import Path
from protocol import ROOT, D, atomic


def rates(rows, key='exact'):
    by = collections.defaultdict(list)
    for x in rows:
        by[x['operation']].append(float(x[key]))
    return {k: statistics.mean(v) for k, v in by.items()}


def macro(values):
    return statistics.mean(values.values()) if values else 0.


def ability(entry):
    return .5 * macro(rates(entry['scores']['binary'])) + .5 * macro(rates(entry['scores']['mixed']))


def staircase(points, end):
    assert end > 0
    previous_x = previous_a = area = 0.
    for x, a in points:
        assert x >= previous_x and 0 <= a <= 1
        clipped = min(x, end)
        area += (clipped - previous_x) * previous_a
        if x >= end:
            return area / end
        previous_x, previous_a = x, a
    return (area + (end - previous_x) * previous_a) / end


def shortcuts(entry):
    result = {}
    for label, rows in entry['scores'].items():
        by = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in rows:
            for name, prediction in r['shortcuts'].items():
                by[r['operation']][name].append(float(prediction == r['reply']))
        result[label] = {op: max(statistics.mean(v) for v in models.values()) for op, models in by.items()}
    return result


def output_diagnostics(entry):
    rows = sum(entry['scores'].values(), [])
    return dict(EOS=statistics.mean(x['status'] == 'EOS' for x in rows) if rows else 0.,
                payload_length=statistics.mean(len(bytes.fromhex(x['actual'])) for x in rows) if rows else 0.,
                truncation=statistics.mean(x['status'] == 'TRUNCATED' for x in rows) if rows else 0.)


def metrics(r, cstar):
    out = {}
    end = r['binary'] + r['mixed'] + sum(r['stages'])
    for name, b in r['branches'].items():
        curve, last = b['curve'], b['curve'][-1]
        out[name] = dict(J_sample=staircase([(x['n'], ability(x)) for x in curve], end),
            J_compute=staircase([(x['cpu'], ability(x)) for x in curve], cstar), final_A=ability(last),
            mixed_macro=macro(rates(last['scores']['mixed'])), shortcut=shortcuts(last),
            **{'binary_' + k: v for k, v in rates(last['scores']['binary']).items()},
            CPU=last['cpu'], final_output=output_diagnostics(last),
            phase_A=[dict(records=x['n'], cpu=x['cpu'], A=ability(x), **output_diagnostics(x)) for x in curve],
            taught_binary=rates(b['taught']['scores']['binary']),
            taught_mixed=rates(b['taught']['scores']['mixed']), resource=b['resource'])
    warm = r['prefixes']['TRUE']
    out['warm'] = dict(A=ability(warm['curve'][-1]),
        taught_binary=rates(warm['taught']['scores']['binary']),
        taught_mixed=rates(warm['taught']['scores']['mixed']),
        output=output_diagnostics(warm['curve'][-1]))
    return out


def load(folder):
    result = []
    for p in sorted((Path(folder) / 'receipts').glob('*.json.gz')):
        certificate = json.loads(p.with_suffix('.audit.json').read_text())
        assert certificate['accepted']
        with gzip.open(p, 'rt') as f:
            r = json.load(f)
        assert r['completed'] and r['technical'] is True
        result.append(r)
    return result


def effects(by):
    result = {c['id']: [] for c in D['required_contrasts']}
    for world, models in sorted(by.items()):
        assert set(models) == {'KEEP', 'G62'}
        candidate = models['G62']['SAFE']
        for contrast in D['required_contrasts']:
            endpoint, reference = contrast['endpoint'], contrast['reference']
            if reference == 'G62_shortcut_envelope':
                value = macro(candidate['shortcut']['mixed']) if endpoint == 'mixed_macro' else candidate['shortcut']['binary'][endpoint[-1]]
            else:
                model, branch = reference.split('_', 1)
                value = models[model][branch][endpoint]
            result[contrast['id']].append(candidate[endpoint] - value)
    return result


def reduce_stage(folder, cstar=None):
    rows = load(folder)
    assert rows
    if cstar is None:
        cstar = statistics.median(r['branches']['OLD']['curve'][-1]['cpu'] for r in rows if r['model'] == 'KEEP')
    by = {}
    for r in rows:
        by.setdefault(r['world'], {})[r['model']] = metrics(r, cstar)
    es = effects(by)
    means = {k: statistics.mean(v) for k, v in es.items()}
    diagnostics = {}
    for r in rows:
        for name, branch in r['branches'].items():
            records = sum((s['records'] for s in branch['segments']), [])
            by_phase = {}
            for phase in ('HALF', 'SPARSE', 'SCALAR'):
                selected = [x for x in records if x['phase'] == phase]
                if not selected:
                    continue
                trials = [x['update'] for x in selected if x['update']['safe']]
                by_phase[phase] = dict(records=len(selected),
                    sampled_EOS=statistics.mean(x['status'] == 'EOS' for x in selected),
                    sampled_payload_length=statistics.mean(len(bytes.fromhex(x['actual'])) for x in selected),
                    actual_reward=statistics.mean(x['reward'] for x in selected),
                    baseline=statistics.mean(x['baseline'] for x in selected),
                    safe_proposals=len(trials),
                    rejected=sum(not x['accepted'] for x in trials),
                    candidate_evaluations=sum(len(x['candidates']) for x in trials))
            diagnostics[f"{r['world']}_{r['model']}_{name}"] = by_phase
    return dict(worlds=sorted(by), cstar=cstar,
        full_dose=all(r['binary'] == D['course']['WARM']['binary'] and r['mixed'] == D['course']['WARM']['mixed'] for r in rows),
        metrics=by, effects=es, effect_means=means,
        all20_descriptive_means_positive=all(v > 0 for v in means.values()),
        committed_jobs=len(rows), executed_records=sum(r['executed_records'] for r in rows),
        correlated_terminal_histories=7 * len(rows), complete_state_archives=16 * len(rows),
        feedback_diagnostics=diagnostics,
        cost=dict(max_job_wall=max(r['wall'] for r in rows), max_job_cpu=max(r['total_cpu'] for r in rows),
                  total_job_cpu=sum(r['total_cpu'] for r in rows),
                  max_worker_rss_bytes=max(r['peak_rss_bytes'] for r in rows)),
        scope='Technical descriptive worlds only; no significance, adoption or DEV/CONFIRM execution.')


def write_report(path, result):
    lines = ['# GATED-FEEDBACK V2 技术试跑结果', '',
        '本报告只描述固定技术世界，不是正式泛化验证，也不采纳新核心。', '',
        f"完整作业 {result['committed_jobs']}；外部记录应用 {result['executed_records']}；相关末态历史 {result['correlated_terminal_histories']}；完整状态档案 {result['complete_state_archives']}。", '',
        '整串输出与自身 EOS 共同计分。保持四则单步宏平均与十六种有序运算对混合题宏平均各占一半；没有强制输出格式、候选答案筛选或算术提示。', '',
        '| 世界 / 模型 | WARM 未见 A | CE_ONLY 末态 A | OLD 末态 A | SAFE 末态 A | FREEZE_SAFE 末态 A | SAFE EOS |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for world, models in result['metrics'].items():
        for model, m in models.items():
            values = [m['warm']['A'], m['CE_ONLY']['final_A'], m['OLD']['final_A'], m['SAFE']['final_A'], m['FREEZE_SAFE']['final_A'], m['SAFE']['final_output']['EOS']]
            lines.append(f'| {world} / {model} | ' + ' | '.join(f'{v:.2%}' for v in values) + ' |')
    lines += ['', '| 固定必要效应 | 配对技术世界均值 |', '|---|---:|']
    lines += [f'| {k} | {v:+.6f} |' for k, v in result['effect_means'].items()]
    lines += ['', f"20 个必要效应均值是否全部大于零：{result['all20_descriptive_means_positive']}。这不是显著性结论。", '',
        (f"计算面积共同上限 Cstar={result['cstar']:.6f} CPU 秒；由本阶段 KEEP_OLD 固定完整剂量技术作业的中位数确定。后续若获准 DEV，不得根据 DEV 结果重新设定。" if result['full_dose'] else
         f"短剂量计算面积临时上限={result['cstar']:.6f} CPU 秒，只用于本轮技术描述，不得充当 DEV 的 Cstar。正式共同上限只能由第三轮完整剂量 KEEP_OLD 预定规则确定。"), '',
        'CE_ONLY 获得更多答案信息，只作信息参照。N_WARM_SAFE 和 SH_WARM_SAFE 控制先前教学的作用；FREEZE 从各自 SPARSE 末态分叉。SAFE 的约束仅作用于当前实际采样前缀，不保证整体策略 KL 或未来题目不退化。', '',
        'SUMMARY.json 保留每阶段已教题回忆、未见题 A、EOS、输出长度、实际奖励、critic 基线、拒绝次数、全部候选评估及实际 CPU。技术资格通过不等于 E3 通过。']
    Path(path).write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=('cycle2', 'cycle3'), required=True)
    a = p.parse_args()
    folder = ROOT / 'cycles' / a.stage
    result = reduce_stage(folder)
    atomic(folder / 'SUMMARY.json', result)
    write_report(folder / 'REPORT.md', result)
    if a.stage == 'cycle3':
        atomic(ROOT / 'CSTAR_LOCK.json', dict(cstar=result['cstar'], rule=D['metrics']['Cstar_rule'], worlds=result['worlds']))
