"""Source/register and task-structure audit only. Never births or trains a core."""
import hashlib
import itertools
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
REG = ROOT / 'PROJECT_DOCUMENTS/MECHANISM_SCREEN_20261007'


def write(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def family_rows(kind, a, b):
    rows = []
    if kind == 'SUM':
        s = a + b
        rows = [('+', a, b, s), ('+', b, a, s),
                ('-', s, a, b), ('-', s, b, a),
                ('-', a, s, -b), ('-', b, s, -a)]
    else:
        p = a * b
        rows = [('*', a, b, p), ('*', b, a, p)]
        if a:
            rows.append(('/', p, a, b))
        if b:
            rows.append(('/', p, b, a))
    # This evaluator-only arithmetic generates observed data, not a learner.
    return sorted(set(rows))


def prompt(row):
    op, a, b, _ = row
    return f'{a}{op}{b}='


def answer(row):
    return str(row[3])


def recover_family(row):
    """Independent evaluator reconstruction, including reversed/inverse cues."""
    op, a, b, y = row
    if op == '+':
        assert a + b == y
        return 'SUM', tuple(sorted((a, b)))
    if op == '-':
        assert a - b == y
        return 'SUM', tuple(sorted((b, y) if y >= 0 else (a, -y)))
    if op == '*':
        assert a * b == y
        return 'PRODUCT', tuple(sorted((a, b)))
    assert op == '/' and b > 0 and a == b * y
    return 'PRODUCT', tuple(sorted((b, y)))


def make_split(seed):
    rng = random.Random(seed)
    all_pairs = list(itertools.combinations_with_replacement(range(8), 2))
    eligible = list(itertools.combinations_with_replacement(range(2, 8), 2))
    assert len(all_pairs) == 36 and len(eligible) == 21
    selected = {}
    for kind in ('SUM', 'PRODUCT'):
        while True:
            held = set(rng.sample(eligible, 12))
            train_rows = [r for a, b in all_pairs if (a, b) not in held
                          for r in family_rows(kind, a, b)]
            held_rows = [r for a, b in all_pairs if (a, b) in held
                         for r in family_rows(kind, a, b)]
            # Character support is about the full teaching pool, not prefixes.
            covered = True
            for op in ('+', '-') if kind == 'SUM' else ('*', '/'):
                tr = [r for r in train_rows if r[0] == op]
                ho = [r for r in held_rows if r[0] == op]
                input_tr = set(''.join(prompt(r) for r in tr))
                output_tr = set(''.join(answer(r) for r in tr))
                covered &= bool(tr and ho)
                covered &= set(''.join(prompt(r) for r in ho)) <= input_tr
                covered &= set(''.join(answer(r) for r in ho)) <= output_tr
            if covered:
                selected[kind] = {'held': sorted(held),
                                  'train': train_rows, 'test': held_rows}
                break
    return selected


def countermodel(train, test, selector):
    groups = defaultdict(Counter)
    for row in train:
        groups[selector(row)][answer(row)] += 1
    predictions = {key: sorted(counts, key=lambda y: (-counts[y], y))[0]
                   for key, counts in groups.items()}
    hits = sum(predictions.get(selector(row)) == answer(row) for row in test)
    return hits / len(test)


def row_operation(row):
    op = row['operation']
    if 'memory_specs' in op:
        return op['memory_specs'][0]
    if 'configurations' in op:
        return op['configurations'][0]
    return op


def disposition(row):
    vid, name = row['version_id'], row['name']
    anchors = {
        'V23/sparse96_fixed': 'G原始版本，固定竞争编码；257输出及公共字节输入为新移植。',
        'V24/sparse96_fixed': 'G直接代码来源；不加入地址可塑性。',
        'V24/bank4_fixed': 'R出生/容量/访问匹配控制；实际原型与4银行保留。',
        'V24/bank4_local': 'R局部原型学习来源；回复整段路由为新序列移植。',
        'SAMPLE_GEN_V2/COV-R': 'C完整相关更新来源及科学比较锚点思想；标量到257读出为新移植。',
        'SAMPLE_GEN_V2/DIAG-R': 'C对角相关控制来源；新增完整矩阵真实计费。',
    }
    if vid in anchors:
        return 'SELECTED_SOURCE_OR_MATCHED_CONTROL', anchors[vid]
    if name in ('TOPO-G', 'TOPO-S', 'TOPO-R'):
        return 'NOT_REUSE_ROLE_SEPARABLE_RECIPE', '源码把16单元分成8左/8右；单独这种结构不能提供旧标量乘法交互，不宣判拓扑整族失败。'
    if 'sparse' in name and 'adaptive' in name:
        return 'NOT_REUSE_ADAPTIVE_ADDRESS_RECIPE', '历史当前配方获取明显较弱且改变记忆坐标；本轮只取固定竞争结构，保留原证据。'
    if name in ('COV-R', 'DIAG-R'):
        return 'REPEATED_REFERENCE', '与选定相关更新方向重复出现的比较锚点；不算一个新机制，也不把跨任务记录混为同次结果。'
    if 'SUPPORT' in name or 'RATE' in name or name in ('COV-I', 'DIAG-I', 'SUPPORT-DIR', 'SUPPORT-RAND', 'SUPPORT-ISO'):
        return 'CURRENT_OR_NOISE_SPECIFIC_RECIPE', '保留现有采纳/负面状态；本轮优先四则规则学习，不重新优化抗噪、支持门或恢复方向。'
    text = json.dumps(row_operation(row), ensure_ascii=False).lower()
    if 'bank' in name or any(t in name for t in ('context_', 'silent_', 'dual_', 'split_', 'structured_', 'e3_local', 'local_base', 'local_slow', 'minimal_')):
        return 'BANK_ROUTE_VARIANT_RESERVED', '本轮先在4银行固定/局部路由上隔离操作；8银行、释放、读取视图、精度和地址/内容组合不自动同时加入。'
    if row['series'] == 'V30_DIRECTION' or 'ridge' in name or name == 'v30_cov':
        return 'DIRECTION_VERSION_RESERVED', '有具体历史相关/方向规则；本轮选已有当前算术依据的COV-R/DIAG-R移植，不把不同协方差算法混算。'
    if row['series'] == 'V31_UNCERTAINTY' or 'bayes' in name:
        return 'UNCERTAINTY_VERSION_RESERVED', '有限视野/恢复规则保留原状态；清洁静态四则首先需要可表达交互的表示，不优先噪声与重开。'
    if any(t in text for t in ('consolidation', 'cascade', 'active_forgetting', 'importance', 'availability', 'projection', 'protection')):
        return 'RETENTION_MECHANISM_RESERVED', '这些操作不能独自补上缺少的跨角色交互；保持仍在本轮测量，只有对应代价出现才提出新实例。'
    if any(t in text for t in ('allocation', 'retrieval', 'replay', 'fastweights', 'repeat_current', 'budget24', 'turnover', 'recycling')):
        return 'STORAGE_OR_REEXPOSURE_VERSION_RESERVED', '有界存储/再暴露仍保留；当前优先从少量不同事实重用到未见家族，不能用更多回忆或未计费回放替代。'
    if 'sparse' in name:
        return 'GEOMETRY_VARIANT_RESERVED', '本轮固定96/8历史规格并与同矩阵线性控制比较；384宽度等不得算免费容量，未来需新预算理由。'
    if name.startswith(('reward_', 'target_')):
        return 'FEEDBACK_TIMING_VERSION_RESERVED', '旧即时/延迟/资格迹与标量反馈版本保留；本轮观察真实字节续写，无新的奖励或延迟归因问题。'
    return 'HISTORICAL_REFERENCE_NOT_NEW_ARM', '保留完整原操作、证据和既有审查状态；不以名称尚未整族否定为新增运行理由。'


def main():
    register_path = REG / 'VERSION_REGISTER_20261011/VERSIONS.json'
    review_path = REG / 'FULL_REVIEW_POST_U034_20261011/VERSION_254_COMPARISON.json'
    rows = json.loads(register_path.read_text())['versions']
    prior = {r['id']: r for r in json.loads(review_path.read_text())}
    assert len(rows) == len(prior) == 254
    assert {r['version_id'] for r in rows} == set(prior)
    review = []
    for row in rows:
        status, reason = disposition(row)
        old = prior[row['version_id']]
        review.append({'version_id': row['version_id'], 'series': row['series'],
                       'name': row['name'], 'original': row,
                       'prior_review_status': old['review_status'],
                       'prior_review_note': old['review_note'],
                       'this_round_disposition': status, 'this_round_reason': reason,
                       'family_wide_exclusion': False, 'new_science_executed': False})
    write('MECHANISM_REVIEW_254.json', review)
    counts = Counter(r['this_round_disposition'] for r in review)
    lines = ['# 完整254版本在四则机制实验中的取舍', '',
             '覆盖254/254，不使用71项子集代替；保留原版本、来源、证据和既有审查状态。', '',
             '表中的暂不进入不是全家族科学排除。新端口未运行，源版本不等于新端口。', '',
             '| 版本 | 本轮用途 | 理由 |', '| --- | --- | --- |']
    lines += [f"| {r['version_id']} | {r['this_round_disposition']} | {r['this_round_reason']} |"
              for r in review]
    (OUT / 'MECHANISM_REVIEW_254.md').write_text('\n'.join(lines) + '\n')

    fixtures = []
    for seed in range(202610120001, 202610120009):
        # Reserved structural fixtures. No science seed namespace or learner.
        split = make_split(seed)
        train = [r for item in split.values() for r in item['train']]
        test = [r for item in split.values() for r in item['test']]
        all_rows = train + test
        pmap = defaultdict(set)
        for r in all_rows:
            pmap[prompt(r)].add(answer(r))
        assert all(len(v) == 1 for v in pmap.values())
        assert set(prompt(r) for r in train).isdisjoint(prompt(r) for r in test)
        tr_families = {recover_family(r) for r in train}
        ho_families = {recover_family(r) for r in test}
        assert tr_families.isdisjoint(ho_families)
        assert ho_families == {(kind, tuple(pair)) for kind, data in split.items()
                               for pair in data['held']}
        per_op = {}
        for op in '+-*/':
            tr = [r for r in train if r[0] == op]
            ho = [r for r in test if r[0] == op]
            assert tr and ho
            per_op[op] = {
                'train_questions': len(tr), 'held_questions': len(ho),
                'maximum_prompt_bytes': max(len(prompt(r)) for r in tr + ho),
                'maximum_answer_bytes': max(len(answer(r)) for r in tr + ho),
                'full_pool_answer_alphabet': ''.join(sorted(set(''.join(answer(r) for r in tr)))),
                'modal_answer_full_pool_accuracy': countermodel(tr, ho, lambda r: None),
                'left_only_full_pool_accuracy': countermodel(tr, ho, lambda r: r[1]),
                'right_only_full_pool_accuracy': countermodel(tr, ho, lambda r: r[2]),
                'exact_or_inverse_same_family_leak': 0,
            }
        fixtures.append({'structural_fixture_seed': seed,
                         'held_families': {k: v['held'] for k, v in split.items()},
                         'operations': per_op, 'learning_records_executed': 0})
    write('TASK_STRUCTURE_CHECK.json', {'fixtures': fixtures,
                                      'scientific_worlds_created': 0,
                                      'core_trajectories_executed': 0})

    src = REG / 'VERSION_REGISTER_20261011/sources/03_elm_fly_local_memory_v24.py'
    current = ROOT / 'addition_sample_generalization_dev_v2_20261011/cores.py'
    source = src.read_text()
    csource = current.read_text()
    for literal in ('class SparseSpecializationMemory', 'width=96, k=8',
                    'address_eta=.015', 'class V24BankMemory', 'address_eta=.05',
                    'self.centers[bank] = unit_rows'):
        assert literal in source, literal
    for literal in ('def local_feature(unit, feature)',
                    '(0 if unit < 8 else 15) + feature',
                    'self.p = np.eye(31) * 10', 'self.p = self.p - factor * np.outer(q, q)'):
        assert literal in csource, literal
    # Old scalar role-only output cross differences vs arithmetic targets.
    additive_cross_difference = (4 + 3) - (4 + 2) - (3 + 3) + (3 + 2)
    product_cross_difference = (4 * 3) - (4 * 2) - (3 * 3) + (3 * 2)
    assert additive_cross_difference == 0 and product_cross_difference == 1

    configs = [f'{g}-{r}-{c}' for g, r, c in itertools.product(
        ('LINEAR', 'COMP'), ('FIXED', 'LOCAL'), ('DIAG', 'FULL'))]
    spec = {'status': 'DESIGN_ONLY_PREPARATION_PENDING',
            'goal': 'Own-byte-output unseen four-operation ability per training sample and compute',
            'full_version_register_rows_reviewed': 254,
            'seed_operands': list(range(8)), 'operators': list('+-*/'),
            'division': 'exact integer inverses of product families; denominator nonzero',
            'family_counts': {'SUM': 36, 'PRODUCT': 36},
            'held_families_per_type': 12, 'training_families_per_type': 24,
            'held_seed_domain': '2 <= a <= b <= 7',
            'noise_rate': 0, 'missing_rate': 0,
            'configs': configs + ['HASH'],
            'scientific_reference': 'LINEAR-FIXED-FULL',
            'selectable_configs': [c for c in configs if c != 'LINEAR-FIXED-FULL'],
            'selection': 'qualified candidates ordered by final W CPU median, state bytes, config name',
            'histories': ['W', 'N_ALL', 'SHUFFLE'],
            'per_operation_sample_checkpoints': [4, 8, 16, 32, 64, 128],
            'records_per_history': 512, 'records_per_job': 1536,
            'fresh_paired_worlds': 48, 'world_config_jobs': 432,
            'correlated_full_histories': 1296, 'record_applications': 663552,
            'statistics': {'paired_unit': 'world', 'predeclared_contrasts': 116,
                           'family_error_rate': .05, 'method': 'two-sided paired t with Bonferroni',
                           'individual_interval_level': 1 - .05 / 116},
            'fixed_accuracy_thresholds': None, 'fixed_pp_gain_thresholds': None,
            'scientific_launch_authorized': False, 'github_upload_authorized': False,
            'three_cycle_exercise_completed': False,
            'session': {'admission_stop_hours': 8, 'hard_limit_hours': 10,
                        'concurrency': 'set from complete technical job cost',
                        'automatic_resume_or_restart': False}}
    assert spec['world_config_jobs'] * spec['records_per_job'] == 663552
    assert len(configs) == 8 and len(set(configs)) == 8
    write('DESIGN.json', spec)
    write('DESIGN_AUDIT.json', {
        'status': 'STATIC_SOURCE_AND_TASK_CHECKS_PASS_LEARNER_PREPARATION_PENDING',
        'register_coverage': len(review), 'missing_or_duplicate_version_ids': [],
        'disposition_counts': dict(counts),
        'source_files': {str(p.relative_to(ROOT)): digest(p)
                         for p in (register_path, review_path, src, current)},
        'source_algorithm_assertions_passed': True,
        'structural_task_fixtures_checked': len(fixtures),
        'family_leakage_and_answer_conflict_checks_passed': True,
        'old_scalar_additive_cross_difference': additive_cross_difference,
        'multiplication_target_cross_difference': product_cross_difference,
        'primary_inferential_contrasts': 116,
        'removed_unsupported_capability_and_gain_thresholds': True,
        'frozen_prior_sources_modified': False,
        'new_learning_trajectories_executed': 0,
        'science_jobs_executed': 0,
        'remaining_preparation': ['port kernels and independent update auditor',
                                  'generic protocol regressions and causal tamper suite',
                                  'complete technical job timings and power/sensitivity estimate',
                                  'cold start/pause/counting/package tests and source freeze']})
    print(json.dumps({'register_rows': len(review), 'fixtures': len(fixtures),
                      'world_config_jobs_proposed': 432,
                      'new_learning_trajectories': 0, 'status': 'static checks pass'},
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
