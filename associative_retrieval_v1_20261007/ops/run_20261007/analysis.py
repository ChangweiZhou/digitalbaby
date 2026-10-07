"""Literal prospective gates; one paired observation per base world."""
import json
import math
from pathlib import Path
import bootstrap_ops as boot
import numpy as np
from analysis_math import lower, harm_upper
from assays import make_world
from contracts import CONDITIONS, roster, require_sources
from io_utils import read_receipt, atomic_json, digest

def accuracy(bd, name, group, condition, intact=None):
    checkpoint = next(p for p in bd['probes'] if p['name'] == name)
    rows = [r for r in checkpoint['rows'] if r['stage'] == group]
    if intact is not None:
        rows = [r for r in rows if r['item'] in intact]
    if not rows:
        raise ValueError('missing endpoint rows')
    return sum(r['policies'][condition]['correct'] for r in rows)/len(rows)

def query_cost(probe):
    costs = [p['costs'] for row in probe['rows'] for p in row['predictions']]
    q = sum(c['query_cpu_s'] for c in costs)
    r = sum(c['retrieval_cpu_s'] for c in costs)
    p = sum(c['control_retrieval_cpu_s'] for c in costs)
    common = probe['physical_probe_cpu_s']-q-r-p
    if common <= 0:
        raise ValueError('invalid measured common final-query CPU')
    return {'ERROR': common, 'LINK': common+q+r, 'PERM': common+q+p}

def operational_cost(bd):
    names = {'ERROR': 'parent_online_cpu_s', 'LINK': 'LINK_online_cpu_s', 'PERM': 'PERM_online_cpu_s'}
    final = next(p for p in bd['probes'] if p['name'] == 'final')
    costs = query_cost(final)
    return {condition: costs[condition]+sum(r[key] for r in bd['records']) for condition, key in names.items()}

def per_world(life, reuse):
    if life['world'] != reuse['world']:
        raise ValueError('paired world mismatch')
    world = life['world']
    intact = make_world(world, 'lifetime')['intact_old_items']
    lb, rb = life['branches']['W'], reuse['branches']['W']
    no = reuse['branches']['N_old']
    lc, rc = operational_cost(lb), operational_cost(rb)
    result = {'world': world, 'conditions': {},
              'physical_cpu_s': life['cpu_s']+reuse['cpu_s'],
              'worker_s': life['worker_s']+reuse['worker_s'],
              'max_rss_bytes': max(life['peak_rss_bytes'], reuse['peak_rss_bytes'])}
    for c in CONDITIONS:
        own = accuracy(rb, 'final', 'heldout', c)-accuracy(no, 'final', 'heldout', c)
        cost = lc[c]+rc[c]
        result['conditions'][c] = {
            'old_E1': accuracy(lb, 'final', 'old', c, intact),
            'new_E1': accuracy(lb, 'final', 'new', c),
            'revision_E1': accuracy(lb, 'final', 'revision', c),
            'reuse_W': accuracy(rb, 'final', 'heldout', c),
            'reuse_N_old': accuracy(no, 'final', 'heldout', c),
            'reuse_effect': own,
            'old_end_W': accuracy(rb, 'old_end', 'heldout', c),
            'old_end_N_old': accuracy(no, 'old_end', 'heldout', c),
            'old_end_effect': accuracy(rb, 'old_end', 'heldout', c)-accuracy(no, 'old_end', 'heldout', c),
            'taught_relation_old_E1': accuracy(rb, 'final', 'old', c),
            'taught_relation_new_E1': accuracy(rb, 'final', 'new', c),
            'operational_W_cpu_s': cost,
            'lifetime_W_operational_cpu_s': lc[c], 'relation_W_operational_cpu_s': rc[c],
            'capability_per_1000_CPU_s': 1000*own/max(cost, 1.),
        }
    return result

def vectors(worlds):
    keys = list(worlds[0]['conditions']['ERROR'])
    return {c: {k: np.asarray([w['conditions'][c][k] for w in worlds], dtype=float)
                for k in keys} for c in CONDITIONS}

def evaluate(worlds, stage):
    n = len(worlds)
    if n != (8 if stage == 'screen' else 64):
        raise ValueError('exact prospective stage sample required')
    v = vectors(worlds)
    l, e, p = (v[c] for c in ('LINK', 'ERROR', 'PERM'))
    gains = l['reuse_effect']-e['reuse_effect']
    mechanism = l['reuse_effect']-p['reuse_effect']
    losses = {k: e[k]-l[k] for k in ('old_E1', 'new_E1', 'revision_E1')}
    harms = [int(max(losses[k][i] for k in losses) > .10) for i in range(n)]
    cpu_ratio = float(l['operational_W_cpu_s'].sum()/e['operational_W_cpu_s'].sum())
    checks = {
        'old_new_revision_mean_loss_at_most_2pp': all(float(a.mean()) <= .02 for a in losses.values()),
        'operational_CPU_ratio_at_most_1_25': cpu_ratio <= 1.25,
    }
    bounds = {}
    if stage == 'screen':
        checks.update({
            'LINK_old_new_revision_at_least_90_percent': all(float(l[k].mean()) >= .90 for k in losses),
            'at_most_one_harmed_world': sum(harms) <= 1,
            'raw_heldout_W_above_chance': float(l['reuse_W'].mean()) > .5,
            'own_prior_teaching_effect_positive': float(l['reuse_effect'].mean()) > 0,
            'causal_gain_at_least_5pp': float(gains.mean()) >= .05,
            'LINK_minus_PERM_causal_gain_positive': float(mechanism.mean()) > 0,
        })
        advance = all(checks.values())
        verdict = 'ADVANCE_LINK' if advance else 'SCREEN_NO_ADVANCE_RETAIN_ERROR'
    else:
        bounds = {
            'causal_gain': lower(gains, .03, width=4),
            'raw_W_improvement': lower(l['reuse_W']-e['reuse_W'], .03, width=2),
            'efficiency_gain': lower(l['capability_per_1000_CPU_s']-e['capability_per_1000_CPU_s'], .03, width=4000),
            'own_prior_teaching': lower(l['reuse_effect'], .01, width=2),
            'raw_W': lower(l['reuse_W'], .01, width=1),
            'correspondence_mechanism': lower(mechanism, .01, width=4),
        }
        hu = harm_upper(harms, .03)
        checks.update({
            'mean_gain_at_least_5pp': float(gains.mean()) >= .05,
            'positive_causal_gain_lower': bounds['causal_gain']['lower'] > 0,
            'positive_raw_W_gain_lower': bounds['raw_W_improvement']['lower'] > 0,
            'positive_efficiency_gain_lower': bounds['efficiency_gain']['lower'] > 0,
            'positive_own_prior_teaching_lower': bounds['own_prior_teaching']['lower'] > 0,
            'raw_W_lower_above_chance': bounds['raw_W']['lower'] > .5,
            'harmed_world_fraction_upper_below_10_percent': hu < .10,
        })
        advance = False
        verdict = ('ADOPT_LINK' if all(checks.values()) else
                   'PROMISING_BUT_UNRESOLVED' if float(gains.mean()) > 0 else 'RETAIN_ERROR')
        bounds['harm_risk_upper'] = hu
    return {'stage': stage, 'n': n, 'worlds': [w['world'] for w in worlds],
            'candidate': 'LINK', 'parent': 'ERROR', 'target': 'final causal held-out reuse',
            'verdict': verdict, 'advance': advance, 'adopted': verdict == 'ADOPT_LINK',
            'checks': checks, 'bounds': bounds, 'CPU_ratio': cpu_ratio,
            'harm_worlds': sum(harms), 'mean_causal_gain': float(gains.mean()),
            'mean_mechanism_contrast': float(mechanism.mean()),
            'means': {c: {k: float(a.mean()) for k, a in x.items()} for c, x in v.items()},
            'per_world': worlds, 'evidence_levels': {'taught': 'E1', 'reuse': 'tested binary-family E3'},
            'Student_intervals': 'approximate; registered Hoeffding range when zero variance',
            'operational_CPU_method': 'qualified event counters plus common final-query work and route-specific timed reads; summed W costs over both tasks'}

def report(d):
    title = 'LINK 筛选结果' if d['stage'] == 'screen' else 'LINK 确认结果'
    md = f'# {title}\n\n裁决：**{d["verdict"]}**。{d["n"]}个新配对世界，固定候选、任务和样本。\n\n'
    md += '| 条件 | 旧键E1 | 新键E1 | 修订E1 | 未教W | 未教N_old | W−N_old | W运行CPU秒 |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
    for c, m in d['means'].items():
        vals = [100*m[k] for k in ('old_E1', 'new_E1', 'revision_E1', 'reuse_W', 'reuse_N_old', 'reuse_effect')]
        md += f'| {c} | '+ ' | '.join(f'{x:.2f}%' for x in vals)+f' | {m["operational_W_cpu_s"]:.2f} |\n'
    md += f'\nLINK相对ERROR的配对因果增量：{100*d["mean_causal_gain"]:+.2f}pp。运行CPU比：{d["CPU_ratio"]:.3f}。\n\n'
    md += '筛选未晋级是本配方的预算停止；不等于整个关联检索家族被证伪。E1查表可解释；E3只涉及既有二元关系、反馈前自选答案，不扩展为普遍智能或自然语言能力。地址关联只有题面活动，没有答案或题目身份。\n\n'
    md += '三种读出条件共享经验证一致的原生生命史；它们不构成三倍独立样本。早期探针和独立审计计入实际实验成本，部署效率使用两项任务W的教学与最终查询CPU估计。\n\n'
    md += '| 既定门槛 | 通过 |\n|---|---|\n'
    for k, passed in d['checks'].items():
        md += f'| {k} | {passed} |\n'
    if d['bounds']:
        md += '\n```json\n'+json.dumps(d['bounds'], ensure_ascii=False, indent=2)+'\n```\n'
    return md

def analyze_stage(folder, stage):
    folder = Path(folder)
    docs = {}
    for j in roster(stage):
        d = read_receipt(folder/'receipts'/f'{j["job"]}.json.gz')
        if (d['world'], d['assay'], d['stage']) != (j['world'], j['assay'], stage):
            raise ValueError('analysis receipt identity mismatch')
        docs[(j['world'], j['assay'])] = d
    ids = sorted({w for w, _ in docs})
    worlds = [per_world(docs[(w, 'lifetime')], docs[(w, 'reuse')]) for w in ids]
    d = evaluate(worlds, stage)
    atomic_json(folder/'SUMMARY.json', d)
    (boot.ROOT/f'{stage.upper()}_REPORT.md').write_text(report(d))
    if stage == 'screen':
        lock = require_sources()
        atomic_json(boot.ROOT/'results/SELECTION_LOCK.json',
                    {'schema': 'LINK_SINGLE_CANDIDATE_SELECTION_V1',
                     'source_identity': lock['native_identity'], 'operations_identity': lock['operations_identity'],
                     'advance': d['advance'], 'candidate': 'LINK' if d['advance'] else None,
                     'target': d['target'], 'screen_summary_digest': digest(d),
                     'confirmation_roster': roster('confirm') if d['advance'] else [],
                     'no_other_candidate': True, 'no_sample_extension': True})
    return d
