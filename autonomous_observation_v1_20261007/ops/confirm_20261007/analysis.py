"""World-level frozen intersection gates; byte rows are never independent n."""
import math
import statistics
from scipy.stats import t
from ops_common import OPS, ROOT, RESULTS, WORLDS, atomic, read, sha, verify_sources, counts


def interval(values):
    if len(values) != 32 or any(not math.isfinite(v) for v in values):
        raise ValueError('confirmation requires all 32 finite world estimates')
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    half = float(t.ppf(.975, 31)) * sd / math.sqrt(32)
    return dict(mean=mean, low=mean-half, high=mean+half, n_worlds=32,
                sample_sd=sd, method='paired-world Student t, two-sided 95%')


def measure(r):
    def acc(stage, branch, label):
        probe = r['probes'][stage][branch][label]
        if label == 'unchanged':
            keys = set(r['inputs']['old']) - set(r['inputs']['revised'])
        else:
            keys = set(r['inputs'][label])
        selected = [row for row in probe['rows'] if row['context'] in keys]
        return sum(row['byte'] == row['emitted'] for row in selected) / len(selected)

    def bpb(branch):
        rows = r['probes']['day2'][branch]['old']['rows']
        return statistics.mean(-math.log2(row['probabilities'][b'0123'.index(row['byte'])]) for row in rows)

    old = acc('old', 'W', 'old')
    retained = acc('day2', 'W', 'old')
    control = acc('day2', 'N_OLD', 'old')
    return dict(world=r['world'], old_acquisition=old, retained=retained, N_OLD=control,
                causal_gain=retained-control, retention_delta=retained-old,
                old_probe_bpb=bpb('W'), all_old_probe_bpb_gain=bpb('N_ALL')-bpb('W'),
                new=acc('day2', 'W', 'new'), revised=acc('revised', 'W', 'revised'),
                unchanged=acc('revised', 'W', 'unchanged'),
                W_training_cpu=sum(v['cpu_seconds'] for k, v in r['training_cost'].items() if k.endswith('/W')),
                world_cpu=r['resources']['cpu_seconds'], world_wall=r['resources']['wall_seconds'])


def gates(est):
    return dict(causal_gain=est['causal_gain']['low'] > 0,
                all_old_probe_bpb_gain=est['all_old_probe_bpb_gain']['low'] > 0,
                retained=est['retained']['mean'] >= .90,
                retention_delta=est['retention_delta']['low'] >= -.05,
                new=est['new']['mean'] >= .80,
                revised=est['revised']['mean'] >= .80,
                unchanged=est['unchanged']['mean'] >= .85)


def analyze():
    source = verify_sources()
    index = read(RESULTS / 'COMMIT_INDEX.json')
    if set(index) != set(map(str, WORLDS)):
        raise ValueError('confirmation roster incomplete; partial data cannot be packaged')
    rows = []
    for world in WORLDS:
        path = RESULTS / 'receipts' / f'WORLD_{world}.json.gz'
        if sha(path) != index[str(world)]['sha256']:
            raise ValueError('committed receipt changed')
        receipt = read(path)
        a = receipt['independent_audit']
        if receipt['world'] != world or a['verdict'] != 'PASS' or a['independently_checked_rows'] != 8240 or a['confirmation_source_lock_sha256'] != source:
            raise ValueError('unaccepted receipt')
        rows.append(measure(receipt))
    est = {key: interval([r[key] for r in rows]) for key in rows[0] if key != 'world'}
    checked = gates(est)
    adopted = all(checked.values())
    summary = dict(verdict='ADOPT_AUTONOMOUS_OBSERVATION_BASELINE' if adopted else 'DO_NOT_ADOPT_AUTONOMOUS_OBSERVATION_BASELINE',
                   estimates=est, gates=checked, world_results=rows, **counts(),
                   E3_tested=False, evidence='limited-alphabet raw-context autonomous acquisition, retention and revision; E0/E1',
                   confirmation_source_lock_sha256=source, inference_unit='world',
                   no_sample_extension=True, byte_rows_not_independent_samples=True,
                   total_learning_and_probe_rows=32*8240)
    atomic(RESULTS / 'SUMMARY.json', summary, exclusive=True)
    lines = ['# 普通字节观察驱动的原生记忆：正式确认', '',
             f"结论：**{summary['verdict']}**。", '',
             '32 个全新世界，96 条学习生命史，122,880 个学习字节；263,680 条学习与探针记录均已逐行独立审计。测试克隆和擦除对照不另算独立生命史。', '',
             '| 固定测量 | 均值 | 世界级配对/均值 95% 区间 | 门槛 |',
             '|---|---:|---:|---|']
    names = [('old_acquisition','旧键获取'), ('retained','第二天旧键保持'), ('N_OLD','禁旧写对照'),
             ('causal_gain','旧学习的因果收益'), ('retention_delta','保持相对获取的变化'),
             ('new','新键获取'), ('revised','内容修订'), ('unchanged','未修订内容'),
             ('all_old_probe_bpb_gain','全旧探针字节 bits/byte 收益')]
    for key, name in names:
        v = est[key]
        if key == 'all_old_probe_bpb_gain':
            value, ci = f"{v['mean']:.4f}", f"[{v['low']:.4f}, {v['high']:.4f}]"
        else:
            value, ci = f"{100*v['mean']:.2f}%", f"[{100*v['low']:.2f}, {100*v['high']:.2f}] pp"
        gate = '通过' if checked.get(key) is True else ('未通过' if checked.get(key) is False else '描述性')
        lines.append(f'| {name} | {value} | {ci} | {gate} |')
    lines += ['', '全部七个门槛按冻结的交集规则判定；不挑选成功的检查点，不扩样。普通到来字节在预测之后生成内部预测误差，更新 Full151 原生快慢状态；没有外部正确性 bit、答案槽教学或可训练旁路预测器。', '',
              '成立范围为 ASCII `0123`、最近四个原始字节的工程固定地址、四份独立原生输出存储。休息的 24 小时按模型时钟推进。本实验没有检验关系泛化、任意长串、自然语言或可扩展的新通用核心；精确上下文计数器也能解决此任务。', '',
              f"W 每世界 1,280 个学习字节的平均 CPU：{est['W_training_cpu']['mean']:.3f} s；三分支及全部探针每世界平均 CPU：{est['world_cpu']['mean']:.3f} s。原生可变数组加上下文约 662,660 bytes/模型，不含固定资产、运行库、缓存与瞬时克隆。", '',
              '正式授权入口仅替换开发入口的 admission guard，并对独立审计的 world/evidence 头部做 namespace adapter；冻结的学习、流、时间、记录和门槛源码没有改动。DEV 样本不参与确认。']
    report = RESULTS / 'REPORT.md'
    with report.open('x') as f:
        f.write('\n'.join(lines)+'\n')
    atomic(RESULTS / 'FINAL_AUDIT.json', dict(status='ACCEPTED_COMPLETE_EXPERIMENT',
           adoption_verdict=summary['verdict'], source_unchanged=True,
           world_roster_exact=True, independently_checked_rows=32*8240,
           completed_worlds=32, completed_training_lives=96,
           committed_training_bytes=122880, gates=checked,
           source_lock_sha256=source, E3_tested=False,
           negative_adoption_is_valid_completed_experiment=True), exclusive=True)
    return summary
