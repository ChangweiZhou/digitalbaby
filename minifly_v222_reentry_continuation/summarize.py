"""Reduce saved outputs for the fixed two-arm continuation."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARMS = ('R8', 'R32-BP')


def load(path):
    return json.loads(path.read_text())


def paired_rows(frozen, causal):
    counts = {'causal_only_exact': 0, 'frozen_only_exact': 0, 'both_exact': 0,
              'output_changed': 0, 'first_byte_changed': 0}
    for a, b in zip(frozen, causal, strict=True):
        assert a['prompt'] == b['prompt'] and a['expected'] == b['expected']
        counts['causal_only_exact'] += bool(b['exact'] and not a['exact'])
        counts['frozen_only_exact'] += bool(a['exact'] and not b['exact'])
        counts['both_exact'] += bool(a['exact'] and b['exact'])
        counts['output_changed'] += a['emitted'] != b['emitted']
        counts['first_byte_changed'] += a['emitted'][0] != b['emitted'][0]
    return counts


def failure_types(rows):
    result = {name: 0 for name in ('exact', 'immediate_eos', 'wrong_first', 'wrong_continuation', 'wrong_termination')}
    for row in rows:
        if row['exact']:
            name = 'exact'
        elif row['immediate_eos']:
            name = 'immediate_eos'
        elif not row['first_byte_correct']:
            name = 'wrong_first'
        elif not row['payload_correct']:
            name = 'wrong_continuation'
        else:
            name = 'wrong_termination'
        result[name] += 1
    return result


def summarize():
    lock = load(ROOT/'LOCK.json')
    out = {'status': 'COMPLETE', 'scope': lock['study_scope'],
           'world_and_birth_seed': lock['world_and_birth_seed'], 'studies': {}}
    for study in ('acquisition', 'curve'):
        out['studies'][study] = {}
        for arm in ARMS:
            folder = ROOT/'results'/study/arm
            rows = []
            for dose, stop in zip(lock['schedule'][study]['doses'], lock['schedule'][study]['symbol_stops']):
                path = folder/f'{dose}.json'
                if not path.exists():
                    out['status'] = 'IN_PROGRESS'
                    break
                row = load(path)
                raw = load(folder/f'{dose}_outputs.json')
                assert row['symbols'] == stop and row['optimizer_updates'] == stop//32
                assert row['dose'] == dose and row['arm'] == arm
                row['paired_probe_comparison'] = {}
                for panel, expected_count in lock['schedule'][study]['panels'].items():
                    frozen = raw['rows']['frozen'][panel]
                    causal = raw['rows']['causal'][panel]
                    assert len(frozen) == len(causal) == expected_count
                    row['paired_probe_comparison'][panel] = paired_rows(frozen, causal)
                    for name, values in [('frozen', frozen), ('causal', causal)]:
                        counts = failure_types(values)
                        assert counts['exact'] == row['metrics'][name][panel]['exact']
                        row['metrics'][name][panel]['failure_types'] = counts
                row['answer_onset_distances'] = {
                    mode: [{**{k: v for k,v in p.items() if k != 'prefix_distances'}, **p['prefix_distances'][-1]}
                           for p in traces] for mode, traces in raw['paired_prompt_traces'].items()}
                rows.append(row)
            out['studies'][study][arm] = rows
    if out['status'] == 'COMPLETE':
        out['worker_totals'] = {study: {arm: load(ROOT/'results'/study/arm/'COMPLETE.json')
                                       for arm in ARMS} for study in out['studies']}
    target = ROOT/'SUMMARY.json'
    tmp = target.with_suffix('.tmp')
    tmp.write_text(json.dumps(out, indent=2) + '\n')
    tmp.replace(target)
    for study, arms in out['studies'].items():
        for arm, rows in arms.items():
            if not rows:
                continue
            row = rows[-1]
            print(study, arm, row['dose'])
            for mode, panels in row['metrics'].items():
                for panel, m in panels.items():
                    print(f'  {mode} {panel} exact {m["exact"]}/{m["n"]}, first {m["first_byte_correct"]}/{m["n"]}, '
                          f'answer {m["all_answer_bits_per_byte"]:.3f} bits/byte, true-boundary EOS {m["eos_top1_at_true_boundary"]}/{m["n"]}')
    print(out['status'])


if __name__ == '__main__':
    summarize()
