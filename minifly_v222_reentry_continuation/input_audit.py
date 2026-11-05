"""One-digit matched prompt audit on the fixed counterfactual panel."""
from run import ARMS, DOSES, ROOT, SEED, Learner, addition, make_world, write_json
from probes import paired_trace, diagnose


def audit(model, pairs, score_answers):
    digest = model.digest()
    rows = {}
    for mode in ('frozen', 'causal'):
        causal = mode == 'causal'
        mode_rows = []
        for left, right in pairs:
            a, gold_a = addition(*left)
            b, gold_b = addition(*right)
            differences = paired_trace(model, a, b, causal)
            row = {
                'left': a.decode(), 'right': b.decode(),
                'gold_left': gold_a.decode(), 'gold_right': gold_b.decode(),
                'onset': differences[-1],
                'sampled_kc_diff_at_any_step': any(p['sampled_kc'] > 0 for p in differences),
                'writer_value_diff_at_any_step': any(p.get('writer_value', 0) > 0 for p in differences),
            }
            if score_answers:
                ra, rb = diagnose(model, a, gold_a, causal), diagnose(model, b, gold_b, causal)
                row.update(both_exact=ra['exact'] and rb['exact'],
                           both_first_correct=ra['first_byte_correct'] and rb['first_byte_correct'],
                           emitted_differ=ra['emitted'] != rb['emitted'])
            mode_rows.append(row)
        rows[mode] = mode_rows
    assert model.digest() == digest
    summary = {}
    for mode, values in rows.items():
        summary[mode] = {'n': len(values), **{
            f'{k}_different_at_onset': sum(r['onset'][k] > 0 for r in values)
            for k in ('sampled_kc', 'workspace', 'memory', 'read', 'logp')
        }, **{k: sum(r[k] for r in values) for k in
              ('sampled_kc_diff_at_any_step', 'writer_value_diff_at_any_step')}}
        if score_answers:
            for key in ('both_exact', 'both_first_correct', 'emitted_differ'):
                summary[mode][key] = sum(r[key] for r in values)
    return {'summary': summary, 'pairs': rows}


def main():
    pairs = make_world(SEED)['counterfactual']
    results = {'birth': audit(Learner(SEED), pairs, False)}
    for arm in ARMS:
        folder = ROOT/'results'/'curve'/arm
        base = folder/f'{DOSES[-1]}.npz'
        exact_dose = folder/f'{DOSES[-1]}_exact_dose.npz'
        if base.exists():
            model = Learner.load(exact_dose if exact_dose.exists() else base)
            results[arm] = audit(model, pairs, True)
    write_json(ROOT/'INPUT_AUDIT.json', results)
    print({arm: values['summary'] for arm, values in results.items()})


if __name__ == '__main__':
    main()
