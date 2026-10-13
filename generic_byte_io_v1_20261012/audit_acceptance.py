"""Reduce saved IO acceptance receipts independently; execute no trajectories."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'technical' / 'acceptance'
spec = json.loads((ROOT / 'ACCEPTANCE_SPEC.json').read_text())
summary = json.loads((OUT / 'SUMMARY.json').read_text())
lock = json.loads((OUT / 'SOURCE_LOCK.json').read_text())
unit = json.loads((ROOT / 'technical' / 'UNIT_TESTS.json').read_text())
independent = json.loads((ROOT / 'technical' / 'INDEPENDENT_AUDIT.json').read_text())
for key, value in lock['files'].items():
    assert hashlib.sha256((ROOT / key).read_bytes()).hexdigest() == value
assert unit['status'] == 'PASS'
assert all(value is True for value in independent['checks'].values())
core_hash = lock['files']['generic_byte_io/core.py']
assert unit['source_sha256']['generic_byte_io/core.py'] == core_hash == independent['source_sha256']
paths = sorted(OUT.glob('FIXTURE_*.json'))
assert len(paths) == len(spec['world_seeds'])
world_metrics = []
scored = 0
target_events = 0
payload_bytes = 0
for path in paths:
    receipt = json.loads(path.read_text())
    assert receipt['seed'] in spec['world_seeds']
    assert receipt['classification'] == 'ENGINEERING_ACCEPTANCE_NOT_SCIENTIFIC_CONFIRMATION'
    branches = receipt['initial']
    all_rows = list(branches.values()) + [rows for cohorts in receipt['final'].values() for rows in cohorts.values()]
    for rows in all_rows:
        assert len(rows) == 16
        for row in rows:
            prompt = bytes.fromhex(row['prompt_hex'])
            expected = bytes.fromhex(row['expected_hex'])
            emitted = bytes.fromhex(row['emitted_hex'])
            correct = row['status'] == 'ENDED' and emitted == expected
            assert row['correct'] is correct
            assert row['selected_events'] == len(emitted) + 1
            scored += 1
    old = receipt['final']['W']['OLD']
    new = receipt['final']['W']['NEW']
    assert len({row['prompt_hex'] for row in old + new}) == 32
    assert set(bytes.fromhex(old[10]['expected_hex'])) == set(range(256))
    assert set(bytes.fromhex(new[10]['expected_hex'])) == set(range(256))
    for rows in (old, new):
        assert all(bytes.fromhex(rows[i]['prompt_hex'])[-4:] == b'\x00=\n\xff' for i in range(4, 10))
        for row in rows:
            plen, alen = len(bytes.fromhex(row['prompt_hex'])), len(bytes.fromhex(row['expected_hex']))
            payload_bytes += (plen + alen) * spec['epochs_per_cohort'] * 2
            target_events += (plen + alen + 1) * spec['epochs_per_cohort'] * 2
    accuracy = lambda rows: sum(row['correct'] for row in rows) / len(rows)
    metrics = {'old_immediate_W': accuracy(branches['W']),
               'old_final_W': accuracy(old), 'new_final_W': accuracy(new),
               'old_final_N': accuracy(receipt['final']['N']['OLD']),
               'old_final_ERASE': accuracy(receipt['final']['ERASE']['OLD'])}
    assert metrics == receipt['metrics']
    world_metrics.append(metrics)
    assert receipt['no_write_frozen'] is True and receipt['inference_learning_frozen'] is True
means = {key: sum(row[key] for row in world_metrics) / len(world_metrics) for key in world_metrics[0]}
assert means == summary['means']
assert scored == summary['scored_sequence_episodes'] == 1024
assert target_events == summary['training']['observed_target_events'] == 373632
assert payload_bytes == summary['training']['payload_bytes'] == 369536
rules = spec['pass']
assert all(means[key] >= rules['exact_sequence_and_model_EOS_mean_min'] for key in ('old_immediate_W', 'old_final_W', 'new_final_W'))
assert all(min(row['old_final_W'], row['new_final_W']) >= rules['each_world_old_and_new_retained_min'] for row in world_metrics)
assert means['old_final_W'] - means['old_final_N'] >= rules['old_gain_over_no_write_and_erase_min']
assert means['old_final_W'] - means['old_final_ERASE'] >= rules['old_gain_over_no_write_and_erase_min']
result = {'verdict': 'ACCEPT_ENGINEERING_IO_BASELINE',
          'classification': 'E0_PLUS_BOUNDED_E1_NO_SCIENTIFIC_PROMOTION',
          'unit_tests_passed': unit['tests_run'],
          'independent_checks_passed': len(independent['checks']),
          'fixtures': len(paths), 'distinct_context_answer_sequences': 256,
          'scored_sequence_episodes': scored,
          'actual_observed_payload_bytes_W_and_N': payload_bytes,
          'actual_observed_target_events_W_and_N': target_events,
          'actual_plastic_updates_W': target_events // 2,
          'actual_plastic_updates_N': 0,
          'means': means, 'source_identity_matches': True,
          'source_sha256': core_hash,
          'frozen_experiments_modified': False,
          'inherited_addition_or_native_Full151_claim': False}
(ROOT / 'FINAL_ENGINEERING_AUDIT.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
