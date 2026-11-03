#!/usr/bin/env python3
"""Reproduce the descriptive quantities used in the V43 analysis narrative.

No fitting, learner updates, new qualification tests, or threshold changes.
Run after the frozen reports and complete-data audit have finished.
"""
from pathlib import Path
import argparse
import collections
import hashlib
import json
import statistics


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def describe(values):
    values = list(values)
    if not values or any(v is None for v in values):
        return {'n': len(values), 'mean': None, 'minimum': None, 'maximum': None}
    return {'n': len(values), 'mean': statistics.mean(values),
            'minimum': min(values), 'maximum': max(values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()
    root = Path(args.out_dir).resolve()
    plan = json.loads((root / 'PLAN.json').read_bytes())
    reports = {}
    hashes = {}
    for name in ('REPORT.json', 'ALL_CELLS_SUMMARY.json', 'EXPOSURE_SCOPE.json',
                 'DIAGNOSTIC_FIRST_COMPLETION.json', 'INDEPENDENT_AUDIT.json'):
        raw = (root / name).read_bytes()
        hashes[name] = sha(raw)
        reports[name] = json.loads(raw)
    assert reports['REPORT.json']['status'] == 'complete'
    assert reports['INDEPENDENT_AUDIT.json']['passed']
    rows = []
    for job in plan['jobs']:
        path = root / 'completed' / f"seed_{job['seed']}_load_{job['load']}.json"
        raw = path.read_bytes()
        envelope = json.loads(raw)
        assert envelope['identity'] == {'plan_sha256': sha(canonical(plan)), 'job': job}
        assert envelope['sha256'] == sha(canonical(envelope['data']))
        rows.append(envelope['data'])
        hashes[str(path.relative_to(root))] = sha(raw)
    assert len(rows) == 64
    invalid = {(x['seed'], x['load']) for x in reports['REPORT.json']['physical_capacity_invalid']}
    result = {
        'scope': 'All16seed cells per load, including physical acquisition failures; descriptive only.',
        'new_learner_updates': 0, 'new_model_fits': 0, 'qualification_changes': False,
        'input_sha256': hashes, 'helper_sha256': sha(Path(__file__).read_bytes()),
        'loads': [], 'gates': {k: v['status'] for k, v in reports['REPORT.json']['gates'].items()},
        'sample_accounting': reports['REPORT.json']['sample_accounting'],
    }
    metrics = ['whole_bpb', 'G_bpb', 'A_after_D_heldout', 'E1_all', 'E2_all', 'oracle_capture']
    for load in (64, 128, 256, 512):
        cells = sorted((x for x in rows if x['load'] == load), key=lambda x: x['seed'])
        assert len(cells) == 16
        item = {'load': load, 'invalid_seeds': [x['seed'] for x in cells if (x['seed'], load) in invalid],
                'policies': {}, 'acquisition': {}, 'paired_descriptive_differences': []}
        item['valid_cells'] = 16 - len(item['invalid_seeds'])
        for checkpoint in ('after_A', 'after_B', 'after_C', 'after_D'):
            acq = [next(a for a in x['audits']['physical_acquisition_current'] if a['checkpoint'] == checkpoint) for x in cells]
            item['acquisition'][checkpoint] = {k: describe(a[k] for a in acq)
                                               for k in ('population_accuracy', 'mature_coverage')}
        for policy in plan['policies']:
            item['policies'][policy] = {m: describe(x['metrics_by_arm'][policy].get(m) for x in cells) for m in metrics}
        pairs = [('LOCAL-BAYES-E2', 'SILENT-HARD'), ('LOCAL-BAYES-E2', 'FIXED-50'),
                 ('LOCAL-BAYES-E2', 'LOCAL-BAYES-0'), ('LOCAL-BAYES-E2', 'SUPER'),
                 ('FIXED-50', 'SUPER'), ('FIXED-50', 'SILENT-HARD'),
                 ('LOCAL-BAYES-0', 'SUPER'), ('LOCAL-BAYES-0', 'FIXED-50')]
        for a, b in pairs:
            for metric in ('whole_bpb', 'G_bpb', 'A_after_D_heldout'):
                values = [x['metrics_by_arm'][a][metric] - x['metrics_by_arm'][b][metric] for x in cells]
                item['paired_descriptive_differences'].append({
                    'a': a, 'b': b, 'metric': metric, **describe(values),
                    'negative_cells': sum(v < 0 for v in values),
                    'equal_cells': sum(v == 0 for v in values),
                    'positive_cells': sum(v > 0 for v in values)})
        item['local_E2_minus_shuffled_permutable_G'] = describe(
            x['metrics_by_arm']['LOCAL-BAYES-E2']['G_permutable_bpb'] -
            x['metrics_by_arm']['SHUFFLED-ALPHA']['G_permutable_bpb'] for x in cells)
        item['memory'] = {
            'full_slot_capacity_cells': sum(x['costs']['slow']['slots'] == 384 for x in cells),
            'slots': describe(x['costs']['slow']['slots'] for x in cells),
            'slot_evictions': describe(x['physical_events'].get('slot_eviction:capacity', 0) for x in cells),
            'maximum_slow_semantic_MiB': max(x['costs']['slow_peak']['semantic_bytes'] for x in cells) / 2**20,
        }
        result['loads'].append(item)
    diagnostic = reports['DIAGNOSTIC_FIRST_COMPLETION.json']
    result['diagnostics'] = {
        'G_events': diagnostic['route_G_events'], 'folds': diagnostic['fold_count'],
        'warnings_by_model': dict(collections.Counter(row['model'] for row in diagnostic['fold_audit'] if row['fit_warnings'])),
        'pooled_descriptive_metrics': diagnostic['pooled_descriptive_metrics'],
        'note': 'Event-pooled held-out-seed metrics, not independent-event inference or deployable read-policy validation.',
    }
    execution = []
    for path in sorted((root / 'execution').glob('seed*.json')):
        raw = path.read_bytes()
        hashes[str(path.relative_to(root))] = sha(raw)
        execution.append(json.loads(raw))
    assert len(execution) == 64
    result['execution_resources'] = {
        'peak_scientific_worker_RSS_MiB': max(x['peak_rss_bytes'] for x in execution) / 2**20,
        'scope': 'Final per-cell execution receipts, including result serialization; distinct from checkpoint-time RSS.',
    }
    target_count = sum(x['groups']['target']['SUPER']['n'] for x in rows)
    assert target_count == result['sample_accounting']['target_presentations']
    path = root / 'ANALYSIS_DIGEST.json'
    path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps({'file': str(path), 'cells': len(rows), 'sha256': sha(path.read_bytes()), 'new_learning': 0}))


if __name__ == '__main__':
    main()
