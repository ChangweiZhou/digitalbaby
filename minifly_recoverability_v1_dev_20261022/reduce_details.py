"""Artifact-only diagnostics and paired contrasts. No native execution or fitting."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import json
from pathlib import Path
from collections import Counter
import numpy as np
from artifact_inference import logits_for

P = Path(__file__).resolve().parent


def emit(z):
    result = []
    for q in z:
        bs = []
        ended = False
        for j in range(9):
            if q[j, 8] > 0:
                ended = True
                break
            if j == 8:
                break
            bs.append(sum(1 << b for b in range(8) if q[j, b] > 0))
        result.append((bs, ended))
    return result


def metric(z, values):
    output = emit(z)
    correct = [ended and bs == v for (bs, ended), v in zip(output, values)]
    bits = [(z[i, j, b] > 0) == bool(t & (1 << b))
            for i, v in enumerate(values) for j, t in enumerate(v) for b in range(8)]
    byte_ok = [all((z[i, j, b] > 0) == bool(t & (1 << b)) for b in range(8))
               for i, v in enumerate(values) for j, t in enumerate(v)]
    nonempty = [i for i, v in enumerate(values) if v]
    payload = [all(all((z[i, j, b] > 0) == bool(t & (1 << b)) for b in range(8))
                   for j, t in enumerate(values[i])) for i in nonempty]
    return dict(exact=float(np.mean(correct)),
                nonempty_exact=float(np.mean([correct[i] for i in nonempty])),
                bit_accuracy=float(np.mean(bits)), byte_accuracy=float(np.mean(byte_ok)),
                forced_length_payload_exact=float(np.mean(payload)),
                early_EOS=float(np.mean([ended and len(bs) < len(v) for (bs, ended), v in zip(output, values)])),
                late_EOS=float(np.mean([ended and len(bs) > len(v) for (bs, ended), v in zip(output, values)])),
                truncation=float(np.mean([not ended for _, ended in output])),
                per_length={str(L): float(np.mean([c for c, v in zip(correct, values) if len(v) == L])) for L in range(9)})


def main():
    detailed = []
    absent = []
    for file in sorted((P / 'traces').glob('*.json')):
        if file.stem.endswith('_decoded'):
            continue
        meta = json.loads(file.read_text())
        if meta['seed'] < 202610229200:
            continue
        dec = json.loads(file.with_name(file.stem + '_decoded.json').read_text())
        with np.load(file.with_suffix('.npz')) as raw:
            for row in dec['results']:
                values = meta['labels'][meta['pools'].index(row['pool'])]
                z = logits_for(P, meta, row, raw)
                identity = {k: meta[k] for k in ('seed', 'life', 'STD', 'arm')}
                identity.update({k: row.get(k) for k in ('pool', 'readout', 'head_seed', 'lesion')})
                if row['pool'] == 'ABSENT':
                    out = emit(z)
                    absent.append(dict(**identity, empty_emission=float(np.mean([end and not bs for bs, end in out])),
                                       nonempty_emission=float(np.mean([bool(bs) for bs, _ in out])),
                                       truncation=float(np.mean([not end for _, end in out]))))
                    continue
                metrics = metric(z, values)
                assert metrics['exact'] == row['metrics']['exact']
                assert metrics['bit_accuracy'] == row['metrics']['bits']
                extra = {}
                if meta['arm'] == 'SH' and row['pool'].startswith('OLD'):
                    extra['actually_taught_nonempty_exact'] = metric(z, meta['observed_OLD'])['nonempty_exact']
                detailed.append(dict(**identity, **metrics, **extra))

    summary = json.loads((P / 'SUMMARY.json').read_text())
    contrasts = []
    for pair in range(202610229301, 202610229305):
        for cohort in ('OLD_final', 'NEW_final'):
            cells = {(r['STD'], r['readout']): r['W_NONE'] for r in summary['pair_rows'] if r['pair'] == pair and r['cohort'] == cohort}
            contrasts.append(dict(pair=pair, cohort=cohort,
                SEP_minus_MEAN_RAW=cells[False, 'SEP'] - cells[False, 'MEAN'],
                SEP_minus_MEAN_STD=cells[True, 'SEP'] - cells[True, 'MEAN'],
                STD_minus_RAW_MEAN=cells[True, 'MEAN'] - cells[False, 'MEAN'],
                STD_minus_RAW_SEP=cells[True, 'SEP'] - cells[False, 'SEP'],
                interaction=(cells[True, 'SEP'] - cells[True, 'MEAN']) - (cells[False, 'SEP'] - cells[False, 'MEAN'])))

    # Pooled predictor is evaluator-only. Fit unique training facts once; their
    # fourteen repeated native exposures have identical empirical frequencies.
    training_values = []
    for seed in range(202610229101, 202610229113):
        meta = json.loads((P / 'traces' / f'{seed}_0_0_W.json').read_text())
        training_values += meta['observed_OLD'] + meta['labels'][meta['pools'].index('NEW_end')]
    lengths = Counter(map(len, training_values))
    payloads = Counter(b for v in training_values for b in v)
    length = min(lengths, key=lambda k: (-lengths[k], k))
    byte = min(payloads, key=lambda k: (-payloads[k], k))
    prediction = [byte] * length
    reference = dict(evaluator_only=True, unique_training_facts=len(training_values),
        repeated_native_exposures_per_fact=14, length_counts=dict(lengths), byte_counts=dict(payloads),
        tie_break='smallest integer', predicted_value=prediction,
        balanced_cohort_all_value_exact=2 / 18 if not prediction else None,
        balanced_cohort_nonempty_exact=0.0 if not prediction else None,
        exact_table='864 W known-value lookups verified in audit.py; not a candidate backend')
    (P / 'DETAIL_METRICS.json').write_text(json.dumps(dict(status='ARTIFACT_REDUCTION_ONLY', rows=detailed, absent_rows=absent, paired_contrasts=contrasts), indent=2) + '\n')
    (P / 'REFERENCE_COUNTERMODELS.json').write_text(json.dumps(reference, indent=2) + '\n')
    print(json.dumps(dict(metric_rows=len(detailed), absent_rows=len(absent), contrasts=len(contrasts), pooled_prediction=prediction)))


if __name__ == '__main__':
    main()
