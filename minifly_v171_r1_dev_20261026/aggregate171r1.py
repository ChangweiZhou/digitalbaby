"""Independent reduction of immutable V171 saved own/TF panels; no model calls.

All-80 reporting is descriptive and does not replace the failed historical NEW
qualification gate. Each world has correlated OLD/NEW items and shared native
history; neither 160 items nor the reference roles are independent worlds.
"""
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PARENT = ROOT / 'minifly_v171_dev_20261026'
ROLES = ('NATIVE', 'EXACT3', 'EXACT_RECORD')


def run(output=HERE):
    t0, w0 = time.process_time(), time.perf_counter()
    hashes, panel_count, row_count, tf_sites = {}, 0, 0, 0

    def read(p, expected=None):
        b = p.read_bytes()
        digest = hashlib.sha256(b).hexdigest()
        if expected is not None:
            assert digest == expected, 'Saved panel receipt mismatch'
        hashes[str(p.relative_to(ROOT))] = digest
        return json.loads(b)

    source = read(PARENT / 'CALIBRATION_RESULTS.json')
    data = read(PARENT / 'local/TRAIN_DATA.json')
    out = {}

    def summarize(rows):
        nll = -sum(x['lp'] for row in rows for x in row['teacher_forced'])
        targets = sum(len(row['teacher_forced']) for row in rows)
        return dict(records=len(rows), complete_own_EOS_correct=sum(r['correct'] for r in rows),
                    TF_targets=targets, TF_negative_log_likelihood=nll,
                    TF_bits_per_target=nll / targets / math.log(2),
                    empty_EOS=sum(r['EOS'] and not r['output_hex'] for r in rows),
                    truncated=sum(r['CAP'] for r in rows))

    def paired(a, b, ids):
        # Positive differences favor b. a/b must describe the same scored items.
        assert set(ids) <= a.keys() and set(ids) <= b.keys()
        ac = {i for i in ids if a[i]['correct']}
        bc = {i for i in ids if b[i]['correct']}
        return dict(records=len(ids), both_correct=len(ac & bc),
                    a_only_correct=len(ac - bc), b_only_correct=len(bc - ac),
                    neither_correct=len(set(ids) - ac - bc),
                    complete_correct_delta_b_minus_a=len(bc) - len(ac))

    for world, wr in source['worlds'].items():
        d = data[world]
        groups = {g: {x['id'] for x in d[g]} for g in ('OLD', 'NEW')}
        groups['ALL80'] = groups['OLD'] | groups['NEW']
        expected_answers = {x['id']: x['answer'] for g in ('OLD', 'NEW') for x in d[g]}
        assert len(groups['OLD']) == 64 and len(groups['NEW']) == 16
        assert len(groups['ALL80']) == 80
        panels = {}
        for tag in ('OLD16', 'NEW256', 'final_OLD', 'delayed'):
            panels[tag] = {}
            for role in ROLES:
                receipt = wr['panels'][tag][role]
                p = read(PARENT / receipt['path'], receipt['sha256'])
                assert p['role'] == role
                rows = p['rows']
                by_id = {r['id']: r for r in rows}
                assert len(by_id) == len(rows)
                wanted = groups['NEW'] if tag == 'NEW256' else (groups['OLD'] if tag == 'final_OLD' else groups['ALL80'])
                assert set(by_id) == wanted
                for r in rows:
                    choices = r['choices']
                    eos = bool(choices) and choices[-1] == 256
                    assert r['EOS'] == eos and r['CAP'] == (not eos)
                    assert all(0 <= k < 256 for k in choices[:-1] if eos) if eos else all(0 <= k < 256 for k in choices)
                    payload = choices[:-1] if eos else choices
                    assert bytes(payload).hex() == r['output_hex']
                    assert r['correct'] == (eos and r['output_hex'] == expected_answers[r['id']])
                    assert [x['y'] for x in r['teacher_forced']] == list(bytes.fromhex(expected_answers[r['id']])) + [256]
                    assert all(math.isfinite(x['lp']) and x['lp'] <= 0 for x in r['teacher_forced'])
                s = summarize(rows)
                assert s['complete_own_EOS_correct'] == receipt['correct']
                assert s['TF_targets'] == receipt['TF_targets']
                assert math.isclose(s['TF_negative_log_likelihood'], receipt['TF_negative_log_likelihood'], rel_tol=1e-12, abs_tol=1e-10)
                panels[tag][role] = by_id
                panel_count += 1
                row_count += len(rows)
                tf_sites += s['TF_targets']
        final = {role: {**panels['final_OLD'][role], **panels['NEW256'][role]} for role in ROLES}
        role_metrics, comparisons = {}, {}
        for role in ROLES:
            old_initial = {i for i in groups['OLD'] if panels['OLD16'][role][i]['correct']}
            native_initial = {i for i in groups['OLD'] if panels['OLD16']['NATIVE'][i]['correct']}
            common = old_initial & native_initial
            old_final = {i for i in groups['OLD'] if final[role][i]['correct']}
            delayed = {i for i in groups['OLD'] if panels['delayed'][role][i]['correct']}
            role_metrics[role] = dict(
                final={g: summarize([final[role][i] for i in sorted(ids)]) for g, ids in groups.items()},
                delayed={g: summarize([panels['delayed'][role][i] for i in sorted(ids)]) for g, ids in groups.items()},
                OLD_initial_correct=len(old_initial), OLD_initial_retained_final=len(old_initial & old_final),
                OLD_initial_lost_final=len(old_initial - old_final),
                OLD_initial_retained_delayed=len(old_initial & delayed),
                shared_initial_OLD_with_native=len(common),
                shared_initial_OLD_retained_final=len(common & old_final),
                shared_initial_OLD_retention_measurable=bool(common),
                OLD_initial_to_final=paired(panels['OLD16'][role], final[role], groups['OLD']),
                final_to_delayed={g: paired(final[role], panels['delayed'][role], ids) for g, ids in groups.items()})
            if role != 'NATIVE':
                comparisons[role] = {g: paired(final['NATIVE'], final[role], ids) for g, ids in groups.items()}
        out[world] = dict(roles=role_metrics, native_a_vs_reference_b_final=comparisons)

    # Hard-coded independent historical totals catch accidental grouping mistakes.
    assert [[out[w]['roles'][r]['final']['ALL80']['complete_own_EOS_correct'] for r in ROLES]
            for w in sorted(out)] == [[4, 4, 4], [11, 10, 10]]
    for name, digest in hashes.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, 'Parent changed during reduction'
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != 'darwin':
        rss *= 1024
    result = dict(version='V171-R1', status='ACCEPTED_SAVED_PANEL_REDUCTION',
                  worlds=out, independent_worlds=2, taught_items_per_world=80,
                  independent_items_claimed=False, shared_history_roles_independent=False,
                  evidence_level='E1 descriptive reanalysis of taught items',
                  qualification_changed=False, historical_verdict=source['verdict'],
                  new_core_adopted=False, model_queries=0, new_training_records=0,
                  assessment_labels_used=False, new_fitted_parameters=0,
                  saved_panel_files_verified=panel_count, saved_rows_verified=row_count,
                  saved_TF_target_reuses=tf_sites, input_sha256=hashes,
                  CPU_seconds=time.process_time() - t0,
                  wall_seconds=time.perf_counter() - w0, peak_RSS_bytes=int(rss),
                  limits=['All80 is additional reporting, not a replacement gate.',
                          'OLD and NEW remain separate; OLD improvements cannot conceal failed NEW acquisition.',
                          'Taught items and reference roles share histories and are not independent replicates.',
                          'No trajectory, training presentation or missing technical record was replayed.'])
    output.mkdir(parents=True, exist_ok=True)
    (output / 'ALL80_RESULTS.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    r = run()
    print(json.dumps({w: {role: x['final']['ALL80']['complete_own_EOS_correct']
                         for role, x in v['roles'].items()} for w, v in r['worlds'].items()}))
