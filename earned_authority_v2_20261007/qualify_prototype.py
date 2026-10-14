"""Prototype checks and bounded-state/cost measurement, no native learning."""
import copy
import gzip
import io
import json
import sys
import time
import unittest
from pathlib import Path
from experience_policy import ExperienceAuthority, Observation, sensory_key, SCALES
import test_experience_policy
from replay_experience import audit_record, make_world

ROOT = Path(__file__).resolve().parent


def deep_bytes(obj, seen=None):
    seen = set() if seen is None else seen
    if id(obj) in seen:
        return 0
    seen.add(id(obj))
    n = sys.getsizeof(obj)
    if isinstance(obj, dict):
        n += sum(deep_bytes(k, seen) + deep_bytes(v, seen) for k, v in obj.items())
    elif isinstance(obj, (tuple, list)):
        n += sum(deep_bytes(v, seen) for v in obj)
    elif isinstance(obj, ExperienceAuthority):
        n += deep_bytes(obj.__dict__, seen)
    return n


def main():
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromModule(test_experience_policy)
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    (ROOT / 'TEST_LOG.txt').write_text(stream.getvalue())
    if not result.wasSuccessful():
        raise AssertionError('prototype tests failed')
    paths = sorted((ROOT.parent / 'next_core_budget_v2_20261006/science/confirm/receipts').glob('*_ERROR_reuse.json.gz'))
    d = json.load(gzip.open(paths[0], 'rt'))
    w = make_world(d['world'], 'reuse')
    original = d['branches']['N_old']['records'][0]
    audit_record(original, w['events'][0], 'N_old')
    bad = []
    for name, mutate in (
        ('metadata_write_true', lambda r: r.update(learn=True)),
        ('one_native_store_writes', lambda r: r['actual_calls'][5].update(write=True)),
        ('nonzero_disabled_write', lambda r: r['actual_calls'][0].update(applied_l1=1)),
        ('disabled_reference_mismatch', lambda r: r['actual_calls'][0].update(no_write_reference_equal=False)),
        ('teacher_substitution', lambda r: r['write'].update(outcome=ord('3') if r['write']['outcome'] != ord('3') else ord('0'))),
        ('late_prediction', lambda r: r.update(prediction_precedes_outcome=False)),
        ('missing_store_call', lambda r: r['actual_calls'].pop()),
        ('wrong_record_cursor', lambda r: r.update(index=999)),
    ):
        t = copy.deepcopy(original)
        mutate(t)
        try:
            audit_record(t, w['events'][0], 'N_old')
        except AssertionError:
            bad.append(name)
        else:
            raise AssertionError('tamper accepted: ' + name)
    empty = ExperienceAuthority()
    full = ExperienceAuthority()
    ob = Observation((0., 1., 2., 3.), (3., 2., 1., 0.))
    for i in range(64):
        full.predict(i.to_bytes(4, 'big'), ob)
        full.observe(48, learn=True)
    full_bytes = deep_bytes(full)
    begin = time.process_time()
    for _ in range(10000):
        full.query((b'1234', b'5678'), (ob, ob))
    query_seconds = (time.process_time() - begin) / 10000
    begin = time.process_time()
    for _ in range(10000):
        full.predict(b'1234', ob)
        full.observe(48, learn=True)
    teacher_seconds = (time.process_time() - begin) / 10000
    report = {'status': 'PROTOTYPE_TECHNICAL_PASS_NOT_NATIVE_QUALIFICATION',
              'unit_test_methods': result.testsRun,
              'snapshot_tamper_cases_rejected': 12, 'actual_write_tamper_cases_rejected': bad,
              'authority_logical_layout_bytes': empty.packed_bytes(),
              'authority_python_heap_empty_bytes': deep_bytes(empty),
              'authority_python_heap_full_bytes': full_bytes,
              'python_heap_method': 'recursive sys.getsizeof, deduplicating shared objects; excludes module/runtime arrays',
              'benchmark_queries': 10000, 'benchmark_teacher_events': 10000,
              'all_five_diagnostic_readouts_query_cpu_s': query_seconds,
              'all_five_diagnostic_readouts_predict_observe_cpu_s': teacher_seconds,
              'benchmark_is_not_native_trajectory': True,
              'native_birth_integration_qualified': False,
              'complete_native_plus_authority_checkpoint_qualified': False,
              'formal_science_launched': False}
    (ROOT / 'QUALIFICATION.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
