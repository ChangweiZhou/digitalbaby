#!/usr/bin/env python3
"""Bounded lifecycle checks for the V43 completion driver; engineering only."""
from __future__ import annotations
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
import contextlib
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import types

import numpy as np

HERE = Path(__file__).resolve().parent
CORE_SHA = 'fb8ebb5355ea03e79d6049e5498db60772b18aa2bd51a54beed0689266a77bd8'
DRIVER_SHA = '456daa818b500a6d9ad128e52b43451626a691f674c7141d018cc605071d4b2d'
CHECKS = []


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(name, **details):
    row = {'name': name, 'passed': True, **details}; CHECKS.append(row)
    print(json.dumps(row), flush=True)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module); return module


def run(source, out, name, extra=(), expected=0):
    log = WORK/(name+'.log')
    command = [sys.executable, str(source), '--smoke', '--jobs', '1', '--checkpoint-records', '2', '--out-dir', str(out), *extra]
    with log.open('wb') as stream:
        proc = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=240)
    text = log.read_text()
    if expected is None:
        assert proc.returncode not in (0, 75), (name, proc.returncode, text[-3000:])
    else: assert proc.returncode == expected, (name, proc.returncode, expected, text[-5000:])
    return text


def completed(out):
    path = out/'completed'/'seed_943901_load_16.json'
    record = json.loads(path.read_bytes())
    return path, out/record['data']['authority_events_file'], record


def compare(out, baseline):
    p, e, record = completed(out); bp, be, baseline_record = completed(baseline)
    assert p.read_bytes() == bp.read_bytes(), 'Scientific completed JSON differs'
    assert e.read_bytes() == be.read_bytes(), 'Scientific authority NPZ differs'
    assert record['identity'] == baseline_record['identity']
    return {'completed_sha256': sha(p), 'authority_npz_sha256': sha(e),
            'physical_byte_events': record['data']['byte_events']}


def clone(src, dst, partial=False):
    dst.mkdir()
    names = ('PLAN.json', 'MANIFEST.json', 'ENGINEERING.json')
    for name in names:
        if (src/name).exists(): shutil.copy2(src/name, dst/name)
    for name in (('checkpoints', 'progress') if partial else ('completed', 'authority_events')):
        if (src/name).exists(): shutil.copytree(src/name, dst/name)
    return dst


def rejected(source, out, name, needle, preserve):
    before = sha(preserve)
    text = run(source, out, name, expected=None)
    assert needle in text, (needle, text[-3000:])
    assert sha(preserve) == before
    assert '"status": "launched"' not in text, 'Worker launched before invalid input rejection'
    return {'diagnostic': needle, 'invalid_input_preserved': True, 'worker_launches': 0}


def synthetic_npz(root):
    root.mkdir(parents=True, exist_ok=True); (root/'authority_events').mkdir()
    n = 16
    arrays = {'route': np.array(['G']*n), 'seed': np.repeat([949901, 949902], 8),
              'load': np.tile([64, 128], 8), 'pF_true': np.full(n, .2), 'pS_true': np.full(n, .4),
              'support': np.full(n, 2), 'slot_age': np.arange(n)+2,
              'time_since_last_use': np.arange(n)+1, 'prior_similarity_use_count': np.arange(n),
              'similarity_top1': np.full(n, .9), 'similarity_top2': np.full(n, .8),
              'similarity_margin': np.full(n, .1), 'fast_slow_agree': np.ones(n),
              'fast_entropy': np.ones(n), 'slow_entropy': np.ones(n),
              'fast_margin': np.full(n, .2), 'slow_margin': np.full(n, .3),
              'js_divergence': np.full(n, .02), 'local_E_before': np.linspace(-1, 1, n),
              'local_w_before': np.linspace(.2, .8, n),
              'revision_age': np.array([-1, 2]*8, dtype=np.int64),
              'was_revised': np.array([0, 1]*8, dtype=np.int8)}
    np.savez_compressed(root/'authority_events'/'synthetic.npz', **arrays)
    return arrays


def main():
    global WORK
    parser = argparse.ArgumentParser()
    parser.add_argument('--driver', type=Path, default=HERE.parent/'elm_v43_fast_slow_authority_complete.py')
    parser.add_argument('--baseline', type=Path, default=HERE/'cli_verification_fb8ebb5355ea'/'uninterrupted')
    args = parser.parse_args(); source = args.driver.resolve(); baseline = args.baseline.resolve()
    assert sha(source) == DRIVER_SHA
    WORK = HERE/('completion_verification_'+DRIVER_SHA[:12]); suffix = 1
    while WORK.exists():
        suffix += 1; WORK = HERE/('completion_verification_'+DRIVER_SHA[:12]+'_attempt'+str(suffix))
    WORK.mkdir()
    baseline_plan = json.loads((baseline/'PLAN.json').read_bytes())
    assert baseline_plan['source_sha256'] == CORE_SHA
    initial_source = sha(source)

    full = WORK/'uninterrupted'; run(source, full, 'uninterrupted')
    check('completion_smoke_exactly_matches_frozen_core', **compare(full, baseline))
    provenance = json.loads((full/'ANALYSIS_IMPLEMENTATION.json').read_bytes())
    assert provenance['scientific_core_sha256'] == CORE_SHA
    assert provenance['completion_driver_sha256'] == DRIVER_SHA
    assert not any(provenance[key] for key in ('physical_learning_changes', 'policy_scoring_changes', 'model_or_hyperparameter_changes', 'gate_changes'))
    assert sha(full/'elm_v43_fast_slow_authority.py') == CORE_SHA
    assert sha(full/'diagnostics_corrected.py') == provenance['corrected_diagnostics_sha256']
    check('core_and_analysis_provenance_verified', corrected_diagnostics_sha256=provenance['corrected_diagnostics_sha256'])

    stopped = WORK/'stopped'; run(source, stopped, 'stopped', extra=('--stop-after-checkpoints', '1'), expected=75)
    progress = json.loads((stopped/'progress'/'seed_943901_load_16.json').read_bytes())
    assert 0 < progress['cursor'] < progress['records'] and not progress['done']
    run(source, stopped, 'stopped_resume')
    check('completion_checkpoint_pause_and_resume_exact', paused_record=progress['cursor'], **compare(stopped, baseline))

    old_partial = HERE/'cli_verification_fb8ebb5355ea'/'paused_snapshot'
    assert (old_partial/'checkpoints'/'seed_943901_load_16.checkpoint').exists()
    old = clone(old_partial, WORK/'original_partial', partial=True)
    old_cursor = json.loads((old/'progress'/'seed_943901_load_16.json').read_bytes())['cursor']
    run(source, old, 'original_partial_resume')
    check('original_core_partial_checkpoint_reused_exactly', original_cursor=old_cursor, **compare(old, baseline))

    old_completed = clone(baseline, WORK/'original_completed')
    text = run(source, old_completed, 'original_completed_reuse')
    assert '"new_completed_trajectories": 0' in text and '"reused_trajectories": 1' in text
    assert '"status": "launched"' not in text
    check('original_completed_results_reused_without_learning', **compare(old_completed, baseline), new_worker_launches=0)

    before = {str(p.relative_to(full)): sha(p) for folder in ('completed', 'authority_events', 'logs') for p in (full/folder).glob('*') if p.is_file()}
    text = run(source, full, 'completed_rerun')
    after = {str(p.relative_to(full)): sha(p) for folder in ('completed', 'authority_events', 'logs') for p in (full/folder).glob('*') if p.is_file()}
    assert before == after and '"new_completed_trajectories": 0' in text and '"status": "launched"' not in text
    check('completed_rerun_zero_learning', new_physical_events=0, unchanged_scientific_outputs=True)

    badcore = clone(baseline, WORK/'bad_core'); p = badcore/'elm_v43_fast_slow_authority.py'; p.write_text('# wrong core\n')
    check('wrong_materialized_core_rejected', **rejected(source, badcore, 'bad_core', 'Existing frozen scientific core differs', p))
    badplan = clone(baseline, WORK/'bad_plan'); p = badplan/'PLAN.json'
    changed = json.loads(p.read_text()); changed['unexpected_mutation'] = True; p.write_text(json.dumps(changed))
    check('wrong_plan_rejected', **rejected(source, badplan, 'bad_plan', 'Existing plan/source mismatch', p))
    badanalysis = clone(baseline, WORK/'bad_analysis'); p = badanalysis/'ANALYSIS_IMPLEMENTATION.json'
    changed = dict(provenance); changed['corrected_diagnostics_sha256'] = 'wrong'; p.write_text(json.dumps(changed))
    check('wrong_analysis_provenance_rejected', **rejected(source, badanalysis, 'bad_analysis', 'Existing analysis implementation identity mismatch', p))
    locked = clone(baseline, WORK/'concurrent_original_writer'); p = locked/'locks'/'runner.lock'; p.parent.mkdir(); p.touch()
    with p.open('a+') as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        check('original_runner_lock_blocks_completion_writer', **rejected(source, locked, 'locked', 'Another writer holds', p))
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    driver = load('v43_completion_verification_driver', source)
    diagnostic_source = driver.unpack('diagnostics')
    direct = types.ModuleType('v43_diagnostics_direct_verification'); sys.modules[direct.__name__] = direct
    exec(compile(diagnostic_source, '<direct-corrected-diagnostic>', 'exec'), direct.__dict__)
    direct_root = WORK/'synthetic_direct'; synthetic_npz(direct_root)
    direct_identity_before = direct._implementation_hash()
    dresult = direct.run(direct_root)
    assert dresult['status'] == 'complete' and dresult['fold_count'] == 6
    assert all(row['constant_training_class'] for row in dresult['fold_audit'])
    assert direct._implementation_hash() == direct_identity_before
    wrapped_root = WORK/'synthetic_wrapped'; synthetic_npz(wrapped_root)
    shutil.copytree(direct_root/'diagnostic_checkpoints', wrapped_root/'diagnostic_checkpoints')
    caches_before = {p.name: sha(p) for p in (wrapped_root/'diagnostic_checkpoints').glob('*')}
    core = driver.prepare_core(wrapped_root)
    with core.writer_lock(wrapped_root/'locks'/'runner.lock'):
        wrapped = core.MODULES['diagnostics'].run(wrapped_root)
    assert wrapped['status'] == 'complete' and wrapped['fit_count'] == 0 and wrapped['reused_fold_count'] == 6
    assert wrapped['input_identity'] == dresult['input_identity']
    assert wrapped['pooled_descriptive_metrics'] == dresult['pooled_descriptive_metrics']
    assert caches_before == {p.name: sha(p) for p in (wrapped_root/'diagnostic_checkpoints').glob('*')}
    p = json.loads((wrapped_root/'ANALYSIS_IMPLEMENTATION.json').read_bytes())
    assert p['diagnostic_implementation_sha256'] == direct_identity_before
    check('guarded_diagnostic_identity_and_direct_cache_reuse', synthetic_seeds=2, synthetic_events=16,
          real_scientific_fits=0, estimator_fits=0, reused_folds=6,
          implementation_sha256=direct_identity_before, input_identity=wrapped['input_identity'])

    # A disposable driver demonstrates the guard around diagnostic execution.
    disposable = WORK/'disposable_complete.py'; shutil.copy2(source, disposable)
    gd = load('v43_completion_guard_fixture', disposable)
    guardroot = WORK/'synthetic_guard'; synthetic_npz(guardroot); gc = gd.prepare_core(guardroot)
    disposable.write_bytes(disposable.read_bytes()+b'\n# verification-only changed driver\n')
    try: gc.MODULES['diagnostics'].run(guardroot)
    except RuntimeError as exc: assert 'Completion driver changed during execution' in str(exc)
    else: raise AssertionError('Diagnostic accepted changed driver')
    assert not (guardroot/'diagnostic_checkpoints').exists()
    check('changed_driver_rejected_before_diagnostic_work', diagnostic_fits=0)

    assert sha(source) == initial_source == DRIVER_SHA
    report = {'passed': True, 'completion_driver_sha256': DRIVER_SHA, 'scientific_core_sha256': CORE_SHA,
              'corrected_diagnostics_sha256': provenance['corrected_diagnostics_sha256'],
              'scientific_trajectories_run': 0, 'synthetic_diagnostic_estimator_fits': 0,
              'checks': CHECKS, 'n_checks': len(CHECKS), 'work_directory': str(WORK)}
    (HERE/'completion_verification.json').write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
    (WORK/'VERIFICATION.json').write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
    text = '# V43 completion-driver verification\n\nAll '+str(len(CHECKS))+' checks passed. No scientific trajectory or real diagnostic fit was run.\n\n'
    text += f'Driver SHA256: `{DRIVER_SHA}`.\n\nFrozen core SHA256: `{CORE_SHA}`.\n\n'
    text += '\n'.join('- '+r['name'].replace('_', ' ')+'.' for r in CHECKS)+'\n\n'
    text += 'Smoke completed JSON and authority NPZ files were byte-identical to the original frozen-core baseline. '
    text += 'The synthetic two-seed diagnostic used constant training targets, so it tested six fold caches and corrected sentinel decoding without fitting an estimator. '
    text += 'Guarded execution reused the direct corrected module’s exact cache identity and all six folds. '
    text += 'The original scientific source and real experiment outputs were not modified.\n'
    (HERE/'completion_verification.md').write_text(text)
    print(json.dumps({'passed': True, 'n_checks': len(CHECKS), 'report': str(HERE/'completion_verification.json')}), flush=True)


if __name__ == '__main__':
    try: main()
    except BaseException as exc:
        if 'WORK' in globals():
            (WORK/'FAILED.json').write_text(json.dumps({'passed': False, 'checks': CHECKS, 'error': repr(exc)}, indent=2)+'\n')
        raise
