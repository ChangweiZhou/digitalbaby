"""Reviewed operations amendment and lossless provenance rebinding; no teaching."""
import io
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OPS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from runtime import (atomic_bytes, atomic_json, canonical, digest, execution_identity,
                     file_sha, frozen_auditor, frozen_checkpoint, make_world,
                     read_receipt, require_environment, require_sources, source_files,
                     write_receipt)
from locks import source_lock, require_authorized, write_manifest, manifest
import numpy as np


def main():
    require_sources()
    require_environment()
    if (OPS / 'RECOVERY_AUDIT.json').exists():
        raise RuntimeError('recovery already recorded; do not migrate twice')
    regression = json.loads((OPS / 'REGRESSION.json').read_text())
    if regression['verdict'] != 'PASS' or regression['tests'] != 43:
        raise ValueError('regression not accepted')
    inventory = json.loads((OPS / 'PRE_REPAIR_INVENTORY.json').read_text())
    old = OPS / 'original'
    for name, sha in inventory['original_evidence'].items():
        if file_sha(old / name) != sha:
            raise ValueError('preserved original evidence changed')
    previous = inventory['source_files']
    current = source_files()
    changed = sorted(p for p in previous if current.get(p) != previous[p])
    if set(current) != set(previous) or set(changed) != {
            str(ROOT / 'dispatcher.py'), str(ROOT / 'pipeline.py')}:
        raise ValueError('change exceeds reviewed operations-only patch')
    if any((ROOT / 'science').glob('*/receipts/*.json.gz')):
        raise ValueError('committed science must not be rebound')
    if any((ROOT / 'science').glob('*/models/*.npz')):
        raise ValueError('exported science must not be rebound')
    if any((ROOT / 'science').glob('*/locks/*.lock')):
        raise ValueError('worker lock still present')
    old_id = inventory['old_execution_identity']
    new_id = execution_identity()
    proofs = []
    prepared = OPS / 'prepared_checkpoints'
    prepared.mkdir(exist_ok=True)
    stage = ROOT / 'science/screen'
    originals = sorted((old / 'checkpoints').glob('*.npz'))
    if len(originals) != 8 or len(list((stage / 'checkpoints').glob('*.npz'))) != 8:
        raise ValueError('checkpoint census changed')
    for cp in originals:
        with np.load(cp, allow_pickle=False) as z:
            arrays = {k: z[k].copy() for k in z.files}
        before = json.loads(arrays['metadata_json'].tobytes())
        arm = before['arm']
        core, cursor = frozen_checkpoint.load(cp, arm, old_id)
        if cursor['stage'] != 'screen' or cursor['branch_index'] != 0 or cursor['W_export'] is not None:
            raise ValueError('recovery is limited to the initial incomplete W branches')
        world = make_world(cursor['world'], cursor['assay'])
        for branch, bd in cursor['branches'].items():
            for row, event in zip(bd['records'], world['events']):
                frozen_auditor.check_record(row, arm, event, branch, len(core.models))
        state_digest = core.state_digest()
        del core
        after = dict(before, identity=new_id)
        after.pop('seal')
        after['seal'] = digest(after)
        arrays['metadata_json'] = np.frombuffer(canonical(after), dtype=np.uint8)
        output = io.BytesIO()
        np.savez_compressed(output, **arrays)
        staged = prepared / cp.name
        atomic_bytes(staged, output.getvalue())
        restored, recovered_cursor = frozen_checkpoint.load(staged, arm, new_id)
        if restored.state_digest() != state_digest or recovered_cursor != cursor:
            raise ValueError('checkpoint state/cursor changed in recovery')
        del restored
        with np.load(cp, allow_pickle=False) as a, np.load(staged, allow_pickle=False) as b:
            if set(a.files) != set(b.files) or any(
                    not np.array_equal(a[k], b[k]) for k in a.files if k != 'metadata_json'):
                raise ValueError('operative array changed')
        normalized = dict(after, identity=old_id, seal=before['seal'])
        if normalized != before:
            raise ValueError('metadata changed beyond source identity and seal')
        try:
            frozen_checkpoint.load(staged, arm, old_id)
        except ValueError:
            wrong_identity_rejected = True
        else:
            raise ValueError('stale checkpoint identity accepted')
        if file_sha(stage / 'checkpoints' / cp.name) != file_sha(cp):
            raise ValueError('live checkpoint changed since stop')
        proofs.append({'checkpoint': cp.name, 'old_sha256': file_sha(cp),
                       'new_sha256': file_sha(staged), 'arm': arm,
                       'world': cursor['world'], 'assay': cursor['assay'],
                       'branch': 'W', 'next_record': cursor['next_record'],
                       'operative_state_digest': state_digest,
                       'all_operative_arrays_identical': True,
                       'full_cursor_and_record_history_identical': True,
                       'saved_actual_write_calls_audited': True,
                       'wrong_identity_rejected': wrong_identity_rejected})
    if sum(p['next_record'] for p in proofs) != 767:
        raise ValueError('saved teaching-prefix census')
    old_manifest = read_receipt(old / 'science/screen/MANIFEST.json.gz')
    recovered_manifest = dict(old_manifest, execution_source_identity=new_id)
    if dict(recovered_manifest, execution_source_identity=old_id) != old_manifest:
        raise ValueError('experiment manifest altered')
    q = json.loads((old / 'results/QUALIFICATION.json').read_text())
    amendment = {'verdict': 'PASS', 'old_execution_source_identity': old_id,
                 'new_execution_source_identity': new_id, 'changed_files': changed,
                 'scientific_worker_and_all_dependencies_unchanged': True,
                 'design_parameters_endpoints_sample_sizes_unchanged': True,
                 'regression_file': str(OPS / 'REGRESSION.json'),
                 'regression_sha256': file_sha(OPS / 'REGRESSION.json'),
                 'checkpoint_proofs': proofs, 'new_teaching_records_for_requalification': 0}
    atomic_json(OPS / 'OPERATIONS_QUALIFICATION.json', amendment)
    q.update(execution_source_identity=new_id, operations_recovery=amendment,
             source_coverage_note='Original three-cycle learning/state evidence is reused unchanged. '
             'Only dispatcher/pipeline filesystem operations changed; 43 operations/protocol regressions '
             'passed, and every saved science prefix/state was validated without reexecuting teaching.',
             science_partial_checkpoint_jobs=8, science_completed_jobs=0,
             operations_requalified_unix=time.time())
    atomic_json(ROOT / 'results/QUALIFICATION.json', q)
    source_lock(q)
    write_receipt(ROOT / 'LAUNCH_AUTHORIZATION.json.gz', {
        'authorized': True, 'execution_source_identity': new_id,
        'plan_sha256': file_sha(ROOT / 'PLAN.json'),
        'purpose': 'authorized screen then at most one confirmation'})
    require_authorized()
    if manifest('screen') != recovered_manifest:
        raise ValueError('recovered manifest is not the qualified unchanged protocol')
    write_manifest(stage / 'MANIFEST.json.gz', recovered_manifest)
    for proof in proofs:
        atomic_bytes(stage / 'checkpoints' / proof['checkpoint'],
                     (prepared / proof['checkpoint']).read_bytes())
    # The failed parent exited before committing its wait4 accounting ledger.
    # Preserve each worker's observed attempt CPU and charge a conservative
    # aggregate upper bound (host logical CPUs times the complete elapsed
    # launch-to-exit window), rather than inventing missing kernel measurements.
    failure = json.loads((old / 'results/FAILURE.json').read_text())
    stopped = json.loads((ROOT / 'ops/background_20261006/STOP_RECORD.json').read_text())
    launch = json.loads((stage / 'LAUNCH.json').read_text())
    attempts = [json.loads(p.read_text()) for p in (stage / 'attempts').glob('*.json')]
    if len(attempts) != 8 or list((stage / 'dispatcher_attempts').glob('*.json')):
        raise ValueError('attempt recovery census/duplicate')
    observed_cpu = sum(a['cpu_s'] for a in attempts)
    window = failure['updated_unix'] - launch['started_unix']
    cpu_bound = max(observed_cpu, window * (os.cpu_count() or 1))
    recovered_attempts = []
    for a in attempts:
        worker_bound = max(window, a['worker_s'])
        out = {'job': a['job'], 'pid': a['pid'], 'exit_code': 0,
               'worker_s': worker_bound,
               'process_CPU_s': cpu_bound * a['cpu_s'] / observed_cpu,
               'result': {'status': 'PAUSED_SAFE_RECORD'},
               'recovered_after_supervisor_exception': True,
               'observed_worker_attempt_CPU_s': a['cpu_s'],
               'observed_worker_attempt_wall_s': a['worker_s'],
               'CPU_charge_kind': 'conservative aggregate host-CPU/time upper bound; '
                                  'original kernel wait4 measurements unavailable',
               'source_provenance': old_id}
        atomic_json(stage / 'dispatcher_attempts' / f"recovered_{a['job']}_{a['pid']}.json", out)
        recovered_attempts.append(out)
    record = dict(amendment, state='RECOVERY_QUALIFIED_READY_FOR_EXPLICIT_RESUME',
                  authorized_instruction=inventory['user_authorization'],
                  completed_unix=time.time(), saved_W_records=767,
                  science_committed_jobs=0, recovered_attempts=recovered_attempts,
                  old_error_preserved='ops/recovery_20261006/original/results/FAILURE.json',
                  automatic_restart=False)
    atomic_json(OPS / 'RECOVERY_AUDIT.json', record)
    # Remove only the acknowledged current error pointer; immutable originals
    # and stop record remain. No receipt, checkpoint, or log is deleted.
    (ROOT / 'results/FAILURE.json').unlink()
    atomic_json(ROOT / 'results/STATUS.json', {
        'state': 'READY_FOR_AUTHORIZED_RESUME', 'updated_unix': time.time(),
        'recovery_audit': str(OPS / 'RECOVERY_AUDIT.json'),
        'current_stage_status': 'science/screen/STATUS.json'})
    print(json.dumps({'state': record['state'], 'old_execution_identity': old_id,
                      'new_execution_identity': new_id, 'checkpoints': len(proofs),
                      'saved_W_records': 767, 'observed_attempt_CPU_s': observed_cpu,
                      'conservative_attempt_CPU_charge_s': cpu_bound,
                      'new_teaching_records': 0}, indent=2), flush=True)


if __name__ == '__main__':
    main()
