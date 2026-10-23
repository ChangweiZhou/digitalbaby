# SPDX-License-Identifier: GPL-3.0-or-later
"""Isolated policy-layer mocks; no real ledger, worker, reservation or network."""
import copy
import json
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import patch

import adapter as a
from durable import create, replace
from integrity import digest_map


def run_checks(directory):
    root = Path(directory) / 'private-policy-tests'
    root.mkdir(parents=True, exist_ok=False)
    checks = []
    def reject(fn):
        try:
            fn()
        except (RuntimeError, OSError, KeyError, ValueError):
            return
        raise AssertionError('private-policy guard accepted invalid state')
    def plan():
        return {'ledger': {'attempts': []}, 'completed': ['cycle3/operations'], 'missing': ['cycle3/pilot', 'cycle3/replay'],
                'pending_barriers': [], 'blocked': ['unresolved_transport'], 'next_key': None}
    pending_path = root / 'operations/TRANSPORT_PENDING.json'
    create(pending_path, {'phase': 'public_tree', 'status': 'uncertain', 'tree': 'mock-fixed-tree'})
    saved = pending_path.read_bytes()
    def filter_local(p):
        return a.filter_plan(root, p, {'original_transport_sha256': a.sha(pending_path)})
    allowed = filter_local(plan())
    assert allowed['next_key'] == 'cycle3/pilot' and allowed['publication_deferred'] is True
    assert pending_path.read_bytes() == saved
    p = plan();p['blocked'].extend(['total worker cap', 'retry_requires_independent_infrastructure_approval'])
    filtered = filter_local(p)
    assert filtered['next_key'] is None and filtered['blocked'] == ['total worker cap', 'retry_requires_independent_infrastructure_approval']
    replace(pending_path, {'phase': 'private_upload', 'status': 'uncertain'})
    assert filter_local(plan())['next_key'] is None
    pending_path.write_bytes(saved)
    create(root / a.POLICY_DIR / 'TRANSPORT_PENDING.json', {'phase': 'private_readback'})
    assert 'unresolved_private_transport' in filter_local(plan())['blocked']
    replace(root / a.POLICY_DIR / 'TRANSPORT_PENDING.json', None)
    p = plan();p['missing'] = ['science/310001'];reject(lambda: filter_local(p))
    p = plan();p['completed'] = [];reject(lambda: filter_local(p))
    checks.append('public_only_blocker_deferred_without_touching_queue_or_other_guards')

    # Policy identity includes the additional layer separately from the base.
    create(root / 'source/private_policy/mock.py', {'isolated': 'source bytes'})
    create(root / a.POLICY_DIR / 'REVIEW.md', {'independent': 'synthetic review'})
    source = {'frozen/mock.py': 'a' * 64}
    policy = {'schema': 'RC-CYCLE3-PRIVATE-POLICY-v1', 'accepted': True, 'independent_review': True,
              'base_source_digest': digest_map(source), 'policy_hashes': a.policy_hashes(root),
              'policy_digest': digest_map(a.policy_hashes(root)), 'allowed_keys': list(a.KEYS), 'mode': 'cycle3',
              'private_backup_required': True, 'github_requirement': 'deferred_not_acknowledged',
              'report_file': a.POLICY_DIR + '/REVIEW.md', 'report_sha256': a.sha(root / a.POLICY_DIR / 'REVIEW.md'),
              'original_transport_sha256': a.sha(pending_path)}
    create(root / a.POLICY_DIR / 'POLICY_ACCEPTED.json', policy)
    with patch.object(a, 'BASE_DIGEST', digest_map(source)), patch.object(a, 'hashes', return_value=source):
        assert a.verify_policy(root) == policy
        replace(pending_path, {'phase': 'public_commit', 'status': 'uncertain'})
        reject(lambda: a.verify_policy(root))
        pending_path.write_bytes(saved)
        mock = root / 'source/private_policy/mock.py';b = mock.read_bytes();mock.write_bytes(b'changed')
        reject(lambda: a.verify_policy(root));mock.write_bytes(b)
        checks.append('policy_audit_hashes_bind_adapter_and_exact_preserved_public_queue')

        # Genuine operations receipt-only ACK can use a pre-policy archive.
        receipt = root / 'receipts/cycle3/operations.json'
        create(receipt, {'passed': True, 'synthetic': True})
        attempt = {'key': 'cycle3/operations', 'status': 'completed', 'source_digest': digest_map(source),
                   'receipt': 'receipts/cycle3/operations.json', 'receipt_sha256': a.sha(receipt)}
        archive_path = root / 'backups/mock.zip';archive_path.parent.mkdir()
        manifest = {'content_identity': 'b' * 64, 'all_payload_sha256': {attempt['receipt']: a.sha(receipt)}}
        with zipfile.ZipFile(archive_path, 'w') as z:
            z.write(receipt, attempt['receipt']);z.writestr('CHECKPOINT_MANIFEST.json', json.dumps(manifest))
        checkpoint = {'status': 'private_backup_readback_verified', 'library_file_id': 'mock-owned-private-file', 'version': 5,
                      'local_path': 'backups/mock.zip', 'sha256': a.sha(archive_path), 'content_identity': 'b' * 64}
        entry = {'key': attempt['key'], 'source_digest': attempt['source_digest'], 'manifest_sha256': attempt['receipt_sha256'],
                 'private_readback_verified': True, 'checkpoint': checkpoint}
        a.verify_private_job(root, attempt, entry)
        create(root / a.POLICY_DIR / 'PRIVATE_PERSISTENCE_STATE.json', {'jobs': {attempt['key']: entry}})
        assert a.private_pending(root, {'attempts': [attempt]}) == []
        bad = copy.deepcopy(entry);bad['github_tree_verified'] = True;reject(lambda: a.verify_private_job(root, attempt, bad))
        bad = copy.deepcopy(entry);bad['manifest_sha256'] = 'f' * 64;reject(lambda: a.verify_private_job(root, attempt, bad))
        saved_archive = archive_path.read_bytes();archive_path.write_bytes(b'bad');reject(lambda: a.verify_private_job(root, attempt, entry));archive_path.write_bytes(saved_archive)
        checks.append('private_barrier_revalidates_real_archive_hash_and_never_claims_github_ack')

        pilot = {'key': 'cycle3/pilot', 'attempt': 1, 'reservation_id': 'c' * 32, 'source_digest': digest_map(source)}
        att = {'schema': 'RC-CYCLE3-PRIVATE-POLICY-ATTESTATION-v1', **pilot, 'policy_digest': policy['policy_digest'],
               'policy_review_sha256': a.sha(root / a.POLICY_DIR / 'POLICY_ACCEPTED.json'), 'ledger_sha256': 'e' * 64,
               'original_transport_sha256': policy['original_transport_sha256']}
        create(a.attestation_path(root, pilot['reservation_id']), att)
        assert a.check_attestation(root, pilot, policy) == att
        changed = dict(pilot, attempt=2);reject(lambda: a.check_attestation(root, changed, policy))
        create(root / 'operations/RESERVATION_ACK.json', {'ledger_sha256': 'e' * 64, 'checkpoint_sha256': checkpoint['sha256'], 'content_identity': checkpoint['content_identity']})
        create(root / 'operations/CHECKPOINT_STATE.json', checkpoint)
        reject(lambda: a.check_attestation(root, pilot, policy, archived=True))
        checks.append('per_attempt_policy_attestation_must_match_and_exist_in_prebirth_archive')

        # Exact replacement surface is restored after use, and science cannot
        # reach the original planner through this private policy context.
        original = (a.recovery._pending_barriers, a.recovery.reconcile_and_plan, a.recovery.require_reservation_ack)
        with a.installed(root):
            reject(lambda: a.recovery.reconcile_and_plan(root, source, {}, 'science'))
            assert a.recovery._pending_barriers is a.private_pending
        assert original == (a.recovery._pending_barriers, a.recovery.reconcile_and_plan, a.recovery.require_reservation_ack)
        checks.append('explicit_runtime_patch_scope_restored_and_science_rejected')
    return checks


if __name__ == '__main__':
    with tempfile.TemporaryDirectory(prefix='rcenter-private-policy-', dir='/tmp') as directory:
        print(json.dumps({'passed': True, 'checks': run_checks(directory), 'native_events': 0}))
