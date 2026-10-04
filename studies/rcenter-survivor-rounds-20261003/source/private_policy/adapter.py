# SPDX-License-Identifier: GPL-3.0-or-later
"""Explicit private-backup-only Cycle3 operational policy adapter.

The frozen 132-file scientific/qualification closure is never edited. This
additional reviewed layer replaces exactly the pending-persistence predicate,
filters only a preserved public_* transport blocker, and adds reservation
attestation checks. Original ledger/recovery/governance/worker methods remain
in use. This adapter cannot authorize science, analysis or operations reruns.
"""
import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
import signal
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'source/runtime'))
import recovery
from durable import create, replace
from integrity import hashes, digest_map

BASE_DIGEST = '5546bbb941d64a4490b1e4469b24461f052d044ae586d62bb4ba9d70cb290077'
POLICY_DIR = 'operations/private_policy'
KEYS = ('cycle3/pilot', 'cycle3/replay')
ALL_KEYS = ('cycle3/operations',) + KEYS


def check(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def policy_hashes(root=ROOT):
    root = Path(root)
    return {str(p.relative_to(root)): sha(p) for p in sorted((root / 'source/private_policy').rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py', '.js', '.json', '.md')}


def require_policy(root=ROOT):
    root = Path(root)
    path = root / POLICY_DIR / 'POLICY_ACCEPTED.json'
    policy = read(path)
    current = policy_hashes(root)
    check(policy.get('schema') == 'RC-CYCLE3-PRIVATE-POLICY-v1' and policy.get('accepted') is True and policy.get('independent_review') is True,
          'independent private-policy acceptance missing')
    check(policy.get('base_source_digest') == BASE_DIGEST == digest_map(hashes()), 'frozen scientific source changed')
    check(current and policy.get('policy_hashes') == current and policy.get('policy_digest') == digest_map(current), 'private-policy source changed')
    check(policy.get('allowed_keys') == list(KEYS) and policy.get('mode') == 'cycle3' and policy.get('private_backup_required') is True,
          'private-policy scope mismatch')
    check(policy.get('github_requirement') == 'deferred_not_acknowledged', 'GitHub must remain explicitly unacknowledged')
    pending, pending_sha = public_pending(root)
    check(policy.get('original_transport_sha256') == pending_sha, 'original publication queue changed since private-policy acceptance')
    check(pending is None or (isinstance(pending, dict) and str(pending.get('phase', '')).startswith('public_')), 'private policy cannot waive original private transport uncertainty')
    report = recovery.relative(root, policy['report_file'])
    check(report.is_relative_to(root / POLICY_DIR) and sha(report) == policy['report_sha256'], 'private-policy review changed')
    return policy


def verify_policy(root=ROOT):
    return require_policy(root)


def public_pending(root):
    path = Path(root) / 'operations/TRANSPORT_PENDING.json'
    value = read(path) if path.exists() else None
    return value, sha(path) if path.exists() else None


def archive(root, state):
    """Verify preserved original bytes identical to the recorded Library readback."""
    root = Path(root)
    check(state.get('status') == 'private_backup_readback_verified' and state.get('library_file_id') and
          type(state.get('version')) is int and state['version'] >= 0, 'verified Library checkpoint required')
    path = recovery.relative(root, state['local_path'])
    check(sha(path) == state['sha256'], 'verified private archive changed/missing')
    z = zipfile.ZipFile(path)
    try:
        m = json.loads(z.read('CHECKPOINT_MANIFEST.json'))
        check(m['content_identity'] == state['content_identity'], 'private archive content identity mismatch')
        check(len(z.namelist()) == len(set(z.namelist())), 'duplicate archive entries')
        check(set(z.namelist()) == set(m['all_payload_sha256']) | {'CHECKPOINT_MANIFEST.json'}, 'private archive payload map mismatch')
    except BaseException:
        z.close()
        raise
    return z, m


def archived_file(z, manifest, name, expected):
    check(manifest['all_payload_sha256'].get(name) == expected, 'required private archive member identity missing: ' + name)
    h = hashlib.sha256()
    with z.open(name) as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    check(h.hexdigest() == expected, 'private archive member changed: ' + name)


def verify_private_job(root, attempt, entry):
    root = Path(root)
    check(entry.get('key') == attempt['key'] and entry.get('manifest_sha256') == attempt['receipt_sha256'] and
          entry.get('source_digest') == attempt['source_digest'] and entry.get('private_readback_verified') is True,
          'private receipt barrier identity mismatch')
    check('github_tree_verified' not in entry and 'public_commit' not in entry, 'private barrier must not claim GitHub acknowledgement')
    z, m = archive(root, entry['checkpoint'])
    try:
        path = root / attempt['receipt']
        manifest_path = path / 'manifest.json' if path.is_dir() else path
        archived_file(z, m, str(manifest_path.relative_to(root)), attempt['receipt_sha256'])
        if path.is_dir():
            for name, part in read(manifest_path)['parts'].items():
                archived_file(z, m, str((path / name).relative_to(root)), part['sha256'])
        if attempt['key'] in KEYS:
            policy = require_policy(root)
            attestation = attestation_path(root, attempt['reservation_id'])
            check_attestation(root, attempt, policy)
            archived_file(z, m, str(attestation.relative_to(root)), sha(attestation))
            archived_file(z, m, POLICY_DIR + '/POLICY_ACCEPTED.json', sha(root / POLICY_DIR / 'POLICY_ACCEPTED.json'))
            archived_file(z, m, policy['report_file'], policy['report_sha256'])
            for rel, h in policy['policy_hashes'].items():
                archived_file(z, m, rel, h)
    finally:
        z.close()


def private_pending(root, ledger):
    root = Path(root)
    state_path = root / POLICY_DIR / 'PRIVATE_PERSISTENCE_STATE.json'
    state = read(state_path) if state_path.exists() else {'jobs': {}}
    pending = []
    for a in ledger['attempts']:
        check(a['key'] in ('cycle1/test', 'cycle2/test') + ALL_KEYS, 'private Cycle3 policy cannot admit official/other jobs')
        if a['status'] != 'completed' or a['key'] in ('cycle1/test', 'cycle2/test'):
            continue
        entry = state.get('jobs', {}).get(a['key'])
        if entry is None:
            pending.append({'key': a['key'], 'receipt': a['receipt'], 'manifest_sha256': a['receipt_sha256']})
        else:
            verify_private_job(root, a, entry)
    check(len(pending) <= 1, 'more than one privately unbacked completed job')
    return pending


def filter_plan(root, plan, policy):
    """Only publication uncertainty is deferred. All other blockers survive."""
    value, pending_sha = public_pending(root)
    check(pending_sha == policy['original_transport_sha256'], 'approved publication queue bytes changed')
    deferred = isinstance(value, dict) and isinstance(value.get('phase'), str) and value['phase'].startswith('public_')
    if deferred:
        plan['blocked'] = [b for b in plan['blocked'] if b != 'unresolved_transport']
    private_queue = Path(root) / POLICY_DIR / 'TRANSPORT_PENDING.json'
    if private_queue.exists() and read(private_queue) is not None:
        plan['blocked'].append('unresolved_private_transport')
    check('cycle3/operations' in plan['completed'], 'accepted completed operations receipt prerequisite required')
    check(all(k in KEYS for k in plan['missing']), 'private policy only continues fixed pilot/replay')
    plan['next_key'] = plan['missing'][0] if plan['missing'] and not plan['blocked'] else None
    plan['publication_deferred'] = deferred
    plan['original_transport_sha256'] = pending_sha
    plan['persistence_policy'] = 'verified_private_backups; GitHub deferred without acknowledgement'
    return plan


def attestation_path(root, reservation):
    check(isinstance(reservation, str) and len(reservation) == 32 and all(c in '0123456789abcdef' for c in reservation), 'invalid policy reservation identity')
    return Path(root) / POLICY_DIR / 'attestations' / (reservation + '.json')


def check_attestation(root, attempt, policy, *, archived=False):
    root = Path(root)
    path = attestation_path(root, attempt['reservation_id'])
    a = read(path)
    check(a.get('schema') == 'RC-CYCLE3-PRIVATE-POLICY-ATTESTATION-v1', 'policy attestation missing')
    for key, expected in (('key', attempt['key']), ('attempt', attempt['attempt']), ('reservation_id', attempt['reservation_id']),
                          ('source_digest', attempt['source_digest']), ('policy_digest', policy['policy_digest']),
                          ('policy_review_sha256', sha(root / POLICY_DIR / 'POLICY_ACCEPTED.json')),
                          ('original_transport_sha256', policy['original_transport_sha256'])):
        check(a.get(key) == expected, 'policy attestation mismatch: ' + key)
    if archived:
        ack = read(root / 'operations/RESERVATION_ACK.json')
        state = read(root / 'operations/CHECKPOINT_STATE.json')
        check(a['ledger_sha256'] == ack['ledger_sha256'] and ack['checkpoint_sha256'] == state['sha256'] and
              ack['content_identity'] == state['content_identity'], 'policy reservation backup identity mismatch')
        z, m = archive(root, state)
        try:
            archived_file(z, m, str(path.relative_to(root)), sha(path))
            archived_file(z, m, 'operations/RUN_LEDGER.json', ack['ledger_sha256'])
            archived_file(z, m, POLICY_DIR + '/POLICY_ACCEPTED.json', a['policy_review_sha256'])
            archived_file(z, m, policy['report_file'], policy['report_sha256'])
            for rel, h in policy['policy_hashes'].items():
                archived_file(z, m, rel, h)
        finally:
            z.close()
    return a


@contextlib.contextmanager
def installed(root, supervisor=None):
    """The full, explicit runtime replacement surface of this adapter."""
    original_pending = recovery._pending_barriers
    original_plan = recovery.reconcile_and_plan
    original_ack = recovery.require_reservation_ack
    def plan(root, source, runtime, mode, **kwargs):
        check(mode == 'cycle3', 'private policy cannot admit official science')
        policy = require_policy(root)
        result = original_plan(root, source, runtime, mode, **kwargs)
        for a in result['ledger']['attempts']:
            if a['key'] in KEYS:
                check_attestation(root, a, policy)
        return filter_plan(root, result, policy)
    def ack(root, attempt):
        check(attempt['key'] in KEYS, 'private policy only starts pilot/replay')
        policy = require_policy(root)
        result = original_ack(root, attempt)
        check_attestation(root, attempt, policy, archived=True)
        return result
    recovery._pending_barriers = private_pending
    recovery.reconcile_and_plan = plan
    recovery.require_reservation_ack = ack
    if supervisor is not None:
        supervisor.reconcile_and_plan = plan
        supervisor.require_reservation_ack = ack
    try:
        yield
    finally:
        recovery._pending_barriers = original_pending
        recovery.reconcile_and_plan = original_plan
        recovery.require_reservation_ack = original_ack
        if supervisor is not None:
            supervisor.reconcile_and_plan = original_plan
            supervisor.require_reservation_ack = original_ack


def ack_private(root, key):
    """Read-only archive verification followed by genuine private-only ACK."""
    root = Path(root)
    policy = require_policy(root)
    check(key in ALL_KEYS, 'unsupported private qualification receipt')
    with recovery.exclusive_lock(root) as lock:
        ledger = recovery.load_ledger(root, lock=lock)
        recovery.validate_attempts(root, ledger, hashes(), recovery.RUNTIME)
        matches = [a for a in ledger['attempts'] if a['key'] == key and a['status'] == 'completed']
        check(len(matches) == 1, 'one completed immutable receipt required')
        attempt = matches[0]
        recovery.validate_completed(root, attempt, hashes(), recovery.RUNTIME)
        if key in KEYS:
            check_attestation(root, attempt, policy)
        entry = {'key': key, 'manifest_sha256': attempt['receipt_sha256'], 'source_digest': attempt['source_digest'],
                 'private_readback_verified': True, 'checkpoint': read(root / 'operations/CHECKPOINT_STATE.json')}
        verify_private_job(root, attempt, entry)
        path = root / POLICY_DIR / 'PRIVATE_PERSISTENCE_STATE.json'
        state = read(path) if path.exists() else {'schema': 'RC-CYCLE3-PRIVATE-PERSISTENCE-v1', 'jobs': {}}
        state['jobs'][key] = entry
        replace(path, state)
        return entry


def main(mode, job='pilot', action='plan', reservation_id=None):
    check(mode == 'cycle3' and job in ('pilot', 'replay'), 'private adapter scope is fixed Cycle3 pilot/replay only')
    policy = require_policy(ROOT)
    spec = importlib.util.spec_from_file_location('unchanged_private_supervisor', ROOT / 'operations/supervise.py')
    supervisor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(supervisor)
    with installed(ROOT, supervisor):
        result = supervisor.main(mode, job, None, action, reservation_id)
    if action == 'reserve':
        a = result['attempt']
        pending, h = public_pending(ROOT)
        create(attestation_path(ROOT, a['reservation_id']), {'schema': 'RC-CYCLE3-PRIVATE-POLICY-ATTESTATION-v1',
               'key': a['key'], 'attempt': a['attempt'], 'reservation_id': a['reservation_id'], 'source_digest': a['source_digest'],
               'policy_digest': policy['policy_digest'], 'policy_review_sha256': sha(ROOT / POLICY_DIR / 'POLICY_ACCEPTED.json'),
               'ledger_sha256': result['ledger_sha256'], 'original_transport_sha256': h,
               'original_transport_phase': pending.get('phase') if isinstance(pending, dict) else None,
               'github_acknowledged': False, 'private_prebirth_backup_required': True})
        result['policy_attestation'] = str(attestation_path(ROOT, a['reservation_id']).relative_to(ROOT))
        result['policy_digest'] = policy['policy_digest']
    return result


if __name__ == '__main__':
    def interrupted(signum, frame):
        raise InterruptedError(f'signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['cycle3', 'ack-private'])
    p.add_argument('--job', choices=['pilot', 'replay'], default='pilot')
    p.add_argument('--action', choices=['plan', 'reserve', 'run'], default='plan')
    p.add_argument('--reservation-id')
    p.add_argument('--key', choices=ALL_KEYS)
    args = p.parse_args()
    result = ack_private(ROOT, args.key) if args.mode == 'ack-private' else main(args.mode, args.job, args.action, args.reservation_id)
    print(json.dumps(result, separators=(',', ':')), flush=True)
    if args.action == 'run' and result.get('status') != 'completed':
        raise SystemExit(1)
