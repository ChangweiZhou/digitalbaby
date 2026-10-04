# SPDX-License-Identifier: GPL-3.0-or-later
"""P2: separately reviewed private persistence for fixed official64 + analysis.

The frozen scientific source and completed P1 policy files are unchanged.
Only operational persistence predicates, policy attestations and the final
analysis bootstrap are layered around their original implementations.
"""
import argparse
import ast
import contextlib
import importlib.util
import inspect
import json
import os
import signal
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'source/runtime'))
import recovery
from durable import create, replace
from integrity import hashes, digest_map

spec = importlib.util.spec_from_file_location('historical_cycle3_private_policy', ROOT / 'source/private_policy/adapter.py')
p1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p1)
check, read, sha = p1.check, p1.read, p1.sha
BASE_DIGEST = p1.BASE_DIGEST
STATE = 'operations/private_official_policy'
WORLDS = tuple(range(310001, 310065))
KEYS = tuple(f'science/{w}' for w in WORLDS) + ('analysis/final',)
CYCLE_KEYS = ('cycle3/operations', 'cycle3/pilot', 'cycle3/replay')


def policy_hashes(root=ROOT):
    root = Path(root)
    return {str(p.relative_to(root)): sha(p) for p in sorted((root / 'source/private_official_policy').rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py', '.js', '.json', '.md')}


def historical_policy(root):
    """Verify completed P1 as historical P1, without reopening its old queue."""
    root = Path(root)
    policy = read(root / p1.POLICY_DIR / 'POLICY_ACCEPTED.json')
    check(policy.get('schema') == 'RC-CYCLE3-PRIVATE-POLICY-v1' and policy.get('accepted') is True and policy.get('independent_review') is True,
          'historical P1 acceptance missing')
    check(policy['base_source_digest'] == BASE_DIGEST and policy['policy_hashes'] == p1.policy_hashes(root) and
          policy['policy_digest'] == digest_map(policy['policy_hashes']), 'historical P1 source changed')
    check(policy['allowed_keys'] == list(p1.KEYS) and policy['mode'] == 'cycle3', 'historical P1 scope changed')
    check(sha(recovery.relative(root, policy['report_file'])) == policy['report_sha256'], 'historical P1 report changed')
    return policy


def launch_guard(root, source):
    root = Path(root)
    lock = read(root / 'protocol/OFFICIAL_LOCK.json')
    launch = read(root / 'audits/LAUNCH_ACCEPTED.json')
    check(lock['hashes'] == source, 'official lock source mismatch')
    check(launch.get('accepted') is True and launch['lock_sha256'] == sha(root / 'protocol/OFFICIAL_LOCK.json'), 'parent official launch approval missing')
    for rel, h in lock['acceptances'].items():
        check(sha(recovery.relative(root, rel)) == h, 'official locked acceptance/evidence drift')
    for cycle in (1, 2, 3):
        rel = f'audits/CYCLE{cycle}_ACCEPTED.json'
        check(rel in lock['acceptances'] and read(root / rel).get('accepted') is True, 'all three accepted cycles required')
    return lock


def verify_policy(root=ROOT):
    root = Path(root)
    policy = read(root / STATE / 'POLICY_ACCEPTED.json')
    source = hashes()
    check(policy.get('schema') == 'RC-OFFICIAL-PRIVATE-POLICY-v1' and policy.get('accepted') is True and policy.get('independent_review') is True,
          'official private-policy acceptance missing')
    check(policy['base_source_digest'] == BASE_DIGEST == digest_map(source), 'frozen scientific source changed')
    for rel, h in source.items():
        check(sha(recovery.relative(root, rel)) == h, 'consumer-local frozen source changed: ' + rel)
    check(policy['policy_hashes'] == policy_hashes(root) and policy['policy_digest'] == digest_map(policy['policy_hashes']), 'official adapter source changed')
    check(policy['mode'] == 'science' and policy['allowed_keys'] == list(KEYS) and policy['private_backup_required'] is True and
          policy['github_requirement'] == 'deferred_not_acknowledged', 'official private-policy scope mismatch')
    historical_policy(root)
    check(policy['cycle3_policy_sha256'] == sha(root / p1.POLICY_DIR / 'POLICY_ACCEPTED.json'), 'historical P1 identity mismatch')
    report = recovery.relative(root, policy['report_file'])
    check(report.is_relative_to(root / STATE) and sha(report) == policy['report_sha256'], 'official private-policy report changed')
    launch_guard(root, source)
    return policy


def attestation_path(root, reservation):
    check(isinstance(reservation, str) and len(reservation) == 32 and all(c in '0123456789abcdef' for c in reservation), 'invalid official reservation identity')
    return Path(root) / STATE / 'attestations' / (reservation + '.json')


def check_attestation(root, attempt, policy, *, archived=False):
    root = Path(root)
    path = attestation_path(root, attempt['reservation_id'])
    att = read(path)
    check(att.get('schema') == 'RC-OFFICIAL-PRIVATE-POLICY-ATTESTATION-v1', 'official policy attestation missing')
    expected = {'key': attempt['key'], 'attempt': attempt['attempt'], 'reservation_id': attempt['reservation_id'],
                'source_digest': attempt['source_digest'], 'policy_digest': policy['policy_digest'],
                'policy_review_sha256': sha(root / STATE / 'POLICY_ACCEPTED.json'),
                'official_lock_sha256': sha(root / 'protocol/OFFICIAL_LOCK.json'),
                'launch_sha256': sha(root / 'audits/LAUNCH_ACCEPTED.json')}
    check(all(att.get(k) == v for k, v in expected.items()), 'official policy attestation identity mismatch')
    if archived:
        ack = read(root / 'operations/RESERVATION_ACK.json')
        current = read(root / 'operations/CHECKPOINT_STATE.json')
        check(att['ledger_sha256'] == ack['ledger_sha256'] and ack['checkpoint_sha256'] == current['sha256'] and
              ack['content_identity'] == current['content_identity'], 'official prebirth archive mismatch')
        required = policy_files(root, policy)
        required[str(path.relative_to(root))] = sha(path)
        required['operations/RUN_LEDGER.json'] = ack['ledger_sha256']
        verify_coverage(root, required)
    return att


def policy_files(root, policy):
    root = Path(root)
    files = dict(policy['policy_hashes'])
    files[STATE + '/POLICY_ACCEPTED.json'] = sha(root / STATE / 'POLICY_ACCEPTED.json')
    files[policy['report_file']] = policy['report_sha256']
    files['protocol/OFFICIAL_LOCK.json'] = sha(root / 'protocol/OFFICIAL_LOCK.json')
    files['audits/LAUNCH_ACCEPTED.json'] = sha(root / 'audits/LAUNCH_ACCEPTED.json')
    return files


def verify_coverage(root, required, *, checkpoint=None):
    """Use a current genuine covering archive, never a fictitious old ZIP."""
    root = Path(root)
    current = checkpoint or read(root / 'operations/CHECKPOINT_STATE.json')
    z, manifest = p1.archive(root, current)
    try:
        for rel, h in required.items():
            p1.archived_file(z, manifest, rel, h)
    finally:
        z.close()
    return current


def receipt_files(root, attempt):
    root = Path(root)
    path = root / attempt['receipt']
    mp = path / 'manifest.json' if path.is_dir() else path
    check(sha(mp) == attempt['receipt_sha256'], 'completed local receipt changed')
    files = {str(mp.relative_to(root)): attempt['receipt_sha256']}
    if path.is_dir():
        files.update({str((path / name).relative_to(root)): part['sha256'] for name, part in read(mp)['parts'].items()})
    return files


def verify_job(root, attempt, entry, policy, *, require_evidence=True):
    root = Path(root)
    check(entry.get('key') == attempt['key'] and entry.get('source_digest') == attempt['source_digest'] and
          entry.get('manifest_sha256') == attempt['receipt_sha256'] and entry.get('private_readback_verified') is True,
          'genuine private job acknowledgement missing/mismatched')
    check('github_tree_verified' not in entry, 'private acknowledgement must not claim GitHub success')
    check(attempt['receipt'] == recovery.receipt_path(attempt['key']) and attempt['source_digest'] == BASE_DIGEST, 'private job canonical path/source mismatch')
    required = receipt_files(root, attempt)
    if attempt['key'] in p1.KEYS:
        historical = historical_policy(root)
        p1.check_attestation(root, attempt, historical)
        path = p1.attestation_path(root, attempt['reservation_id'])
        required[str(path.relative_to(root))] = sha(path)
        required.update(historical['policy_hashes'])
        required[p1.POLICY_DIR + '/POLICY_ACCEPTED.json'] = sha(root / p1.POLICY_DIR / 'POLICY_ACCEPTED.json')
        required[historical['report_file']] = historical['report_sha256']
    elif attempt['key'] in KEYS:
        if require_evidence:
            evidence_path = recovery.relative(root, entry['evidence_file'])
            check(evidence_path.is_relative_to(root / STATE / 'barrier_evidence') and sha(evidence_path) == entry['evidence_sha256'], 'immutable private barrier evidence changed')
            evidence = read(evidence_path)
            check(all(evidence.get(k) == entry.get(k) for k in ('key', 'source_digest', 'manifest_sha256', 'policy_digest')), 'private barrier evidence identity mismatch')
        check_attestation(root, attempt, policy)
        path = attestation_path(root, attempt['reservation_id'])
        required[str(path.relative_to(root))] = sha(path)
        required.update(policy_files(root, policy))
    return verify_coverage(root, required)


def private_pending(root, ledger):
    root = Path(root)
    policy = verify_policy(root)
    states = {}
    for folder in (p1.POLICY_DIR, STATE):
        path = root / folder / 'PRIVATE_PERSISTENCE_STATE.json'
        if path.exists():
            states.update(read(path).get('jobs', {}))
    pending = []
    for attempt in ledger['attempts']:
        check(attempt['key'] in ('cycle1/test', 'cycle2/test') + CYCLE_KEYS + KEYS, 'unregistered job in official ledger')
        if attempt['status'] != 'completed' or attempt['key'] in ('cycle1/test', 'cycle2/test'):
            continue
        entry = states.get(attempt['key'])
        if entry is None:
            pending.append({'key': attempt['key'], 'receipt': attempt['receipt'], 'manifest_sha256': attempt['receipt_sha256']})
        else:
            verify_job(root, attempt, entry, policy)
    check(len(pending) <= 1, 'more than one privately unbacked job')
    return pending


def filter_plan(root, plan):
    pending, pending_sha = p1.public_pending(root)
    deferred = isinstance(pending, dict) and isinstance(pending.get('phase'), str) and pending['phase'].startswith('public_')
    if deferred:
        plan['blocked'] = [b for b in plan['blocked'] if b != 'unresolved_transport']
    for folder in (p1.POLICY_DIR, STATE):
        q = Path(root) / folder / 'TRANSPORT_PENDING.json'
        if q.exists() and read(q) is not None:
            plan['blocked'].append('unresolved_private_transport')
    check(set(CYCLE_KEYS) <= set(plan['completed']), 'completed Cycle3 operations/pilot/replay required')
    check(all(key in KEYS for key in plan['missing']), 'official fixed missing-only sequence required')
    plan['next_key'] = plan['missing'][0] if plan['missing'] and not plan['blocked'] else None
    plan.update(publication_deferred=deferred, observed_public_queue_sha256=pending_sha,
                persistence_policy='verified-private-only; public_* uncertainty independent')
    return plan


def prelaunch_guard(root, policy):
    state = read(Path(root) / 'operations/PERSISTENCE_STATE.json')
    check(state.get('prelaunch_verified') is True and state.get('prelaunch_source_digest') == BASE_DIGEST and
          state.get('prelaunch_persistence_policy') == 'verified-private-only-v1' and
          state.get('private_official_policy_digest') == policy['policy_digest'], 'truthful private prelaunch verification missing')


def private_input_gate(root, world, manifest_sha):
    """Replacement for the single analysis operational persistence predicate."""
    root = Path(root)
    check(world in WORLDS, 'unregistered analysis input')
    policy = verify_policy(root)
    ledger = read(root / 'operations/RUN_LEDGER.json')
    matches = [a for a in ledger['attempts'] if a['key'] == f'science/{world}' and a['status'] == 'completed']
    check(len(matches) == 1 and matches[0]['receipt_sha256'] == manifest_sha, 'analysis input not completed/identical')
    states = read(root / STATE / 'PRIVATE_PERSISTENCE_STATE.json')
    verify_job(root, matches[0], states.get('jobs', {}).get(f'science/{world}', {}), policy)


@contextlib.contextmanager
def analysis_gate(root=ROOT):
    """Replace exactly one frozen AST check; all numerical/schema code stays."""
    import validate_analysis
    original = validate_analysis.inputs
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    replacements = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == 'check':
            args = node.value.args
            if len(args) == 2 and isinstance(args[1], ast.Constant) and args[1].value == 'analysis input awaiting private/public persistence barrier':
                node.value = ast.Call(func=ast.Name(id='_private_input_gate', ctx=ast.Load()),
                                      args=[ast.Name(id=n, ctx=ast.Load()) for n in ('root', 'world', 'h')], keywords=[])
                replacements += 1
    check(replacements == 1, 'frozen analysis operational predicate changed; refuse adaptation')
    namespace = dict(vars(validate_analysis))
    namespace['_private_input_gate'] = private_input_gate
    exec(compile(ast.fix_missing_locations(tree), str(ROOT / 'source/private_official_policy/adapter.py'), 'exec'), namespace)
    validate_analysis.inputs = namespace['inputs']
    try:
        yield
    finally:
        validate_analysis.inputs = original


@contextlib.contextmanager
def installed(root, supervisor=None):
    original_pending, original_plan, original_ack = recovery._pending_barriers, recovery.reconcile_and_plan, recovery.require_reservation_ack
    def plan(root, source, runtime, mode, **kwargs):
        check(mode == 'science', 'P2 admits official science/analysis only')
        policy = verify_policy(root)
        prelaunch_guard(root, policy)
        result = original_plan(root, source, runtime, mode, **kwargs)
        for attempt in result['ledger']['attempts']:
            if attempt['key'] in KEYS:
                check_attestation(root, attempt, policy)
        return filter_plan(root, result)
    def ack(root, attempt):
        check(attempt['key'] in KEYS, 'P2 reservation scope mismatch')
        policy = verify_policy(root)
        result = original_ack(root, attempt)
        check_attestation(root, attempt, policy, archived=True)
        return result
    recovery._pending_barriers, recovery.reconcile_and_plan, recovery.require_reservation_ack = private_pending, plan, ack
    old_wrapper = None
    if supervisor is not None:
        supervisor.reconcile_and_plan, supervisor.require_reservation_ack = plan, ack
        old_wrapper = supervisor.WRAPPER
        final_call = "runpy.run_path(os.environ['ENTRY'],run_name='__main__')"
        check(old_wrapper.count(final_call) == 1, 'original parent-death wrapper changed')
        bootstrap = str(Path(root) / 'source/private_official_policy/analysis_bootstrap.py')
        supervisor.WRAPPER = old_wrapper.replace(final_call, "if os.environ.get('SURVIVOR_JOB_KEY')=='analysis/final':\n runpy.run_path(" + repr(bootstrap) + ",run_name='__main__')\nelse:\n " + final_call)
    try:
        with analysis_gate(root):
            yield
    finally:
        recovery._pending_barriers, recovery.reconcile_and_plan, recovery.require_reservation_ack = original_pending, original_plan, original_ack
        if supervisor is not None:
            supervisor.reconcile_and_plan, supervisor.require_reservation_ack, supervisor.WRAPPER = original_plan, original_ack, old_wrapper


def ack_private(root, key):
    root = Path(root)
    check(key in KEYS, 'P2 cannot relabel historical qualification receipts')
    policy = verify_policy(root)
    with recovery.exclusive_lock(root) as lock, analysis_gate(root):
        ledger = recovery.load_ledger(root, lock=lock)
        recovery.validate_attempts(root, ledger, hashes(), recovery.RUNTIME)
        matching = [a for a in ledger['attempts'] if a['key'] == key and a['status'] == 'completed']
        check(len(matching) == 1, 'one completed official receipt required')
        attempt = matching[0]
        recovery.validate_completed(root, attempt, hashes(), recovery.RUNTIME)
        entry = {'key': key, 'manifest_sha256': attempt['receipt_sha256'], 'source_digest': attempt['source_digest'],
                 'private_readback_verified': True, 'policy_digest': policy['policy_digest'],
                 'coverage_mode': 'current_verified_covering_archive'}
        checkpoint = verify_job(root, attempt, entry, policy, require_evidence=False)
        evidence_path = root / STATE / 'barrier_evidence' / (attempt['reservation_id'] + '.json')
        if not evidence_path.exists():
            create(evidence_path, {'schema': 'RC-OFFICIAL-PRIVATE-BARRIER-EVIDENCE-v1', **entry, 'checkpoint': checkpoint})
        else:
            evidence = read(evidence_path)
            check(all(evidence.get(k) == entry.get(k) for k in ('key', 'source_digest', 'manifest_sha256', 'policy_digest')), 'existing barrier evidence differs')
        entry['evidence_file'] = str(evidence_path.relative_to(root))
        entry['evidence_sha256'] = sha(evidence_path)
        path = root / STATE / 'PRIVATE_PERSISTENCE_STATE.json'
        state = read(path) if path.exists() else {'schema': 'RC-OFFICIAL-PRIVATE-PERSISTENCE-v1', 'jobs': {}}
        state['jobs'][key] = entry
        replace(path, state)
        return entry


def mark_prelaunch(root=ROOT):
    root = Path(root)
    policy = verify_policy(root)
    with recovery.exclusive_lock(root) as lock:
        ledger = recovery.load_ledger(root, lock=lock)
        recovery.validate_attempts(root, ledger, hashes(), recovery.RUNTIME)
        check(set(CYCLE_KEYS) <= {a['key'] for a in ledger['attempts'] if a['status'] == 'completed'}, 'Cycle3 prerequisites missing')
        check(not private_pending(root, ledger), 'unverified prelaunch private receipts')
        required = dict(hashes());required.update(policy_files(root, policy))
        official_lock = read(root / 'protocol/OFFICIAL_LOCK.json')
        required.update(official_lock['acceptances'])
        checkpoint = verify_coverage(root, required)
        path = root / 'operations/PERSISTENCE_STATE.json'
        state = read(path) if path.exists() else {'worlds': {}, 'jobs': {}}
        state.update(prelaunch_verified=True, prelaunch_source_digest=BASE_DIGEST,
                     prelaunch_persistence_policy='verified-private-only-v1', private_official_policy_digest=policy['policy_digest'],
                     prelaunch_private_version=checkpoint['version'], prelaunch_private_archive_sha256=checkpoint['sha256'],
                     github_required_for_execution=False)
        replace(path, state)
        return state


def main(mode, job='operations', world=None, action='plan', reservation_id=None):
    check(mode == 'science' and ((action == 'plan' and world is None and job == 'operations') or (job == 'analysis' and world is None) or (job != 'analysis' and world in WORLDS)), 'fixed official job required')
    policy = verify_policy(ROOT)
    spec = importlib.util.spec_from_file_location('original_official_supervisor', ROOT / 'operations/supervise.py')
    supervisor = importlib.util.module_from_spec(spec);spec.loader.exec_module(supervisor)
    with installed(ROOT, supervisor):
        result = supervisor.main(mode, job, world, action, reservation_id)
    if action == 'reserve':
        attempt = result['attempt']
        create(attestation_path(ROOT, attempt['reservation_id']), {'schema': 'RC-OFFICIAL-PRIVATE-POLICY-ATTESTATION-v1',
               'key': attempt['key'], 'attempt': attempt['attempt'], 'reservation_id': attempt['reservation_id'],
               'source_digest': attempt['source_digest'], 'policy_digest': policy['policy_digest'],
               'policy_review_sha256': sha(ROOT / STATE / 'POLICY_ACCEPTED.json'), 'ledger_sha256': result['ledger_sha256'],
               'official_lock_sha256': sha(ROOT / 'protocol/OFFICIAL_LOCK.json'), 'launch_sha256': sha(ROOT / 'audits/LAUNCH_ACCEPTED.json'),
               'private_prebirth_backup_required': True, 'github_acknowledged': False})
        result['policy_attestation'] = str(attestation_path(ROOT, attempt['reservation_id']).relative_to(ROOT))
        result['policy_digest'] = policy['policy_digest']
    return result


if __name__ == '__main__':
    def interrupted(signum, frame):
        raise InterruptedError(f'signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    p = argparse.ArgumentParser();p.add_argument('mode', choices=['science', 'ack-private', 'prelaunch'])
    p.add_argument('--job', choices=['operations', 'analysis'], default='operations');p.add_argument('--world', type=int)
    p.add_argument('--action', choices=['plan', 'reserve', 'run'], default='plan');p.add_argument('--reservation-id');p.add_argument('--key', choices=KEYS)
    args = p.parse_args()
    result = mark_prelaunch(ROOT) if args.mode == 'prelaunch' else ack_private(ROOT, args.key) if args.mode == 'ack-private' else main(args.mode, args.job, args.world, args.action, args.reservation_id)
    print(json.dumps(result, separators=(',', ':')), flush=True)
    if args.action == 'run' and result.get('status') != 'completed':
        raise SystemExit(1)
