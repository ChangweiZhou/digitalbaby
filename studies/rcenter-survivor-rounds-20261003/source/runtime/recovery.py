# SPDX-License-Identifier: GPL-3.0-or-later
"""Same-lock production recovery, write-ahead accounting and missing-only planning.

A reservation charges the complete worker cap *before* it can be backed up and
started. Thus restoring the pre-worker checkpoint cannot forget an unknown run.
Only a normal, observed finish refunds unused reserved seconds. Recovery never
refunds. Neither this module nor the supervisor writes the host-wall ledger.
All paths persisted here are root-relative. No network or learner is imported.
"""
import contextlib
import copy
import fcntl
import json
import math
import os
import shutil
import time
import uuid
from pathlib import Path

from durable import create, replace, journal, validate_journal, sha, fsync_dir
from integrity import digest_map

CAP = 900.0
RSS = 768 * 1024**2
TOTAL = 12 * 3600.0
WALL = 18 * 3600.0
RESERVE = 3 * 1024**3
OUTPUT_CAP = 1024**3
OFFICIAL_WORLDS = tuple(range(310001, 310065))
QUALIFICATION_KEYS = ('cycle3/operations', 'cycle3/pilot', 'cycle3/replay')
ANALYSIS_KEY = 'analysis/final'
RUNTIME = {'python': '3.11.15', 'numpy': '2.2.6', 'scipy': '1.14.1', 'numba': '0.61.2'}


def check(value, message):
    if not value:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def relative(root, name):
    p = Path(name)
    check(not p.is_absolute() and '..' not in p.parts, 'unsafe persisted relative path')
    q = Path(root) / p
    check(q.resolve().is_relative_to(Path(root).resolve()), 'path or symlink outside study')
    return q


def mkdir(path):
    p = Path(path)
    if p.exists():
        check(p.is_dir() and not p.is_symlink(), 'directory replaced or symlinked')
        return
    mkdir(p.parent)
    p.mkdir()
    fsync_dir(p.parent)
    fsync_dir(p)


class Lock:
    def __init__(self, root, handle):
        self.root = Path(root).resolve()
        self.handle = handle
        self.pid = os.getpid()
        self.active = True

    def require(self, root):
        check(self.active and self.pid == os.getpid() and self.root == Path(root).resolve(), 'exclusive supervisor lock required')
        check(not self.handle.closed, 'supervisor lock closed')
        fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)


@contextlib.contextmanager
def exclusive_lock(root):
    root = Path(root).resolve()
    mkdir(root / 'operations')
    with (root / 'operations/SUPERVISOR.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock = Lock(root, handle)
        try:
            yield lock
        finally:
            lock.active = False
            # Closing the parent's descriptor leaves an inherited worker
            # descriptor locked until PDEATHSIG has actually killed that worker.


def process_identity(pid=None):
    pid = os.getpid() if pid is None else int(pid)
    try:
        stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return {'pid': pid, 'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                'start_ticks': int(stat[19]), 'state': stat[0]}
    except FileNotFoundError:
        return None


def writer_gone(identity):
    check(isinstance(identity, dict) and {'pid', 'boot_id', 'start_ticks'} <= set(identity), 'missing old process identity')
    now = process_identity(identity['pid'])
    return (now is None or now['state'] == 'Z' or
            any(now[k] != identity[k] for k in ('pid', 'boot_id', 'start_ticks')))


def receipt_path(key):
    if key == ANALYSIS_KEY:
        return 'results/final'
    if key in ('cycle1/test', 'cycle2/test'):
        return f'receipts/{key.split("/")[0]}/smoke.json'
    if key == 'cycle3/operations':
        return 'receipts/cycle3/operations.json'
    if key in ('cycle3/pilot', 'cycle3/replay'):
        return f'receipts/cycle3/{key.split("/")[1]}-310200'
    if key.startswith('science/'):
        w = int(key.split('/')[1])
        check(w in OFFICIAL_WORLDS and key == f'science/{w}', 'noncanonical official world')
        return f'receipts/science/{w}'
    raise RuntimeError('unapproved attempt key')


def output_size(root):
    # Include artifacts, isolated mocks, quarantine, journals and accounting.
    # Runtime installs/vendor inputs/backups have separate storage budgets.
    root = Path(root)
    total = 0
    for name in ('receipts', 'results'):
        total += sum(p.stat().st_size for p in (root / name).rglob('*') if p.is_file())
    excluded_inputs = {'venv', 'python', 'uv-cache', 'license-staging', 'staging'}
    for p in (root / 'operations').rglob('*'):
        rel = p.relative_to(root / 'operations')
        if rel.parts[0] not in excluded_inputs and p.is_file():
            total += p.stat().st_size
    return total


def limits(root, ledger, *, active_wall_s=None, extra_worker_s=0.0, output_bytes=None, free_bytes=None):
    worker = float(ledger['worker_s']) + extra_worker_s
    wall = float(ledger['active_wall_s'] if active_wall_s is None else active_wall_s)
    check(math.isfinite(worker) and 0 <= worker <= TOTAL, 'total worker cap')
    check(math.isfinite(wall) and float(ledger['active_wall_s']) <= wall <= WALL, 'total active host-wall cap')
    size = output_size(root) if output_bytes is None else output_bytes
    free = shutil.disk_usage(root).free if free_bytes is None else free_bytes
    check(size <= OUTPUT_CAP and free >= RESERVE, 'output cap or disk reserve')


def _body(ledger):
    return {k: v for k, v in ledger.items() if k != 'journal_anchor'}


def _digest(ledger):
    return digest_map(_body(ledger))


def _historical(root, attempt):
    """Validate accepted old receipts against old bytes, not today's closure."""
    cycle = int(attempt['key'][5])
    acceptance = read(root / f'audits/CYCLE{cycle}_ACCEPTED.json')
    review_path = root / f'audits/CYCLE{cycle}_SOURCE_REVIEW.json'
    review = read(review_path)
    check(acceptance.get('accepted') is True and review.get('accepted') is True, 'historical cycle not accepted')
    check(sha(review_path) == acceptance['source_review_sha256'], 'historical review drift')
    for rel, h in acceptance['input_hashes'].items():
        check(sha(relative(root, rel)) == h, 'historical acceptance input drift: ' + rel)
    check(sha(relative(root, acceptance['report_file'])) == acceptance['report_sha256'], 'historical report drift')
    check(sha(root / f'audits/CYCLE{cycle}_SOURCE_REVIEW.md') == review['report_sha256'], 'historical source report drift')
    source = review['source_hashes']
    check(digest_map(source) == attempt['source_digest'] == acceptance['source_digest'], 'historical source identity mismatch')
    snapshot = root / f'history/cycle{cycle}'
    metadata = read(snapshot / 'SNAPSHOT.json')
    check(metadata['source_review_sha256'] == sha(review_path) and metadata['exact_accepted_source_map'] == source, 'historical snapshot map mismatch')
    for rel, h in source.items():
        check(sha(relative(snapshot / 'source_snapshot', rel)) == h, 'historical source bytes drift: ' + rel)
    path = root / receipt_path(attempt['key'])
    check(sha(path) == attempt['receipt_sha256'] == acceptance['receipt_sha256'], 'historical receipt missing/tampered')
    receipt = read(path)
    check(receipt.get('passed') is True, 'historical test did not pass')
    if 'source' in receipt:
        check(receipt['source'] == source, 'historical receipt source mismatch')
    snap_path = relative(root, acceptance['ledger_snapshot'])
    check(sha(snap_path) == acceptance['ledger_snapshot_sha256'], 'historical ledger snapshot drift')
    check(attempt in read(snap_path)['attempts'], 'historical completed attempt mutated')
    check(attempt['runtime'] == RUNTIME, 'historical runtime mismatch')


def _legacy_base(root, ledger):
    check(ledger.get('schema') in ('RC-SURVIVOR-LEDGER-v1', 'RC-SURVIVOR-LEDGER-v3'), 'unsupported ledger baseline')
    check(all(a['key'] in ('cycle1/test', 'cycle2/test') and a['status'] == 'completed' for a in ledger['attempts']), 'unjournalled modern attempts prohibited')
    check(len({a['key'] for a in ledger['attempts']}) == len(ledger['attempts']), 'duplicate historical attempts')
    for attempt in ledger['attempts']:
        _historical(root, attempt)
    accepted = [c for c in (1, 2) if (root / f'audits/CYCLE{c}_ACCEPTED.json').exists()]
    if accepted:
        acceptance = read(root / f'audits/CYCLE{max(accepted)}_ACCEPTED.json')
        baseline = read(relative(root, acceptance['ledger_snapshot']))
        check(ledger['attempts'] == baseline['attempts'], 'accepted historical attempts missing or modified')
        check(ledger['active_wall_s'] == baseline['active_wall_s'], 'legacy wall baseline mutated')
    check(math.isclose(ledger['worker_s'], sum(a['charged_s'] for a in ledger['attempts']), abs_tol=1e-9), 'historical worker total mismatch')


def load_ledger(root, *, lock):
    """Verify WAL and anchor; replay only a validated append-before-replace tail."""
    root = Path(root)
    lock.require(root)
    path = root / 'operations/RUN_LEDGER.json'
    ledger = read(path) if path.exists() else {'schema': 'RC-SURVIVOR-LEDGER-v3', 'worker_s': 0.0, 'active_wall_s': 0.0, 'attempts': []}
    directory = root / 'operations/journal'
    validate_journal(directory)
    rows = sorted(directory.glob('*.json'))
    anchor = ledger.get('journal_anchor')
    if not rows:
        check(anchor is None, 'ledger journal tail deleted')
        _legacy_base(root, ledger)
        return ledger
    bodies = [read(p) for p in rows]
    first = bodies[0]
    check(first.get('event') == 'ledger_transition' and first['before_ledger'] is not None, 'missing ledger journal genesis')
    _legacy_base(root, first['before_ledger'])
    prior_digest = _digest(first['before_ledger'])
    for i, row in enumerate(bodies):
        check(rows[i].name == f'{i:06d}.json', 'noncanonical journal sequence')
        check(row.get('event') == 'ledger_transition' and row['before_digest'] == prior_digest, 'ledger WAL before identity mismatch')
        check(_digest(row['after_ledger']) == row['after_digest'], 'ledger WAL after identity mismatch')
        prior_digest = row['after_digest']
    if anchor is None:
        check(_digest(ledger) == bodies[0]['before_digest'], 'unanchored ledger differs from WAL genesis')
        start = 0
    else:
        n = anchor['sequence']
        check(isinstance(n, int) and 0 <= n < len(rows), 'journal anchor missing')
        check(sha(rows[n]) == anchor['sha256'] and _digest(ledger) == bodies[n]['after_digest'], 'ledger or anchor tampered')
        start = n + 1
    for n in range(start, len(rows)):
        check(_digest(ledger) == bodies[n]['before_digest'], 'uncommitted journal tail does not follow ledger')
        ledger = copy.deepcopy(bodies[n]['after_ledger'])
        ledger['journal_anchor'] = {'sequence': n, 'sha256': sha(rows[n])}
    if start < len(rows):
        replace(path, ledger)
    return ledger


def commit_ledger(root, ledger, new_ledger, reason, *, lock):
    root = Path(root)
    lock.require(root)
    current = load_ledger(root, lock=lock)
    check(current == ledger, 'stale ledger writer')
    after = _body(copy.deepcopy(new_ledger))
    row = {'event': 'ledger_transition', 'reason': reason, 'before_digest': _digest(ledger),
           'after_digest': _digest(after), 'before_ledger': _body(ledger) if not ledger.get('journal_anchor') else None,
           'after_ledger': after}
    result = journal(root / 'operations/journal', row)
    path = Path(result['path'])
    after['journal_anchor'] = {'sequence': int(path.stem), 'sha256': result['sha256']}
    replace(root / 'operations/RUN_LEDGER.json', after)
    return after


def source_transition(root, attempt, source, runtime):
    """Validate one independently authorized failed-source transition.

    This authenticates a preserved old failed entry; it never relabels its
    source, refunds its charge, or admits an old completed science receipt.
    The private per-reservation record remains necessary after later retry
    approvals replace RECOVERY_ACCEPTED.json.
    """
    root = Path(root)
    check(attempt['status'] in ('failed', 'interrupted'), 'source transition requires preserved failed/interrupted attempt')
    check(attempt['runtime'] == runtime == RUNTIME, 'source transition runtime mismatch')
    reservation = attempt['reservation_id']
    check(isinstance(reservation, str) and len(reservation) == 32 and all(c in '0123456789abcdef' for c in reservation), 'source transition reservation identity malformed')
    path = root / f'operations/RECOVERY_TRANSITIONS/{reservation}.json'
    check(path.is_file(), 'attempt source/runtime mismatch: independently accepted source transition missing')
    transition = read(path)
    check(transition.get('schema') == 'RC-SURVIVOR-SOURCE-TRANSITION-v1' and
          transition.get('accepted') is True and transition.get('independent_review') is True and
          transition.get('classification') == 'infrastructure' and transition.get('disposition') == 'retry',
          'independently accepted infrastructure source transition required')
    for field, expected in (('key', attempt['key']), ('attempt_count', attempt['attempt']),
                            ('runtime', runtime), ('reservation_id', reservation),
                            ('source_digest', attempt['source_digest']), ('target_source_digest', digest_map(source)),
                            ('attempt_entry_sha256', digest_map(attempt))):
        check(transition.get(field) == expected, 'source transition exact identity mismatch: ' + field)
    check(transition['source_digest'] != transition['target_source_digest'], 'source transition must name distinct exact revisions')
    snapshot_name = transition['snapshot_dir']
    check(Path(snapshot_name).parts[0] == 'history', 'source transition snapshot must be preserved history')
    snapshot = relative(root, snapshot_name)
    metadata_path = snapshot / 'SNAPSHOT.json'
    check(sha(metadata_path) == transition['snapshot_sha256'], 'source transition snapshot metadata changed')
    metadata = read(metadata_path)
    old_review_path = snapshot / 'CYCLE3_SOURCE_REVIEW.json'
    old_review = read(old_review_path)
    check(old_review.get('accepted') is True and sha(old_review_path) == metadata['source_review_sha256'], 'old source admission missing/changed')
    check(sha(snapshot / 'CYCLE3_SOURCE_REVIEW.md') == metadata['report_sha256'] == old_review['report_sha256'], 'old source report changed')
    old_source = metadata['source_hashes']
    check(old_source and old_source == old_review['source_hashes'] and
          digest_map(old_source) == metadata['source_digest'] == attempt['source_digest'], 'old source map identity mismatch')
    source_root = snapshot / 'source_snapshot'
    check({str(p.relative_to(source_root)) for p in source_root.rglob('*') if p.is_file() and '__pycache__' not in p.parts} == set(old_source),
          'old source snapshot must contain the complete exact accepted closure')
    for name, expected in old_source.items():
        check(sha(relative(source_root, name)) == expected, 'old accepted source bytes changed: ' + name)
    failed_path = snapshot / 'FAILED_LEDGER.json'
    check(sha(failed_path) == metadata['failed_ledger_sha256'], 'old failed-ledger snapshot changed')
    matching = [a for a in read(failed_path)['attempts'] if a.get('reservation_id') == reservation]
    check(len(matching) == 1 and matching[0] == attempt, 'old attempt entry/charge changed')
    check(sha(snapshot / 'worker.log') == metadata['worker_log_sha256'] == sha(relative(root, attempt['log'])), 'old worker log changed/missing')
    current_review_path = root / 'audits/CYCLE3_SOURCE_REVIEW.json'
    current_review = read(current_review_path)
    check(sha(current_review_path) == transition['current_source_review_sha256'] and current_review.get('accepted') is True,
          'new source independent acceptance missing/changed')
    check(current_review.get('source_hashes') == source and current_review.get('source_digest') == digest_map(source),
          'new accepted source revision does not match transition')
    check(sha(root / 'audits/CYCLE3_SOURCE_REVIEW.md') == current_review['report_sha256'], 'new source report changed')
    return {'record': transition, 'sha256': sha(path)}


def validate_attempts(root, ledger, source, runtime):
    root = Path(root)
    check(runtime == RUNTIME, 'runtime mismatch')
    counts = {}
    complete = set()
    accepted = [c for c in (1, 2) if (root / f'audits/CYCLE{c}_ACCEPTED.json').exists()]
    if accepted:
        acceptance = read(root / f'audits/CYCLE{max(accepted)}_ACCEPTED.json')
        baseline = read(relative(root, acceptance['ledger_snapshot']))
        check(ledger['attempts'][:len(baseline['attempts'])] == baseline['attempts'], 'accepted historical attempts not preserved')
        check(ledger['active_wall_s'] == baseline['active_wall_s'], 'legacy wall baseline mutated')
    for a in ledger['attempts']:
        key = a['key']
        check(key not in complete, 'attempt after completed key prohibited')
        if key == ANALYSIS_KEY:
            check(set(QUALIFICATION_KEYS) <= complete and all(f'science/{w}' in complete for w in OFFICIAL_WORLDS), 'analysis attempt before all64 completed worlds')
        if key.startswith('science/'):
            check(ANALYSIS_KEY not in complete, 'science attempt after final analysis prohibited')
            world = int(key.split('/')[1])
            check(set(QUALIFICATION_KEYS) <= complete, 'science attempt before completed qualification')
            check(all(f'science/{w}' in complete for w in range(310001, world)), 'science attempt skipped earlier world')
        elif key in QUALIFICATION_KEYS:
            check(set(QUALIFICATION_KEYS[:QUALIFICATION_KEYS.index(key)]) <= complete, 'qualification attempt skipped prerequisite')
        counts[key] = counts.get(key, 0) + 1
        check(counts[key] <= 3, 'exact-key attempt cap')
        receipt_path(key)
        check(a['status'] in ('completed', 'reserved', 'running', 'failed', 'interrupted', 'receipt_complete_pending_review'), 'unknown attempt status')
        check(math.isfinite(a['charged_s']) and a['charged_s'] >= 0, 'invalid attempt charge')
        if key in ('cycle1/test', 'cycle2/test'):
            _historical(root, a)
        else:
            check(a['attempt'] == counts[key], 'attempt count identity mismatch')
            check(a['runtime'] == runtime, 'attempt source/runtime mismatch')
            if a['source_digest'] != digest_map(source):
                source_transition(root, a, source, runtime)
            check(a['receipt'] == receipt_path(key), 'attempt receipt path mismatch')
            check(isinstance(a['reservation_id'], str) and len(a['reservation_id']) == 32, 'attempt reservation identity mismatch')
            check(a.get('cap_reserved_s') == CAP, 'attempt lacks conservative pre-job reservation')
            check(isinstance(a.get('host_token'), str) and a['host_token'], 'reservation host session missing')
            if a['status'] in ('reserved', 'running', 'interrupted', 'receipt_complete_pending_review'):
                check(a['charged_s'] == CAP, 'unknown duration must retain complete cap')
        if a['status'] == 'completed':
            check(key not in complete, 'duplicate completed key')
            complete.add(key)
    check(math.isclose(ledger['worker_s'], sum(a['charged_s'] for a in ledger['attempts']), abs_tol=1e-7), 'ledger worker sum mismatch')


def validate_completed(root, attempt, source, runtime, *, receipt_loader=None):
    root = Path(root)
    key = attempt['key']
    if key in ('cycle1/test', 'cycle2/test'):
        _historical(root, attempt)
        return {'manifest_sha256': attempt['receipt_sha256']}
    path = root / receipt_path(key)
    if key == ANALYSIS_KEY:
        from validate_analysis import load as load_analysis
        result = load_analysis(root, path, source, runtime, attempt=attempt, receipt_loader=receipt_loader)
    elif key == 'cycle3/operations':
        d = read(path)
        check(d['schema'] == 'RC-SURVIVOR-CYCLE3-OPERATIONS-v1' and d['passed'] is True and d['mock_only'] is True and d['native_teaching_events'] == 0, 'invalid pure operations receipt')
        check(d['source'] == source and d['runtime'] == runtime, 'operations source/runtime mismatch')
        check(d['key'] == key and d['attempt'] == attempt['attempt'] and d['reservation_id'] == attempt['reservation_id'], 'operations attempt identity mismatch')
        result = {'manifest_sha256': sha(path), 'manifest': {'resources': d['resources']}}
    else:
        if receipt_loader is None:
            from validate_receipt import load
            acceptance = sha(root / ('protocol/OFFICIAL_LOCK.json' if key.startswith('science/') else 'protocol/ROUND1_DESIGN_CYCLE3.md'))
            def receipt_loader(path, world, source):
                return load(path, world, source, expected_kind='science' if key.startswith('science/') else 'technical', expected_acceptance=acceptance)
        world = int(key.split('/')[1]) if key.startswith('science/') else 310200
        result = receipt_loader(path, world, source)
        check(result['manifest']['kind'] == ('science' if key.startswith('science/') else 'technical'), 'receipt kind mismatch')
        check(result['header']['runtime'] == runtime, 'receipt runtime mismatch')
        if key == 'cycle3/replay':
            first = receipt_loader(root / receipt_path('cycle3/pilot'), 310200, source)
            comparison = read(root / 'receipts/cycle3/replay_comparison.json')
            check(comparison.get('passed') is True and comparison.get('fresh_process') is True and comparison.get('same_world') == 310200, 'independent replay comparison missing')
            check(comparison['first_manifest_sha256'] == first['manifest_sha256'] and comparison['replay_manifest_sha256'] == result['manifest_sha256'], 'replay comparison manifest mismatch')
            check(comparison['scientific_digest'] == result['scientific_digest'] == first['scientific_digest'], 'scientific replay mismatch')
    if attempt.get('receipt_sha256'):
        check(result['manifest_sha256'] == attempt['receipt_sha256'], 'completed local receipt missing/tampered')
    measured = result['manifest']['resources']
    check(math.isfinite(measured['wall_s']) and 0 <= measured['wall_s'] <= CAP, 'receipt time exceeds cap')
    check(0 <= measured['peak_rss_bytes'] <= RSS, 'receipt RSS exceeds cap')
    return result


def recovery_approval(root, attempt, count, disposition, *, source=None):
    r = read(Path(root) / 'operations/RECOVERY_ACCEPTED.json')
    check(r.get('accepted') is True and r.get('independent_review') is True and r.get('classification') == 'infrastructure', 'independent infrastructure classification required')
    check(r.get('key') == attempt['key'] and r.get('attempt_count') == count and r.get('source_digest') == attempt['source_digest'], 'recovery approval key/count/source mismatch')
    check(r.get('runtime') == attempt['runtime'] and r.get('disposition') == disposition, 'recovery approval runtime/disposition mismatch')
    check(r.get('reservation_id') == attempt['reservation_id'], 'recovery approval attempt identity mismatch')
    if source is not None and attempt['source_digest'] != digest_map(source):
        check(disposition == 'retry', 'old-source receipt cannot be promoted to current completion')
        transition = source_transition(root, attempt, source, attempt['runtime'])
        check(r.get('target_source_digest') == digest_map(source) and r.get('source_transition_sha256') == transition['sha256'],
              'retry approval does not bind exact independently accepted source transition')
    return r


def _quarantine(root, ledger, index, *, lock, preserve_entry=False):
    a = ledger['attempts'][index]
    if any(x['key'] == a['key'] and x.get('attempt', 0) > a['attempt'] for x in ledger['attempts']):
        # The canonical destination now belongs to the later attempt. Never
        # steal its new receipt while rechecking an older failed attempt.
        if a.get('quarantine_path'):
            check(relative(root, a['quarantine_path']).exists(), 'preserved quarantine artifact missing')
        return ledger
    source = relative(root, a['receipt'])
    dest_rel = a.get('quarantine_path') or f'operations/quarantine/{a["key"]}/attempt-{a["attempt"]}-{a["reservation_id"]}'
    dest = relative(root, dest_rel)
    if not source.exists() and not dest.exists():
        return ledger
    if 'quarantine_path' not in a:
        check(not preserve_entry, 'old-source partial output must be preserved/quarantined before transition is frozen')
        new = copy.deepcopy(ledger)
        new['attempts'][index]['quarantine_path'] = dest_rel
        ledger = commit_ledger(root, ledger, new, 'quarantine_intent', lock=lock)
    check(not (source.exists() and dest.exists()), 'both partial and quarantine exist; preserve and stop')
    if source.exists():
        mkdir(dest.parent)
        os.rename(source, dest)
        fsync_dir(source.parent)
        fsync_dir(dest.parent)
    return ledger


def _pending_barriers(root, ledger):
    p = Path(root) / 'operations/PERSISTENCE_STATE.json'
    persistence = read(p) if p.exists() else {'worlds': {}, 'jobs': {}}
    pending = []
    for a in ledger['attempts']:
        if a['status'] != 'completed' or a['key'] in ('cycle1/test', 'cycle2/test'):
            continue
        key = a['key']
        ack = persistence.get('worlds', {}).get(key.split('/')[1]) if key.startswith('science/') else persistence.get('jobs', {}).get(key)
        if not (ack and ack.get('manifest_sha256') == a['receipt_sha256'] and ack.get('private_readback_verified') is True and ack.get('github_tree_verified') is True):
            pending.append({'key': key, 'receipt': a['receipt'], 'manifest_sha256': a['receipt_sha256']})
    check(len(pending) <= 1, 'more than one unbacked completed job')
    return pending


def reconcile_and_plan(root, source, runtime, mode, *, lock, receipt_loader=None, keep_reservation_id=None, active_wall_s=None):
    """Reconcile dead writers and return deterministic missing-only plan.

    The inherited exclusive lock plus recorded /proc birth identity establishes
    the old writer is gone. A valid complete receipt in a crash window remains
    blocked pending independent infrastructure disposition; no silent promotion.
    """
    root = Path(root)
    lock.require(root)
    check(mode in ('cycle3', 'science'), 'unsupported plan mode')
    ledger = load_ledger(root, lock=lock)
    validate_attempts(root, ledger, source, runtime)
    for a in ledger['attempts']:
        if a['status'] == 'completed':
            validate_completed(root, a, source, runtime, receipt_loader=receipt_loader)
    for i in range(len(ledger['attempts'])):
        a = ledger['attempts'][i]
        if a['status'] in ('reserved', 'running'):
            if a['status'] == 'reserved' and a['reservation_id'] == keep_reservation_id:
                continue
            if a['status'] == 'running':
                check(a.get('parent_death_guard') == 'SIGKILL+inherited-exclusive-lock', 'missing old writer death guarantee')
                check(writer_gone(a['supervisor_identity']) and writer_gone(a['child_identity']), 'old writer still alive; do not reconcile')
            result = None
            error = None
            try:
                result = validate_completed(root, a, source, runtime, receipt_loader=receipt_loader)
            except (OSError, ValueError, KeyError, AssertionError, RuntimeError) as exc:
                error = str(exc)
            new = copy.deepcopy(ledger)
            target = new['attempts'][i]
            target.update(status='receipt_complete_pending_review' if result else 'interrupted', charged_s=CAP,
                          recovery_error=error, recovered_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
            if result:
                target['receipt_sha256'] = result['manifest_sha256']
            new['worker_s'] = sum(x['charged_s'] for x in new['attempts'])
            ledger = commit_ledger(root, ledger, new, 'interruption_reconciled_at_reserved_cap', lock=lock)
            a = ledger['attempts'][i]
        if a['status'] == 'receipt_complete_pending_review':
            validate_completed(root, a, source, runtime, receipt_loader=receipt_loader)
            approval = root / 'operations/RECOVERY_ACCEPTED.json'
            if approval.exists():
                count = sum(x['key'] == a['key'] for x in ledger['attempts'])
                recovery_approval(root, a, count, 'accept_completed_receipt', source=source)
                new = copy.deepcopy(ledger)
                new['attempts'][i]['status'] = 'completed'
                ledger = commit_ledger(root, ledger, new, 'independently_accepted_complete_crash_receipt', lock=lock)
        elif a['status'] in ('failed', 'interrupted'):
            ledger = _quarantine(root, ledger, i, lock=lock, preserve_entry=a['source_digest'] != digest_map(source))
    validate_attempts(root, ledger, source, runtime)
    pending = _pending_barriers(root, ledger)
    completed = {a['key'] for a in ledger['attempts'] if a['status'] == 'completed'}
    sequence = QUALIFICATION_KEYS if mode == 'cycle3' else tuple(f'science/{w}' for w in OFFICIAL_WORLDS) + (ANALYSIS_KEY,)
    missing = [k for k in sequence if k not in completed]
    blocked = []
    if pending:
        blocked.append('persistence_barrier')
    transport = root / 'operations/TRANSPORT_PENDING.json'
    if transport.exists() and read(transport) is not None:
        blocked.append('unresolved_transport')
    if any(a['status'] == 'receipt_complete_pending_review' for a in ledger['attempts']):
        blocked.append('complete_receipt_requires_independent_review')
    if mode == 'science' and not set(QUALIFICATION_KEYS) <= completed:
        blocked.append('qualification_prerequisites_missing')
    if any(a['status'] == 'reserved' for a in ledger['attempts']):
        blocked.append('reserved_attempt')
    if missing:
        prior = [a for a in ledger['attempts'] if a['key'] == missing[0]]
        if prior and prior[-1]['status'] in ('failed', 'interrupted'):
            if len(prior) >= 3:
                blocked.append('exact_key_attempt_cap')
            else:
                try:
                    recovery_approval(root, prior[-1], len(prior), 'retry', source=source)
                except (OSError, KeyError, ValueError, RuntimeError) as exc:
                    blocked.append('retry_requires_independent_infrastructure_approval')
    try:
        limits(root, ledger, active_wall_s=active_wall_s)
    except RuntimeError as exc:
        blocked.append(str(exc))
    return {'ledger': ledger, 'missing': missing, 'completed': sorted(completed), 'pending_barriers': pending,
            'blocked': blocked, 'next_key': missing[0] if missing and not blocked else None}


def prepare_attempt(root, source, runtime, mode, key, *, lock, host_token=None, active_wall_s=None, receipt_loader=None):
    root = Path(root)
    check(isinstance(host_token, str) and host_token, 'active host token required for reservation')
    plan = reconcile_and_plan(root, source, runtime, mode, lock=lock, active_wall_s=active_wall_s, receipt_loader=receipt_loader)
    check(plan['next_key'] == key, 'not the next admitted missing key: ' + repr(plan['blocked']))
    ledger = plan['ledger']
    prior = [a for a in ledger['attempts'] if a['key'] == key]
    check(len(prior) < 3, 'exact-key attempt cap')
    if prior:
        recovery_approval(root, prior[-1], len(prior), 'retry', source=source)
    limits(root, ledger, active_wall_s=active_wall_s, extra_worker_s=CAP)
    # Require room for the complete job in both independent budgets.
    check((ledger['active_wall_s'] if active_wall_s is None else active_wall_s) + CAP <= WALL, 'insufficient active-wall reserve')
    check(not (root / receipt_path(key)).exists(), 'unowned/partial output must be reconciled before admission')
    number = len(prior) + 1
    attempt_dir = root / f'operations/attempts/{key}/attempt-{number}'
    check(not attempt_dir.exists(), 'attempt artifact directory already exists')
    mkdir(attempt_dir)
    a = {'key': key, 'attempt': number, 'status': 'reserved', 'reservation_id': uuid.uuid4().hex, 'host_token': host_token,
         'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'charged_s': CAP, 'cap_reserved_s': CAP,
         'source_digest': digest_map(source), 'runtime': runtime, 'receipt': receipt_path(key),
         'log': str((attempt_dir / 'worker.log').relative_to(root))}
    new = copy.deepcopy(ledger)
    new['schema'] = 'RC-SURVIVOR-LEDGER-v3'
    new['attempts'].append(a)
    new['worker_s'] += CAP
    ledger = commit_ledger(root, ledger, new, 'prejob_full_cap_reservation', lock=lock)
    return {'attempt': a, 'ledger_sha256': sha(root / 'operations/RUN_LEDGER.json'), 'journal_anchor': ledger['journal_anchor']}


def require_reservation_ack(root, attempt):
    """Host writes this only after archived reservation readback is verified."""
    r = read(Path(root) / 'operations/RESERVATION_ACK.json')
    check(r.get('reservation_id') == attempt['reservation_id'] and r.get('key') == attempt['key'] and r.get('source_digest') == attempt['source_digest'], 'reservation backup identity mismatch')
    check(r.get('charged_s') == CAP and r.get('private_readback_verified') is True, 'full-cap reservation not durably read back')
    check(r.get('ledger_sha256') == sha(Path(root) / 'operations/RUN_LEDGER.json'), 'reservation acknowledged against different ledger')
    check(isinstance(r.get('checkpoint_sha256'), str) and len(r['checkpoint_sha256']) == 64, 'reservation checkpoint identity missing')
    check(isinstance(r.get('content_identity'), str) and len(r['content_identity']) == 64, 'reservation checkpoint content identity missing')
    check(isinstance(r.get('library_file_id'), str) and r['library_file_id'] and isinstance(r.get('version'), int) and r['version'] >= 0, 'reservation Library readback identity missing')
    return r


def start_reserved_attempt(root, reservation_id, source, runtime, mode, *, lock, child_identity, host_token=None, active_wall_s=None, receipt_loader=None):
    root = Path(root)
    plan = reconcile_and_plan(root, source, runtime, mode, lock=lock, keep_reservation_id=reservation_id,
                              active_wall_s=active_wall_s, receipt_loader=receipt_loader)
    check(set(plan['blocked']) <= {'reserved_attempt'}, 'reserved start blocked: ' + repr(plan['blocked']))
    ledger = plan['ledger']
    candidates = [(i, a) for i, a in enumerate(ledger['attempts']) if a.get('reservation_id') == reservation_id]
    check(len(candidates) == 1 and candidates[0][1]['status'] == 'reserved', 'reservation missing or consumed')
    i, a = candidates[0]
    check(host_token and a['host_token'] == host_token, 'reservation belongs to a different/lost host session')
    check(plan['missing'] and plan['missing'][0] == a['key'], 'reserved key no longer next')
    require_reservation_ack(root, a)
    new = copy.deepcopy(ledger)
    new['attempts'][i].update(status='running', supervisor_identity=process_identity(), child_identity=child_identity,
                              parent_death_guard='SIGKILL+inherited-exclusive-lock')
    ledger = commit_ledger(root, ledger, new, 'reserved_worker_started', lock=lock)
    return ledger, ledger['attempts'][i]


def finish_attempt(root, reservation_id, source, runtime, duration, high_rss, error, exit_code, *, lock, receipt_loader=None):
    root = Path(root)
    ledger = load_ledger(root, lock=lock)
    validate_attempts(root, ledger, source, runtime)
    matches = [(i, a) for i, a in enumerate(ledger['attempts']) if a.get('reservation_id') == reservation_id]
    check(len(matches) == 1 and matches[0][1]['status'] == 'running', 'attempt not running or already finalized')
    i, a = matches[0]
    check(math.isfinite(duration) and duration >= 0, 'invalid measured worker duration')
    result = None
    if not error:
        check(exit_code == 0 and duration <= CAP and high_rss <= RSS, 'observed worker cap/exit violation')
        result = validate_completed(root, a, source, runtime, receipt_loader=receipt_loader)
    new = copy.deepcopy(ledger)
    target = new['attempts'][i]
    # A measured overrun is never hidden by truncating it to the nominal cap.
    charge = duration
    target.update(status='failed' if error else 'completed', charged_s=charge, measured_s=duration,
                  peak_rss_bytes=high_rss, error=error, exit_code=exit_code)
    if result:
        target['receipt_sha256'] = result['manifest_sha256']
    new['worker_s'] = sum(x['charged_s'] for x in new['attempts'])
    ledger = commit_ledger(root, ledger, new, 'observed_worker_finish', lock=lock)
    return ledger['attempts'][i]
