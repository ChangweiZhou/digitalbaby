"""Additive, fail-closed final audit and local return bundle for frozen Package A V3.

No simulation, analyzer import, partial science reduction, network or upload. The
CLI requires the complete primary/replay/qualified-analysis admission contract.
Pure functions are testable with synthetic data and world-190000 technical data.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime
import fcntl
import gzip
import hashlib
import json
import math
import os
import resource
import sys
import time
import uuid
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
ARMS = ('R0', 'R1', 'R1_rand', 'R0_signed', 'R3', 'R3_randtarget',
        'Z0_resource', 'Z2', 'Z2_rand', 'P0', 'P1', 'P2', 'P4')
WORLDS = tuple(range(190001, 190065))
BRANCHES = ('W', 'N_old_fact', 'N_old_rel', 'N_new_fact', 'N_new_rel')
PRIMARY = {'R1': 'R0', 'R3': 'R0_signed', 'Z2': 'Z0_resource',
           'P1': 'P0', 'P2': 'P0', 'P4': 'P0'}
DIAGNOSTIC = {'R1': 'R1_rand', 'R3': 'R3_randtarget', 'Z2': 'Z2_rand'}
DEPENDENCIES = {'R1_rand': 'R1', 'Z2_rand': 'Z2'}
CANONICAL_B = '32a3726cb25a8255c1aef5696ff40aa303138c5d061a598004d2087d4da5f964'
LOCK_SHA = 'f321f6c00e43a7f476728522357d50514444e7e5d7cfe2ca3996e2c8b8976572'
QUALIFICATION_SHA = '2e963204efd6b973a2d1254c031d24767aeb31cdd3f547c6b6d1babc1cf9d17a'
M, ALPHA = 26, .05
REPLAY_SCOPE = {'specification': 'SPEC_LOCK.md:148-152', 'branch': 'W', 'records': 600,
    'Z_and_P_store': 0, 'R_novelty': 'private pre-answer codes',
    'signed_R_shared_values': 'all four shared stores, all 600 records',
    'P_concentration_checkpoints': [336, 600], 'probe_recalculation': False}
OUTPUTS = ('FINAL_AUDIT.json', 'RESOURCE_REPORT.json', 'REPORT.md',
           'INDEPENDENT_METRICS.json', 'FIXTURE_EXPOSURE_AUDIT.json',
           'PER_ITEM_FAILURES.jsonl.gz', 'FINALIZATION_COMMIT.json')
# All selectors are literal declarations, independent of the frozen reducer.
SELECTORS = {
    'old_fact_final_W': ('final', 'W', 'old_fact', 'canonical'),
    'old_fact_final_N': ('final', 'N_old_fact', 'old_fact', 'canonical'),
    'heldout_final_W': ('final', 'W', 'old_relation_heldout', None),
    'heldout_final_N_old_rel': ('final', 'N_old_rel', 'old_relation_heldout', None),
    'old_fact_old_end_W': ('old_end', 'W', 'old_fact', 'canonical'),
    'new_fact_final_W': ('final', 'W', 'new_fact', 'canonical'),
    'new_fact_final_N_new_fact': ('final', 'N_new_fact', 'new_fact', 'canonical'),
    'new_relation_taught_final_W': ('final', 'W', 'new_relation_taught', None),
    'new_relation_taught_final_N_new_rel': ('final', 'N_new_rel', 'new_relation_taught', None),
    'old_relation_taught_final_W': ('final', 'W', 'old_relation_taught', None),
    'old_relation_taught_final_N_old_rel': ('final', 'N_old_rel', 'old_relation_taught', None),
    'heldout_old_end_W': ('old_end', 'W', 'old_relation_heldout', None),
    'old_fact_spacing_final_W': ('final', 'W', 'old_fact', 'spacing'),
    'old_fact_prefix_final_W': ('final', 'W', 'old_fact', 'prefix'),
    'old_fact_marker_final_W': ('final', 'W', 'old_fact', 'inner_marker'),
}


class FinalAuditError(AssertionError):
    pass


def need(condition, message):
    if not condition:
        raise FinalAuditError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def read_json(path):
    return json.loads(Path(path).read_bytes())


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=1, allow_nan=False)
        stream.write('\n')


def safe_path(root, name):
    p = Path(name)
    need(not p.is_absolute() and '..' not in p.parts, f'unsafe relative path: {name}')
    result = root / p
    need(result.resolve().is_relative_to(root.resolve()), f'path escaped root: {name}')
    return result


def receipt_name(arm, world):
    return f'results/science/{arm}/{world}.json.gz'


def roster_names():
    return {receipt_name(a, w) for a in ARMS for w in WORLDS}


def verify_source(root):
    """Hash every frozen file independently before asking original verifier too."""
    need(file_sha(root / 'SOURCE_LOCK.json') == LOCK_SHA, 'wrong pinned SOURCE_LOCK bytes')
    locked = read_json(root / 'SOURCE_LOCK.json')
    need(sha(canonical({k: v for k, v in locked.items() if k != 'lock_digest'})) == locked['lock_digest'],
         'source lock body digest')
    need(locked['arms'] == list(ARMS) and locked['worlds'] == list(WORLDS) and locked['family_size_m'] == M,
         'frozen roster/family changed')
    for name, expected in locked['files'].items():
        need(file_sha(safe_path(root, name)) == expected, f'wrong source hash: {name}')
    expected_code = {n for n in locked['files'] if n.startswith(('src/', 'tests/')) and n.endswith('.py')}
    actual_code = {p.relative_to(root).as_posix() for d in ('src', 'tests') for p in (root / d).glob('*.py')}
    need(actual_code == expected_code, 'new/missing locked scientific module')
    sys.path.insert(0, str(root / 'src'))
    import lock
    need(Path(lock.__file__).resolve() == root / 'src/lock.py', 'wrong lock module')
    need(lock.verify_lock() == locked, 'frozen runtime/source validation changed')
    source = {n: locked['files'][n] for n in expected_code}
    for n in ('package/MANIFEST.json', 'SPEC_LOCK.md', 'ARM_ROSTER.json', 'results/calibration/R_CALIBRATION.json'):
        source[n] = locked['files'][n]
    return locked, source


def verify_terminal_persistence(root, status, ledger, backup_path=None):
    """Reconcile Driver.finish's stale cache using exact append-only evidence.

    Frozen finish() persists directly without updating ledger.persisted_receipts.
    This checks raw hashes/metadata only; no science receipt is decoded.
    """
    need(status.get('state') == 'complete' and status.get('validated_receipts') == 832 and
         status.get('running') == [] and status.get('failure') is None and ledger.get('failure') is None,
         'terminal persistence requires complete quiescent status')
    need(type(ledger.get('persisted_receipts')) is int and 0 <= ledger['persisted_receipts'] <= 832,
         'invalid cached persistence count')
    local_path = root / 'results/recovery/LOCAL_STATUS.json'
    local_raw = local_path.read_bytes()
    local = json.loads(local_raw)
    need(local.get('local_validated_receipts') == 832, 'full local persistence required')
    chain, pinned = [], {local_path: sha(local_raw)}
    for path in (root / 'results/recovery/local_queue').glob('*.json'):
        raw = path.read_bytes(); doc = json.loads(raw)
        need(doc.get('schema') == 'MINIFLY-A3-LOCAL-CHECKPOINT-v2' and type(doc.get('sequence')) is int,
             'invalid local checkpoint schema/sequence')
        chain.append((doc['sequence'], path, sha(raw), doc))
    chain.sort(key=lambda entry: entry[0])
    need(chain, 'local checkpoint chain absent')
    previous, prior = None, None
    for sequence, path, digest, doc in chain:
        need(sequence == (1 if previous is None else previous['sequence'] + 1) and
             doc.get('previous_checkpoint') == previous, 'broken/duplicate local checkpoint chain')
        roster = doc['receipt_sha256']
        need(doc.get('lock_digest') == ledger['lock_digest'] == status['lock_digest'] and
             doc.get('validated_receipts') == len(roster) and set(roster) <= roster_names(),
             'checkpoint lock/count/roster mismatch')
        if prior is not None:
            need(all(roster.get(k) == v for k, v in prior['receipt_sha256'].items()),
                 'checkpoint replaced or omitted prior receipt')
            need(all(doc['ledger_snapshot'][k] >= prior['ledger_snapshot'][k]
                     for k in ('active_wall_s', 'worker_s')), 'checkpoint cumulative counters decreased')
        previous = {'path': path.relative_to(root).as_posix(), 'sha256': digest, 'sequence': sequence}
        prior = doc; pinned[path] = digest
    need(local.get('checkpoint') == previous['path'] and local.get('checkpoint_sha256') == previous['sha256'],
         'LOCAL_STATUS does not identify the exact terminal checkpoint')
    need(prior['validated_receipts'] == 832 and set(prior['receipt_sha256']) == roster_names() and
         prior['ledger_snapshot'] == ledger and prior['status_snapshot'] == status,
         'terminal checkpoint not exact full current roster/ledger/status')
    for name, digest in prior['receipt_sha256'].items():
        path = safe_path(root, name)
        need(file_sha(path) == digest, 'terminal persisted receipt changed: ' + name)
        pinned[path] = digest
    # A private count/ACK alone is insufficient: verify the independently read
    # back archive and its archived source/ledger/status and full receipt map.
    backup_path = Path(backup_path) if backup_path else root.parent.parent / 'minifly-recovery-evidence' / \
        'library-readback/MiniFly_A_V3_checkpoint_832.zip'
    ack_path = root / 'results/recovery/PRIVATE_BACKUP_STATUS.json'
    ack_raw = ack_path.read_bytes(); ack = json.loads(ack_raw)
    need(ack.get('schema') == 'MINIFLY-A3-VERIFIED-PRIVATE-BACKUP-v1' and ack.get('verified_at_utc') and
         ack.get('library_file_id') and ack.get('lock_digest') == ledger['lock_digest'] and
         ack.get('receipt_sha256') == prior['receipt_sha256'], 'terminal private ACK identity/roster mismatch')
    need(file_sha(backup_path) == ack.get('archive_sha256'), 'terminal private readback archive changed')
    with zipfile.ZipFile(backup_path) as archive:
        names = archive.namelist()
        need(len(names) == len(set(names)), 'duplicate terminal private archive member')
        raw_manifest = archive.read('RECOVERY_MANIFEST.json')
        manifest = json.loads(raw_manifest)
        need(manifest.get('schema') == 'MINIFLY-A3-PRIVATE-INCREMENTAL-v1' and
             manifest.get('lock_digest') == ledger['lock_digest'] and
             manifest.get('receipt_sha256') == prior['receipt_sha256'] and
             manifest.get('total_recoverable_with_base_and_prior_backups') == 832,
             'terminal private archive roster/lock mismatch')
        need(set(names) == set(manifest['file_sha256']) | {'RECOVERY_MANIFEST.json'},
             'terminal private archive manifest coverage')
        for name, digest in manifest['file_sha256'].items():
            safe_path(Path('/virtual-bundle'), name)
            need(sha(archive.read(name)) == digest, 'terminal private archived file changed: ' + name)
        for name in ('SOURCE_LOCK.json', 'results/science/RUN_LEDGER.json', 'results/science/RUN_STATUS.json'):
            need(archive.read('minifly_a_v3/' + name) == (root / name).read_bytes(),
                 'terminal private snapshot differs: ' + name)
    pinned[ack_path] = sha(ack_raw); pinned[backup_path] = ack['archive_sha256']
    recheck_inputs(pinned)
    return {'authoritative_persisted_receipts': 832, 'ledger_cached_persisted_receipts': ledger['persisted_receipts'],
        'cache_is_stale': ledger['persisted_receipts'] != 832, 'checkpoint': previous,
        'chain_records': len(chain), 'checkpoint_sha256': {p.relative_to(root).as_posix(): h
            for p, h in pinned.items() if p.parent == root / 'results/recovery/local_queue'},
        'local_status_sha256': sha(local_raw), 'private_backup_ack_sha256': sha(ack_raw),
        'private_readback_path': str(backup_path), 'private_readback_sha256': ack['archive_sha256'],
        'private_manifest_sha256': sha(raw_manifest),
        'explanation': 'Frozen Driver.finish calls env.persist directly and does not update the ledger cache; '
            'the unchanged cache is disclosed, while exact terminal journal and independently read-back backup contents prove persistence'}, pinned


def preflight_roster(root, terminal_backup_path=None):
    """Metadata/filesystem gate only: must precede any science receipt decoding."""
    status = read_json(root / 'results/science/RUN_STATUS.json')
    ledger = read_json(root / 'results/science/RUN_LEDGER.json')
    need(status.get('validated_receipts') == 832 and status.get('roster') == 832,
         'full 832 validated roster required; no partial reduction')
    need(status.get('running') == [] and status.get('failure') is None and ledger.get('failure') is None,
         'science active or failed')
    if ledger.get('persisted_receipts') != 832:
        verify_terminal_persistence(root, status, ledger, terminal_backup_path)
    actual = {p.relative_to(root).as_posix() for p in (root / 'results/science').glob('*/*.json.gz')}
    need(actual == roster_names(), 'exact receipt roster missing/extra')
    for name in ('results/FINAL_METRICS.json', 'results/FINAL_ANALYSIS_STORAGE_VERIFICATION.json'):
        need((root / name).is_file(), f'qualified final analysis absent: {name}')
    return status, ledger


def verify_analysis(root, qualification, locked):
    need(file_sha(qualification) == QUALIFICATION_SHA, 'not the approved storage qualification bytes')
    q = read_json(qualification)
    good = ('pass', 'test_only', 'byte_exact_final_metrics', 'byte_exact_stdout',
            'identical_scientific_call_order_arguments_and_returns', 'original_audits_enabled',
            'source_immutable', 'source_lock_immutable', 'qualification_code_identity_unchanged')
    need(q.get('schema') == 'MINIFLY-A3-BOUNDED-ANALYSIS-QUALIFICATION-v1' and
         all(q.get(k) is True for k in good) and q.get('numerical_tolerance_used') is None,
         'qualified storage exact-equivalence evidence absent')
    marker = read_json(root / 'results/FINAL_ANALYSIS_STORAGE_VERIFICATION.json')
    metrics = read_json(root / 'results/FINAL_METRICS.json')
    need(set(metrics) == {'schema', 'lock_digest', 'family_size_m', 'package_contrasts_available',
         'package_contrasts_unavailable', 't_critical', 'contrasts', 'diagnostics', 'absolute',
         'per_world', 'receipt_sha256'}, 'unexpected final analysis fields')
    need(marker.get('schema') == 'MINIFLY-A3-BOUNDED-ANALYSIS-ACCEPTANCE-v1' and marker.get('pass') is True,
         'analysis storage acceptance missing')
    need(marker['qualification_sha256'] == file_sha(qualification), 'qualification hash changed')
    need(marker['adapter_sha256'] == q['adapter_sha256'] == file_sha(root / 'ops/bounded_analysis.py'),
         'qualified adapter changed')
    need(q['test_sha256'] == file_sha(root / 'ops/test_bounded_analysis.py'), 'qualification test changed')
    need(marker['runtime'] == q['runtime'] and marker['runtime']['lock_digest'] == locked['lock_digest'] and
         marker['runtime']['source_lock_sha256'] == LOCK_SHA and
         marker['runtime']['analyzer_sha256'] == locked['files']['src/analyze_a.py'] and
         marker['runtime']['environment'] == locked['environment'], 'analysis runtime identity')
    need(marker['final_metrics_sha256'] == file_sha(root / 'results/FINAL_METRICS.json'),
         'unaccepted FINAL_METRICS bytes')
    storage = marker['storage']
    need(storage.get('registered') == 832 and storage.get('complete_rechecks', 0) >= 1 and
         storage.get('decoded_receipts_cached') == 0, 'incomplete full-analysis storage verification')
    names = {f'{a}/{w}' for a in ARMS for w in WORLDS}
    need(set(storage['receipt_sha256']) == names and storage['receipt_sha256'] == metrics['receipt_sha256'],
         'analysis acceptance receipt coverage')
    need(metrics.get('schema') == 'MINIFLY-A3-CLAUDE-FINAL-METRICS-v1' and
         metrics.get('lock_digest') == locked['lock_digest'] and metrics.get('family_size_m') == 26 and
         metrics.get('package_contrasts_available') == 12 and
         set(metrics.get('package_contrasts_unavailable', {})) == {'P3'} and
         set(metrics['package_contrasts_unavailable']['P3']) == {'dE1', 'dE3'}, 'final analysis identity/family/P3')
    for name, expected in metrics['receipt_sha256'].items():
        a, w = name.split('/')
        need(file_sha(root / receipt_name(a, int(w))) == expected, f'accepted receipt changed: {name}')
    return metrics, marker


def verify_replay(root, summary_path, locked, source, hashes):
    """Verify each immutable replay record and its exact input/source/context key."""
    summary = read_json(summary_path)
    need(summary.get('schema') == 'MINIFLY-A3-SCIENCE-REPLAY-RECOVERY-v1-SUMMARY' and
         summary.get('pass') is True and summary.get('sample_arms') == list(ARMS) and
         summary.get('sample_worlds') == [190001, 190002, 190003, 190004], 'complete prescribed replay summary required')
    ctx = summary['context']
    need(ctx['lock_digest'] == locked['lock_digest'] and ctx['locked_environment'] == locked['environment'] and
         ctx['receipt_source_sha256'] == source and
         ctx['wrapper_sha256'] == file_sha(root / 'ops/audit_science_replay.py') and
         ctx.get('scope') == REPLAY_SCOPE, 'replay source context/scope changed')
    records = summary['records']
    need(len(records) == 52 and {(r['arm'], r['world']) for r in records} ==
         {(a, w) for a in ARMS for w in range(190001, 190005)}, 'replay duplicate/missing sample')
    verified, mechanism_checks, elapsed, peak = {}, {}, 0., 0
    for entry in records:
        arm, world = entry['arm'], entry['world']
        path = safe_path(root, entry['record'])
        need(entry.get('pass') is True and file_sha(path) == entry['record_sha256'], 'replay record altered/failed')
        r = read_json(path)
        identity = {**ctx, 'arm': arm, 'world': world,
                    'receipt_sha256': {f'{arm}/{world}': hashes[f'{arm}/{world}']}}
        if arm in DEPENDENCIES:
            name = f'{DEPENDENCIES[arm]}/{world}'
            identity['receipt_sha256'][name] = hashes[name]
        need(r.get('schema') == 'MINIFLY-A3-SCIENCE-REPLAY-RECOVERY-v1' and r.get('pass') is True and
             r.get('error') is None and r.get('identity') == identity and
             r.get('key') == entry['key'] == sha(canonical(identity)), 'replay input/key identity')
        log, replay = r['checks']['independent_log_audit'], r['checks']['independent_replay']
        need(log.get('pass') is True and log.get('arm') == arm and log.get('world') == world and
             replay.get('arm') == arm, 'replay check identity')
        if arm.startswith('P'):
            need(replay.get('branch') == 'W' and replay.get('store') == 0 and
                 replay.get('p_updates_checked') == 600 and replay.get('pd_checkpoints_checked') == 2,
                 'incomplete P replay')
        elif arm.startswith('Z'):
            need(replay.get('branch') == 'W' and replay.get('store') == 0 and
                 type(replay.get('z_events_checked')) is int and 0 < replay['z_events_checked'] <= 600,
                 'incomplete Z replay')
        else:
            need(replay.get('novelty_checked') == 600, 'incomplete R replay')
            if arm in ('R0_signed', 'R3', 'R3_randtarget'):
                need(replay.get('shared_values', {}).get('records_checked') == 600 and
                     replay['shared_values'].get('arm') == arm, 'incomplete signed-R replay')
        verified[entry['record']] = entry['record_sha256']
        mechanism_checks[f'{arm}/{world}'] = replay
        need(type(r['resources'].get('elapsed_s')) in (int, float) and
             math.isfinite(r['resources']['elapsed_s']) and r['resources']['elapsed_s'] >= 0 and
             type(r['resources'].get('process_peak_rss_bytes')) is int and
             r['resources']['process_peak_rss_bytes'] > 0, 'invalid replay resources')
        elapsed += r['resources']['elapsed_s']
        peak = max(peak, r['resources']['process_peak_rss_bytes'])
    return {'pass': True, 'records_checked': 52, 'record_sha256': verified,
            'summary_sha256': file_sha(summary_path), 'mechanism_checks': mechanism_checks,
            'summed_record_elapsed_s': elapsed,
            'max_process_peak_rss_bytes': peak,
            'scope': 'W-branch mechanism reconstruction; Z/P store 0; R novelty and signed shared values; no probe replay'}


def effective_key(cue):
    return bytes(b for b in cue if b != 32)[-4:]


def forms(cue):
    a, b = cue[8], cue[10]
    return {'canonical': cue, 'spacing': b' ' * 7 + bytes((a, 32, 43, b, 61)),
            'prefix': b'Q' + b' ' * 7 + bytes((a, 43, b, 61)),
            'inner_marker': b' ' * 7 + bytes((a, 43, 126, b, 61))}


def expected_probes(fixture):
    """Independent construction; no import of historical _expected_probes."""
    for pair in (('old_end', 'old_day'), ('new_end', 'final')):
        for branch in BRANCHES:
            for stage in pair:
                for cohort in ('old', 'new'):
                    if cohort == 'new' and stage in ('old_end', 'old_day'):
                        continue
                    group = fixture[cohort + '_fact']
                    for item, (cue, label) in enumerate(zip(group['cues_hex'], group['labels'], strict=True)):
                        for form, shown in forms(bytes.fromhex(cue)).items():
                            if form != 'canonical' and stage != 'final':
                                continue
                            yield dict(stage=stage, branch=branch, set=cohort + '_fact', item=item,
                                       form=form, cue_hex=shown.hex(), target=48 + label)
                for cohort in ('old', 'new'):
                    if cohort == 'new' and stage in ('old_end', 'old_day'):
                        continue
                    for item, row in enumerate(fixture[cohort + '_relation']['taught']):
                        yield dict(stage=stage, branch=branch, set=cohort + '_relation_taught',
                                   item=item, cue_hex=row['cue_hex'], target=48 + row['label'])
                held = fixture['old_relation']['heldout']
                for item in range(3):
                    positive, negative = held[2 * item:2 * item + 2]
                    for order in (0, 1):
                        options = [positive['cue_hex'], negative['cue_hex']]
                        if order:
                            options.reverse()
                        yield dict(stage=stage, branch=branch, set='old_relation_heldout', item=item,
                                   order=order, option_hex=options, target=76 if order == 0 else 82)


def fixture_evidence(fixture):
    from audit_fixture import audit
    passed = audit(fixture)
    exposure = []
    taught = defaultdict(set)
    for row in fixture['records']:
        cue = bytes.fromhex(row['cue_hex'])
        taught[(row['stage'], row['domain'])].add(row['cue_hex'])
        exposure.append({**row, 'effective_content_key_hex': effective_key(cue).hex()})
    fact_audit = {}
    for cohort in ('old', 'new'):
        group = fixture[cohort + '_fact']
        table = dict(zip(group['cues_hex'], group['labels'], strict=True))
        keys = {effective_key(bytes.fromhex(cue)): label for cue, label in table.items()}
        for form in ('canonical', 'spacing', 'prefix', 'inner_marker'):
            exact_hits, key_hits, exact_correct, key_correct = 0, 0, 0, 0
            for cue, target in table.items():
                shown = forms(bytes.fromhex(cue))[form]
                exact_hits += shown.hex() in table
                key_hits += effective_key(shown) in keys
                # Fixed label 0 fallback; never tuned to observed model outputs.
                exact_correct += table.get(shown.hex(), 0) == target
                key_correct += keys.get(effective_key(shown), 0) == target
            fact_audit[cohort + '_' + form] = dict(n=16, exact_key_hits=exact_hits, content_key_hits=key_hits,
                exact_table_accuracy=exact_correct / 16, content_table_accuracy=key_correct / 16)
    held = fixture['old_relation']['heldout']
    trained = taught[('old', 'relation')]
    keys = {effective_key(bytes.fromhex(cue)) for cue in trained}
    need(not ({h['cue_hex'] for h in held} & trained), 'heldout exact-pair leakage')
    need(not ({effective_key(bytes.fromhex(h['cue_hex'])) for h in held} & keys), 'heldout effective-key leakage')
    symbol_frequency = Counter(b for cue in trained for b in bytes.fromhex(cue)[9:11])
    need(set(symbol_frequency.values()) == {4}, 'unequal old-symbol frequency')
    # Demonstrate a learned one-bit class rule, derived only from taught labels.
    learned_class = {}
    for row in fixture['old_relation']['taught']:
        first, second = bytes.fromhex(row['cue_hex'])[9:11]
        for symbol, label in ((first, row['label']), (second, 1 - row['label'])):
            need(symbol not in learned_class or learned_class[symbol] == label, 'class-rule inconsistent training')
            learned_class[symbol] = label
    correct = sum(learned_class[bytes.fromhex(row['cue_hex'])[9]] == row['label'] for row in held)
    return {**passed, 'exposure_records': exposure, 'fact_address_countermodels': fact_audit,
            'heldout': {'directions': held, 'exact_taught_pair_hits': 0, 'content_key_hits': 0,
                        'constant_option_L_accuracy': .5, 'constant_option_R_accuracy': .5,
                        'training_symbol_frequency': dict(sorted(symbol_frequency.items())),
                        'frequency_only_cannot_separate_classes': True,
                        'learned_first_symbol_class_accuracy': correct / 6,
                        'limit': 'Symbol-position/class rule can solve the assay; this is limited synthetic E3, not broad reasoning'}}


def values_emission(values, n):
    need(isinstance(values, list) and len(values) == 4 and all(isinstance(v, (int, float)) and
         math.isfinite(v) for v in values), 'nonfinite/invalid four-channel values')
    return 48 + max(range(n), key=lambda j: values[j])


def validate_receipt(r, arm, world, fixture, source, lock_digest, p_support, *, kind='science'):
    expected = {'schema': 'MINIFLY-A3-CLAUDE-RECEIPT-v1', 'arm': arm, 'world': world,
                'kind': kind, 'lock_digest': lock_digest, 'source_sha256': source,
                'fixture_digest': fixture['digest'], 'branches': list(BRANCHES),
                'canonical_B_sha256': CANONICAL_B, 'technical': kind != 'science'}
    for k, v in expected.items():
        need(r.get(k) == v, f'receipt identity/source/graph/branch field {k}: {arm}/{world}')
    births = r['births']
    banks = ('shared', 'private') if arm.startswith('R') else ('native',)
    need(len(births) == 4 * len(banks) and {(b['bank'], b['store']) for b in births} ==
         {(b, j) for b in banks for j in range(4)} and
         all(b['canonical_B_sha256'] == CANONICAL_B for b in births), 'wrong graph or separate birth roster')
    if arm.startswith('P'):
        need(r['params']['support_sha256'] == p_support, 'wrong graph support')
    need(len(r['first']) == 3000, 'first-response count')
    for index, row in enumerate(r['first']):
        record, branch = divmod(index, 5)
        spec = fixture['records'][record]
        need(row['record'] == record and row['branch'] == BRANCHES[branch], 'first-response order/branch/record')
        target_time = record * 165. + (86400. if record >= 336 else 0.) + 12 * (30. / 14.)
        need(abs(row['teacher_at'] - target_time) < 1e-9, 'first-response teacher timing')
        need(row['emitted'] == values_emission(row['values'], 4 if spec['domain'] == 'fact' else 2),
             'first-response emitted/value arithmetic')
    expected_rows = list(expected_probes(fixture))
    need(len(r['probes']) == len(expected_rows) == 1380, 'dropped/extra probe')
    bins = defaultdict(list)
    for index, (row, expected_row) in enumerate(zip(r['probes'], expected_rows, strict=True)):
        for k, v in expected_row.items():
            need(row.get(k) == v, f'probe target/roster field {k}, index {index}')
        if row['set'] == 'old_relation_heldout':
            option_values = row['option_values']
            need(isinstance(option_values, list) and len(option_values) == 2, 'relation option value count')
            for values in option_values:
                values_emission(values, 2)
            scores = [v[1] - v[0] for v in option_values]
            need(row['option_scores'] == scores, 'relation score arithmetic')
            emitted = 76 if scores[0] >= scores[1] else 82
        else:
            emitted = values_emission(row['values'], 4 if row['set'].endswith('_fact') else 2)
        need(row['emitted'] == emitted, 'tampered emitted answer')
        correct = int(emitted == expected_row['target'])
        need(type(row['correct']) is int and row['correct'] == correct, 'tampered correctness flag')
        bins[(row['stage'], row['branch'], row['set'], row.get('form'))].append(correct)
    return bins


def reduce_bins(bins):
    result = {}
    for label, selector in SELECTORS.items():
        values = bins.get(selector)
        need(values is not None and len(values) == (16 if 'fact' in selector[2] else
             12 if selector[2] == 'old_relation_taught' else 6), f'endpoint count {label}')
        result[label] = sum(values) / len(values)
    result['E1'] = result['old_fact_final_W'] - result['old_fact_final_N']
    result['E3'] = result['heldout_final_W'] - result['heldout_final_N_old_rel']
    return result


def interval(values, simultaneous=True):
    from scipy.stats import t
    n = len(values)
    need(n >= 2 and all(math.isfinite(v) for v in values), 'invalid interval sample')
    mean = math.fsum(values) / n
    sd = math.sqrt(math.fsum((v - mean) ** 2 for v in values) / (n - 1))
    if simultaneous:
        if sd == 0:
            half = 4 * math.sqrt(math.log(2 * M / ALPHA) / (2 * n))
            name = 'Hoeffding bounded (zero paired variance), Bonferroni m=26'
        else:
            half = float(t.ppf(1 - ALPHA / (2 * M), n - 1)) * sd / math.sqrt(n)
            name = 'two-sided Student-t, Bonferroni m=26'
        return dict(n_worlds=n, mean=mean, sd_world=sd, lower=mean-half, upper=mean+half, interval=name)
    half = float(t.ppf(.975, n - 1)) * sd / math.sqrt(n)
    return dict(mean=mean, lower95=mean-half, upper95=mean+half)


def compare(actual, expected, path='metrics'):
    """Recursive shape/exact-label comparison, with 2e-12 numeric arithmetic tolerance."""
    if isinstance(expected, dict):
        need(isinstance(actual, dict) and set(actual) == set(expected), f'{path}: key mismatch')
        for k, v in expected.items():
            compare(actual[k], v, path + '/' + str(k))
    elif isinstance(expected, list):
        need(isinstance(actual, list) and len(actual) == len(expected), f'{path}: sequence mismatch')
        for i, v in enumerate(expected):
            compare(actual[i], v, path + '/' + str(i))
    elif type(expected) in (int, float):
        need(type(actual) in (int, float) and math.isfinite(actual) and
             math.isclose(actual, expected, rel_tol=2e-12, abs_tol=2e-12), f'{path}: arithmetic mismatch')
    else:
        need(type(actual) is type(expected) and actual == expected, f'{path}: value mismatch')


def aggregate(per):
    contrasts = {}
    for candidate, control in PRIMARY.items():
        entry = {'control': control}
        for axis in ('E1', 'E3'):
            differences = [per[candidate][str(w)][axis] - per[control][str(w)][axis] for w in WORLDS]
            ci = interval(differences)
            entry['d' + axis] = ci
            entry['d' + axis + '_improvement_claim'] = ci['lower'] > 0
            entry['per_world_d' + axis] = differences
        contrasts[candidate] = entry
    diagnostics = {c: {'control': b, **{'d' + axis: interval(
        [per[c][str(w)][axis] - per[b][str(w)][axis] for w in WORLDS], False)
        for axis in ('E1', 'E3')}} for c, b in DIAGNOSTIC.items()}
    absolute = {a: {k: interval([per[a][str(w)][k] for w in WORLDS], False)
                    for k in (*SELECTORS, 'E1', 'E3')} for a in ARMS}
    return {'per_world': per, 'contrasts': contrasts, 'diagnostics': diagnostics, 'absolute': absolute}


def coverage(root, hashes, lock_digest):
    expected = {receipt_name(*name.split('/')): h for name, h in hashes.items()}
    channels, union = {}, {}
    definitions = (('github', 'BACKUP_STATUS.json', 'MINIFLY-A3-VERIFIED-BACKUP-v1'),
                   ('private_recoverable', 'PRIVATE_BACKUP_STATUS.json', 'MINIFLY-A3-VERIFIED-PRIVATE-BACKUP-v1'))
    for label, name, schema in definitions:
        path = root / 'results/recovery' / name
        if not path.exists():
            channels[label] = {'receipts': 0, 'status': 'no verified ACK present'}
            continue
        ack = read_json(path)
        need(ack.get('schema') == schema and ack.get('verified_at_utc') and ack.get('lock_digest') == lock_digest,
             'invalid off-host ACK')
        table = ack['receipt_sha256']
        for key, digest in table.items():
            need(expected.get(key) == digest, 'backup ACK does not match current receipt')
            need(key not in union or union[key] == digest, 'backup channels disagree')
            union[key] = digest
        channels[label] = {'receipts': len(table), 'ack_sha256': file_sha(path),
            'verified_at_utc': ack['verified_at_utc'], 'verification': ack.get('verification'),
            'commit': ack.get('commit'), 'library_file_id': ack.get('library_file_id'),
            'dependencies': ack.get('dependencies', []),
            'interpretation': 'Prior independently verified ACK rechecked against current bytes; no fresh remote request in finalizer'}
    need(len(union) == 832, 'all 832 receipts must have verified off-host recovery coverage')
    return {'channels': channels, 'combined_receipts': len(union), 'unprotected_primary_receipts': 832-len(union),
            'new_final_outputs_off_host': False,
            'final_delivery_limit': 'This finalizer creates a local ZIP only; its later private upload/readback must be verified separately'}


def resource_report(root, status, ledger, per, replay, backup, analysis_resource_path, audit_elapsed):
    budget = read_json(root / 'RESOURCE_BUDGET.json')
    adjustments = ledger.get('recovery_adjustments', [])
    wall_reserve = sum(x.get('reservation_wall_s', x.get('additional_wall_reservation_s', 0)) for x in adjustments)
    worker_reserve = sum(x.get('reservation_worker_s', x.get('additional_worker_reservation_s', 0)) for x in adjustments)
    evidence = read_json(analysis_resource_path) if analysis_resource_path else None
    if evidence is not None:
        need(isinstance(evidence, dict) and evidence.get('schema') == 'MINIFLY-A3-ANALYSIS-RESOURCES-v1' and
             type(evidence.get('wall_s')) in (int, float) and math.isfinite(evidence['wall_s']) and
             evidence['wall_s'] >= 0 and type(evidence.get('peak_rss_bytes')) is int and
             evidence['peak_rss_bytes'] > 0 and evidence.get('measurement_method'), 'invalid analysis resource evidence')
        need(evidence.get('final_metrics_sha256') == file_sha(root / 'results/FINAL_METRICS.json'),
             'analysis resource evidence must identify accepted metrics')
    per_arm = {}
    for arm in ARMS:
        rows = list(per[arm].values())
        per_arm[arm] = {'surviving_life_s_sum': math.fsum(r['life_s'] for r in rows),
                        'max_life_s': max(r['life_s'] for r in rows),
                        'max_worker_lifetime_peak_rss_bytes': max(r['peak_rss_bytes'] for r in rows),
                        'compressed_receipt_bytes': sum((root / receipt_name(arm, w)).stat().st_size for w in WORLDS)}
    return {'schema': 'MINIFLY-A3-FINAL-RESOURCE-REPORT-v1', 'budget': budget,
        'science_cumulative_charged': {'active_wall_s': max(status['active_wall_s'], ledger['active_wall_s']),
            'worker_s': max(status['worker_s'], ledger['worker_s']),
            'definition': 'Ledger dispatch-to-exit worker time, including lost/killed work and conservative interruption reservations; not CPU time'},
        'explicit_reservations_already_in_ledger': {'wall_s': wall_reserve, 'worker_s': worker_reserve,
            'note': 'Informational decomposition only; do not add again. Some reservations deliberately overlap measured time.',
            'events': adjustments},
        'cleared_infrastructure_stops': ledger.get('cleared', []), 'sessions': ledger.get('sessions', []),
        'operational_concurrency': {'original_frozen_workers': 4, 'approved_recovery_workers': 6,
            'amendment': 'ops/CONCURRENCY_AMENDMENT_20261002.md',
            'limit': 'Concurrency-only amendment; frozen wall, worker-time, per-job and disk ceilings unchanged'},
        'per_arm_surviving_receipts': per_arm,
        'host_loss': {'original_receipts_retained': 690, 'accepted_receipts_lost': 67,
            'same_seed_reconstructions_authorized': 67,
            'limit': 'Reconstructed receipts are not claimed byte-identical to unavailable original receipts; original lost compute remains charged'},
        'replay': {k: replay[k] for k in ('summed_record_elapsed_s', 'max_process_peak_rss_bytes', 'scope')},
        'qualified_analysis': evidence if evidence else {'status': 'measurement unavailable',
            'limit': 'No wall/RSS number is inferred from science counters or qualification runs'},
        'finalizer_audit_before_packaging': {'elapsed_s': audit_elapsed,
            'process_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)},
        'reporting_limits': ['Surviving receipt life seconds exclude lost/repeated work and must not replace ledger totals',
            'Receipt RSS is a worker-process lifetime high-water mark, not a summed concurrent peak',
            'Replay summed per-record elapsed is not parallel wall time; cached original resource records are not double-counted',
            'Final ZIP construction and later transfer costs are outside this pre-packaging snapshot',
            'No GPU or accelerated speedup claim'], 'off_host_recovery': backup}


def format_ci(ci, simultaneous=False):
    lo, hi = ('lower', 'upper') if simultaneous else ('lower95', 'upper95')
    return f"{ci['mean']:.6f} [{ci[lo]:.6f}, {ci[hi]:.6f}]"


def make_report(data, resources, audit):
    lines = ['# MiniFly Package A V3 final report', '',
        '## Scope and integrity',
        'All 832 frozen world-arm receipts and the prescribed 52 replay audits passed. '
        'This additive audit independently rebuilt the sealed probe roster, emitted-answer arithmetic, '
        'per-world scores and paired contrasts, then compared them with the qualified frozen analysis.',
        '64 worlds are the sampling units. All 26 program-wide primary comparisons remain in the family; '
        '12 Package-A comparisons are available and P3’s two remain unavailable. '
        'Package B V2 E3 is invalid and excluded from any joint claim.', '',
        '## Every assigned variant',
        'E0 below denotes technical/mechanism qualification only, not a separately scored efficacy endpoint. '
        'E1 is old exact-key retention; E2 is presentation sensitivity; E3 is limited synthetic held-out relation transfer. '
        'Separate successful architectures would not establish one integrated learner.', '']
    for arm in ARMS:
        absolute = data['absolute'][arm]
        role = 'candidate' if arm in PRIMARY else 'diagnostic outside primary family' if arm in DIAGNOSTIC.values() else 'matched control'
        lines += [f'### {arm}: instantiated; {role}',
            'E0: technical qualification, canonical-birth and literal causal-write audits passed; '
            'limited W-branch replay coverage is stated below.',
            'E1 absolute final W / N_old_fact (descriptive 95%): ' + format_ci(absolute['old_fact_final_W']) +
            ' / ' + format_ci(absolute['old_fact_final_N']),
            'E3 absolute final W / N_old_rel (descriptive 95%): ' + format_ci(absolute['heldout_final_W']) +
            ' / ' + format_ci(absolute['heldout_final_N_old_rel']),
            'E2 final W spacing / prefix / inner-marker: ' + ' / '.join(format_ci(absolute[k]) for k in
                ('old_fact_spacing_final_W', 'old_fact_prefix_final_W', 'old_fact_marker_final_W')),
            'Old taught-relation guard W / N_old_rel: ' + format_ci(absolute['old_relation_taught_final_W']) +
            ' / ' + format_ci(absolute['old_relation_taught_final_N_old_rel']),
            'New fact W / N_new_fact: ' + format_ci(absolute['new_fact_final_W']) + ' / ' +
            format_ci(absolute['new_fact_final_N_new_fact']),
            'New taught relation W / N_new_rel: ' + format_ci(absolute['new_relation_taught_final_W']) +
            ' / ' + format_ci(absolute['new_relation_taught_final_N_new_rel'])]
        if arm in PRIMARY:
            contrast = data['contrasts'][arm]
            lines.append('Predeclared matched control: ' + contrast['control'])
            for axis in ('dE1', 'dE3'):
                ci = contrast[axis]
                verdict = 'positive improvement evidence' if ci['lower'] > 0 else 'negative contrast evidence' if ci['upper'] < 0 else 'no positive improvement claim'
                lines.append(f"{axis}: {format_ci(ci, True)}; {ci['interval']}; {verdict}")
        lines += ['']
    lines += ['### P3 and P3_shuffle: NOT_INSTANTIATED',
        'P3 is technically unqualified/unresolved; the historical runtime-failure report is unverified and no receipt was recovered. '
        'P3_shuffle is unrun. E0–E3 evidence and P3’s two primary comparisons are unavailable; no replacement or family-size reduction.', '',
        '## Diagnostics and uncertainty',
        'Primary intervals are two-sided simultaneous 95% Student-t, df=63, Bonferroni m=26. '
        'Exactly zero paired variance uses the predeclared Hoeffding bound on [-2,2], without a zero-width victory interval. '
        'Absolute scores and diagnostic intervals are descriptive unadjusted 95% world-level Student-t intervals, '
        'not simultaneous claims; they can extend outside [0,1] and may be zero-width when every world agrees. '
        'Chance references are 0.25 for four-way facts and 0.5 for binary relation output. '
        'An E3 difference with W near chance does not establish useful output.']
    for candidate, entry in data['diagnostics'].items():
        lines.append(f"{candidate} versus {entry['control']}: dE1 {format_ci(entry['dE1'])}; dE3 {format_ci(entry['dE3'])}")
    lines += ['', '## Exposure, countermodels and audit limits',
        'FIXTURE_EXPOSURE_AUDIT.json records every teaching cue, answer, index, cohort and last-four-nonspace-byte effective address '
        'for all 64 sealed fixtures. Each world has 600 records: old fact 192, old relation 144, new fact 192, new relation 72. '
        'Every taught item appears 12 times. Fact labels and relation choices are balanced.',
        'Canonical old facts are explainable by exact-key recall. Spacing and prefix preserve the CONTENT key; '
        'inner-marker changes it. No E2 form is E3. Exact/raw and effective-key tables have no held-out relation entries. '
        'Always-left/right baselines score 0.5; training symbol frequencies are balanced, but a class rule learned from '
        'symbol position and labels solves the held-out relation task. This is an explicit assay limitation.',
        'All receipts: independent raw answer/probe arithmetic plus frozen literal per-record write and aggregate causal gates. '
        'Source/receipt hashes, paired-yoke hashes and first-response timing/identity were verified. '
        'Read-only clone isolation and native finite-state checks rely on locked technical and runtime assertions. '
        'The 52 science replays reconstruct W-branch mechanisms only: Z/P store 0, R novelty, and signed-R shared values. '
        'They are not all-store/all-branch state reconstruction or full probe replay. '
        'Consistent malicious changes to un-replayed values cannot be ruled out by log arithmetic alone; immutable input hashes '
        'and the provenance chain bound this limitation.',
        'PER_ITEM_FAILURES.jsonl.gz contains every failed final-checkpoint probe for every branch/arm/world, with target, '
        'emitted answer and item/form/order identity. Other checkpoints remain in the compact raw receipts. '
        'FINAL_AUDIT.json records correct/total counts by endpoint and all source/input hashes.', '',
        '## Resources and recovery',
        f"Cumulative science charge: {resources['science_cumulative_charged']['active_wall_s']/3600:.3f} active-wall hours; "
        f"{resources['science_cumulative_charged']['worker_s']/3600:.3f} worker-hours. "
        'Worker-hours are dispatch-to-exit wall time including conservative interruption reservations, not CPU hours.',
        'The frozen resource budget retains its original four-worker plan. Recovery used the separately approved six-worker '
        'concurrency amendment; the wall, worker-time, per-job and disk ceilings were unchanged.',
        '690 original receipts survived the host loss; 67 previously accepted receipts were lost and explicitly authorized '
        'for same-seed reconstruction under the unchanged lock. Their new bytes are not claimed identical to unavailable originals. '
        'Prior computation and interruption charges remain in the ledger. RESOURCE_REPORT.json retains every available reservation and stop record.',
        f"Verified prior recovery ACK coverage: GitHub {resources['off_host_recovery']['channels']['github']['receipts']}; "
        f"private recoverable {resources['off_host_recovery']['channels']['private_recoverable']['receipts']}; "
        f"combined {resources['off_host_recovery']['combined_receipts']}/832. These are distinct channels, not extra independent science receipts.",
        f"Persistence reconciliation: the ledger cache reports {resources['terminal_persistence']['ledger_cached_persisted_receipts']}; "
        f"the authoritative verified terminal evidence covers {resources['terminal_persistence']['authoritative_persisted_receipts']}/832. "
        'Frozen Driver.finish does not refresh the persistence cache. The ledger was preserved unchanged; '
        'FINAL_AUDIT.json records the exact journal, receipt roster and independently read-back backup checks.',
        'The return ZIP is local at creation. Final artifacts and ZIP require separately verified private upload/readback; '
        'primary receipt backup coverage does not prove the final deliverables are off host.', '',
        '## Return bundle',
        'BUNDLE_MANIFEST.json hashes every other ZIP member. It includes the frozen spec/source lock, complete candidate/control '
        'source and tests, technical qualification, all 832 compact science receipts, replay records, qualified metrics/acceptance, '
        'this report, independent metrics, per-item failures, recovery/resource evidence and operational finalizer/tests. '
        'The manifest itself is hashed in the separate FINALIZATION_COMMIT.json together with the ZIP hash.']
    return '\n'.join(lines) + '\n'


def verify_bundle(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        need(len(names) == len(set(names)), 'duplicate ZIP member')
        manifest_raw = archive.read('BUNDLE_MANIFEST.json')
        manifest = json.loads(manifest_raw)
        need(set(names) == set(manifest['files']) | {'BUNDLE_MANIFEST.json'}, 'manifest ZIP coverage')
        for name, meta in manifest['files'].items():
            safe_path(Path('/virtual-bundle'), name)
            h = hashlib.sha256()
            size = 0
            with archive.open(name) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    h.update(block); size += len(block)
            need(h.hexdigest() == meta['sha256'] and size == meta['bytes'], f'ZIP hash mismatch: {name}')
    return {'files': len(manifest['files']), 'manifest_sha256': sha(manifest_raw), 'bundle_sha256': file_sha(path)}


def build_bundle(path, files, lock_digest):
    need(not path.exists(), 'refusing to overwrite return bundle')
    manifest = {'schema': 'MINIFLY-A3-RETURN-BUNDLE-MANIFEST-v1', 'lock_digest': lock_digest,
                'files': {name: {'sha256': file_sha(p), 'bytes': p.stat().st_size} for name, p in sorted(files.items())},
                'self_exclusion': 'Manifest hashes all other members; its hash is in the separate finalization commit'}
    with zipfile.ZipFile(path, 'x', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for name, file in sorted(files.items()):
            safe_path(Path('/virtual-bundle'), name)
            archive.write(file, name)
        archive.writestr('BUNDLE_MANIFEST.json', json.dumps(manifest, sort_keys=True, indent=1) + '\n')
    return verify_bundle(path)


def recheck_inputs(pinned):
    for path, expected in pinned.items():
        need(file_sha(path) == expected, 'admitted input changed during finalization: ' + str(path))


def finalize(root, replay_summary, qualification, bundle_path, analysis_resource_path=None, terminal_backup_path=None):
    root = root.resolve()
    bundle_path = bundle_path.resolve()
    need(not bundle_path.exists() and not bundle_path.is_relative_to(root / 'results'),
         'return ZIP must be new and outside results to avoid the science disk budget')
    need(not any((root / 'results' / n).exists() for n in OUTPUTS), 'final outputs already exist; no overwrite')
    started = time.monotonic()
    # Admission locks coordinate with both restart supervision and qualified analysis.
    with contextlib.ExitStack() as stack:
        for name in ('scratch/supervision/supervisor.lock', 'scratch/bounded_analysis_runs/FULL_ANALYSIS.lock'):
            path = root / name
            need(path.exists(), f'existing admission lock missing: {name}')
            handle = stack.enter_context(path.open('a+b'))
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise FinalAuditError('science supervisor or analysis still active') from exc
        # Pin mutable admission evidence before interpreting it, never after scoring.
        admitted_inputs = {root / name: file_sha(root / name) for name in
            ('results/science/RUN_STATUS.json', 'results/science/RUN_LEDGER.json')}
        status, ledger = preflight_roster(root, terminal_backup_path)
        for path in [root / 'SOURCE_LOCK.json', qualification, replay_summary,
                     root / 'results/FINAL_METRICS.json',
                     root / 'results/FINAL_ANALYSIS_STORAGE_VERIFICATION.json',
                     root / 'ops/bounded_analysis.py', root / 'ops/test_bounded_analysis.py',
                     root / 'ops/audit_science_replay.py', Path(__file__),
                     root / 'ops/test_finalize_a_v3.py', *([analysis_resource_path] if analysis_resource_path else [])]:
            admitted_inputs[path] = file_sha(path)
        for name in ('BACKUP_STATUS.json', 'PRIVATE_BACKUP_STATUS.json'):
            path = root / 'results/recovery' / name
            if path.exists():
                admitted_inputs[path] = file_sha(path)
        locked, source = verify_source(root)
        admitted_inputs.update({root / n: h for n, h in locked['files'].items()})
        need(status['lock_digest'] == ledger['lock_digest'] == locked['lock_digest'], 'run lock identity')
        metrics, marker = verify_analysis(root, qualification, locked)
        replay = verify_replay(root, replay_summary, locked, source, metrics['receipt_sha256'])
        backup = coverage(root, metrics['receipt_sha256'], locked['lock_digest'])
        persistence = {'authoritative_persisted_receipts': 832,
                       'ledger_cached_persisted_receipts': ledger['persisted_receipts'], 'cache_is_stale': False}
        if ledger['persisted_receipts'] != 832 or (root / 'results/recovery/LOCAL_STATUS.json').exists():
            persistence, journal_inputs = verify_terminal_persistence(root, status, ledger, terminal_backup_path)
            admitted_inputs.update(journal_inputs)
        admitted_inputs.update({root / receipt_name(*n.split('/')): h
                                for n, h in metrics['receipt_sha256'].items()})
        admitted_inputs.update({root / n: h for n, h in replay['record_sha256'].items()})
        recheck_inputs(admitted_inputs)
        # All gates passed; only now decode and independently score the complete roster.
        import numpy as np
        from scipy.stats import t
        import audit_a
        from fixture import make_world
        with np.load(root / 'package/FULL151_CANONICAL_B.npz') as anchor:
            p_support = [sha(np.ascontiguousarray(anchor[k]).tobytes()) for k in ('indices', 'indptr')]
        cal = read_json(root / 'results/calibration/R_CALIBRATION.json')
        kw = {'theta': cal['theta'], 'scales': (cal['scales']['shared'], cal['scales']['private'])}
        stage = root / 'scratch/finalization' / uuid.uuid4().hex
        stage.mkdir(parents=True, exist_ok=False)
        per = {a: {} for a in ARMS}
        log_audits, counts, fixtures, resources = {}, {}, {}, {}
        receipt_hashes = metrics['receipt_sha256']
        def load_receipt(arm, world):
            raw = (root / receipt_name(arm, world)).read_bytes()
            need(sha(raw) == receipt_hashes[f'{arm}/{world}'], 'receipt mutated during final audit')
            return json.loads(gzip.decompress(raw))
        with gzip.open(stage / 'PER_ITEM_FAILURES.jsonl.gz', 'wt', encoding='utf-8') as failures:
            for world in WORLDS:
                fixture = make_world(world)
                fixtures[str(world)] = fixture_evidence(fixture)
                for arm in ARMS:
                    receipt = load_receipt(arm, world)
                    bins = validate_receipt(receipt, arm, world, fixture, source, locked['lock_digest'], p_support)
                    if arm.startswith('Z') and world in range(190001, 190005):
                        checked = replay['mechanism_checks'][f'{arm}/{world}']['z_events_checked']
                        expected_count = sum(row[0] == 'Z' and row[2] == 0
                                             for row in receipt['mechanism_events']['W'])
                        need(checked == expected_count, 'incomplete Z replay against accepted receipt')
                    extra = {}
                    if arm in DEPENDENCIES:
                        dep = DEPENDENCIES[arm]
                        need(receipt['params'][f'paired_{dep}_receipt_sha256'] == receipt_hashes[f'{dep}/{world}'],
                             'paired yoke receipt hash changed')
                        extra['r1_receipt' if arm == 'R1_rand' else 'z2_receipt'] = load_receipt(dep, world)
                    log_audits[f'{arm}/{world}'] = audit_a.audit_receipt(receipt, **kw, **extra)
                    need(log_audits[f'{arm}/{world}'].get('pass') is True, 'frozen receipt audit failed')
                    need(type(receipt['resources'].get('life_s')) in (int, float) and
                         math.isfinite(receipt['resources']['life_s']) and receipt['resources']['life_s'] >= 0 and
                         type(receipt['resources'].get('peak_rss_bytes')) is int and
                         receipt['resources']['peak_rss_bytes'] > 0, 'invalid receipt resources')
                    per[arm][str(world)] = {**reduce_bins(bins), 'life_s': receipt['resources']['life_s'],
                                           'peak_rss_bytes': receipt['resources']['peak_rss_bytes']}
                    for selector, values in bins.items():
                        if selector[0] == 'final':
                            key = '/'.join(str(v) for v in (arm, *selector))
                            count = counts.setdefault(key, {'correct': 0, 'total': 0})
                            count['correct'] += sum(values); count['total'] += len(values)
                    for row in receipt['probes']:
                        if row['stage'] == 'final' and row['emitted'] != row['target']:
                            identity = {k: v for k, v in row.items() if k not in ('values', 'option_values', 'option_scores')}
                            failures.write(json.dumps({'arm': arm, 'world': world, **identity}, sort_keys=True) + '\n')
        independent = aggregate(per)
        for key in ('per_world', 'contrasts', 'diagnostics', 'absolute'):
            compare(metrics[key], independent[key], key)
        compare(metrics['t_critical'], float(t.ppf(1-ALPHA/(2*M), 63)), 't_critical')
        for values in metrics['package_contrasts_unavailable']['P3'].values():
            need(values == 'NOT_INSTANTIATED (technically unqualified/unresolved; no replacement)',
                 'P3 improperly available')
        audit = {'schema': 'MINIFLY-A3-INDEPENDENT-FINAL-AUDIT-v1', 'pass': True,
            'lock_digest': locked['lock_digest'], 'source_lock_sha256': LOCK_SHA,
            'locked_file_sha256': locked['files'], 'receipt_sha256': receipt_hashes,
            'independent_score_arithmetic': {'pass': True, 'receipts': 832, 'worlds': list(WORLDS),
                'all_probe_rows_checked': 832 * 1380, 'analyzer_imported': False,
                'numeric_compare_tolerance': {'relative': 2e-12, 'absolute': 2e-12},
                'score_method': 'Emissions re-derived from logged values, sealed target re-derived, then counted; never trusts correct flags'},
            'frozen_literal_and_aggregate_gate_audits': log_audits, 'science_replay': replay,
            'terminal_persistence': persistence,
            'final_probe_counts': counts, 'qualified_analysis_acceptance': marker,
            'technical_audit_sha256': file_sha(root / 'results/technical_final/TECHNICAL_AUDIT.json'),
            'fixture_scope': 'Independent audit of frozen generator output, exposure/key/label checks; not a separately invented generator',
            'unavailable': {'P3': 'NOT_INSTANTIATED; two comparisons unavailable', 'P3_shuffle': 'NOT_INSTANTIATED; unrun'},
            'family_size_m': 26, 'package_B_V2_E3': 'INVALID; excluded from combined claims',
            'limits': replay['scope'], 'finalizer_sha256': file_sha(Path(__file__)),
            'test_sha256': file_sha(root / 'ops/test_finalize_a_v3.py')}
        resources = resource_report(root, status, ledger, per, replay, backup, analysis_resource_path, time.monotonic()-started)
        resources['terminal_persistence'] = persistence
        write_json(stage / 'INDEPENDENT_METRICS.json', independent)
        write_json(stage / 'FIXTURE_EXPOSURE_AUDIT.json', fixtures)
        write_json(stage / 'RESOURCE_REPORT.json', resources)
        write_json(stage / 'FINAL_AUDIT.json', audit)
        (stage / 'REPORT.md').write_text(make_report(independent, resources, audit))
        files = {name: root / name for name in locked['files']}
        for name in roster_names():
            files[name] = root / name
        for directory in ('ops', 'results/recovery/host_loss_20261001', 'results/recovery/interruptions',
                          'results/recovery/concurrency_transition', 'results/recovery/local_queue'):
            for p in (root / directory).rglob('*'):
                if p.is_file() and p.suffix in ('.py', '.md', '.json') and '__pycache__' not in p.parts:
                    files[p.relative_to(root).as_posix()] = p
        for name in ('SOURCE_LOCK.json', 'results/FINAL_METRICS.json', 'results/FINAL_ANALYSIS_STORAGE_VERIFICATION.json',
                     'results/science/RUN_STATUS.json', 'results/science/RUN_LEDGER.json',
                     'results/recovery/BACKUP_STATUS.json', 'results/recovery/PRIVATE_BACKUP_STATUS.json',
                     'results/recovery/LOCAL_STATUS.json'):
            if (root / name).exists():
                files[name] = root / name
        files['results/recovery/science_replay/FINAL_SUMMARY.json'] = replay_summary
        files.update({name: root / name for name in replay['record_sha256']})
        for name in OUTPUTS[:-1]:
            files['results/' + name] = stage / name
        files['TECHNICAL_AUDIT.json'] = root / 'results/technical_final/TECHNICAL_AUDIT.json'
        if analysis_resource_path:
            files['results/ANALYSIS_RESOURCE_EVIDENCE.json'] = analysis_resource_path
        if 'private_readback_path' in persistence:
            files['results/recovery/FINAL_PRIMARY_PRIVATE_READBACK.zip'] = Path(persistence['private_readback_path'])
        recheck_inputs(admitted_inputs)
        pinned_inputs = {n: file_sha(p) for n, p in files.items() if not p.is_relative_to(stage)}
        temporary_zip = stage / 'RETURN_BUNDLE.zip'
        bundle_info = build_bundle(temporary_zip, files, locked['lock_digest'])
        for name, digest in pinned_inputs.items():
            need(file_sha(files[name]) == digest, 'input changed during bundle: ' + name)
        need(verify_source(root)[0] == locked, 'source changed during finalization')
        preflight_roster(root, terminal_backup_path)
        recheck_inputs(admitted_inputs)
        # Publish only after full bundle readback and all source/input rechecks.
        bundle_path.parent.mkdir(parents=True, exist_ok=True)
        os.link(temporary_zip, bundle_path)
        for name in OUTPUTS[:-1]:
            os.link(stage / name, root / 'results' / name)
        commit = {'schema': 'MINIFLY-A3-FINALIZATION-COMMIT-v1', 'pass': True,
            'created_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            **bundle_info, 'bundle_path': str(bundle_path), 'bundle_bytes': bundle_path.stat().st_size,
            'final_outputs_sha256': {n: file_sha(stage / n) for n in OUTPUTS[:-1]},
            'elapsed_including_packaging_s': time.monotonic()-started,
            'local_only': True, 'off_host_final_bundle_verified': False}
        write_json(stage / 'FINALIZATION_COMMIT.json', commit)
        os.link(stage / 'FINALIZATION_COMMIT.json', root / 'results/FINALIZATION_COMMIT.json')
        return commit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--finalize', action='store_true', help='Explicit full-roster finalization only')
    parser.add_argument('--replay-summary', type=Path, required=True)
    parser.add_argument('--qualification', type=Path, default=ROOT / 'ops/BOUNDED_ANALYSIS_QUALIFICATION_20261002_APPROVED.json')
    parser.add_argument('--analysis-resources', type=Path)
    parser.add_argument('--terminal-backup', type=Path, help='Independently materialized 832-primary private backup ZIP')
    parser.add_argument('--bundle', type=Path, required=True)
    args = parser.parse_args()
    if not args.finalize:
        parser.error('--finalize is required; importing or testing never runs finalization')
    print(json.dumps(finalize(ROOT, args.replay_summary.resolve(), args.qualification.resolve(),
                             args.bundle, args.analysis_resources, args.terminal_backup), sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
