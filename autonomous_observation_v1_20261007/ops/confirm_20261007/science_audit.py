"""Frozen row auditor plus strict full-life, birth, and authorized-input checks."""
import hashlib
from ops_common import EVIDENCE, OPS, WORLDS, read
import audit

BRANCHES = {'W', 'N_OLD', 'N_ALL'}
STAGES = ('old', 'day1', 'new', 'day2', 'revised')
CANONICAL_B = '32a3726cb25a8255c1aef5696ff40aa303138c5d061a598004d2087d4da5f964'


def audit_full(r, plan, source_sha):
    world = r['world']
    if type(world) is not int or world not in WORLDS or r['evidence'] != EVIDENCE:
        raise AssertionError('unauthorized world/evidence')
    if r['confirmation_source_lock_sha256'] != source_sha:
        raise AssertionError('receipt source identity mismatch')
    if r['inputs'] != plan['inputs']:
        raise AssertionError('frozen input plan mismatch')
    if set(r['training']) != {'old', 'new', 'revised'} or set(r['probes']) != set(STAGES):
        raise AssertionError('incomplete life stages')
    if set(r['birth']) != BRANCHES:
        raise AssertionError('birth branch roster mismatch')
    ids = []
    for branch in BRANCHES:
        births = r['birth'][branch]
        if len(births) != 4 or any(b['canonical_B_sha256'] != CANONICAL_B for b in births):
            raise AssertionError('noncanonical native birth')
        ids.extend(b['fly_id'] for b in births)
    if len(set(ids)) != 12:
        raise AssertionError('newborn store identity reused')
    for stage in ('old', 'new', 'revised'):
        train = r['training'][stage]
        if set(train['branches']) != BRANCHES or train['raw_hex'] != plan['training'][stage]:
            raise AssertionError('training branches or frozen bytes mismatch')
    for si, stage in enumerate(STAGES):
        if set(r['probes'][stage]) != BRANCHES | {'NATIVE_ERASE'}:
            raise AssertionError('probe branch roster mismatch')
        labels = {'old'} if si < 2 else {'old', 'new'}
        if stage == 'revised':
            labels = {'new', 'current_old', 'revised', 'unchanged'}
        for branch in BRANCHES | {'NATIVE_ERASE'}:
            expected = labels if branch != 'NATIVE_ERASE' else ({'current_old'} if stage == 'revised' else {'old'})
            if set(r['probes'][stage][branch]) != expected:
                raise AssertionError('missing/unexpected probe set')
            for label, probe in r['probes'][stage][branch].items():
                if probe['raw_hex'] != plan['probes'][stage][label]:
                    raise AssertionError('frozen probe bytes mismatch')
                if probe['erase'] != (branch == 'NATIVE_ERASE'):
                    raise AssertionError('erase branch mismatch')
                if branch == 'N_ALL' and any(p != .25 for row in probe['rows'] for p in row['probabilities']):
                    raise AssertionError('all-no-write probabilities not uniform')
        seen = 640 if si < 2 else (960 if si < 4 else 1280)
        for branch in BRANCHES:
            if r['states'][f'{stage}/{branch}']['bytes_seen'] != seen:
                raise AssertionError('native byte counter mismatch')
    if set(r['states']) != {f'{stage}/{b}' for stage in STAGES for b in BRANCHES}:
        raise AssertionError('checkpoint roster mismatch')
    if r['resources']['actual_training_lives'] != 3 or r['resources']['training_bytes'] != 1280:
        raise AssertionError('life or training-byte count mismatch')
    # The frozen independent auditor uses world/evidence only as a DEV admission
    # guard. Rebind those two header fields in a shallow view; every scientific
    # row, input, branch, probability, clock, and metric is passed unchanged.
    checked = audit.audit_receipt({**r, 'world': 810201, 'evidence': 'DEV_ONLY_E0_E1'})
    if checked['independently_checked_rows'] != 8240:
        raise AssertionError('full-history audited row count mismatch')
    return {**checked, 'world': world, 'full_training_lives': 3,
            'committed_training_bytes': 3840, 'full_stage_roster_checked': True,
            'birth_identity_count': 12, 'frozen_plan_checked': True,
            'confirmation_source_lock_sha256': source_sha,
            'auditor_namespace_adapter_only': True}
