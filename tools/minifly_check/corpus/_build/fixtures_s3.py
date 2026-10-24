"""Supplementary fixtures S46-S50 (integration round 2, beyond the SPEC 12.2 table).

The round-2 verifier found two input-tampering paths the corpus did not cover (SPEC 3.2 cycle cap; SPEC 3.3 /
8 / 9.9 claim of an official INVALID). Each fixture adds one documented defect on the base delivery; nothing in
an existing fixture changes. Same base and conventions as fixtures_s.py and fixtures_s2.py.
"""
import copy

import base_artifacts as BA
import base_files as BF
import fixtures_d as FD
import mutate as M
from base_timeline import build_base
from mrk_util import dump_jsonl

D = BF.D
P = D + '/'
CYCLES = P + 'runlog/CYCLES.jsonl'
# the stages whose base commits write runlog/CYCLES.jsonl (cycle close rows 1-4)
CYCLE_LABELS = ('testbed', 'roster', 'dev_close', 'rehearsal')


def b_s46():
    """The committed runlog/CYCLES.jsonl is deleted in the worktree while every attempt stays (probe P4)."""
    tl, b = build_base()
    tl.worktree[CYCLES] = None
    tl.dirty.append(CYCLES)
    return tl, b


def b_s47():
    """No cycle is ever closed: the CYCLES rows are never written, while testbed, technical, dev, rehearsal and
    official attempts are all recorded."""
    return build_base({'drop_files': {lab: [CYCLES] for lab in CYCLE_LABELS}})


def b_s48():
    """T1 dev-only delivery (C00e) that never closes a cycle: unlimited dev attempts would bypass the cap."""
    return FD.b_c00e(drop_files={lab: [CYCLES] for lab in CYCLE_LABELS})


def b_s49():
    """A committed CYCLES row is rewritten in a later commit (the reason of the dev close)."""
    def edit(b):
        rows = copy.deepcopy(b.cycles)
        rows[2]['reason'] = 'Dev cycle closed; numbers looked fine.'
        b.commit('cycles_edit', {CYCLES: dump_jsonl(rows)}, utc='20261103T130000Z')
    return build_base({'after': {'report': [edit]}})


INVALID_HEADLINE = ('INVALID | earned NONE (attempted E3) | held exact own-EOS 85.2% [S2+ determined, n=128] vs '
                    'best shortcut DIGIT_NO_CARRY 50.0% | G 28.9pp | OLD final 96.9% (drop 3.1pp) | 8 MAIN worlds '
                    '| best of k=1')


def b_s50():
    """The control is not delivered during the official run only: every official-world NPT receipt carries W's
    writes and channel digests (technical, dev and rehearsal receipts stay valid; OUTPUTS rows are unchanged, so
    INERT-01 stays quiet). decide gives INVALID TREATMENT_NOT_DELIVERED on the 8 official MAIN worlds; the
    executor ran `claim`, filled findings_none_reason and committed the INVALID CLAIM with REPORT.md, and
    publishes the honest INVALID."""
    def rec(r, label):
        if r['arm'] == 'NPT' and r['world_role'] == 'official':
            w = BA.receipt(r['world'], r['world_role'], 'W', r['attempt_id'], r['native_closure_sha'],
                           r['params_sha256'], r['endpoint_lock_sha256'])
            for k in ('writes_by_channel', 'channel_digests', 'state_digest_end'):
                r[k] = copy.deepcopy(w[k])
        return r
    def claim(b):
        return M.claim_doc(b, primary_gate='INVALID', level_earned='NONE', invalid_reason='TREATMENT_NOT_DELIVERED',
                           stop_reason='INVALID')
    return build_base({'receipt_patch': rec, 'claim_doc': claim, 'headline': INVALID_HEADLINE})
