"""Supplementary fixtures S26-S43 (integration round 1, beyond the SPEC 12.2 table).

SPEC 20.2: every Section 7.4 check id is either implemented and exercised by the corpus or registered as
planned. `selftest --mutate` showed implemented ids that no fixture exercised (an always-PASS replacement
noticed by no dirty fixture) or constrained (an always-FAIL replacement noticed by no clean fixture). Each
fixture below adds one documented, realistic defect (or one clean configuration) for such ids; nothing in
an existing fixture changes. Same base and conventions as fixtures_s.py.
"""
import copy
import json

import base_artifacts as BA
import base_files as BF
import fixtures_c as C
import fixtures_d as FD
import fixtures_g as G
import fixtures_s as S
import mutate as M
from base_timeline import FAMILY_ID, build_base
from fixtures_c import _set
from mrk_util import Timeline, dump_compact, dump_json, dump_jsonl, sha256_bytes

D = BF.D
P = D + '/'
KIT = 'tools/minifly_check/'
LEG = 'minifly_corpus_legacy_dev_20261001'          # a legacy directory name (no _vN_): never a v1 delivery
LEG_POINTER = 'PROJECT_DOCUMENTS/CURRENT_ADDITION_LEGACY_POINTER.json'
LEG_RETRO = 'PROJECT_DOCUMENTS/KB/RETRO/RETRO_ANNOTATION_%s.json' % LEG


def b_s26():
    """paired_lcb without alpha: the rule is undecidable, so no rule qualification can hold."""
    return S.lock_only(lock_patch=_set('primary.aggregator', {'type': 'paired_lcb'}))


def b_s27():
    """T1 dev delivery (C00e) whose ADMISSION cites a parent admission although lineage.parent is null."""
    def adm(a):
        a['parent_admission_sha'] = 'f' * 64
    return FD.b_c00e(admission_patch=adm)


def b_s28():
    """Committed CLAIM.json says level_earned E4 (one delivery never earns E4); publish without re-claiming."""
    def claim(b):
        return M.claim_doc(b, level_earned='E4')
    return build_base({'claim_doc': claim})


def b_s29():
    """A results/official file added after the official run (and its lock) that the run never produced."""
    def posthoc(b):
        b.commit('posthoc', {P + 'results/official/EXTRA_ANALYSIS.json':
                             dump_json({'note': 'reanalysis written after the official run'})},
                 utc='20261103T130000Z')
    return build_base({'after': {'report': [posthoc]}})


def b_s30():
    """A committed GATE report whose checker_sha256 is not the checker committed at its git_head."""
    def foreign(b):
        g = BA.gate_report('publish', False, '20261103T125000Z', b.tl.head_sha(), b.lock_sha, b.sl['sha'],
                           'gate', FAMILY_ID)
        g['checker_sha256'] = 'f' * 64
        b.commit('foreign_report', {P + 'checks/GATE_publish_20261103T125000Z.json': dump_compact(g)},
                 utc='20261103T125000Z')
    return build_base({'after': {'report': [foreign]}})


def b_s31():
    """CONSTRAINTS.json without tier_thresholds: the tier cannot be computed."""
    c = copy.deepcopy(BF.CONSTRAINTS)
    del c['tier_thresholds']
    return S.lock_only(kb={'constraints': c})


def _finding(b, delivery_note):
    return {'id': 'F-20261102-codex-1', 'kind': 'finding', 'refers_to': None,
            'claim': 'Pair-rectangle units compose to held single-digit sums (%s).' % delivery_note, 'level': 'E3',
            'conditions': {'substrate': 'corpus stub', 'input_code': 'raw bytes', 'priors_supplied': 'none',
                           'feedback': 'gradual_warmup', 'dose': 'full', 'comparator': 'NPT',
                           'best_shortcut': 'DIGIT_NO_CARRY', 'worlds': '2 DEV', 'metric_definition': 'exact_own_eos'},
            'effect': {'value': 0.281, 'ci': [0.25, 0.31], 'unit': 'abs'},
            'source': D + '/REPORT.md:1@' + b.tl.head_sha()[:8], 'status': 'proposed', 'delivery': D,
            'author_agent': 'codex', 'utc': '20261102T121000Z', 'caveat': None}


def b_s32():
    """T1 DEV delivery (C00e): codex proposes a finding and claude (the other family) activates it."""
    def findings(b):
        f1 = _finding(b, 'dev only')
        f2 = {'id': 'F-20261102-claude-1', 'kind': 'activation', 'refers_to': f1['id'], 'claim': 'activated',
              'level': 'E3', 'conditions': f1['conditions'], 'effect': f1['effect'], 'source': f1['source'],
              'status': 'active', 'delivery': D, 'author_agent': 'claude:s-corpus-review',
              'utc': '20261102T121500Z', 'caveat': None, 'evidence': D + '/REPORT.md:1'}
        b.commit('finding', {BF.KB + 'FINDINGS.jsonl': dump_jsonl(list(BF.FINDINGS_BASE) + [f1])},
                 utc='20261102T121000Z')
        b.commit('activate', {BF.KB + 'FINDINGS.jsonl': dump_jsonl(list(BF.FINDINGS_BASE) + [f1, f2])},
                 who='claude', utc='20261102T121500Z')
    return FD.b_c00e(after={'report': [findings]})


def b_s33():
    """Governance range on the miniature kit while a kit file is modified in the worktree."""
    tl = G._tl_base()
    G._add(tl, 'g_skill_text', '20261105T100000Z', 'codex',
           {'.claude/skills/minifly-e3-design/SKILL.md': G.skill_md('minifly-e3-design') + b'\nOne more line.\n'})
    tl.worktree[KIT + 'lockcheck.py'] = G.LOCKCHECK.encode('utf-8') + b'\n# local edit, not committed\n'
    tl.dirty.append(KIT + 'lockcheck.py')
    return tl, None


def _legacy_pointer_timeline(with_retro):
    ud = {'id': 'UD-20261105-1', 'utc': '20261105T090000Z',
          'verbatim': 'Point the addition track at %s as the legacy reference until a v1 delivery exists.' % LEG,
          'session': 'owner-chat-20261105', 'channel': 'chat', 'scope': ['adoption:%s' % LEG],
          'numbers_restated': [], 'recorded_by': 'codex'}
    files = dict(BF.kb_files(ud=[ud]))
    files[LEG + '/REPORT.md'] = b'Legacy delivery (pre schema v1); read only.\n'
    if with_retro:
        files[LEG_RETRO] = dump_json({'schema': 'minifly.retro_annotation.v1', 'target': LEG,
                                      'rows': [{'adapter': 'R-CLAIM', 'status': 'PASS',
                                                'reason': 'no gate prose in %s' % LEG}]})
    ptr = {'schema': 'minifly.pointer.v1', 'track': 'addition_legacy', 'delivery': LEG, 'claim': None,
           'claim_sha256': None, 'level_earned': 'legacy', 'primary_gate': 'legacy', 'gate_type': 'legacy',
           'worlds_role': 'legacy', 'objective_metric': {'held_exact_own_eos': 'not_measured'},
           'narrowing_rows': [], 'updated_utc': '20261105T100000Z', 'previous_snapshot': None,
           'user_decision': 'UD-20261105-1'}
    if with_retro:
        ptr['retro_annotation'] = LEG_RETRO
    tl = Timeline()
    tl.add('lp_kb', '20261105T090000Z', BF.CODEX['author'], BF.CODEX['trailers'], [BF.CODEX_BRANCH], files)
    tl.add('lp_pointer', '20261105T100000Z', BF.CODEX['author'], BF.CODEX['trailers'], [BF.CODEX_BRANCH],
           {LEG_POINTER: dump_json(ptr)})
    return tl, None


def b_s34():
    """A pointer to a legacy delivery with an owner UD but no KB/RETRO annotation."""
    return _legacy_pointer_timeline(with_retro=False)


def b_s35():
    """Clean: a pointer to a legacy delivery with an owner UD (adoption scope) and a RETRO annotation."""
    return _legacy_pointer_timeline(with_retro=True)


def b_s36():
    """REPORT prose quotes a number about earlier work without an F-id or catalog id."""
    body = ('Corpus base delivery for the MRK-1 regression corpus. The earlier delivery '
            'minifly_corpus_peer_dev_v1_20261020 reached 62.5% held exact own-EOS on its own split.\n')
    return build_base({'report_body': body})


def b_s37():
    """W answers one constant on the NEW pool in both dev worlds (modal output fraction 1.0)."""
    return build_base({'behaviour': {'w_new_const': '7', 'w_new_const_worlds': ['w_dev_1', 'w_dev_2']}})


def b_s38():
    """An earlier delivery's HELD manifest shares 4 canonical held items with this delivery's held set."""
    prior = 'minifly_corpus_prior_dev_v1_20261001'
    rows = BF.manifest_rows('HELD')[:4]
    return build_base({'extra_files': {'kb': {prior + '/manifests/HELD.jsonl': dump_jsonl(rows)}}})


def b_s39():
    """A pointer to this delivery while a newer v1 delivery with the same task family exists."""
    newer = 'minifly_corpus_next_dev_v1_20261201'

    def ptr(b):
        claim_bytes = b.files[P + 'CLAIM.json']
        claim = json.loads(claim_bytes.decode())
        p = {'schema': 'minifly.pointer.v1', 'track': 'addition_e3', 'delivery': D, 'claim': D + '/CLAIM.json',
             'claim_sha256': sha256_bytes(claim_bytes), 'level_earned': claim['level_earned'],
             'primary_gate': claim['primary_gate'], 'gate_type': claim['gate_type'],
             'worlds_role': claim['worlds_role'], 'objective_metric': claim['objective_metric'],
             'narrowing_rows': [], 'updated_utc': '20261203T130000Z', 'previous_snapshot': None,
             'user_decision': None}
        lock = json.loads(b.files[P + 'ENDPOINT_LOCK.json'].decode())
        lock['delivery'] = newer
        lock['protocol_id'] = newer + '-p1'
        lock['created_utc'] = '20261201T090000Z'
        lock['worlds']['official']['ids'] = ['w_next_%d' % (i + 1) for i in range(8)]
        lock['worlds']['dev'] = ['w_next_dev_1', 'w_next_dev_2']
        lock['worlds']['rehearsal'] = ['w_next_reh_1']
        lock['worlds']['technical'] = ['w_next_tech_%d' % (i + 1) for i in range(8)]
        b.commit('newer_lock', {newer + '/ENDPOINT_LOCK.json': dump_json(lock)}, utc='20261201T090000Z')
        b.commit('pointer', {'PROJECT_DOCUMENTS/CURRENT_ADDITION_E3_POINTER.json': dump_json(p)},
                 utc='20261203T130000Z')
    return build_base({'after': {'report': [ptr]}})


def b_s40():
    """Canary on a delivery whose causal control already fails ADM-CTRL-ZERO (C02a)."""
    return C.b_c02a()


def b_s41():
    """Canary on a delivery that stops at its lock: almost no operator has a target."""
    return S.lock_only()


def b_s42():
    """Clean single-agent T2 dev delivery (SPEC 5): no cross-family review; a REVIEW_DEBT row recorded before
    the dev spend; no rehearsal or official run; the claim is DEV with a dev shadow."""
    def debt(b):
        row = {'id': 'DEBT-20261101-1', 'delivery': D, 'utc': '20261101T150000Z',
               'reason': 'codex unavailable for the review event; dev cycle only', 'recorded_by': 'codex',
               'repaid_by': None}
        b.commit('debt', {BF.KB + 'REVIEW_DEBT.jsonl': dump_jsonl([row])}, utc='20261101T150000Z')

    def claim(b):
        return M.claim_doc(b, rows=b.dev_rows, primary_gate='NOT_OPENED', level_earned='NONE',
                           dev_shadow_verdict='FAIL', worlds_role='DEV', n_worlds=2, review=None,
                           headline=FD.C00E_HEADLINE)
    return build_base({'skip': ['review', 'ack', 'rehearsal', 'official'], 'after': {'technical': [debt]},
                       'headline': FD.C00E_HEADLINE, 'claim_doc': claim})


HELPER_PY = b'''"""Repo-level helper module imported by the corpus learner (S43); parsed, never executed."""


def mix(a, b):
    return (a * 31 + b) % 997
'''


def b_s43():
    """The receipts' write_counter_site is a repo module under tools/ that SOURCE_LOCK.native_closure_sha256
    records but closure_sha256 omits: the canary copies the recorded closure only, so its unmutated copy cannot
    resolve the counter site the original repo resolves."""
    from mrk_util import sha256_json

    def sl(doc):
        doc['native_closure_sha256']['tools/corpus_helper.py'] = sha256_bytes(HELPER_PY)
        doc['native_closure_sha'] = sha256_json(doc['native_closure_sha256'])
        return doc

    def rec(r, stage):
        r['write_counter_site'] = 'tools/corpus_helper.py:mix'
        return r
    return build_base({'extra_files': {'kb': {'tools/corpus_helper.py': HELPER_PY, 'tools/__init__.py': b''}},
                       'source_lock_patch': sl, 'receipt_patch': rec})


# ---------------------------------------------------------------------------------------------
# Held-out mechanism fixtures (S44 clean, S45 dirty). They test the commitment/seal/reveal mechanics of SPEC
# 12.3 with a one-fixture toy batch whose plaintext is visible in this builder: they are NOT held-out canaries
# and never count towards held-out recall. The bundle format is re-stated here (stdlib only, no checker import).
import hashlib  # noqa: E402
import hmac  # noqa: E402

from mrk_util import canonical_json, commit_sha  # noqa: E402

HB_MAGIC = b'MRKHB1\n'
HELD_BATCH = 'MECH1'
HELD_SEALED = KIT + 'corpus/heldout/sealed/%s.bin' % HELD_BATCH
HELD_KEY = hashlib.sha256(b'mrk1-corpus-heldout-owner-key').digest()          # 32 bytes, outside repo/
HELD_NONCE = hashlib.sha256(b'mrk1-corpus-heldout-nonce').digest()[:16]


def _hb_keys(key, batch_id):
    enc = hmac.new(key, b'minifly-heldout|enc|' + batch_id.encode('utf-8'), hashlib.sha256).digest()
    mac = hmac.new(key, b'minifly-heldout|mac|' + batch_id.encode('utf-8'), hashlib.sha256).digest()
    return enc, mac


def _xor(enc_key, nonce, data):
    out = bytearray(len(data))
    for block, off in enumerate(range(0, len(data), 32)):
        ks = hmac.new(enc_key, nonce + block.to_bytes(8, 'big'), hashlib.sha256).digest()
        chunk = data[off:off + 32]
        x = int.from_bytes(chunk, 'big') ^ int.from_bytes(ks[:len(chunk)], 'big')
        out[off:off + len(chunk)] = x.to_bytes(len(chunk), 'big')
    return bytes(out)


def seal_toy_batch(batch_id=HELD_BATCH, key=HELD_KEY):
    files = sorted({'KEY.json': dump_json({'H01': ['FM-14']}),
                    'H01/timeline.json': dump_json({'head': 'h' * 40, 'commits': []}),
                    'H01/repo/README.md': b'Toy held-out mechanism fixture (visible; never held-out).\n'}.items())
    manifest = {'schema': 'minifly.heldout_bundle.v1', 'batch_id': batch_id,
                'files': [{'path': rel, 'size': len(b), 'sha256': sha256_bytes(b)} for rel, b in files]}
    mbytes = canonical_json(manifest)
    plain = len(mbytes).to_bytes(8, 'big') + mbytes + b''.join(b for _, b in files)
    enc, mac = _hb_keys(key, batch_id)
    body = HB_MAGIC + HELD_NONCE + _xor(enc, HELD_NONCE, plain)
    return body + hmac.new(mac, body, hashlib.sha256).digest()


HELD_KIT_FILE = KIT + 'lockextra.py'
HELD_KIT_PY = b'"""Kit module committed as the checker version under test (held-out mechanism fixture)."""\n'


def _heldout_timeline(dirty):
    blob = seal_toy_batch()
    tl = Timeline()
    kit_label = 'hk_kit'
    row = {'batch_id': HELD_BATCH, 'author_identity': 'codex' if dirty else 'owner-zhou', 'utc': '20261105T090000Z',
           'bundle_sha256': sha256_bytes(blob), 'key_sha256': sha256_bytes(HELD_KEY), 'n': 1, 'n_clean': 0,
           'fm_coverage': ['FM-14'], 'checker_sha_under_test': commit_sha(kit_label), 'revealed_utc': None}
    kb = dict(BF.kb_files(canary=[] if dirty else [row]))
    if dirty:
        blob = blob[:40] + bytes([blob[40] ^ 1]) + blob[41:]   # one ciphertext byte flipped after the commitment
    kb[HELD_SEALED] = blob
    tl.add('hk_kb', '20261105T090000Z', BF.CODEX['author'], BF.CODEX['trailers'], [BF.CODEX_BRANCH], kb)
    tl.add(kit_label, '20261105T100000Z', BF.CODEX['author'], BF.CODEX['trailers'], [BF.CODEX_BRANCH],
           {HELD_KIT_FILE: HELD_KIT_PY})
    if dirty:  # the commitment row only after the checker version under test merged
        tl.add('hk_row', '20261105T110000Z', BF.CODEX['author'], BF.CODEX['trailers'], [BF.CODEX_BRANCH],
               {BF.KB + 'CANARY_COMMITMENTS.jsonl': dump_jsonl([row])})
    tl.root_files = {'owner.key': HELD_KEY}
    return tl, None


HELD_REVEAL = ['heldout', 'reveal', '--batch', HELD_BATCH, '--key-file', '../owner.key', '--out', '../revealed']


def b_s44():
    """Clean: a committed commitment row, then the checker version under test; reveal with the owner key."""
    return _heldout_timeline(dirty=False)


def b_s45():
    """Dirty: the commitment row lands after the kit merge, its author is the kit committer, and one bundle
    byte changed after the commitment."""
    return _heldout_timeline(dirty=True)
