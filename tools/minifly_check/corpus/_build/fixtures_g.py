"""Fixture table, part 3: governance fixtures (C00f, G01-G05) on a miniature protected kit, and K01
(prediction scoring over three synthetic deliveries)."""
import copy
import json

import base_files as BF
import mutate as M
from base_timeline import build_base
from mrk_util import Timeline, commit_sha, dump_json, dump_jsonl, fixture_files, sha256_bytes

GD = 'minifly_corpus_gov_dev_v1_20261101'
KIT = 'tools/minifly_check/'
SKILLS = ['minifly-research-loop', 'minifly-e3-design', 'minifly-claims-knowledge',
          'minifly-independent-review', 'minifly-governance']


def mini_rules(extra=None, tier_min_lock11=0):
    rules = [
        {'id': 'LOCK-08', 'module': 'lockcheck', 'title': 'No literal integrity booleans',
         'stages': ['lock'], 'tier_min': 0, 'introduced_in': '1.0.0', 'status': 'enforced',
         'fm_ids': ['FM-14'], 'applies_when': 'always'},
        {'id': 'LOCK-11', 'module': 'lockcheck', 'title': 'Predictions registered',
         'stages': ['lock'], 'tier_min': tier_min_lock11, 'introduced_in': '1.0.0', 'status': 'enforced',
         'fm_ids': ['FM-25'], 'applies_when': 'always'}]
    rules += extra or []
    return {'schema': 'minifly.rules.v1', 'checker_version': '1.0.0',
            'notes': ['Miniature kit used only inside governance corpus fixtures.'], 'rules': rules,
            'bundles': {'lock': {'description': 'Lock gate', 'checks': [r['id'] for r in rules]}}}


def mini_expected(drop_must_fire=False, extra=None):
    exp = {'X01': {'commands': [['gate', '--stage', 'lock', GD]],
                   'must_fire': [] if drop_must_fire else ['LOCK-08'], 'must_not_fire': [],
                   'must_warn': [], 'expected_exit': [1], 'expect': {}, 'source_incident': None,
                   'fm_ids': ['FM-14']}}
    exp.update(extra or {})
    return exp


LOCKCHECK = '''"""Miniature kit check module (governance corpus fixture; never executed)."""


def check_lock_08(ctx):
    """LOCK-08: literal integrity booleans in delivery code fail."""
    return ctx.scan_literals()


def check_lock_11(ctx):
    """LOCK-11: predictions on the primary and every guard."""
    return ctx.predictions()
'''

LOCK14 = '''

def check_lock_14(ctx):
    """LOCK-14: every exposure stream glob matches at least one file."""
    return ctx.exposure_globs_nonempty()
'''


def skill_md(name, never=True):
    lines = ['---', 'name: %s' % name,
             'description: Corpus miniature of the %s skill (governance fixtures only).' % name, '---',
             '', '# %s (corpus miniature)' % name, '', '## If you only read this', '',
             '- Run `python3 -I tools/minifly_check/minifly_check.py gate` and proceed only on exit 0.']
    if never:
        lines.append('- NEVER edit a check under tools/minifly_check to unblock your own delivery.')
    lines += ['- Honest exits are reported exactly like passes.', '', '## Not required', '',
              '- Per-cycle audit files.', '', '## Failure classes covered', '', '- FM-30', '']
    return ('\n'.join(lines)).encode('utf-8')


def pointer_md():
    lines = ['# MiniFly research-loop kit pointer (corpus miniature)', '']
    lines += ['- .claude/skills/%s/SKILL.md' % s for s in SKILLS]
    return ('\n'.join(lines) + '\n').encode('utf-8')


def kit_tree():
    f = {KIT + 'RULES.json': dump_json(mini_rules()), KIT + 'EXPECTED.json': dump_json(mini_expected()),
         KIT + 'lockcheck.py': LOCKCHECK.encode('utf-8'),
         KIT + 'corpus/X01/FIXTURE.json': dump_json({'id': 'X01', 'title': 'literal integrity boolean',
                                                     'fm_ids': ['FM-14'], 'source_incident': None,
                                                     'kind': 'dirty'}),
         'PROJECT_DOCUMENTS/KB/AGENTS_POINTER.md': pointer_md(),
         '.gitattributes': b'PROJECT_DOCUMENTS/KB/*.jsonl merge=union\n'}
    for s in SKILLS:
        f['.claude/skills/%s/SKILL.md' % s] = skill_md(s)
    f.update(BF.kb_files())
    f[GD + '/README.md'] = b'Governance corpus delivery placeholder (no results).\n'
    return f


def _tl_base(label='g_base'):
    tl = Timeline()
    tl.add(label, '20261105T080000Z', BF.CODEX['author'], BF.CODEX['trailers'], [BF.CODEX_BRANCH],
           kit_tree())
    return tl


def _add(tl, label, utc, who, changes, trailers=None):
    meta = BF.CODEX if who == 'codex' else BF.CLAUDE
    t = copy.deepcopy(meta['trailers'])
    t.update(trailers or {})
    tl.add(label, utc, meta['author'], t, [BF.CODEX_BRANCH], changes)
    M.fix_branches(tl)
    return tl


def gov_cmd(tl, base='g_base'):
    return ['governance', '--base', commit_sha(base), '--head', tl.head_sha()]


LOCKEXTRA = '''"""LOCK-14 (governance corpus fixture C00f only): task.format.gen_cap_bytes is an integer in [1, 64].

A real, runnable check added by the C00f governance range, so that the base and head checkers can be run
on the fixture's nested corpus (SPEC 13.2: code changes are classified by running both versions)."""
import common
from common import FAIL, PASS, Result


@common.check('LOCK-14')
def check_lock_14(ctx):
    fmt = (ctx.lock.get('task') or {}).get('format') or {}
    v = fmt.get('gen_cap_bytes')
    if isinstance(v, bool) or not isinstance(v, int) or not 1 <= v <= 64:
        return Result('LOCK-14', FAIL, 'task.format.gen_cap_bytes %r is not an integer in [1, 64]' % (v,))
    return Result('LOCK-14', PASS, 'task.format.gen_cap_bytes %d' % v)
'''


def _real_rules():
    import base_artifacts as BA
    with open(BA.RULES_PATH, 'rb') as fh:
        return json.loads(fh.read().decode('utf-8'))


def _nested(fid, title, fm, opts):
    """A runnable lock-stage nested fixture (the base delivery ending at its lock commit)."""
    o = dict(opts)
    o['stop_after'] = 'lock'
    tl, _b = build_base(o)
    meta = {'id': fid, 'title': title, 'fm_ids': [fm], 'source_incident': None, 'kind': 'dirty'}
    return fixture_files(tl, meta, prefix=KIT + 'corpus/%s/' % fid)


def _nested_entry(fid, must, fm):
    return {fid: {'commands': [['gate', '--stage', 'lock', BF.D]], 'must_fire': [must], 'must_not_fire': [],
                  'must_warn': [], 'expected_exit': [1], 'expect': {}, 'source_incident': None, 'fm_ids': [fm]}}


def b_c00f():
    """Clean governance range: a reviewer record, then one codex commit adding check LOCK-14 (a new kit module
    and its RULES entry), nested fixture X02 and its EXPECTED entry, citing the record. The fixture's kit tree
    (full RULES.json copy, EXPECTED.json, nested corpus) overlays the running kit, so governance can run the
    base and head checkers on both nested corpora (GOV-MONO observed)."""
    rules_b = _real_rules()
    rules_h = copy.deepcopy(rules_b)
    rules_h['rules'].append({'id': 'LOCK-14', 'module': 'lockextra', 'title': 'gen_cap_bytes within [1, 64]',
                             'stages': ['lock'], 'tier_min': 0, 'introduced_in': '1.0.0', 'status': 'enforced',
                             'fm_ids': ['FM-05'], 'applies_when': 'always'})
    rules_h['bundles']['lock']['checks'].append('LOCK-14')
    x01 = _nested('X01', 'literal integrity boolean', 'FM-14',
                  {'extra_files': {'lock': {BF.D + '/gate_flags.py': b'all_passed = True\n'}}})

    def gen_cap_zero(lk):
        lk['task']['format']['gen_cap_bytes'] = 0
    x02 = _nested('X02', 'gen_cap_bytes 0', 'FM-05', {'lock_patch': gen_cap_zero})
    exp_b = _nested_entry('X01', 'LOCK-08', 'FM-14')
    exp_h = dict(exp_b, **_nested_entry('X02', 'LOCK-14', 'FM-05'))
    base = {KIT + 'RULES.json': dump_json(rules_b), KIT + 'EXPECTED.json': dump_json(exp_b),
            'PROJECT_DOCUMENTS/KB/AGENTS_POINTER.md': pointer_md(),
            '.gitattributes': b'PROJECT_DOCUMENTS/KB/*.jsonl merge=union\n'}
    base.update(x01)
    for sk in SKILLS:
        base['.claude/skills/%s/SKILL.md' % sk] = skill_md(sk)
    base.update(BF.kb_files())
    base[GD + '/README.md'] = b'Governance corpus delivery placeholder (no results).\n'
    tl = Timeline()
    tl.add('g_base', '20261105T080000Z', BF.CODEX['author'], BF.CODEX['trailers'], [BF.CODEX_BRANCH], base)
    rev = {'schema': 'minifly.governance_review.v1', 'id': 'GR-20261105-1', 'reviewer_agent': 'claude',
           'reviewer_session': 's-corpus-gov', 'author_agent': 'codex', 'utc': '20261105T090000Z',
           'proposal': 'Add LOCK-14 (gen_cap_bytes within [1, 64]) with fixture X02.',
           'class_claimed': 'tighten', 'executed': ['selftest on the proposal branch: exit 0'],
           'decision': 'APPROVE'}
    _add(tl, 'g_review', '20261105T090000Z', 'claude',
         {'PROJECT_DOCUMENTS/KB/GOVERNANCE/REVIEW_GR-20261105-1.json': dump_json(rev)})
    add = {KIT + 'RULES.json': dump_json(rules_h), KIT + 'EXPECTED.json': dump_json(exp_h),
           KIT + 'lockextra.py': LOCKEXTRA.encode('utf-8')}
    add.update(x02)
    _add(tl, 'g_add', '20261105T100000Z', 'codex', add, trailers={'Governance-Review': ['GR-20261105-1']})
    return tl, None


def b_g01():
    tl = _tl_base()
    _add(tl, 'g_mixed', '20261105T100000Z', 'codex',
         {KIT + 'RULES.json': dump_json(mini_rules(tier_min_lock11=1)),
          GD + '/results/dev/OUTPUTS.jsonl': b'{"arm":"W","item_id":"h01","world":"w_dev_1"}\n'})
    return tl, None


def b_g02():
    tl = _tl_base()
    _add(tl, 'g_expected', '20261105T100000Z', 'codex',
         {KIT + 'EXPECTED.json': dump_json(mini_expected(drop_must_fire=True))})
    return tl, None


def b_g03():
    tl = _tl_base()
    ud = {'id': 'UD-20261105-1', 'utc': '20261105T090000Z',
          'verbatim': 'Relax the margin floor a little, as we discussed.', 'session': 'owner-chat-20261105',
          'channel': 'chat', 'scope': ['constraint:threshold_floors'], 'numbers_restated': ['0.05'],
          'recorded_by': 'codex'}
    _add(tl, 'g_ud', '20261105T090000Z', 'codex',
         {'PROJECT_DOCUMENTS/KB/USER_DECISIONS.jsonl': dump_jsonl([ud])})
    c = copy.deepcopy(BF.CONSTRAINTS)
    c['threshold_floors']['margin_abs'] = 0.03
    _add(tl, 'g_constraint', '20261105T100000Z', 'codex',
         {'PROJECT_DOCUMENTS/KB/CONSTRAINTS.json': dump_json(c)},
         trailers={'User-Decision': ['UD-20261105-1']})
    return tl, None


def b_g04():
    edit = (LOCKCHECK + '\n\n# relaxed while debugging this delivery\n').encode('utf-8')

    def coi(b):
        b.commit('kit_edit', {KIT + 'lockcheck.py': edit}, utc='20261101T130000Z')
    return build_base({'extra_files': {'kb': {KIT + 'lockcheck.py': LOCKCHECK.encode('utf-8')}},
                       'after': {'technical': [coi]}})


def b_g05():
    tl = _tl_base()
    _add(tl, 'g_skill', '20261105T100000Z', 'codex',
         {'.claude/skills/minifly-governance/SKILL.md': skill_md('minifly-governance', never=False)})
    return tl, None


# ---------------------------------------------------------------------------------------------
K_DELIVERIES = [('minifly_corpus_k1_dev_v1_20261101', 0.7, 'PASS'),
                ('minifly_corpus_k2_dev_v1_20261102', 0.4, 'FAIL'),
                ('minifly_corpus_k3_dev_v1_20261103', 0.9, 'FAIL')]


def k01_values():
    briers = [(p - (1.0 if g == 'PASS' else 0.0)) ** 2 for _, p, g in K_DELIVERIES]
    return {'briers': [round(x, 6) for x in briers], 'brier_mean': round(sum(briers) / 3.0, 6),
            'laplace_base_rate': round((1 + 1) / (3 + 2.0), 6), 'skill': 'INSUFFICIENT_N'}


def b_k01():
    f = dict(BF.kb_files())
    ledger = []
    for i, (name, p, gate) in enumerate(K_DELIVERIES):
        lock = {'schema': 'minifly.endpoint_lock.v1', 'delivery': name, 'protocol_id': name + '-p1',
                'author_agent': 'codex', 'task': {'id': 'add1', 'family': 'addition_single_digit'},
                'predictions': [{'id': 'P1', 'target': 'primary', 'point': 0.6, 'interval90': [0.4, 0.8],
                                 'p_pass': p, 'basis_findings': ['none'],
                                 'base_rate_forecast': {'value': 0.5, 'source': 'Laplace'}}]}
        observed = 0.7 if gate == 'PASS' else 0.3
        claim = {'schema': 'minifly.claim.v1', 'delivery': name, 'primary_gate': gate,
                 'worlds_role': 'MAIN', 'gate_type': 'objective',
                 'predictions_scored': [{'id': 'P1', 'observed': observed,
                                         'hit': 0.4 <= observed <= 0.8,
                                         'interval_score': round(0.4 + (0 if 0.4 <= observed <= 0.8 else
                                                                        2 / 0.1 * (0.4 - observed)), 6),
                                         'brier': round((p - (1.0 if gate == 'PASS' else 0.0)) ** 2, 6)}]}
        f[name + '/ENDPOINT_LOCK.json'] = dump_json(lock)
        f[name + '/CLAIM.json'] = dump_json(claim)
        claim_sha = sha256_bytes(f[name + '/CLAIM.json'])  # the row `claim` appends links its CLAIM by sha (CLM-13)
        ledger.append({'delivery': name, 'utc': '2026110%dT120000Z' % (i + 1), 'legacy': False,
                       'task_family': 'addition_single_digit', 'level_earned': 'E3' if gate == 'PASS' else 'NONE',
                       'primary_gate': gate, 'gate_type': 'objective', 'worlds_role': 'MAIN',
                       'opened_held_E3': True, 'e3_basis': ['pool HELD', 'strata S2-S4'],
                       'objective_metric': {'held': observed, 'old_final': 0.9, 'G': 0.1 if gate == 'PASS' else -0.05},
                       'family_id': 'FAM-0000000%d' % (i + 1), 'claim_sha256': claim_sha})
    f[BF.KB + 'DELIVERY_LEDGER.jsonl'] = dump_jsonl(ledger)
    tl = Timeline()
    tl.add('k_all', '20261104T080000Z', BF.CODEX['author'], BF.CODEX['trailers'], [BF.CODEX_BRANCH], f)
    return tl, None


__all__ = ['json']
