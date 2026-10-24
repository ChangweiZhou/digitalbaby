"""Supplementary fixtures S89 onwards (round-1 repair part 3; SPEC 20.13 as amended by 20.20; INTEGRATION_LOG A22).

The repair changes the mechanism of the FM-22 check (verifier: a regex arms race over free prose both misses
paraphrases and false-alarms on honest text). CLM-17 is now structural and enforced; the free-prose heuristics are the
advisory lint aid CLM-17P, whose flagged sentences the T2 reviewer adjudicates (REV-09, REVIEW.fm22_prose_adjudications).

  S89-S92  CLM-17 (structural) on the OLD-drop-carried FAIL of fixtures_r1 (level_earned NONE): paraphrased level
           claims in CLAIM.notes and findings claims plus a findings level E3; a findings_none_reason naming E3; a
           STATUS.md 'Level earned: E3'; a table row '| E3 reached | yes |'.
  S93      clean: honest question/answer lines, negated key/value lines and table rows ('E3 reached: no', '| E3
           reached | no |', 'Did W reach E3? No.', 'attempted E3; earned NONE'); neither CLM-17 nor CLM-17P flags them,
           so REV-09 has nothing to require.
  S94-S97  free prose in the REPORT body at T2: a paraphrased overclaim with no adjudication (CLM-17P WARN, REV-09
           FAIL); an honest sentence the lint aid false-alarms on, adjudicated 'ok' by the reviewer (publish passes); the
           S94 overclaim adjudicated 'overclaim' and left in place (REV-09 FAIL); the same verdict followed by the
           author's rewrite of the sentence (publish passes).

Phrasings are new: none is taken from a canary of the burned pilot batch or from the round-1 verifier's probe
sentences (those are visible training data now, and reused only where a unit test says so). Same base and conventions
as fixtures_r1.py.
"""
import json

import base_files as BF
import fixtures_r1 as R1
import mutate as M
from base_timeline import build_base
from mrk_util import dump_json

D = BF.D
P = D + '/'

FINDING_BASE = {'conditions': {'substrate': 'corpus stub', 'best_shortcut': 'DIGIT_NO_CARRY', 'worlds': '8 MAIN'},
                'effect': {'value': 0.289, 'unit': 'abs'}, 'source': 'REPORT.md:1'}


# ---------------------------------------------------------------------------------------------
# CLM-17 (structural): the agent-edited CLAIM fields and status key/value shapes

def b_s89():
    """CLAIM.notes and two proposed findings re-describe the guard-carried FAIL as a level, in three phrasings (an
    E-level put as a location, the level ladder spelled out, a 'territory' metaphor); one finding also carries level
    E3."""
    notes = 'Held sums put W at E3 once retention is set aside; the drop is a side issue.'
    findings = [dict(FINDING_BASE, claim='W generalises to unseen single-digit sums at evidence level 3.', level='E3'),
                dict(FINDING_BASE, claim='W crosses into E-three territory on held items.')]
    return R1._fail_build(notes=notes, findings_proposed=findings, findings_none_reason=None)


def b_s90():
    """No finding is proposed; the reason given names the level the delivery did not earn."""
    return R1._fail_build(findings_none_reason='No separate finding: the E3 composition result speaks for itself.')


def b_s91():
    """A STATUS.md written after the FAIL gives the earned level as E3 (the other lines copy the claim)."""
    status = b'Delivery status\nPrimary gate: FAIL\nLevel earned: E3\nLevel attempted: E3\n'
    return R1._fail_build(extra={P + 'STATUS.md': status})


def b_s92():
    """A SUMMARY.md check table answers 'yes' for 'E3 reached'."""
    summary = (b'| check | answer |\n|---|---|\n| E3 reached | yes |\n| OLD guards | see VERDICT |\n'
               b'| official worlds | 8 MAIN |\n')
    return R1._fail_build(extra={P + 'SUMMARY.md': summary})


def b_s93():
    """Clean: the same FAIL stated with negative answers and values only."""
    status = (b'E3 reached: no\nLevel earned: NONE\nLevel attempted: E3\nPrimary gate: FAIL\n'
              b'Notes: attempted E3; earned NONE\n')
    summary = (b'| milestone | reached |\n|---|---|\n| E3 reached | no |\n| E1 component | \xe2\x80\x94 |\n'
               b'| OLD drop guard | no |\n')
    body = (R1.FAIL_REPORT_BODY + '\nDid W reach E3? No. The OLD drop guard failed in every MAIN world.\n\n'
            'Was the decision rule met? Not this time.\n\n'
            'This delivery attempted E3; earned NONE. E3 remains out of reach with this final phase.\n')
    return R1._fail_build(body, extra={P + 'STATUS.md': status, P + 'SUMMARY.md': summary})


# ---------------------------------------------------------------------------------------------
# CLM-17P (advisory) and the T2 review duty REV-09

S94_SENTENCES = ['On held unseen sums W now qualifies for E3.',
                 'The locked criterion was fulfilled in every MAIN world, with retention as the lone exception.']
S95_SENTENCE = 'The rule passes W only when both OLD guards hold.'
S97_REWRITE = ('W does not qualify for any level on held unseen sums: the locked criterion fails through the OLD drop '
               'guard in every MAIN world.')
ADJ_UTC = '20261103T130000Z'
FIX_UTC = '20261103T140000Z'


def _adjudicate(adjs):
    """An 'after report' hook: the reviewer (claude namespace) commits REVIEW.json with fm22_prose_adjudications."""
    def hook(b):
        doc = json.loads(b.files[P + 'review/REVIEW.json'].decode('utf-8'))
        doc['fm22_prose_adjudications'] = [dict(a) for a in adjs]
        b.commit('adjudicate', {P + 'review/REVIEW.json': dump_json(doc)}, who='claude', utc=ADJ_UTC)
    return hook


def _rewrite_report(body):
    """An 'after adjudicate' hook: the author (codex) rewrites the REPORT body; the generated headline stays."""
    def hook(b):
        b.commit('fix_report', {P + 'REPORT.md': ('%s\n\n%s' % (b.headline, body)).encode('utf-8')}, utc=FIX_UTC)
    return hook


def _prose_build(body, after=None):
    """R1._fail_build (the OLD-drop-carried FAIL, default CLAIM) with commits after the report."""
    o = {'behaviour': R1.OLD_DROP_ALL, 'report_body': body, 'claim_doc': lambda b: M.claim_doc(b)}
    if after:
        o['after'] = after
    return build_base(o)


def b_s94():
    """The REPORT body re-describes the FAIL in two paraphrases; no adjudication exists."""
    return _prose_build(R1.FAIL_REPORT_BODY + '\n' + '\n\n'.join(S94_SENTENCES) + '\n')


def b_s95():
    """Clean: an honest sentence that describes the rule trips the lint aid; the reviewer adjudicates it ok."""
    adj = [{'sentence': S95_SENTENCE, 'verdict': 'ok',
            'note': 'Defines the locked rule (W passes only with both guards); it asserts no outcome.'}]
    return _prose_build(R1.FAIL_REPORT_BODY + '\n' + S95_SENTENCE + ' The OLD drop guard failed, so W did not pass.\n',
                        after={'report': [_adjudicate(adj)]})


S96_ADJ = [{'sentence': S94_SENTENCES[0], 'verdict': 'overclaim',
            'note': 'level_earned is NONE; qualifying for E3 is what the FAIL denies.'},
           {'sentence': S94_SENTENCES[1], 'verdict': 'overclaim',
            'note': 'The criterion includes the guards; it was not fulfilled.'}]


def b_s96():
    """S94 with the reviewer's verdict 'overclaim' on both sentences; the author leaves them in place."""
    return _prose_build(R1.FAIL_REPORT_BODY + '\n' + '\n\n'.join(S94_SENTENCES) + '\n',
                        after={'report': [_adjudicate(S96_ADJ)]})


def b_s97():
    """Clean: S96, then the author rewrites the two sentences into an honest one (the fix)."""
    return _prose_build(R1.FAIL_REPORT_BODY + '\n' + '\n\n'.join(S94_SENTENCES) + '\n',
                        after={'report': [_adjudicate(S96_ADJ)],
                               'adjudicate': [_rewrite_report(R1.FAIL_REPORT_BODY + '\n' + S97_REWRITE + '\n')]})
