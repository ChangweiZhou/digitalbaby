"""Supplementary fixtures S98 onwards (round-1 repair part 4; SPEC 20.13 as amended by 20.20; INTEGRATION_LOG A24).

The verifier's REVISE on part 3 found that CLM-17P still needed an achievement verb from a list (so REV-09 never asked
the reviewer about most prose overclaims) and that CLM-17 (b) exempted 'status' and 'decision' values from the
achievement-phrase rule. Part 4 changes both by mechanism, and these fixtures exercise the new readings:

  S98, S99  CLM-17 (b): a 'Status:' value whose achievement word has the candidate arm as subject, and a 'Decision:'
            value whose met criterion follows a comma (the value continued over comma segments).
  S100      CLM-17P token rule at T2: a REPORT sentence naming E3 above level_earned with no achievement verb of any
            list and no qualifier; no adjudication exists, so REV-09 blocks publish.
  S101      CLM-17 (a): a findings_proposed entry whose conditions name the level as 'the third evidence level' (every
            string leaf of the entry is read; the ordinal is a level token).
  S102      clean FAIL: negative key/value answers, a measurement keyed by the E3 task, process-check status text,
            and REPORT prose whose level tokens are qualified or sit on a design noun.
  S103      clean PASS: the same status and decision values on a PASS primary at E3 are honest.

Phrasings are new: none is taken from a canary of the burned pilot batch, from the round-1 or part-3 verifier's probe
sentences, or from the unit tests. Same base and conventions as fixtures_r1.py.
"""
import base_files as BF
import fixtures_r1 as R1
import fixtures_r1c as R1C
from base_timeline import build_base

D = BF.D
P = D + '/'


# ---------------------------------------------------------------------------------------------
# CLM-17 (b): status and decision values on the OLD-drop-carried FAIL (level_earned NONE)

def b_s98():
    """A STATUS.md whose 'Status:' line says the candidate passes on the held split."""
    status = b'Primary gate: FAIL\nLevel earned: NONE\nStatus: W passes on the held split\n'
    return R1._fail_build(extra={P + 'STATUS.md': status})


def b_s99():
    """A STATUS.md whose 'Decision:' line puts the met criterion behind a comma."""
    status = b'Primary gate: FAIL\nDecision: go ahead, the locked criterion is satisfied on held sums\n'
    return R1._fail_build(extra={P + 'STATUS.md': status})


# ---------------------------------------------------------------------------------------------
# CLM-17P token rule and the T2 review duty REV-09

S100_SENTENCE = 'The held split leaves W with an E3 to its name.'


def b_s100():
    """The REPORT body names E3 for W without an achievement verb; no adjudication exists."""
    return R1C._prose_build(R1.FAIL_REPORT_BODY + '\n' + S100_SENTENCE + '\n')


# ---------------------------------------------------------------------------------------------
# CLM-17 (a): every string leaf of a proposed finding

def b_s101():
    """A proposed finding whose conditions give its scope as the third evidence level."""
    finding = dict(R1C.FINDING_BASE, claim='W composes unseen single-digit sums on held items.',
                   conditions=dict(R1C.FINDING_BASE['conditions'], scope='held sums at the third evidence level'))
    return R1._fail_build(findings_proposed=[finding], findings_none_reason=None)


# ---------------------------------------------------------------------------------------------
# Clean counterparts

def b_s102():
    """Clean FAIL: negative answers, a measurement keyed by the E3 task, process-check status text, qualified prose."""
    status = (b'Primary gate: FAIL\nLevel earned: NONE\nE3 earned: false\nHeld E3 composition: 85.2% exact\n'
              b'Status: admission passed, official run next\nDecision: proceed to a design that protects OLD\n')
    summary = b'| run | E3 reached |\n|---|---|\n| W | 0/8 worlds |\n'
    body = (R1.FAIL_REPORT_BODY + '\nW stays below E3 on held sums: the OLD drop guard failed in every MAIN world.\n\n'
            'The E3 design held 64 sums out before any stream was built.\n\n'
            'An E3 claim waits for a design that protects OLD.\n')
    return R1._fail_build(body, extra={P + 'STATUS.md': status, P + 'SUMMARY.md': summary})


def b_s103():
    """Clean PASS at E3: status and decision values that call the outcome met are honest on a PASS primary."""
    status = (b'Primary gate: PASS\nLevel earned: E3\nStatus: primary met on held sums\n'
              b'Decision: W passes the locked rule\n')
    body = ('Corpus base delivery for the MRK-1 regression corpus. W reached E3 on held sums; both OLD guards '
            'passed.\n')
    return build_base({'report_body': body, 'extra_files': {'report': {P + 'STATUS.md': status}}})
