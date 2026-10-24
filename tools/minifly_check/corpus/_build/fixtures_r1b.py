"""Supplementary fixtures S81 onwards (round-1 repair after the verifier's REVISE; SPEC 20.11, 20.12, 20.13, 20.14 as
amended; INTEGRATION_LOG A20).

Each re-implements a verifier probe CLASS through this builder with new names, fields and layouts (the verifier's
probe files are not copied):

  S81-S83  CLM-17 key/value shapes: a STATUS key/value block and a REPORT table that state the level and the rule as
           values (the clause split used to cut key and value apart), and the honest counterpart.
  S84-S86  ADM-EXPO packaging: a held answer inside a JSON-encoded string payload, a prompt and its answer in
           consecutive records of a tab-separated tape, and the clean counterpart (JSON payloads with held prompts only,
           played back to back).
  S87      role 'candidate' on a control arm (LOCK-02; the decision comparator also takes such an arm).
  S88      CLM-04: a confirm word split by markdown link syntax.
Same base and conventions as fixtures_r1.py.
"""
import json

import base_files as BF
import base_items as BI
import fixtures_d as FD
import fixtures_r1 as R1
from base_timeline import build_base
from mrk_util import dump_jsonl, hexs

D = BF.D
P = D + '/'


# ---------------------------------------------------------------------------------------------
# CLM-17: the level and the rule stated as values of a key/value block or a table (OLD-drop-carried FAIL of R1)

def b_s81():
    """STATUS.md is a key/value block written after the FAIL: the evidence level and the decision criterion are values,
    not sentences (no verb next to the level)."""
    status = ('run: official, 8 MAIN worlds\n'
              'evidence_level: E3\n'
              'Decision criterion: satisfied\n'
              'OLD guards: see VERDICT\n').encode('utf-8')
    return R1._fail_build(extra={P + 'STATUS.md': status})


def b_s82():
    """The REPORT body carries a quantity/value table whose rows give the earned level and the locked rule's state."""
    body = (R1.FAIL_REPORT_BODY + '\n| quantity | value |\n|---|---|\n| held exact own-EOS | 85.2% |\n'
            '| level earned | E3 |\n| locked rule | satisfied in 8/8 worlds |\n')
    return R1._fail_build(body)


def b_s83():
    """Clean: the same FAIL with an honest key/value STATUS block, an honest REPORT table and a NOTES.md, none of which
    states a level above NONE or a met rule."""
    status = ('Level earned: NONE\nLevel attempted: E3\nPrimary gate: FAIL\n'
              'Decision rule: W passes when the paired LCB exceeds 5pp in every world\n'
              'Guard old_drop: FAIL\n').encode('utf-8')
    notes = b'E3 was not reached. Next: aim at E3 with a gentler final phase.\n'
    body = (R1.FAIL_REPORT_BODY + '\n| quantity | value |\n|---|---|\n| level earned | NONE |\n'
            '| level attempted | E3 |\n| guard old_drop | FAIL |\n| guard old_acq | PASS |\n\n'
            'The roster gate for this E3 design passed; the primary FAILs on the OLD drop guard.\n')
    return R1._fail_build(body, extra={P + 'STATUS.md': status, P + 'NOTES.md': notes})


# ---------------------------------------------------------------------------------------------
# ADM-EXPO: packaging of held answers in declared non-feedback streams

def b_s84():
    """A declared non-feedback 'session' stream logs tutor events whose body is a JSON-encoded string: the held prompt
    and, in another key of the same encoded object, its answer."""
    rows = []
    for n, (_iid, a, b) in enumerate(BI.pool_items()['HELD']):
        rows.append({'t': n, 'event': 'utter',
                     'body': json.dumps({'text': '%d+%d=' % (a, b), 'next_text': BI.truth(a, b)}, sort_keys=True)})
    return build_base({'lock_patch': R1._declare('manifests/SESSION/*.jsonl', 'session'),
                       'extra_files': {'testbed': {P + 'manifests/SESSION/log.jsonl': dump_jsonl(rows)}}})


def b_s85():
    """A declared non-feedback tape (tab-separated text, one utterance per line) plays each commuted held prompt and,
    on the next line, its answer."""
    lines = []
    tick = 0
    for _iid, a, b in BI.pool_items()['HELD']:
        tick += 1
        lines.append('%d\t%d+%d=' % (tick, b, a))
        tick += 1
        lines.append('%d\t%s' % (tick, BI.truth(a, b)))
    return build_base({'lock_patch': R1._declare('manifests/TAPE/*.tsv', 'tape'),
                       'extra_files': {'testbed': {P + 'manifests/TAPE/side_a.tsv':
                                                   ('\n'.join(lines) + '\n').encode('ascii')}}})


def b_s86():
    """Clean: the same session stream with JSON-encoded bodies that carry held prompts only, back to back (each next
    utterance starts with an operand, never with the answer)."""
    rows = []
    for n, (_iid, a, b) in enumerate(BI.pool_items()['HELD']):
        rows.append({'t': n, 'event': 'utter', 'body': json.dumps({'text': '%d+%d=' % (b, a), 'next_text': '?'},
                                                                   sort_keys=True),
                     'body_hex': hexs('%d+%d=' % (a, b))})
    return build_base({'lock_patch': R1._declare('manifests/SESSION/*.jsonl', 'session'),
                       'extra_files': {'testbed': {P + 'manifests/SESSION/log.jsonl': dump_jsonl(rows)}}})


# ---------------------------------------------------------------------------------------------
# Role 'candidate' on a control arm

def b_s87():
    """The FROZEN control is relabelled role 'candidate' (primary.candidate_arm stays W) and answers every official
    HELD item like W."""
    def lock(lk):
        for a in lk['arms']:
            if a['id'] == 'FROZEN':
                a['role'] = 'candidate'
    return build_base({'lock_patch': lock, 'behaviour': {'frozen_like_w_held': list(BI.OFF_WORLDS)}})


# ---------------------------------------------------------------------------------------------
# CLM-04: markdown link syntax inside a DEV confirm word

def b_s88():
    """C00e (T1, DEV) with a SUMMARY.md whose confirm word is split by an empty markdown link."""
    return FD.b_c00e(extra_files={'report': {P + 'SUMMARY.md': b'Carry reuse con[]()firmed on the dev worlds.\n'}})
