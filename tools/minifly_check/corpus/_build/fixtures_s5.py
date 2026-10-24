"""Supplementary fixtures S56 onwards (hardening round 4, beyond the SPEC 12.2 table; INTEGRATION_LOG A14).

Round-4 hardening (SPEC 20.5-20.9) adds: confirm wording inside identifiers and behind zero-width characters
(CLM-04, S56-S58); the decision comparator includes every declared no_prior_teaching, shuf, frozen and lesion
arm (S59); the runtime closure of every counted run (RUN-03, S60; the base now writes one per counted attempt);
resumable official runs with a determinism replay (RUN-02 amended, RUN-04, S61-S64). Each fixture adds one
documented defect (or one clean configuration) on the base delivery. Same base and conventions as fixtures_s.py .. fixtures_s4.py.
"""
import json

import base_artifacts as BA
import base_files as BF
import base_items as BI
import base_timeline as TL
import fixtures_d as FD
import mutate as M
from base_timeline import build_base
from mrk_util import dump_compact, dump_json, dump_jsonl, sha256_bytes

D = BF.D
P = D + '/'


# ---------------------------------------------------------------------------------------------
# CLM-04: confirm wording in identifiers and behind format characters (verifier round 4, blocking)

def b_s56():
    """C00e (T1, DEV) with a SUMMARY.json in the project's own snake_case vocabulary (probe a)."""
    summary = dump_json({'verdict': 'GAIN_CONFIRMED', 'axis_confirmed': True})
    return FD.b_c00e(extra_files={'report': {P + 'SUMMARY.json': summary}})


def b_s57():
    """C00e (T1, DEV) whose REPORT body states a CamelCase confirmation label (probe f)."""
    return FD.b_c00e(report_body='NativeConfirmation achieved on dev.\n')


def b_s58():
    """C00e (T1, DEV) with a STATUS.md whose confirm word is split by a zero-width space (probe g)."""
    status = 'Reuse con\u200bfirmed on the held set.\n'.encode('utf-8')
    return FD.b_c00e(extra_files={'report': {P + 'STATUS.md': status}})


# ---------------------------------------------------------------------------------------------
# Decision comparator completeness (SPEC 20.6; verifier probe 2c)

def b_s59():
    """In every official world the FROZEN arm (role frozen, not in comparator_arms) answers every HELD item exactly
    as W does; every other pool keeps the base behaviour, so no INERT alarm applies. Under the amended SPEC 9.5
    the decision comparator includes FROZEN: d_w = 0 in every world and the honest verdict is FAIL (the
    round-3 checker gave PASS)."""
    return build_base({'behaviour': {'frozen_like_w_held': list(BI.OFF_WORLDS)}})


# ---------------------------------------------------------------------------------------------
# RUN-03: runtime closure (S60; SPEC 20.7; LEARNED-BANK lesson: a hand-listed lock of 6 files against a runtime
# closure of 55)

BOOTSTRAP_PY = (b'"""Corpus stub: a runtime helper that run.py loads through importlib (never executed)."""\n\n'
                b'HINT = "learner"\n')


def b_s60():
    """bootstrap.py is committed with the learner but reached only by a runtime import: lock build's static closure
    (SOURCE_LOCK.closure_sha256) misses it, LOCK-05 passes, and the official run's runtime closure lists it."""
    sha = sha256_bytes(BOOTSTRAP_PY)
    return build_base({'extra_native': {P + 'bootstrap.py': BOOTSTRAP_PY},
                       'runtime_extra': {'official': [(P + 'bootstrap.py', sha)]}})


# ---------------------------------------------------------------------------------------------
# Resumable official runs with a determinism replay (S61-S64; SPEC 20.8; LEARNED-BANK lesson: the container
# restarted mid-run, the session resumed only the missing histories under the identical lock and re-ran one
# completed history to show bit-identical determinism)

RESUME_START, RESUME_END, RESUME_COMMIT = '20261103T100000Z', '20261103T101500Z', '20261103T103000Z'
REPLAY_UNIT = ('w_off_1', 'W')


def unit_digests(rows):
    """The ATTEMPTS `units` field launch records for an official attempt: [{world, arm, n_rows, digest}], the digest
    as in fixtures_d.replay_digest (SCHEMAS 13; attempt_id excluded)."""
    units = sorted(set((r['world'], r['arm']) for r in rows))
    return [{'world': w, 'arm': a, 'n_rows': sum(1 for r in rows if (r['world'], r['arm']) == (w, a)),
             'digest': FD.replay_digest(rows, w, a)} for w, a in units]


def _resume_build(variant):
    """Official run in two launches. Attempt A (killed, exit -9) completes every arm of w_off_1..w_off_5; attempt B,
    `launch --stage official --resume` under the identical lock, runs every arm of w_off_6..w_off_8 (rows appended to
    results/official/OUTPUTS.jsonl.gz, A's rows byte-identical) and re-runs the unit launch chose, (w_off_1, W), into
    results/official_replay/OUTPUTS.jsonl. variant: clean | rewrite | no_replay | mismatch."""
    def run(b):
        worlds = b.off_worlds()
        done, rest = worlds[:5], worlds[5:]
        lock_sha, sl_sha = b.lock_sha, b.sl['sha']
        # attempt A: interrupted after five worlds
        aid_a = BA.att_id('20261103T083000Z', 'official')
        rows_a, rec_a = b.native_run(done, 'official', aid_a, 'official')
        out_a = b.rows_file(rows_a)
        gp_a = 'checks/GATE_official_20261103T083000Z.json'
        g_a = dump_compact(BA.gate_report('official', False, '20261103T083000Z', b.tl.head_sha(), lock_sha, sl_sha,
                                          'launch', TL.FAMILY_ID))
        prod_a = [('results/official/OUTPUTS.jsonl.gz', sha256_bytes(out_a))]
        prod_a += sorted((p[len(P):], sha256_bytes(x)) for p, x in rec_a.items())
        f_a = {}
        b.add_runtime_closure(aid_a, 'official', f_a, prod_a)
        argv = ['python3', '-I', 'run.py', '--worlds', ','.join(worlds)]
        a = BA.attempt(aid_a, 'official', argv, '20261103T083000Z', '20261103T084500Z', b.tl.head_sha(), lock_sha,
                       b.sl, prod_a, gp_a, cpu=300.0)
        a.update({'exit_code': -9, 'interrupted': True, 'units': unit_digests(rows_a)})
        b.attempts.append(a)
        f_a.update({P + 'results/official/OUTPUTS.jsonl.gz': out_a, P + gp_a: g_a,
                    P + 'runlog/ATTEMPTS.jsonl': dump_jsonl(b.attempts)})
        f_a.update(rec_a)
        b.commit('official', f_a, utc=TL.UTC['official'])
        # attempt B: the resume
        aid_b = BA.att_id(RESUME_START, 'official-resume')
        rows_b, rec_b = b.native_run(rest, 'official', aid_b, 'official')
        kept_a = list(rows_a)
        if variant == 'rewrite':
            # B also re-runs a unit A completed, writing it into the official rows and its receipt (overwrite)
            redo = ('w_off_1', 'NPT')
            kept_a = [r for r in rows_a if (r['world'], r['arm']) != redo]
            rows_b = [dict(r, attempt_id=aid_b) for r in rows_a if (r['world'], r['arm']) == redo] + rows_b
            rp = P + 'results/receipts/%s/%s.json' % redo
            rc = json.loads(rec_a[rp].decode('utf-8'))
            rc['attempt_id'] = aid_b
            rec_b[rp] = dump_compact(rc)
        out_b = b.rows_file(kept_a + rows_b)
        replay = [dict(r, attempt_id=aid_b) for r in rows_a if (r['world'], r['arm']) == REPLAY_UNIT]
        if variant == 'mismatch':
            # the replayed history is not bit-identical: one HELD item answers differently
            for r in replay:
                if r['pool'] == 'HELD' and r['item_id'] == 'h02':
                    stream = BI.stream_of('99')
                    r.update({'stream_hex': stream.encode('ascii').hex(), 'emitted_hex': b'99'.hex(), 'own_eos': True})
        gp_b = 'checks/GATE_official_20261103T095500Z.json'
        g_b = dump_compact(BA.gate_report('official', False, '20261103T095500Z', b.tl.head_sha(), lock_sha, sl_sha,
                                          'launch', TL.FAMILY_ID))
        prod_b = [('results/official/OUTPUTS.jsonl.gz', sha256_bytes(out_b))]
        f_b = {P + 'results/official/OUTPUTS.jsonl.gz': out_b, P + gp_b: g_b}
        if variant != 'no_replay':
            rep = dump_jsonl(replay)
            f_b[P + 'results/official_replay/OUTPUTS.jsonl'] = rep
            prod_b.append(('results/official_replay/OUTPUTS.jsonl', sha256_bytes(rep)))
        prod_b += sorted((p[len(P):], sha256_bytes(x)) for p, x in rec_b.items())
        b.add_runtime_closure(aid_b, 'official-resume', f_b, prod_b)
        a2 = BA.attempt(aid_b, 'official', argv, RESUME_START, RESUME_END, b.tl.head_sha(), lock_sha, b.sl, prod_b,
                        gp_b, cpu=180.0)
        a2.update({'units': unit_digests(rows_b), 'resume_of': aid_a, 'replay_requested': [list(REPLAY_UNIT)],
                   'replay_units': [] if variant == 'no_replay' else [list(REPLAY_UNIT)]})
        b.attempts.append(a2)
        f_b[P + 'runlog/ATTEMPTS.jsonl'] = dump_jsonl(b.attempts)
        f_b.update(rec_b)
        b.commit('official_resume', f_b, utc=RESUME_COMMIT)
        b.official_rows = kept_a + rows_b
        b.resume_ids = (aid_a, aid_b)
        b.resumed_units = sorted(set((r['world'], r['arm']) for r in rows_b))

    def claim(b):
        prov = {'interrupted': True, 'resume_attempts': [b.resume_ids[1]],
                'resumed_units': [list(u) for u in b.resumed_units],
                'replay_units': [] if variant == 'no_replay' else [list(REPLAY_UNIT)]}
        return M.claim_doc(b, provenance=prov)
    return build_base({'skip': ['official'], 'after': {'rehearsal': [run]}, 'claim_doc': claim})


def b_s61():
    """Clean legal resume: A interrupted, B resumes the missing units under the identical lock and replays one."""
    return _resume_build('clean')


def b_s62():
    """B also rewrites the completed unit (w_off_1, NPT): its rows and its receipt now carry B's attempt id."""
    return _resume_build('rewrite')


def b_s63():
    """B resumes the missing units but re-runs no completed unit (no determinism replay)."""
    return _resume_build('no_replay')


def b_s64():
    """B's replay of (w_off_1, W) differs from the original on one HELD item (not bit-identical)."""
    return _resume_build('mismatch')
