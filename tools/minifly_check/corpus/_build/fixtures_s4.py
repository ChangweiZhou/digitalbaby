"""Supplementary fixtures S51-S54 (integration round 3, beyond the SPEC 12.2 table).

The round-3 verifier found two paths the corpus did not cover: dev work relabelled as testbed (SPEC 3.2: the
testbed is not counted against the cycle cap) and a held set that the lock fixes by hash but nothing verified
(SPEC 6.2: "The held set is fixed in the lock (split.held_set_sha256) before any stream exists"). Each fixture
adds one documented defect on the base delivery; nothing in an existing fixture changes. Same base and
conventions as fixtures_s.py, fixtures_s2.py and fixtures_s3.py.
"""
import base_artifacts as BA
import base_files as BF
import base_items as BI
import base_timeline as TL
from base_timeline import build_base
from mrk_util import dump_compact, dump_jsonl, hexs, sha256_bytes

D = BF.D
P = D + '/'
RELABEL_UTC = '20261102T130000Z'
FOREIGN = (2, 7)   # 2+7=, an item no split manifest holds


def b_s51():
    """With the cycle cap of 3 used up (technical, dev and rehearsal closed), a `launch --stage testbed` attempt
    reruns the dev runner on w_dev_1: it rewrites W's w_dev_1 dev rows (now carrying the testbed attempt id) and
    a testbed CYCLES row covers it. RUN-01 is satisfied (the file is launch-recorded); the cap is not."""
    def relabel(b):
        aid = BA.att_id(RELABEL_UTC, 'testbed-relabel')
        rows = [dict(r, attempt_id=aid) if (r['arm'] == 'W' and r['world'] == 'w_dev_1') else r for r in b.dev_rows]
        out = b.rows_file(rows)
        gp = 'checks/GATE_lock_%s.json' % RELABEL_UTC
        g = dump_compact(BA.gate_report('lock', False, RELABEL_UTC, b.tl.head_sha(), b.lock_sha, b.sl['sha'],
                                        'launch', TL.FAMILY_ID))
        a = BA.attempt(aid, 'testbed', ['python3', '-I', 'run.py', '--worlds', 'w_dev_1'], RELABEL_UTC,
                       '20261102T131500Z', b.tl.head_sha(), b.lock_sha, b.sl,
                       [('results/dev/OUTPUTS.jsonl.gz', sha256_bytes(out))], gp, cpu=60.0)
        b.attempts.append(a)
        b.cycles.append(BA.cycle(len(b.cycles) + 1, 'testbed', 'revise', 'Testbed rerun.', '20261102T132000Z',
                                 b.sl, [aid], None, True, getattr(b, 'designs_k', 1)))
        b.commit('relabel', {P + 'results/dev/OUTPUTS.jsonl.gz': out, P + gp: g,
                             P + 'runlog/ATTEMPTS.jsonl': dump_jsonl(b.attempts),
                             P + 'runlog/CYCLES.jsonl': dump_jsonl(b.cycles)}, utc='20261102T133000Z')
    return build_base({'stop_after': 'rehearsal', 'after': {'rehearsal': [relabel]}})


def held_manifest_edited():
    rows = BF.manifest_rows('HELD')
    rows[4] = dict(rows[4], prompt_hex=hexs(BI.prompt(*FOREIGN)), expected_hex=hexs(BI.truth(*FOREIGN)))
    return dump_jsonl(rows)


def b_s52():
    """After the official run, a commit edits one item of manifests/HELD.jsonl (its prompt becomes 2+7=); the
    lock's split.held_set_sha256 and split_sha256 are unchanged and no amendment exists (probe F)."""
    def edit(b):
        b.commit('held_edit', {P + 'manifests/HELD.jsonl': held_manifest_edited()}, utc='20261103T100000Z')
    return build_base({'after': {'official': [edit]}})


def b_s53():
    """In every official world the first HELD item (h01) is replaced, in every arm, by the out-of-manifest item
    2+7= (W answers it exactly); OUTPUTS, receipts and the ATTEMPTS shas are consistent with the edited rows, the
    manifest and the lock are unchanged (probe E2)."""
    def patch(r):
        if r['pool'] == 'HELD' and r['item_id'] == 'h01':
            t = BI.truth(*FOREIGN)
            em = t if r['arm'] == 'W' else bytes.fromhex(r['emitted_hex']).decode('ascii')
            stream = BI.stream_of(em)
            cut = stream.find(BI.EOS)
            r = dict(r, prompt_hex=hexs(BI.prompt(*FOREIGN)), expected_hex=hexs(t), stream_hex=hexs(stream),
                     emitted_hex=hexs(stream if cut < 0 else stream[:cut]), own_eos=cut >= 0)
        return r
    return build_base({'row_patch': {'official': patch}})


def b_s54():
    """After the official run, a commit deletes manifests/OLD.jsonl (deletion sweep): the lock still fixes it."""
    def drop(b):
        b.commit('old_delete', {P + 'manifests/OLD.jsonl': None}, utc='20261103T100000Z')
    return build_base({'after': {'official': [drop]}})


def b_s55():
    """After the official run, a commit deletes the committed roster gate report (cycle 2's latest_gate), the
    deletion sweep the verifier ran: CLM-15 (T2) keeps every committed report."""
    def drop(b):
        b.commit('report_delete', {P + 'checks/GATE_roster_20261101T155000Z.json': None}, utc='20261103T100000Z')
    return build_base({'after': {'official': [drop]}})
