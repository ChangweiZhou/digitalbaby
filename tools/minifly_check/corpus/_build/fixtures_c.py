"""Fixture table, part 1: C00a, C00b and the dirty C-series (SPEC 12.2). Each entry names the one
documented mutation applied to the base (FIXTURE.json 'mutation')."""
import json

import base_files as BF
import base_items as BI
import mutate as M
from base_timeline import build_base
from mrk_util import dump_json, dump_jsonl, hexs

D = BF.D
P = D + '/'
LOCK = ['gate', '--stage', 'lock', D]
ROSTER = ['gate', '--stage', 'roster', D]
OFFICIAL = ['gate', '--stage', 'official', D]
PUBLISH = ['gate', '--stage', 'publish', D]
SC_OFF = ['score', D, '--role', 'official']
SC_ADM = ['score', D, '--role', 'admission']
SC_DEV = ['score', D, '--role', 'dev']
CLAIM = ['claim', D]
CANARY = ['canary', D, '--seed', 'c0ffee']


def lp(fn):
    """lock_patch helper: fn(lock) edits the lock in place."""
    return {'lock_patch': fn}


def _set(path, value):
    def fn(lock):
        d = lock
        keys = path.split('.')
        for k in keys[:-1]:
            d = d[k]
        d[keys[-1]] = value
    return fn


def jsonl_map(b, fn):
    rows = [json.loads(x) for x in b.decode('utf-8').splitlines() if x.strip()]
    return dump_jsonl([fn(r) for r in rows])


# ---------------------------------------------------------------------------------------------
def b_c00a():
    return build_base()


def b_c00b():
    wrong = {}
    for w in BI.OFF_WORLDS + BI.DEV_WORLDS + BI.REH_WORLDS + [BI.NATIVE_TECH_WORLD]:
        k = int(w.rsplit('_', 1)[1])
        right = 6 if w == 'w_off_8' else 3
        wrong[w] = ([(k - 1) % 8], list(range(right, 8)))
    return build_base({'behaviour': {'w_held_wrong': wrong}})


def b_c01():
    def patch(row):
        if row['arm'] != 'W':
            return row
        a, b = [int(x) for x in bytes.fromhex(row['prompt_hex']).decode()[:-1].split('+')]
        if a + b >= 10:
            row['expected_hex'] = hexs(str((a + b) % 10))       # truth that drops the carry
            row['emitted_hex'] = row['emitted_hex'][:2]           # decoder keeps the first byte only
        return row
    return build_base({'row_patch': {'dev': patch, 'rehearsal': patch, 'official': patch}})


def b_c02a():
    def rec(r, stage):
        if r['arm'] == 'NPT':
            r['writes_by_channel']['pair_memory'] = 288
            r['channel_digests']['post_first']['pair_memory'] = \
                BI_digest(r['world'], 'W', 'pair_memory')
        return r
    return build_base({'behaviour': {'npt_mode': 'copy_w'}, 'receipt_patch': rec})


def BI_digest(world, arm, ch):
    import base_artifacts as BA
    return BA._dg('post_first', world, arm, ch)


def b_c02b():
    def rec(r, stage):
        if r['arm'] == 'NPT':
            r['realized_dose'] = {'teaching_withheld_fraction': 0.00312}
        return r
    return build_base({'receipt_patch': rec})


def b_c03():
    def rec(r, stage):
        if r['arm'] == 'NPT':
            r['channel_digests']['post_first']['readout'] = BI_digest(r['world'], 'NPT', 'readout')
        return r
    return build_base({'receipt_patch': rec})


def b_c04a():
    def fx(rows):
        out = []
        for r in rows:
            r = dict(r)
            if 'port' in r:
                r['params'] = dict(r['params'], pair_density=0.02)
            if r.get('gate_id') == 'stop_gate' and r['kind'] == 'negative':
                continue
            out.append(r)
        return out
    return build_base({'fixtures_patch': fx})


def b_c04b():
    # SHUF labels: a derangement of the orbit answers (cyclic shift), agreement 0 with the truth.
    orbits = sorted(set(tuple(sorted(p)) for p in BI.TAUGHT_PAIRS))
    ans = [BI.truth(*o) for o in orbits]
    shift = {o: ans[(i + 1) % len(ans)] for i, o in enumerate(orbits)}

    def tp(data, path):
        if not path.endswith('/SHUF.jsonl'):
            return data

        def f(r):
            a, b = [int(x) for x in bytes.fromhex(r['prompt_hex']).decode()[:-1].split('+')]
            r['answer_hex'] = hexs(shift[tuple(sorted((a, b)))])
            return r
        return jsonl_map(data, f)
    return build_base({'taught_patch': tp})


def b_c05a():
    return build_base({'behaviour': {'ceiling_mode': 'fail'}})


def b_c05b():
    return build_base({'behaviour': {'floor_pass': ['FRF_W4096_LR0.02']}})


def b_c05c():
    return build_base(lp(_set('primary.margin_abs', 0.9)))


def b_c06a():
    extra = [(7, 8), (8, 7), (6, 8), (8, 6)]

    def lock(lk):
        lk['primary']['n_items_per_pool']['HELD'] = 20
    return build_base({'bi': {'HELD_S3': list(BI.HELD_S3) + extra},
                       'behaviour': {'w_mode': 'fixed2', 'npt_mode': 'zero'}, 'lock_patch': lock})


def b_c06b():
    s0 = [(0, 3), (4, 0), (0, 7), (6, 0)]
    wrong = {}
    for w in BI.OFF_WORLDS + BI.DEV_WORLDS + BI.REH_WORLDS + [BI.NATIVE_TECH_WORLD]:
        k = int(w.rsplit('_', 1)[1])
        wrong[w] = ([], list(range(1 if k % 2 else 0, 8)))

    def lock(lk):
        lk['primary']['n_items_per_pool']['HELD'] = 20
    return build_base({'bi': {'HELD_EXTRA': s0}, 'lock_patch': lock,
                       'behaviour': {'w_held_wrong': wrong, 'npt_s0_wrong': True}})


def b_c06c():
    return build_base({'behaviour': {'w_old_mode': 'dnc'}})


def b_c07():
    def claim(b):
        return M.claim_doc(b, gate_type='objective')
    return build_base({'lock_patch': _set('primary.unconditional', False), 'claim_doc': claim})


def b_c08a():
    def amend(b):
        old = b.lock_sha
        lock = json.loads(b.files[P + 'ENDPOINT_LOCK.json'].decode())
        lock['task']['id'] = 'add1_parity'
        lock['primary']['task_id'] = 'add1_parity'
        lb = dump_json(lock)
        import hashlib
        row = {'schema': 'minifly.amendment.v1', 'id': 'AMD-1', 'utc': '20261102T103000Z',
               'old_lock_sha256': old, 'new_lock_sha256': hashlib.sha256(lb).hexdigest(),
               'new_protocol_id': lock['protocol_id'], 'fields_changed': ['task.id', 'primary.task_id'],
               'results_seen': ['results/dev/OUTPUTS.jsonl.gz'], 'reason': 'measurement_defect',
               'evidence': 'results/dev/SCORES.json', 'objective_distance_delta': {},
               'original_endpoint_kept_as_coprimary': False, 'user_decision': None,
               'author_agent': 'codex'}
        b.commit('amend', {P + 'ENDPOINT_LOCK.json': lb, P + 'AMENDMENTS.jsonl': dump_jsonl([row])},
                 utc='20261102T103000Z')
    return build_base({'after': {'dev_close': [amend]}, 'stop_after': 'dev_close'})



def b_c08b():
    def edit(b):
        lock = json.loads(b.files[P + 'ENDPOINT_LOCK.json'].decode())
        lock['primary']['margin_abs'] = 0.1
        b.commit('lock_edit', {P + 'ENDPOINT_LOCK.json': dump_json(lock)}, utc='20261101T091500Z')
    return build_base({'after': {'lock': [edit]}, 'stop_after': 'lock'})


def _ledger(rows):
    return {'kb': {'ledger': rows}}


def _lrow(name, utc, level, gate, role, opened, task='addition_single_digit'):
    return {'delivery': name, 'utc': utc, 'legacy': False, 'task_family': task, 'level_earned': level,
            'primary_gate': gate, 'gate_type': 'objective' if level != 'NONE' else 'component',
            'worlds_role': role, 'opened_held_E3': opened,
            'e3_basis': ['pool HELD', 'strata S2-S4'] if opened is True else [],
            'objective_metric': {'held': 'not_measured', 'old_final': 0.9, 'G': 'not_measured'},
            'family_id': 'FAM-0000000%d' % (len(name) % 10), 'claim_sha256': '0' * 64}


def b_c09a():
    rows = [_lrow('minifly_corpus_a_dev_v1_20261020', '20261020T120000Z', 'E1', 'PASS', 'MAIN', False),
            _lrow('minifly_corpus_b_dev_v1_20261025', '20261025T120000Z', 'E1', 'PASS', 'MAIN', False)]
    o = _ledger(rows)
    o.update(lp(_set('claim_level_target', 'E1')))
    o['stop_after'] = 'lock'
    return build_base(o)


def b_c09b():
    rows = [_lrow('minifly_corpus_a_dev_v1_20261010', '20261010T120000Z', 'E3', 'PASS', 'MAIN', True),
            _lrow('minifly_corpus_b_dev_v1_20261020', '20261020T120000Z', 'E1', 'PASS', 'MAIN', False),
            _lrow('minifly_corpus_c_dev_v1_20261025', '20261025T120000Z', 'NONE',
                  'CANNOT_TEST_CLAIM_AS_SPECIFIED', 'MAIN', True)]
    o = _ledger(rows)
    o.update(lp(_set('claim_level_target', 'E1')))
    o['stop_after'] = 'lock'
    return build_base(o)


def b_c10():
    v2 = BF.LEARNER_PY.replace('Corpus stub learner', 'Corpus stub learner (revision 2)').encode()
    v3 = BF.LEARNER_PY.replace('Corpus stub learner', 'Corpus stub learner (revision 3)').encode()
    worlds3 = BI.DEV_WORLDS + BI.REH_WORLDS
    return build_base({'dev_redesigns': [v2, v3], 'dev_redesign_worlds': {3: worlds3}})


def b_c11a():
    other = {'minifly_corpus_other_dev_20261020/WORLDS.json':
             dump_json({'note': 'earlier delivery world list', 'worlds': ['w_dev_9', 'w_off_3']})}
    return build_base({'extra_files': {'kb': other}})


def b_c11b():
    body = ('Sealed confirmation passed: the confirmed E3 result holds on 8 worlds.\n')
    return build_base({'lock_patch': _set('worlds.official_role_intended', 'SEALED_CONFIRM'),
                       'report_body': body})


def b_c12():
    def lock(lk):
        lk['exposure_streams'].append({'path_glob': 'manifests/CALIBRATION/*/*.jsonl',
                                       'feedback_bearing': True, 'phase': 'calibration'})
    cal = dump_jsonl([{'world': BI.NATIVE_TECH_WORLD, 'arm': 'W', 'prompt_hex': hexs('8+2='),
                       'answer_hex': hexs('10'), 'phase': 'calibration', 'feedback_bearing': True,
                       'presentations': 2}])
    extra = dump_jsonl([{'world': BI.NATIVE_TECH_WORLD, 'arm': 'W', 'prompt_hex': hexs('1+5='),
                         'answer_hex': hexs('6'), 'phase': 'warmup', 'feedback_bearing': True,
                         'presentations': 1}])
    return build_base({'lock_patch': lock, 'extra_lock_files': {
        P + 'manifests/CALIBRATION/w_tech_1/W.jsonl': cal,
        P + 'manifests/EXTRA/w_tech_1/W.jsonl': extra}})


def b_c13a():
    tl, b = build_base()
    M.merge_into(tl, 'lock', 'testbed')
    return tl, b


def b_c13b():
    learner = BF.LEARNER_PY.replace('import hashlib\n', 'import hashlib\n\nimport bootstrap\n').encode()
    boot = b'"""Corpus stub: helper imported by learner.py but left out of SOURCE_LOCK."""\nSEED = 7\n'
    return build_base({'learner_py': learner, 'extra_files': {'learner': {P + 'bootstrap.py': boot}}})


def b_c13c():
    tl, b = build_base()
    M.move_after(tl, 'review', 'dev', '20261102T093000Z')
    M.fix_branches(tl)
    return tl, b


def b_c14():
    audit = (b'"""Corpus stub audit helper (never executed)."""\n\n\ndef summary(rows):\n'
             b'    exposure_overlap = 0\n    return {"rows": len(rows), "exposure_overlap": exposure_overlap}\n')
    return build_base({'lock_patch': _set('primary.rule_type', 'custom'),
                       'extra_lock_files': {P + 'audit_results.py': audit}})


def b_c15():
    def edit(b):
        new = BF.ADMIT_PY.replace('never executed)."""', 'never executed).\n\nEdited after the '
                                  'testbed gate failed; no amendment recorded."""').encode()
        import hashlib
        b.admit_sha = hashlib.sha256(new).hexdigest()
        b.commit('admit_edit', {P + 'admit.py': new}, utc='20261101T103000Z')
    return build_base({'after': {'testbed': [edit]}})


def b_c16():
    def tech(row):
        if row['arm'] == 'W' and row['pool'] == 'TAUGHT':
            row['stream_hex'] = hexs('0\n\n\n')
            row['emitted_hex'] = hexs('0')
        return row

    def adm(a):
        a['input_contrast'].update({'analytic_snr': 0.027, 'ratio': 0.014,
                                    'same_input_diff_noise': 0.41, 'diff_input_same_noise': 0.0057,
                                    'prompt_id_loo_acc': 0.07})
    return build_base({'lock_patch': _set('admission_thresholds.analytic_input_snr', 0.027),
                       'admission_patch': adm, 'row_patch': {'technical': tech}})


def b_c17():
    def fx(rows):
        out = []
        for r in rows:
            r = dict(r)
            if r.get('sign') == '+':
                r['target_margin_delta_vs_passive'] = -0.012
            if r.get('sign') == '-':
                r['zero_replay_applications'] = 32
            out.append(r)
        return out
    return build_base({'fixtures_patch': fx})


def b_c18():
    sib = 'minifly_corpus_sibling_dev_20261015'
    learner = BF.LEARNER_PY.replace('import hashlib\n', "import hashlib\nimport sys\n\nsys.path.insert(0, "
                                    "'../%s')\nimport core\n" % sib).encode()
    core = b'"""Sibling delivery core (corpus stub)."""\nWIDTH = 4\n'
    return build_base({'learner_py': learner, 'extra_files': {'kb': {sib + '/core.py': core}}})


def b_c19():
    return build_base({'drop_produced': {'official': ['results/official/OUTPUTS.jsonl.gz']}})


def b_c20():
    def rec(r, stage):
        if r['world'] == 'w_off_5' and r['arm'] == 'NPT':
            return False
        return r

    def rows(row):
        if row['world'] == 'w_off_5' and row['arm'] == 'NPT':
            return None
        return row
    return build_base({'lock_patch': _set('schedule.paired_atomic', False), 'receipt_patch': rec,
                       'row_patch': {'official': rows}})
