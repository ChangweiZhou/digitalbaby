"""Unit tests for scorer.py (WP3). Synthetic deliveries in temp dirs (TimelineGit); stdlib only.

Hand-computed reference design (single-digit addition, gen_cap 4, eos 0x0a):
- HELD H = {(a, b): a in 1..5, b in 5..8} (20 items). TAUGHT T = all (x, y) in 1..9 x 1..9 minus H minus the
  reversed H. Every held item is determined by the rectangle a' = 9, b' = 9; no held orbit is taught (no S1);
  10 items have no carry (S2, answers 6..9) and 10 carry (S3, answers 10..13).
- The best shortcut on HELD is DIGIT_NO_CARRY: exact on the 10 S2 items, wrong on the 10 S3 items -> 0.5. A
  constant answer hits at most 4 items (0.2); FIXED_1 / FIXED_2 are bounded by the candidate's own correct
  1- / 2-digit items (<= 0.5).
- Candidate W is wrong on [1,2,3,2,1,2,3,2] held items in the 8 official worlds -> rates 0.95..0.85, mean 0.9;
  NPT is right on items 0 and 10 only (0.1). d_w = rate - 0.5, mean 0.4, sd = sqrt(0.01/7) = 0.03780,
  LCB = 0.4 - 1.8946 * 0.03780 / sqrt(8) = 0.3747 > 0.05 -> PASS. G = 0.4.
- OLD (10 items: 5 one-digit no-carry, 5 two-digit carry): W wrong on one 2-digit item at old_end and final ->
  0.9 / 0.9, drop 0; best exposure-independent shortcut 0.5 (DIGIT_NO_CARRY = FIXED_1) -> acq 0.4.
"""
import argparse
import contextlib
import gzip
import io
import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]
if str(KIT) not in sys.path:
    sys.path.insert(0, str(KIT))

import common  # noqa: E402
import decide  # noqa: E402
import scorer  # noqa: E402

NAME = 'minifly_wp3fx_dev_v1_20261101'
EOS = b'\n'
CAP = 4
FMT = {'kind': 'addition', 'sep_hex': '2b', 'terminator_hex': '3d', 'eos_hex': '0a', 'gen_cap_bytes': CAP,
       'normalize_leading_zeros': True}
OFFICIAL = ['wo%d' % i for i in range(1, 9)]
DEV = ['wd1', 'wd2']
REH = ['wr1']
TECH = ['wt1']
CONSTRAINTS = {
    'schema': 'minifly.constraints.v1',
    'tier_thresholds': {'t0_max_cpu_h': 1, 't1_max_cpu_h': 4, 't1_max_records': 20000, 'family_t1_max_cpu_h': 12},
    'threshold_floors': {'margin_abs': 0.05, 'min_items_per_stratum': 8, 'oc_min_power': 0.8, 'min_worlds_nonpilot': 8},
    'default_grant_caps': {'max_cpu_h': 2, 'max_records': 5000, 'max_cycles': 3},
}

H = [(a, b) for a in range(1, 6) for b in range(5, 9)]
T_ALL = [(x, y) for x in range(1, 10) for y in range(1, 10) if (x, y) not in H and (y, x) not in H]
OLD_ITEMS = [(1, 1), (1, 2), (2, 1), (1, 3), (3, 1), (9, 1), (9, 2), (9, 3), (1, 9), (2, 9)]
TAUGHT_ITEMS = [(1, 4), (4, 1), (2, 2), (2, 3), (3, 2), (9, 4), (9, 5), (9, 6), (3, 9), (4, 9)]
NEW_ITEMS = [(2, 4), (4, 2), (3, 3), (4, 4), (3, 4), (6, 9), (7, 9), (8, 9), (9, 7), (9, 8)]
POOL_ITEMS = {'HELD': H, 'OLD': OLD_ITEMS, 'TAUGHT': TAUGHT_ITEMS, 'NEW': NEW_ITEMS}
W_WRONG = {'wo1': 1, 'wo2': 2, 'wo3': 3, 'wo4': 2, 'wo5': 1, 'wo6': 2, 'wo7': 3, 'wo8': 2, 'wd1': 1, 'wd2': 2,
           'wr1': 1, 'wt1': 1}


def ph(a, b):
    return ('%d+%d=' % (a, b)).encode('ascii').hex()


def truth(a, b):
    return str(a + b).encode('ascii')


def stream_of(content: bytes, eos: bool = True) -> bytes:
    s = content + (EOS if eos else b'')
    pad = EOS if eos else b'z'
    s = (s + pad * CAP)[:CAP]
    return s


def row(world, role, arm, pool, ck, iid, prompt_hex, expected: bytes, stream: bytes, answer_path='native', **over):
    pos = stream.find(EOS)
    em = stream[:pos] if pos >= 0 else stream
    r = {'world': world, 'world_role': role, 'arm': arm, 'pool': pool, 'checkpoint': ck, 'item_id': iid,
         'prompt_hex': prompt_hex, 'expected_hex': expected.hex(), 'stream_hex': stream.hex(), 'emitted_hex': em.hex(),
         'own_eos': pos >= 0, 'answer_path': answer_path, 'native_closure_sha': 'n' * 64, 'params_sha256': 'p' * 64,
         'attempt_id': 'ATT-1'}
    r.update(over)
    return r


def wrong_of(exp: bytes) -> bytes:
    return b'5' if exp != b'5' else b'4'


def base_lock(**over):
    lock = {
        'schema': common.SCHEMA_ENDPOINT_LOCK, 'delivery': NAME, 'protocol_id': NAME + '-p1', 'author_agent': 'claude',
        'tier_declared': 1, 'intent': 'diagnostic', 'claim_level_target': 'E3',
        'task': {'id': 'add1', 'family': 'addition_single_digit', 'format': dict(FMT)},
        'units': {'witness': 'pair_rectangle', 'commutative': False},
        'split': {'taught_manifest': 'manifests/TAUGHT', 'held_manifest': 'manifests/HELD.jsonl',
                  'old_manifest': 'manifests/OLD.jsonl', 'new_manifest': 'manifests/NEW.jsonl'},
        'arms': [{'id': 'W', 'role': 'candidate', 'zero_channels': []},
                 {'id': 'NPT', 'role': 'no_prior_teaching', 'zero_channels': ['pair_memory']},
                 {'id': 'SHUF', 'role': 'shuf', 'zero_channels': []}],
        'primary': {'task_id': 'add1', 'metric': 'exact_own_eos', 'unconditional': True, 'pool': 'HELD',
                    'checkpoint': 'final', 'strata_counted': ['S2', 'S3'], 'candidate_arm': 'W',
                    'comparator_arms': ['NPT'], 'ladder': 'auto', 'chance': 0.1, 'margin_abs': 0.05,
                    'absolute_floor': 0.15, 'per_world_floor': -0.05, 'aggregator': {'type': 'paired_lcb', 'alpha': 0.05},
                    'min_worlds': 8, 'n_items_per_pool': {'HELD': 20, 'OLD': 10, 'TAUGHT': 10, 'NEW': 10},
                    'rule_type': 'declarative'},
        'gates': [],
        'guards': [{'id': 'old_drop', 'kind': 'old_drop', 'pool': 'OLD', 'max_drop': 0.1},
                   {'id': 'old_acq', 'kind': 'old_acq', 'pool': 'OLD', 'margin': 0.05}],
        'udi': None,
        'checkpoints': {'prefix': 100, 'old_end': 200, 'final': 300},
        'worlds': {'technical': list(TECH), 'dev': list(DEV), 'rehearsal': list(REH),
                   'official': {'mode': 'listed', 'ids': list(OFFICIAL), 'n': 8}, 'official_role_intended': 'MAIN'},
        'budget': {'max_records': 1000, 'max_cpu_h': 1},
        'admission_thresholds': {'taught_acq_margin_min': 0.05},
        'testbed': {'ceiling_grid': [{'id': 'EXACT_BP', 'learner': 'EXACT_BP'}],
                    'floor_grid': [{'id': 'FRF', 'learner': 'FROZEN_RANDOM_FEATURES'}]},
        'files': {'admit': 'admit.py', 'native': ['native.py'], 'runner': ['run.py'], 'params': 'params.json'},
    }
    for k, v in over.items():
        lock[k] = v
    return lock


def jsonl(rows):
    return ''.join(json.dumps(r, sort_keys=True) + '\n' for r in rows)


def write(path, data):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, (dict, list)):
        data = json.dumps(data, indent=1)
    if isinstance(data, str):
        data = data.encode('utf-8')
    p.write_bytes(data)
    return p


def arm_behaviour(world, arm, pool, ck, idx, a, b):
    """(content_ok, eos_ok) for the reference design."""
    exp = truth(a, b)
    if arm == 'W':
        if pool == 'HELD':
            return idx not in set(range(19, 19 - W_WRONG[world], -1)), True
        if pool == 'OLD':
            return idx != 9, True
        return idx != 0, True
    if arm == 'NPT':
        return idx % 10 == 0, True
    if arm == 'SHUF':
        return idx % 5 == 0, True
    return False, True


def make_rows(worlds, role, arms=('W', 'NPT', 'SHUF'), pools=('HELD', 'OLD', 'TAUGHT', 'NEW'),
              cks=('prefix', 'old_end', 'final'), behaviour=arm_behaviour):
    rows = []
    for w in worlds:
        for arm in arms:
            for pool in pools:
                for ck in cks:
                    for idx, (a, b) in enumerate(POOL_ITEMS[pool]):
                        exp = truth(a, b)
                        ok, eos = behaviour(w, arm, pool, ck, idx, a, b)
                        content = exp if ok else wrong_of(exp)
                        rows.append(row(w, role, arm, pool, ck, '%s%02d' % (pool[0].lower(), idx), ph(a, b), exp,
                                        stream_of(content, eos)))
    return rows


def taught_rows(world, arm, pairs=None, label=None):
    out = []
    for (x, y) in (pairs if pairs is not None else T_ALL):
        ans = label(x, y) if label else truth(x, y)
        out.append({'world': world, 'arm': arm, 'prompt_hex': ph(x, y), 'answer_hex': ans.hex(), 'phase': 'teach',
                    'feedback_bearing': True, 'presentations': 12})
    return out


class DeliveryCase(unittest.TestCase):
    """A v1 delivery inside a TimelineGit fixture (no real git needed)."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix='mfc_wp3_'))
        self.fx = self.tmp / 'fx'
        self.root = self.fx / 'repo'
        self.d = self.root / NAME
        write(self.root / 'PROJECT_DOCUMENTS' / 'KB' / 'CONSTRAINTS.json', CONSTRAINTS)
        self.set_lock(base_lock())
        write(self.d / 'SOURCE_LOCK.json', {'schema': common.SCHEMA_SOURCE_LOCK, 'delivery': NAME,
                                            'checker_version': common.CHECKER_VERSION, 'closure_sha256': {},
                                            'native_closure_sha': 'n' * 64, 'params_sha256': 'p' * 64})
        write(self.fx / 'timeline.json', {'head': 'a' * 40, 'commits': [
            {'sha': 'a' * 40, 'utc': '20261101T000000Z', 'author': 't', 'trailers': {}, 'branches': [],
             'paths': [NAME + '/ENDPOINT_LOCK.json', 'PROJECT_DOCUMENTS/KB/CONSTRAINTS.json']}]})
        self._orig_ctrl = scorer._adm_ctrl_status
        scorer._adm_ctrl_status = lambda ctx: ('PASS', 'test stub: ADM-CTRL established by WP4 tests')

    def tearDown(self):
        scorer._adm_ctrl_status = self._orig_ctrl
        shutil.rmtree(self.tmp, ignore_errors=True)

    def set_lock(self, lock):
        self.lock = lock
        write(self.d / 'ENDPOINT_LOCK.json', lock)

    def ctx(self, stage=None):
        return common.Ctx(repo_root=self.root, delivery=self.d, git=common.TimelineGit(self.fx), kb=common.KB(self.root),
                          rules=common.load_rules(), stage=stage)

    def write_rows(self, role, rows, gz=False):
        rel = 'results/%s/OUTPUTS.jsonl' % role
        if gz:
            p = self.d / (rel + '.gz')
            p.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(str(p), 'wb') as fh:
                fh.write(jsonl(rows).encode('utf-8'))
        else:
            write(self.d / rel, jsonl(rows))

    def write_taught(self, worlds, arms=('W', 'SHUF'), pairs=None, label=None):
        for w in worlds:
            for a in arms:
                lab = label if (label and a == 'SHUF') else None
                write(self.d / 'manifests' / 'TAUGHT' / w / ('%s.jsonl' % a), jsonl(taught_rows(w, a, pairs, lab)))

    def write_manifests(self):
        for pool, key in (('HELD', 'HELD'), ('OLD', 'OLD'), ('NEW', 'NEW')):
            write(self.d / 'manifests' / ('%s.jsonl' % key), jsonl(
                [{'item_id': '%s%02d' % (pool[0].lower(), i), 'prompt_hex': ph(a, b), 'expected_hex': truth(a, b).hex()}
                 for i, (a, b) in enumerate(POOL_ITEMS[pool])]))

    def write_testbed(self, ceiling_ok=True, floor_ok=False, items=None):
        """Testbed rows (HELD/final, technical worlds): items = [(prompt_hex, expected bytes)]."""
        items = items if items is not None else [(ph(a, b), truth(a, b)) for (a, b) in H]
        rows = []
        for w in TECH:
            for arm, good in (('CEILING_EXACT_BP', ceiling_ok), ('FLOOR_FRF', floor_ok)):
                for idx, (p_hex, exp) in enumerate(items):
                    ok = good or idx % 10 == 0
                    rows.append(row(w, 'technical', arm, 'HELD', 'final', 'h%02d' % idx, p_hex, exp,
                                    stream_of(exp if ok else wrong_of(exp)), answer_path='evaluator',
                                    native_closure_sha=None, params_sha256=None))
        return rows

    def full(self, official=True, testbed=True):
        """The reference delivery: TAUGHT manifests, admission (testbed + technical), dev, rehearsal, official."""
        self.write_taught(TECH + DEV + REH + OFFICIAL)
        self.write_manifests()
        adm = make_rows(TECH, 'technical') + (self.write_testbed() if testbed else [])
        self.write_rows('admission', adm)
        self.write_rows('dev', make_rows(DEV, 'dev'))
        self.write_rows('rehearsal', make_rows(REH, 'rehearsal'))
        if official:
            self.write_rows('official', make_rows(OFFICIAL, 'official'))


# ------------------------------------------------------------------------------------------------
class TestScoreRows(unittest.TestCase):
    def lock(self):
        return base_lock()

    def taught(self, worlds):
        return {(w, 'W'): taught_rows(w, 'W') for w in worlds}

    def test_recompute_and_met_mismatches(self):
        lock = self.lock()
        good = row('wo1', 'official', 'W', 'HELD', 'final', 'h00', ph(1, 5), truth(1, 5), stream_of(b'6'))
        bad_em = dict(good, item_id='h01', prompt_hex=ph(1, 6), expected_hex=truth(1, 6).hex(),
                      stream_hex=stream_of(b'7').hex(), emitted_hex='37' + '0a')
        bad_eos = dict(good, item_id='h02', prompt_hex=ph(1, 7), expected_hex=truth(1, 7).hex(),
                       stream_hex=stream_of(b'8').hex(), emitted_hex='38', own_eos=False)
        short = dict(good, item_id='h03', prompt_hex=ph(1, 8), expected_hex=truth(1, 8).hex(),
                     stream_hex=b'9\n'.hex(), emitted_hex='39', own_eos=True)
        no_eos = row('wo1', 'official', 'W', 'HELD', 'final', 'h04', ph(2, 5), truth(2, 5), stream_of(b'7', eos=False))
        head = row('wo1', 'official', 'W', 'HELD', 'final', 'h05', ph(2, 6), truth(2, 6), stream_of(b'8'),
                   answer_path='head')
        # dropped carry in the runner's expected_hex (C01): 2+8 -> '0'
        carry = row('wo1', 'official', 'W', 'HELD', 'final', 'h06', ph(2, 8), truth(2, 8), stream_of(b'10'),
                    expected_hex=b'0'.hex())
        c = scorer.score_rows([good, bad_em, bad_eos, short, no_eos, head, carry], lock, self.taught(['wo1']))
        counts = c['met_mismatch_counts']
        self.assertEqual(counts.get('emitted_hex'), 1)
        self.assertEqual(counts.get('own_eos'), 1)
        self.assertEqual(counts.get('stream_len'), 1)
        self.assertEqual(c['met02_mismatch_count'], 1)
        self.assertEqual(c['met02_mismatches'][0]['recomputed'], '3130')
        self.assertEqual(c['non_native_rows'], 1)
        cells = [x for x in c['cells'] if x['arm'] == 'W']
        # exact: good (h00), bad_em and bad_eos (recomputed values are right), carry (truth recomputed) = 4;
        # short stream, no own EOS and answer_path head never count
        self.assertEqual(sum(x['n_exact'] for x in cells), 4)
        self.assertEqual(sum(x['n_rows'] for x in cells), 7)

    def test_conflicts_raise(self):
        lock = self.lock()
        r1 = row('wo1', 'official', 'W', 'HELD', 'final', 'h00', ph(1, 5), truth(1, 5), stream_of(b'6'))
        with self.assertRaises(common.InvalidInput):
            scorer.score_rows([r1, dict(r1)], lock, self.taught(['wo1']))
        r2 = dict(r1, arm='NPT', prompt_hex=ph(1, 6))
        with self.assertRaises(common.InvalidInput):
            scorer.score_rows([r1, r2], lock, self.taught(['wo1']))
        with self.assertRaises(common.InvalidInput):
            scorer.score_rows([dict(r1, pool='NOPE')], lock, self.taught(['wo1']))
        with self.assertRaises(common.InvalidInput):
            scorer.score_rows([dict(r1, checkpoint='middle')], lock, self.taught(['wo1']))
        bad_fmt = base_lock(task={'id': 'x', 'family': 'addition_single_digit',
                                  'format': dict(FMT, normalize_leading_zeros=False)})
        with self.assertRaises(common.InvalidInput):
            scorer.score_rows([r1], bad_fmt, self.taught(['wo1']))

    def test_reference_ladder_strata_and_cells(self):
        lock = self.lock()
        rows = make_rows(['wo1'], 'official')
        c = scorer.score_rows(rows, lock, self.taught(['wo1']))
        lad = c['ladder']['wo1|HELD|final']
        self.assertAlmostEqual(lad['DIGIT_NO_CARRY'], 0.5)
        self.assertLessEqual(lad['CONST'], 0.2)
        self.assertEqual(c['best_shortcut']['name'], 'DIGIT_NO_CARRY')
        self.assertAlmostEqual(c['best_shortcut']['value'], 0.5)
        # exposure-independent subset only on OLD/TAUGHT/NEW
        self.assertNotIn('EXACT', c['ladder']['wo1|OLD|old_end'])
        self.assertNotIn('NEAREST', c['ladder']['wo1|TAUGHT|prefix'])
        self.assertIn('NEAREST', lad)
        held_w = [x for x in c['cells'] if x['arm'] == 'W' and x['pool'] == 'HELD' and x['checkpoint'] == 'final']
        by = {(x['stratum'], x['determined']): x for x in held_w}
        self.assertEqual(by[('S2', True)]['n'], 10)
        self.assertEqual(by[('S3', True)]['n'], 10)
        self.assertEqual(sum(x['n_exact'] for x in held_w), 19)  # W wrong on 1 item in wo1
        # the OLD pool: DIGIT_NO_CARRY on the 5 no-carry items, FIXED_1 on W's 5 correct 1-digit items
        old = c['ladder']['wo1|OLD|old_end']
        self.assertAlmostEqual(old['DIGIT_NO_CARRY'], 0.5)
        self.assertAlmostEqual(old['FIXED_1'], 0.5)
        self.assertAlmostEqual(old['FIXED_2'], 0.4)

    def test_balanced_metric_cells_carry_answers(self):
        lock = base_lock()
        lock['primary']['metric'] = 'balanced_exact_own_eos'
        c = scorer.score_rows(make_rows(['wo1'], 'official', arms=('W',), pools=('HELD',), cks=('final',)), lock,
                              self.taught(['wo1']))
        self.assertTrue(all('answer' in x for x in c['cells']))
        idx = decide.index_cells(c['cells'])
        v, n, _ = decide.group_rate(idx[('wo1', 'W', 'HELD', 'final')], decide.counted_filter(lock),
                                    'balanced_exact_own_eos')
        # W is wrong on h19 = (5, 8) -> answer 13 is the only item with answer 13: balanced rate = 7/8
        self.assertAlmostEqual(v, 7.0 / 8.0)

    def test_nearest_and_fixed(self):
        ti = scorer.TaughtIndex([{'prompt_hex': ph(1, 2), 'answer_hex': b'3'.hex()},
                                 {'prompt_hex': ph(4, 4), 'answer_hex': b'8'.hex()},
                                 {'prompt_hex': ph(1, 3), 'answer_hex': b'4'.hex()}], FMT, 't')
        self.assertEqual(ti.nearest(bytes.fromhex(ph(1, 9))), b'3')  # ties (1+2=, 1+3=) -> smallest prompt
        self.assertEqual(ti.nearest(bytes.fromhex(ph(5, 4))), b'8')
        self.assertEqual(ti.const, b'3')  # ties -> lexicographically smallest
        p = ti.predictions(ph(0, 7), bytes.fromhex(ph(0, 7)), True)
        self.assertEqual(p['ID_ZERO'], b'7')
        self.assertEqual(p['DIGIT_NO_CARRY'], b'7')
        self.assertEqual(scorer._digit_no_carry('15', '7'), '12')
        self.assertEqual(scorer._digit_no_carry('5', '5'), '0')


class TestScoreDelivery(DeliveryCase):
    def test_reference_official_verdict_pass_and_files(self):
        self.full()
        ctx = self.ctx()
        res = scorer.score(ctx, 'official')
        s = res['scores']
        self.assertAlmostEqual(s['G'], 0.4)
        self.assertEqual(s['best_shortcut']['name'], 'DIGIT_NO_CARRY')
        self.assertEqual(s['strata'], {'S2': 80, 'S3': 80})
        self.assertEqual(s['determinacy'], {'determined': 160, 'total': 160})
        self.assertAlmostEqual(s['old']['end'], 0.9)
        self.assertAlmostEqual(s['old']['drop'], 0.0)
        self.assertAlmostEqual(s['old']['acq'], 0.4)
        self.assertAlmostEqual(s['held_exact_own_eos'], 0.9)
        self.assertEqual(s['inert_alarms'], [])
        self.assertEqual(s['met_mismatches'], [])
        self.assertEqual(s['designs_tried_k'], 1)
        v = res['verdict']
        self.assertEqual(v['verdict'], 'PASS', v['reasons'])
        self.assertAlmostEqual(v['primary']['mean_cand'], 0.9)
        self.assertAlmostEqual(v['primary']['mean_comp'], 0.5)
        self.assertAlmostEqual(v['primary']['lcb'], 0.4 - 1.8946 * (0.01 / 7) ** 0.5 / 8 ** 0.5, places=3)
        self.assertEqual(v['guards'], {'old_drop': 'PASS', 'old_acq': 'PASS'})
        self.assertTrue((self.d / 'results/official/VERDICT.json').is_file())
        self.assertTrue((self.d / 'results/official/SCORES.json').is_file())
        stored = json.loads((self.d / 'results/official/VERDICT.json').read_text())
        self.assertEqual(stored['cells_sha256'], v['cells_sha256'])
        # DEC-01: the stored verdict equals a fresh decide
        self.assertEqual(decide.check_dec_01(self.ctx('publish')).status, common.PASS)
        self.assertEqual(stored['source_lock_sha256'], common.sha256_file(self.d / 'SOURCE_LOCK.json'))
        stored['verdict'] = 'FAIL'
        write(self.d / 'results/official/VERDICT.json', stored)
        r = decide.check_dec_01(self.ctx('publish'))
        self.assertEqual(r.status, common.FAIL)
        self.assertIn('verdict', r.detail)
        # a VERDICT without lock shas is never accepted, and a rebuilt SOURCE_LOCK invalidates the old VERDICT
        scorer.score(self.ctx(), 'official')
        (self.d / 'SOURCE_LOCK.json').write_text('{"schema": "minifly.source_lock.v1", "rebuilt": true}')
        r = decide.check_dec_01(self.ctx('publish'))
        self.assertEqual(r.status, common.FAIL)
        self.assertIn('source_lock_sha256', r.detail)
        (self.d / 'SOURCE_LOCK.json').unlink()
        scorer.score(self.ctx(), 'official')
        r = decide.check_dec_01(self.ctx('publish'))
        self.assertEqual(r.status, common.FAIL)
        self.assertIn('lacks source_lock_sha256', r.detail)

    def test_held_answer_in_any_stream_makes_the_orbit_s1(self):
        # SPEC 9.3 as amended in 20.11 (round 1, INC-HB1-A3): a held orbit fed with its answer in ANY stream
        # (here a declared non-feedback listening stream) is S1 (taught) in every world and never counted.
        self.full()
        lock = dict(self.lock, exposure_streams=[
            {'path_glob': 'manifests/TAUGHT/*/*.jsonl', 'feedback_bearing': True, 'phase': 'teach'},
            {'path_glob': 'manifests/LISTEN/*.jsonl', 'feedback_bearing': False, 'phase': 'listening'}])
        self.set_lock(lock)
        s0 = scorer.score(self.ctx(), 'official')['scores']
        self.assertEqual(s0['strata'], {'S2': 80, 'S3': 80})
        a, b = H[0]
        # held prompts alone are not exposure: the strata do not move
        write(self.d / 'manifests/LISTEN/l.jsonl', jsonl([{'utterance': '%d+%d=' % (b, a)}]))
        s1 = scorer.score(self.ctx(), 'official')['scores']
        self.assertEqual(s1['strata'], {'S2': 80, 'S3': 80})
        # the commuted orbit with its answer, in a field that is neither prompt_hex nor answer_hex
        write(self.d / 'manifests/LISTEN/l.jsonl', jsonl([{'utterance': '%d+%d=%d' % (b, a, a + b)}]))
        ctx = self.ctx()
        sr = scorer.score_role(ctx, 'official')
        self.assertEqual(sr['exposed_orbits'], [(str(a), str(b))])
        s2 = scorer.score(self.ctx(), 'official')['scores']
        # one item per world moves to S1 and leaves the counted strata
        self.assertEqual(s2['strata'].get('S1'), len(OFFICIAL))
        self.assertEqual(s2['strata'].get('S2', 0) + s2['strata'].get('S3', 0), 160 - len(OFFICIAL))
        self.assertEqual(s2['determinacy']['total'], 160 - len(OFFICIAL))
        held_w = [x for x in sr['cells']['cells'] if x['arm'] == 'W' and x['pool'] == 'HELD' and x['world'] == 'wo1'
                  and x['checkpoint'] == 'final']
        self.assertEqual(sum(x['n'] for x in held_w if x['stratum'] == 'S1'), 1)

    def test_admission_testbed_acquisition_shuf(self):
        self.full(official=False)
        self.write_taught(TECH, arms=('SHUF',), label=lambda x, y: str((x + y) % 7 + 1).encode())
        res = scorer.score(self.ctx(), 'admission')
        tb = res['testbed']
        self.assertEqual(tb['result'], 'PASS', tb['reason'])
        self.assertTrue(tb['ceiling_any_pass'])
        self.assertTrue(tb['floors_all_fail'])
        self.assertEqual(tb['best_floor']['arm'], 'FLOOR_FRF')
        self.assertTrue((self.d / 'results/admission/TESTBED.json').is_file())
        acq = res['scores']['taught_acquisition']
        self.assertEqual(acq['status'], 'PASS')  # W 0.9 on TAUGHT/prefix vs best 0.5 + 0.05
        sh = res['scores']['shuf']
        self.assertEqual(sh['status'], 'OK')
        wt = [a for a in sh['arms'] if a['world'] == 'wt1'][0]
        self.assertEqual(wt['agreement'], 0.0)  # label (x+y)%7+1 never equals x+y
        self.assertTrue(wt['orbit_consistent'])  # the label depends on x+y only
        wd = [a for a in sh['arms'] if a['world'] == 'wd1'][0]
        self.assertEqual(wd['agreement'], 1.0)

    def test_testbed_cannot_test_when_ceiling_fails_or_floor_passes(self):
        self.write_taught(TECH)
        self.write_manifests()
        self.write_rows('admission', self.write_testbed(ceiling_ok=False))
        tb = scorer.testbed(self.ctx())
        self.assertEqual(tb['result'], 'CANNOT_TEST_CLAIM_AS_SPECIFIED')
        self.assertFalse(tb['ceiling_any_pass'])
        self.write_rows('admission', self.write_testbed(ceiling_ok=True, floor_ok=True))
        tb = scorer.testbed(self.ctx())
        self.assertEqual(tb['result'], 'CANNOT_TEST_CLAIM_AS_SPECIFIED')
        self.assertFalse(tb['floors_all_fail'])

    def test_official_without_testbed_is_incomplete(self):
        self.full(testbed=False)
        v = scorer.score(self.ctx(), 'official')['verdict']
        self.assertEqual(v['verdict'], 'INCOMPLETE')
        self.assertEqual(v['shadow_verdict'], 'PASS')

    def test_dev_decision_is_dev_not_evidence(self):
        self.full(official=False)
        s = scorer.score(self.ctx(), 'dev')['scores']
        self.assertEqual(s['decision']['verdict'], 'DEV_NOT_EVIDENCE')
        self.assertEqual(s['decision']['shadow_verdict'], 'FAIL')  # 2 dev worlds < min_worlds 8
        self.assertTrue(any('min_worlds' in r for r in s['decision']['reasons']))
        self.assertEqual(s['weak_comparator'], ['NPT'])
        self.assertFalse((self.d / 'results/dev/VERDICT.json').exists())


class TestStopOnlyAndZeroOperand(DeliveryCase):
    def test_stop_only_like_fixed1_beats_candidate(self):
        """Non-addition recall task with mixed answer lengths (items 0-9 answer one byte, 10-19 two bytes). The
        candidate has the right first byte everywhere and always stops after one byte (except item 0 in odd worlds):
        it is exact on the 10 one-byte items (9 in odd worlds) -> 0.5 / 0.45. FIXED_1 of its own stream is 0.5 in
        every world; DIRECT (the comparator) is at 0. Verdict FAIL, best shortcut FIXED_1. (With a single answer
        length FIXED_L of the candidate's own stream bounds its own-EOS exactness, so even the ceiling cannot pass.)"""
        fmt = {'kind': 'other', 'eos_hex': '0a', 'gen_cap_bytes': CAP}
        lock = base_lock(task={'id': 'recall', 'family': 'non_arithmetic', 'format': fmt}, units=None,
                         arms=[{'id': 'W', 'role': 'candidate'}, {'id': 'DIRECT', 'role': 'other'}],
                         guards=[], split={'taught_manifest': 'manifests/TAUGHT'})
        lock['primary'].update({'comparator_arms': ['DIRECT'], 'n_items_per_pool': {'HELD': 20},
                                'strata_counted': [], 'chance': 0.0, 'absolute_floor': 0.05})
        self.set_lock(lock)
        rows = []
        letters = 'abcdefghijklmnopqrst'
        items = [(('k%s?' % ch).encode().hex(), (str(i) if i < 10 else '1%d' % (i - 10)).encode())
                 for i, ch in enumerate(letters)]
        for wi, w in enumerate(OFFICIAL):
            for i, (prompt, exp) in enumerate(items):
                stop_ok = not (wi % 2 == 1 and i == 0)
                st = stream_of(exp[:1], eos=stop_ok)
                rows.append(row(w, 'official', 'W', 'HELD', 'final', 'h%02d' % i, prompt, exp, st))
                rows.append(row(w, 'official', 'DIRECT', 'HELD', 'final', 'h%02d' % i, prompt, exp, stream_of(b'y')))
        self.write_rows('official', rows)
        for w in OFFICIAL + TECH:
            write(self.d / 'manifests/TAUGHT' / w / 'W.jsonl', jsonl(
                [{'world': w, 'arm': 'W', 'prompt_hex': ('t%d?' % j).encode().hex(), 'answer_hex': b'Q'.hex(),
                  'phase': 'teach', 'feedback_bearing': True, 'presentations': 1} for j in range(5)]))
        self.write_rows('admission', self.write_testbed(items=items))
        ctx = self.ctx()
        res = scorer.score(ctx, 'official')
        s, v = res['scores'], res['verdict']
        self.assertEqual(scorer.testbed(ctx)['result'], 'PASS')
        self.assertEqual(s['best_shortcut']['name'], 'FIXED_1')
        self.assertAlmostEqual(s['best_shortcut']['value'], 0.5)
        self.assertAlmostEqual(v['primary']['mean_cand'], 0.475)
        self.assertEqual(v['verdict'], 'FAIL', v['reasons'])
        self.assertTrue(all(d <= 0 for d in v['primary']['per_world']))
        self.assertTrue(all(x['comparator'] == 'shortcut:FIXED_1' for x in v['primary']['per_world_detail']))
        cells = scorer.score_role(ctx, 'official')['cells']
        self.assertEqual(cells['ladder_skipped_members'], sorted(decide.OPERAND_MEMBERS))
        self.assertNotIn('DIGIT_NO_CARRY', cells['ladder']['wo1|HELD|final'])

    def test_v6_like_zero_operand_items_are_s0_and_not_counted(self):
        """Held set with 10 zero-operand items (0+b, a+0) where the candidate is perfect and 10 S2/S3 items where it
        is wrong: the gain is only on S0, so nothing counted improves and the verdict is FAIL."""
        lock = base_lock()
        lock['primary']['n_items_per_pool'] = {'HELD': 20}
        lock['guards'] = []
        lock['split'] = {'taught_manifest': 'manifests/TAUGHT'}
        self.set_lock(lock)
        zero = [(0, b) for b in range(1, 6)] + [(a, 0) for a in range(1, 6)]
        held = zero + H[:5] + H[-5:]
        # SPEC 20.6: the declared SHUF arm belongs to the decision comparator, so its rows are required (without
        # them the verdict is INCOMPLETE, checked below); SHUF is wrong everywhere here
        rows = []
        for w in OFFICIAL:
            for arm in ('W', 'NPT'):
                for i, (a, b) in enumerate(held):
                    exp = truth(a, b)
                    ok = (arm == 'W' and i < 10) or (arm == 'NPT' and i == 15)
                    rows.append(row(w, 'official', arm, 'HELD', 'final', 'h%02d' % i, ph(a, b), exp,
                                    stream_of(exp if ok else wrong_of(exp))))
        self.write_rows('official', rows)
        self.write_taught(OFFICIAL + TECH, arms=('W',))
        self.write_rows('admission', self.write_testbed(items=[(ph(a, b), truth(a, b)) for (a, b) in held]))
        v0 = scorer.score(self.ctx(), 'official')['verdict']
        self.assertEqual(v0['verdict'], 'INCOMPLETE', v0['reasons'])
        self.assertTrue(any('SHUF/HELD/final (comparator)' in r for r in v0['reasons']), v0['reasons'])
        rows = []
        for w in OFFICIAL:
            for arm in ('W', 'NPT', 'SHUF'):
                for i, (a, b) in enumerate(held):
                    exp = truth(a, b)
                    ok = (arm == 'W' and i < 10) or (arm == 'NPT' and i == 15)
                    rows.append(row(w, 'official', arm, 'HELD', 'final', 'h%02d' % i, ph(a, b), exp,
                                    stream_of(exp if ok else wrong_of(exp))))
        self.write_rows('official', rows)
        self.write_taught(OFFICIAL + TECH, arms=('W',))
        self.write_rows('admission', self.write_testbed(items=[(ph(a, b), truth(a, b)) for (a, b) in held]))
        res = scorer.score(self.ctx(), 'official')
        s, v = res['scores'], res['verdict']
        self.assertEqual(s['strata'].get('S0'), 80)
        self.assertEqual(s['strata'].get('S2', 0) + s['strata'].get('S3', 0), 80)
        self.assertEqual(v['verdict'], 'FAIL', v['reasons'])
        self.assertTrue(all(x['v_cand'] == 0.0 for x in v['primary']['per_world_detail']))
        self.assertTrue(all(x['n_counted'] == 10 for x in v['primary']['per_world_detail']))
        self.assertEqual(s['held_by_stratum']['S0']['rate'], 1.0)
        self.assertEqual(s['held_by_stratum']['S0']['n_determined'], 0)


class TestChecks(DeliveryCase):
    def test_clean_reference_checks_pass_at_official_and_publish(self):
        self.full()
        for stage in ('official', 'publish'):
            ctx = self.ctx(stage)
            for fn in (scorer.check_met_01, scorer.check_met_02, scorer.check_lad_01, scorer.check_lad_02,
                       scorer.check_str_01, scorer.check_det_01, scorer.check_inert_01):
                r = fn(ctx)
                self.assertEqual(r.status, common.PASS, '%s %s: %s' % (stage, r.id, r.detail))
        r = scorer.check_udi_01(self.ctx('publish'))
        self.assertEqual(r.status, common.NOT_APPLICABLE)

    def test_lad01_missing_cell_and_role_conflict(self):
        self.full()
        rows = [r for r in make_rows(DEV, 'dev') if not (r['world'] == 'wd2' and r['arm'] == 'NPT'
                                                         and r['pool'] == 'NEW' and r['checkpoint'] == 'final')]
        rows[0]['world_role'] = 'official'
        self.write_rows('dev', rows)
        r = scorer.check_lad_01(self.ctx('official'))
        self.assertEqual(r.status, common.FAIL)
        self.assertIn('wd2/NPT/NEW/final missing', r.detail)
        self.assertIn('world_role conflicts', r.detail)

    def test_lad01_implied_cells_follow_the_lock(self):
        # the cells the rule, guards and SPEC measurement points read; no full pool x checkpoint product
        pcs = scorer.implied_pool_checkpoints(self.lock)
        self.assertIn((self.lock['primary']['pool'], self.lock['primary']['checkpoint']), pcs)
        self.assertIn(('OLD', 'old_end'), pcs)
        self.assertIn(('OLD', 'final'), pcs)
        self.assertIn(('TAUGHT', 'prefix'), pcs)
        self.assertNotIn(('HELD', 'prefix'), pcs)
        lk = json.loads(json.dumps(self.lock))
        lk['primary']['n_items_per_pool'] = {'HELD': 4}
        self.assertEqual(scorer.implied_pool_checkpoints(lk), [('HELD', 'final')])
        # an extra cell of a declared pool/checkpoint is allowed only when complete
        self.full()
        rows = [r for r in make_rows(DEV, 'dev') if not (r['world'] == 'wd2' and r['arm'] == 'NPT'
                                                         and r['pool'] == 'HELD' and r['checkpoint'] == 'prefix'
                                                         and r['item_id'] == 'h00')]
        self.write_rows('dev', rows)
        r = scorer.check_lad_01(self.ctx('official'))
        self.assertEqual(r.status, common.FAIL, r.detail)
        self.assertIn('extra cell wd2/NPT/HELD/prefix', r.detail)

    def test_met01_fires_on_stray_output_rows(self):
        self.full()
        write(self.d / 'results/dev/OUTPUTS_extra.jsonl', jsonl(make_rows(['wd1'], 'dev', arms=('W',), pools=('HELD',),
                                                                       cks=('final',))))
        r = scorer.check_met_01(self.ctx('official'))
        self.assertEqual(r.status, common.FAIL)
        self.assertIn('OUTPUTS_extra', r.detail)

    def test_met01_fires_on_buggy_decoder_and_met02_on_dropped_carry(self):
        self.full()
        rows = make_rows(DEV, 'dev')
        rows[3]['emitted_hex'] = rows[3]['stream_hex']
        rows[5]['expected_hex'] = b'9'.hex()
        self.write_rows('dev', rows)
        self.assertEqual(scorer.check_met_01(self.ctx('official')).status, common.FAIL)
        self.assertEqual(scorer.check_met_02(self.ctx('official')).status, common.FAIL)

    def test_inert_identical_arms_alarm_and_invalid_verdict(self):
        def same(world, arm, pool, ck, idx, a, b):
            return arm_behaviour(world, 'W' if arm == 'NPT' else arm, pool, ck, idx, a, b)
        self.write_taught(TECH + OFFICIAL)
        self.write_manifests()
        self.write_rows('admission', make_rows(TECH, 'technical') + self.write_testbed())
        self.write_rows('official', make_rows(OFFICIAL, 'official', behaviour=same))
        ctx = self.ctx('publish')
        r = scorer.check_inert_01(ctx)
        self.assertEqual(r.status, common.FAIL)
        self.assertIn('identical stream_hex', r.detail)
        self.assertIn('exactly 0', r.detail)
        v = scorer.score(self.ctx(), 'official')['verdict']
        self.assertEqual(v['verdict'], 'INVALID')
        self.assertEqual(v['invalid_reason'], 'INTEGRITY_ALARM')

    def test_lad03_weak_comparator_and_lad04_collapse(self):
        self.full()
        r = scorer.check_lad_03(self.ctx('official'))
        self.assertEqual(r.status, common.WARN)  # NPT 0.1 < best shortcut 0.5 on dev
        self.assertIn('WEAK_COMPARATOR', r.detail)

        def collapse(world, arm, pool, ck, idx, a, b):
            return (False, True) if arm == 'W' else arm_behaviour(world, arm, pool, ck, idx, a, b)
        self.write_rows('dev', make_rows(DEV, 'dev', behaviour=collapse))
        r = scorer.check_lad_04(self.ctx('official'))
        self.assertEqual(r.status, common.WARN)
        self.assertIn('modal 1.000', r.detail)

    def test_str01_and_det01_failures(self):
        self.full()
        lock = base_lock()
        lock['primary']['strata_counted'] = ['S2', 'S3', 'S4']
        self.set_lock(lock)
        r = scorer.check_str_01(self.ctx('official'))
        self.assertEqual(r.status, common.FAIL)
        self.assertIn('S4 has 0 items', r.detail)
        lock['primary']['strata_counted'] = ['S0', 'S2']
        self.set_lock(lock)
        r = scorer.check_str_01(self.ctx('official'))
        self.assertIn('not valid counted strata at E3', r.detail)
        # DET-01: remove the witness corner 9 so no held item is determined
        self.set_lock(base_lock())
        pairs = [p for p in T_ALL if 9 not in p]
        self.write_taught(DEV + REH, pairs=pairs)
        r = scorer.check_det_01(self.ctx('official'))
        self.assertEqual(r.status, common.FAIL, r.detail)
        self.assertIn('fewer than 8', r.detail)
        lock = base_lock(units=None)
        self.set_lock(lock)
        self.assertEqual(scorer.check_det_01(self.ctx('official')).status, common.NOT_APPLICABLE)

    def test_scope_missing_and_vacuous(self):
        self.write_taught(TECH)
        self.write_manifests()
        self.write_rows('admission', self.write_testbed())
        # official stage with no dev/rehearsal rows: MISSING
        self.assertEqual(scorer.check_lad_01(self.ctx('official')).status, common.MISSING)
        self.assertEqual(scorer.check_met_01(self.ctx('official')).status, common.MISSING)
        # publish of a testbed-only delivery (honest stop): learner checks pass vacuously, MET-01 checks testbed rows
        self.assertEqual(scorer.check_lad_01(self.ctx('publish')).status, common.PASS)
        r = scorer.check_met_01(self.ctx('publish'))
        self.assertEqual(r.status, common.PASS)
        self.assertIn('admission', r.detail)
        # a launch-recorded dev output with no readable rows: MISSING
        write(self.d / 'runlog' / 'ATTEMPTS.jsonl', jsonl([{'attempt_id': 'A', 'stage': 'dev', 'produced_paths': [
            {'path': 'results/dev/OUT.bin', 'sha256': 'x'}], 'cpu_children_user_s': 1, 'cpu_children_sys_s': 0}]))
        self.assertEqual(scorer.check_lad_01(self.ctx('publish')).status, common.MISSING)

    def test_rows_must_carry_the_locked_split_items(self):
        # verifier round-3 probe E2: an out-of-manifest held item swapped in for a missed one in every world
        self.full()
        self.assertEqual(scorer.check_lad_01(self.ctx('publish')).status, common.PASS)
        rows = make_rows(OFFICIAL, 'official')
        for r in rows:
            if r['pool'] == 'HELD' and r['item_id'] == 'h00':
                r['prompt_hex'] = ph(2, 7)
        self.write_rows('official', rows)
        with self.assertRaises(common.InvalidInput) as cm:
            scorer.score_role(self.ctx(), 'official')
        self.assertIn('not in the locked split manifest', str(cm.exception))
        ctx = self.ctx('publish')
        r = common.run_check(ctx, common.rule_index(ctx.rules)['LAD-01'])[0]
        self.assertEqual(r.status, common.FAIL)
        self.assertIn('not in the locked split manifest', r.detail)
        # a new item id the manifest does not hold, with its own prompt
        rows = make_rows(OFFICIAL, 'official')
        rows = [dict(r, item_id='hx') if (r['pool'] == 'HELD' and r['item_id'] == 'h00') else r for r in rows]
        self.write_rows('official', rows)
        self.assertEqual(common.run_check(self.ctx('publish'), common.rule_index(ctx.rules)['LAD-01'])[0].status,
                         common.FAIL)

    def test_lad01_cells_hold_exactly_the_manifest(self):
        self.full()
        # the manifest gains an item no row carries: every HELD cell now misses it
        rows = [json.loads(l) for l in (self.d / 'manifests/HELD.jsonl').read_text().splitlines()]
        rows.append({'item_id': 'h99', 'prompt_hex': ph(9, 9), 'expected_hex': truth(9, 9).hex()})
        write(self.d / 'manifests/HELD.jsonl', jsonl(rows))
        r = scorer.check_lad_01(self.ctx('publish'))
        self.assertEqual(r.status, common.FAIL)
        self.assertIn("items differ from the locked HELD manifest (missing ['h99']", r.detail)

    def test_gz_rows_and_cmd_score(self):
        self.full(official=False)
        self.write_rows('official', make_rows(OFFICIAL, 'official'), gz=True)
        ns = argparse.Namespace(dir=str(self.d), role='official', repo_root=str(self.root),
                                git='timeline:%s' % self.fx, json=False, extra=[])
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = scorer.cmd_score(ns)
        self.assertEqual(rc, 0)
        self.assertIn('verdict PASS', out.getvalue())
        self.assertEqual(json.loads((self.d / 'results/official/VERDICT.json').read_text())['verdict'], 'PASS')
        ns.dir = str(self.root / 'minifly_absent_dev_v1_20261101')
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertEqual(scorer.cmd_score(ns), 2)
        self.assertIn('legacy delivery', err.getvalue())


class TestUdi(DeliveryCase):
    def test_udi_known_value(self):
        """Two base worlds, each with a variant that removes the taught pair (6, 8): the dependent item (1, 8) loses
        its only rectangle, the independent item (3, 5) keeps its own. The candidate is right on both items in the
        base worlds; in variant wAv it is wrong on D, in wBv right. UDI per world: wA (1-0)-(1-1) = 1, wB 0 ->
        UDI 0.5, LCB = 0.5 - t(0.95, 1) * sd / sqrt(2) with sd = sqrt(0.5)."""
        lock = base_lock()
        lock['udi'] = {'variants_manifest': 'manifests/UDI_VARIANTS.json', 'alpha': 0.05}
        lock['primary']['n_items_per_pool'] = {'HELD': 2}
        lock['guards'] = []
        lock['split'] = {'taught_manifest': 'manifests/TAUGHT'}
        lock['worlds']['official'] = {'mode': 'listed', 'ids': ['wA', 'wB'], 'n': 2}
        self.set_lock(lock)
        import addition
        # each held item has exactly one rectangle: D = (1, 8) via a'=6 b'=2; I = (3, 5) via a'=7 b'=4
        taught = [(1, 2), (6, 8), (6, 2),
                  (3, 4), (7, 5), (7, 4)]
        T = addition.Taught(taught)
        self.assertTrue(addition.determined((1, 8), T, 'pair_rectangle'))
        self.assertTrue(addition.determined((3, 5), T, 'pair_rectangle'))
        self.assertEqual(addition.support((1, 8), T, 'pair_rectangle'), frozenset({('1', '2'), ('6', '8'), ('6', '2')}))
        removed = [(6, 8)]
        variant_taught = [p for p in taught if p not in removed]
        held = [(1, 8), (3, 5)]
        rows = []
        for w, var_d_ok in (('wA', False), ('wB', True)):
            for world, tset, d_ok in ((w, taught, True), (w + 'v', variant_taught, var_d_ok)):
                for arm in ('W', 'NPT'):
                    for i, (a, b) in enumerate(held):
                        exp = truth(a, b)
                        ok = arm == 'W' and (i == 1 or d_ok)
                        rows.append(row(world, 'official', arm, 'HELD', 'final', 'h%d' % i, ph(a, b), exp,
                                        stream_of(exp if ok else wrong_of(exp))))
                write(self.d / 'manifests/TAUGHT' / world / 'W.jsonl', jsonl(taught_rows(world, 'W', tset)))
        self.write_rows('official', rows)
        write(self.d / 'manifests/UDI_VARIANTS.json', {'variants': [
            {'base_world': w, 'variant_world': w + 'v', 'removed_taught_prompt_hex': [ph(6, 8)],
             'dependent_item_ids': ['h0'], 'independent_item_ids': ['h1']} for w in ('wA', 'wB')]})
        res = scorer.udi(self.ctx(), 'official')
        self.assertTrue(res['valid'], res['problems'])
        # wA: (1 - 0) - (1 - 1) = 1; wB: (1 - 1) - 0 = 0 -> UDI 0.5
        self.assertAlmostEqual(res['value'], 0.5)
        self.assertAlmostEqual(res['lcb'], 0.5 - decide.t_ppf(0.95, 1) * (0.5 ** 0.5) / (2 ** 0.5), places=6)
        self.assertEqual(res['lcb_method'], 't')
        self.assertEqual(scorer.check_udi_01(self.ctx('publish')).status, common.PASS)
        # an invalid variant: the "independent" item loses its rectangle too
        write(self.d / 'manifests/UDI_VARIANTS.json', {'variants': [
            {'base_world': 'wA', 'variant_world': 'wAv', 'removed_taught_prompt_hex': [ph(6, 8)],
             'dependent_item_ids': ['h1'], 'independent_item_ids': ['h0']}]})
        r = scorer.check_udi_01(self.ctx('publish'))
        self.assertEqual(r.status, common.FAIL)
        self.assertIn('stays determined', r.detail)
        # SEALED_CONFIRM: the seal-derived draw of D is not verifiable, so a valid UDI is still MISSING
        write(self.d / 'manifests/UDI_VARIANTS.json', {'variants': [
            {'base_world': w, 'variant_world': w + 'v', 'removed_taught_prompt_hex': [ph(6, 8)],
             'dependent_item_ids': ['h0'], 'independent_item_ids': ['h1']} for w in ('wA', 'wB')]})
        lock['worlds']['official_role_intended'] = 'SEALED_CONFIRM'
        self.set_lock(lock)
        r = scorer.check_udi_01(self.ctx('publish'))
        self.assertEqual(r.status, common.MISSING)
        self.assertIn('seal-derived', r.detail)


class TestPerformance(unittest.TestCase):
    def test_one_million_rows_under_15s(self):
        """1e6 synthetic 3-digit addition rows: 8 worlds x 5 arms x 3 checkpoints x 8334 items."""
        fmt = dict(FMT, gen_cap_bytes=6)
        lock = base_lock(task={'id': 'add3', 'family': 'addition_multi_digit', 'format': fmt})
        lock['arms'] = [{'id': a, 'role': r} for a, r in (('W', 'candidate'), ('NPT', 'no_prior_teaching'),
                                                          ('SHUF', 'shuf'), ('FROZEN', 'frozen'),
                                                          ('ORACLE', 'oracle_plumbing'))]
        pools = {'HELD': 3000, 'OLD': 2500, 'NEW': 1500, 'TAUGHT': 1334}
        lock['primary']['n_items_per_pool'] = pools
        worlds = ['p%d' % i for i in range(8)]
        lock['worlds']['official']['ids'] = worlds

        def pairs(seed, n):
            out = []
            x = seed
            for _ in range(n):
                x = (x * 1103515245 + 12345) % (2 ** 31)
                out.append((100 + x % 900, 100 + (x // 900) % 900))
            return out
        items = {pool: pairs(i + 7, n) for i, (pool, n) in enumerate(sorted(pools.items()))}
        taught = {(w, 'W'): [{'prompt_hex': ('%d+%d=' % p).encode().hex(), 'answer_hex': str(p[0] + p[1]).encode().hex(),
                              'feedback_bearing': True, 'presentations': 3} for p in pairs(99 + j, 2000)]
                  for j, w in enumerate(worlds)}
        prompts = {pool: [('%d+%d=' % p).encode().hex() for p in its] for pool, its in items.items()}
        exps = {pool: [str(p[0] + p[1]).encode() for p in its] for pool, its in items.items()}

        def gen():
            k = 0
            for w in worlds:
                for arm in ('W', 'NPT', 'SHUF', 'FROZEN', 'ORACLE'):
                    for ck in ('prefix', 'old_end', 'final'):
                        for pool in sorted(pools):
                            for i in range(pools[pool]):
                                exp = exps[pool][i]
                                ok = (i + k) % 3 != 0
                                content = exp if ok else b'9'
                                st = (content + b'\n' * 6)[:6]
                                pos = st.find(b'\n')
                                yield {'world': w, 'world_role': 'official', 'arm': arm, 'pool': pool,
                                       'checkpoint': ck, 'item_id': 'i%d' % i, 'prompt_hex': prompts[pool][i],
                                       'expected_hex': exp.hex(), 'stream_hex': st.hex(), 'emitted_hex': st[:pos].hex(),
                                       'own_eos': True, 'answer_path': 'native'}
                    k += 1
        t0 = time.perf_counter()
        c = scorer.score_rows(gen(), lock, taught)
        dt = time.perf_counter() - t0
        self.assertEqual(c['n_rows'], 8 * 5 * 3 * sum(pools.values()))
        self.assertGreaterEqual(c['n_rows'], 1000000)
        self.assertEqual(c['met_mismatches'], [])
        self.assertLessEqual(dt, 15.0, 'scored %d rows in %.1f s' % (c['n_rows'], dt))
        sys.stderr.write('\n[perf] score_rows: %d rows in %.2f s\n' % (c['n_rows'], dt))


if __name__ == '__main__':
    unittest.main()
