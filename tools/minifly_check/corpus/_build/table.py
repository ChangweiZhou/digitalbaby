"""The fixture table: one entry per SPEC 12.2 row. 'mutation' states the only difference from the base
(or from the named variant). Check ids are exactly the spec's. Expected exits: 0 PASS, 1 FAIL."""
import fixtures_c as C
import fixtures_d as F
import fixtures_g as G
from fixtures_c import CANARY, CLAIM, LOCK, OFFICIAL, PUBLISH, ROSTER, SC_ADM, SC_DEV, SC_OFF, D

BASE_HEADLINE = ('PASS | earned E3 (attempted E3) | held exact own-EOS 85.2% [S2+ determined, n=128] vs '
                 'best shortcut DIGIT_NO_CARRY 50.0% | G 28.9pp | OLD final 96.9% (drop 3.1pp) | 8 MAIN '
                 'worlds | best of k=1')
C00B_HEADLINE = ('FAIL | earned NONE (attempted E3) | held exact own-EOS 64.8% [S2+ determined, n=128] vs '
                 'best shortcut DIGIT_NO_CARRY 50.0% | G 8.6pp | OLD final 96.9% (drop 3.1pp) | 8 MAIN '
                 'worlds | best of k=1')
C00D_HEADLINE = BASE_HEADLINE.replace('8 MAIN worlds', '8 SEALED_CONFIRM worlds')


def E(fid, title, fm, inc, kind, src, mutation, expectation, build, commands, exits, must_fire=(),
      must_warn=(), must_not_fire=(), expect=None, tier=2, sealed=False):
    return {'id': fid, 'title': title, 'fm_ids': list(fm), 'source_incident': inc, 'kind': kind,
            'built_from': src, 'mutation': mutation, 'expectation': expectation, 'build': build,
            'commands': commands, 'expected_exit': list(exits), 'must_fire': list(must_fire),
            'must_warn': list(must_warn), 'must_not_fire': list(must_not_fire),
            'expect': dict(expect or {}), 'tier': tier, 'sealed': sealed}


T = [
    E('C00a', 'clean T2 MAIN pass', [], None, 'clean', 'base, T2 MAIN pass', 'none (the base itself)',
      'All stages exit 0; hand-computed SCORES and VERDICT values (EXPECTED_VALUES.md).', C.b_c00a,
      [LOCK, ROSTER, OFFICIAL, SC_OFF, CLAIM, PUBLISH, CANARY], [0, 0, 0, 0, 0, 0, 0],
      expect={'/verdict': 'PASS', '/primary_gate': 'PASS', '/level_earned': 'E3',
              '/worlds_role': 'MAIN', '/headline': BASE_HEADLINE,
              '/objective_metric/best_shortcut/name': 'DIGIT_NO_CARRY', '/designs_tried_k': 1}),
    E('C00b', 'clean honest FAIL: candidate at comparator level', [], None, 'clean', 'base',
      'W\'s HELD answers lowered to the comparator level: per world one S2 item wrong and only 3 S3 items '
      'right (6 in w_off_8), so d_w = 0.0625 in 7 worlds and 0.25 in one; paired LCB 0.0415 <= margin.',
      'verdict FAIL; publish exit 0', C.b_c00b, [OFFICIAL, SC_OFF, CLAIM, PUBLISH], [0, 0, 0, 0],
      expect={'/verdict': 'FAIL', '/primary_gate': 'FAIL', '/headline': C00B_HEADLINE}),
    E('C00c', 'clean honest exit: CANNOT_TEST after a testbed_invalid amendment', [], None, 'clean',
      'base', 'All ceiling configs answer at DIGIT_NO_CARRY level; cycle 1 closes with revise; AMD-1 '
      '(reason testbed_invalid, no UD, results_seen = the testbed files) adds ceiling EXACT_BP_LONG; the '
      'rerun testbed fails again; cycle 2 closes with stop. The timeline ends there (no learner code).',
      'stop_reason CANNOT_TEST; publish exit 0', F.b_c00c, [SC_ADM, CLAIM, PUBLISH], [0, 0, 0],
      expect={'/primary_gate': 'CANNOT_TEST_CLAIM_AS_SPECIFIED', '/stop_reason': {'contains': 'CANNOT_TEST'},
              '/headline': F.C00C_HEADLINE}),
    E('C00d', 'clean sealed confirmation', [], None, 'clean', 'base',
      'worlds.official mode sealed (role SEALED_CONFIRM); after the rehearsal: SEAL_COMMIT (reviewer), '
      'owner GO naming both lock shas, SEAL_REVEAL, CONFIRM_LEDGER attempt 0; official worlds derived '
      'from the seal (fixtures_d.derive_worlds) and run with the base behaviour of w_off_1..8; post-run '
      'replay of 4 histories; REPORT uses confirm wording.',
      'SEALED_CONFIRM; confirm wording allowed', F.b_c00d, [OFFICIAL, SC_OFF, CLAIM, PUBLISH],
      [0, 0, 0, 0], sealed=True,
      expect={'/verdict': 'PASS', '/worlds_role': 'SEALED_CONFIRM', '/headline': C00D_HEADLINE}),
    E('C00e', 'clean T1 dev-only delivery without review', [], None, 'clean', 'base',
      'tier_declared 1, no official worlds (role PILOT), AUTHORIZATION official false; the review, ack, '
      'rehearsal and official commits are absent.',
      'primary_gate NOT_OPENED with dev shadow; publish exit 0', F.b_c00e,
      [LOCK, ROSTER, SC_DEV, CLAIM, PUBLISH], [0, 0, 0, 0, 0], tier=1,
      expect={'/primary_gate': 'NOT_OPENED', '/worlds_role': 'DEV',
              '/headline': {'contains': 'dev shadow'}}),
    E('C00f', 'clean governance: a new check plus fixture with a GOV-REV record', [], None, 'clean',
      'governance', 'runnable overlay kit (full RULES.json copy, nested EXPECTED.json and lock-stage nested '
      'fixture X01); claude review record GR-20261105-1, then a codex commit adding check LOCK-14 (new module '
      'lockextra.py and its RULES entry), nested fixture X02 and its EXPECTED entry, with trailer '
      'Governance-Review.',
      'governance exit 0', G.b_c00f, lambda tl: [G.gov_cmd(tl)], [0]),
    E('C01', 'buggy decoder and carry-dropping truth', ['FM-01'], 'INC-SEED-01', 'dirty',
      'minifly_b/REPORT.md:12-26; archive FC08',
      'W rows with a carry answer on dev, rehearsal and official: emitted_hex keeps only the first byte '
      '(buggy decoder) and expected_hex drops the carry ((a+b) mod 10).', 'MET-01, MET-02', C.b_c01,
      [OFFICIAL], [1], ['MET-01', 'MET-02']),
    E('C02a', 'lesion arm undelivered: NPT writes and equals W', ['FM-02'], 'INC-SEED-01', 'dirty',
      'minifly_b/REPORT.md:12-26',
      'NPT receipts: writes_by_channel.pair_memory 288 and post_first pair_memory equal to W; NPT OUTPUTS '
      'identical to W on every item in every world.', 'ADM-CTRL-ZERO, INERT-01', C.b_c02a,
      [ROSTER, OFFICIAL], [1, 1], ['ADM-CTRL-ZERO', 'INERT-01']),
    E('C02b', 'control dose too small', ['FM-02'], None, 'dirty',
      'minifly_recoverability_v1_r2_dev_20261023/REPORT.md:30-39',
      'NPT receipts realized_dose teaching_withheld_fraction 0.00312 (min_realized_dose 0.05).',
      'ADM-CTRL-DOSE', C.b_c02b, [ROSTER], [1], ['ADM-CTRL-DOSE']),
    E('C03', 'confounded control: NPT also changes readout at post_first', ['FM-03'], None, 'dirty',
      'minifly_address_capacity_design_v1_20261023/RECEIVED_REVIEW.md:36',
      'NPT receipts: post_first readout digest differs from W (readout is neither a zero channel nor '
      'declared_changed; it stands for the incident\'s stop channel).', 'ADM-CTRL-ISO', C.b_c03,
      [ROSTER], [1], ['ADM-CTRL-ISO']),
    E('C04a', 'fixture/production mismatch and a gate without a negative fixture', ['FM-04'],
      'INC-SEED-02', 'dirty', 'V51 AUDIT.json A4',
      'FIXTURES.jsonl: directional rows carry pair_density 0.02 (params.json 0.0354); the stop_gate '
      'negative fixture row is removed.', 'ADM-FIX, ADM-GATE', C.b_c04a, [ROSTER], [1],
      ['ADM-FIX', 'ADM-GATE']),
    E('C04b', 'SHUF derangement', ['FM-04'], 'INC-SEED-03', 'dirty', 'LEARNED-BANK NOTES section 1',
      'SHUF TAUGHT labels: a cyclic derangement of the 8 orbit answers (agreement with the truth 0, '
      'outside shuf_band).', 'ADM-SHUF', C.b_c04b, [ROSTER], [1], ['ADM-SHUF']),
    E('C05a', 'ceiling grid fails', ['FM-05'], 'INC-SEED-04', 'dirty', 'LEARNED-BANK CYCLE2.md:10-36',
      'Both CEILING configs answer at DIGIT_NO_CARRY level on every technical world.',
      'ADM-TB-CEIL; expect stop_reason CANNOT_TEST', C.b_c05a, [ROSTER, SC_OFF, CLAIM], [1, 0, 0],
      ['ADM-TB-CEIL'], expect={'/verdict': 'CANNOT_TEST_CLAIM_AS_SPECIFIED',
                               '/stop_reason': {'contains': 'CANNOT_TEST'}}),
    E('C05b', 'floor passes', ['FM-05'], 'INC-SEED-04', 'dirty', 'LEARNED-BANK CARRY/CMP',
      'FLOOR_FRF_W4096_LR0.02 answers like CEILING_EXACT_BP (passes the rule).', 'ADM-TB-FLOOR',
      C.b_c05b, [ROSTER], [1], ['ADM-TB-FLOOR']),
    E('C05c', 'unreachable rule', ['FM-05'], 'INC-SEED-05', 'dirty',
      'persistent_core_v2_trial/results/SUMMARY.json',
      'primary.margin_abs 0.9: no clean positive (candidate 0.9, all else at chance 0.05, d = 0.85) can '
      'clear it. The incident\'s 0.6 was unreachable only with its Bonferroni alpha and 8 items; with this '
      'base 0.6 is reachable, so the fixture uses a margin no synthetic positive can reach.', 'RED-01',
      C.b_c05c, [ROSTER], [1], ['RED-01']),
]

T += [
    E('C06a', 'candidate at its own-stream truncation shortcut; comparator at 0', ['FM-06'],
      'INC-SEED-06', 'dirty', 'minifly_address_capacity_v1_dev_20261023/REPORT.md:7',
      'HELD gains 4 carry items (20 items, 8 S2 + 12 S3); W is wrong on S2 and emits every S3 answer '
      'with the right 2-byte prefix but closes it with EOS on only 6 (odd worlds) or 5 (even worlds); NPT '
      '(the declared comparator) answers "0". FIXED_2 of W\'s own stream is 0.6 > DIGIT_NO_CARRY 0.4. '
      'FIXED_1 cannot be the unique best on single-digit addition (DIGIT_NO_CARRY is right on every '
      '1-digit answer), so the incident\'s FIXED_1 becomes FIXED_2 here.',
      'expect /verdict FAIL, headline best shortcut FIXED_2; must_warn LAD-03', C.b_c06a,
      [OFFICIAL, SC_OFF, CLAIM], [0, 0, 0], must_warn=['LAD-03'],
      expect={'/verdict': 'FAIL', '/objective_metric/best_shortcut/name': 'FIXED_2',
              '/headline': {'contains': 'vs best shortcut FIXED_2 60.0%'}}),
    E('C06b', 'gains only on zero-operand items', ['FM-06'], 'INC-SEED-07', 'dirty',
      'minifly_v6_r1_dev_20261023/v6_data.py:7-34',
      'HELD gains 4 S0 items with a zero operand (20 items); W is right on them and at or below NPT on '
      'the counted S2/S3 items; NPT is wrong on the S0 items.', 'expect /verdict FAIL; S0 not counted',
      C.b_c06b, [SC_OFF], [0], expect={'/verdict': 'FAIL'}),
    E('C06c', 'OLD protected by never learning OLD', ['FM-06'], 'INC-SEED-07', 'dirty',
      'V6 audit_results.py:91 (NO_OLD guard)',
      'W answers the OLD pool at DIGIT_NO_CARRY level at old_end and final (OLD_END 0.5 = best '
      'exposure-independent shortcut, drop 0).', 'expect /verdict FAIL (old_acq)', C.b_c06c, [SC_OFF], [0],
      expect={'/verdict': 'FAIL', '/guards/old_acq': 'FAIL'}),
    E('C07', 'conditional primary and an edited CLAIM', ['FM-07'], 'INC-SEED-08', 'dirty',
      'minifly_stop_only_confirm_v1_20261023/DESIGN.md:5',
      'primary.unconditional false; a committed CLAIM.json whose gate_type was edited to objective.',
      'LOCK-03 (forced component), CLM-02', C.b_c07, [LOCK, SC_OFF, PUBLISH], [1, 0, 1],
      ['LOCK-03', 'CLM-02']),
    E('C08a', 'task switched after learner outcomes without a UD', ['FM-08'], 'INC-SEED-09', 'dirty',
      'LEARNED-BANK CYCLE2.md:39-103',
      'After the dev cycle: AMD-1 changes task.id and primary.task_id (add1 -> add1_parity) with '
      'results_seen = dev outputs, no UD, protocol_id not bumped, original endpoint not kept. The timeline '
      'ends at that amendment.', 'AMD-02', C.b_c08a, [LOCK], [1], ['AMD-02']),
    E('C08b', 'margin changed without an AMENDMENTS row', ['FM-08'], None, 'dirty', 'synthetic',
      'A second commit of ENDPOINT_LOCK.json raises primary.margin_abs 0.05 -> 0.1 with no AMENDMENTS '
      'row; the timeline ends there (lock stage).', 'AMD-01', C.b_c08b, [LOCK], [1], ['AMD-01']),
    E('C09a', 'E3 cadence due, component lock, no waiver', ['FM-09'], None, 'dirty', 'CATALOG.md:36-43',
      'DELIVERY_LEDGER holds 2 earlier v1 rows that did not open held E3; the lock targets E1 (not '
      'E3-opening) without e3_cadence.waiver_user_decision; the timeline ends at the lock.', 'LOCK-09',
      C.b_c09a, [LOCK], [1], ['LOCK-09']),
    E('C09b', 'honest exit does not reset the E3 cadence', ['FM-09'], None, 'dirty', 'adversarial judge',
      'DELIVERY_LEDGER: an older admitted MAIN held-E3 PASS, then a non-E3 delivery, then an E3 attempt '
      'that ended CANNOT_TEST_CLAIM_AS_SPECIFIED (row claims opened_held_E3 true); the lock targets E1 '
      'without a waiver; the timeline ends at the lock.', 'LOCK-09 (no reset)', C.b_c09b, [LOCK], [1],
      ['LOCK-09']),
    E('C10', 'rehearsal world used in dev; best of k=3', ['FM-10'], None, 'dirty',
      'LEARNED-BANK CYCLE3.md:46; addition_rule cycles',
      'Two outcome-informed learner revisions inside the dev cycle (3 distinct native closures, each with '
      'a SOURCE_LOCK and a dev attempt); the third dev attempt also runs w_reh_1.',
      "CYC-03; expect headline 'best of k=3'", C.b_c10, [OFFICIAL, SC_OFF, CLAIM], [1, 0, 0], ['CYC-03'],
      expect={'/headline': {'contains': 'best of k=3'}, '/designs_tried_k': 3}),
    E('C11a', 'official world id present in another delivery', ['FM-11'], 'INC-SEED-10', 'dirty',
      'minifly_cls_dev_20261017/REPORT.md:3',
      'An earlier directory minifly_corpus_other_dev_20261020/WORLDS.json lists w_off_3.',
      'WLD-01, AMD-04', C.b_c11a, [OFFICIAL], [1], ['WLD-01', 'AMD-04']),
    E('C11b', 'SEALED_CONFIRM without a seal', ['FM-11'], None, 'dirty', 'adversarial judge',
      'official_role_intended SEALED_CONFIRM with listed worlds and no seal files; REPORT.md says '
      '"Sealed confirmation passed".', 'SEAL-01, CLM-04', C.b_c11b, [OFFICIAL, SC_OFF, CLAIM, PUBLISH],
      [1, 0, 0, 1], ['SEAL-01', 'CLM-04'], sealed=True),
    E('C12', 'held orbit in a feedback-bearing stream; undeclared stream file', ['FM-12'], None, 'dirty',
      'minifly_v5_design_review_20261020/REVIEW.md:25-31',
      'A declared feedback-bearing calibration stream manifests/CALIBRATION/w_tech_1/W.jsonl teaches 8+2=10 '
      '(the commuted twin of held h09 2+8); an undeclared stream file manifests/EXTRA/w_tech_1/W.jsonl.',
      'ADM-EXPO', C.b_c12, [ROSTER], [1], ['ADM-EXPO']),
    E('C13a', 'lock and results in one commit', ['FM-13'], 'INC-SEED-11', 'dirty', 'commit 43d04b5e',
      'The lock commit and the testbed-results commit are one commit.', 'LOCK-06', C.b_c13a, [ROSTER], [1],
      ['LOCK-06']),
    E('C13b', 'native import outside SOURCE_LOCK', ['FM-13'], 'INC-SEED-13', 'dirty',
      'LEARNED-BANK make_lock.py:12',
      'learner.py imports bootstrap.py (committed in the delivery); SOURCE_LOCK.closure_sha256 omits it.',
      'LOCK-05', C.b_c13b, [ROSTER], [1], ['LOCK-05']),
    E('C13c', 'T2 dev output before any attestation', ['FM-13'], 'INC-SEED-13', 'dirty',
      'LEARNED-BANK 1ce1eadb to 84c530d4',
      'The review commit (the first reviewer attestation) moves after the dev-output commit.', 'WIT-01',
      C.b_c13c, [OFFICIAL], [1], ['WIT-01']),
    E('C14', 'literal integrity field and a custom reducer without a UD', ['FM-14'], 'INC-SEED-14',
      'dirty', 'V6 audit_results.py:97; LEARNED-BANK summarize.py',
      'audit_results.py assigns exposure_overlap = 0; primary.rule_type custom with no UD, reviewer or '
      'property fixtures.', 'LOCK-08, DEC-01', C.b_c14, [LOCK, ROSTER], [1, 1], ['LOCK-08', 'DEC-01']),
    E('C15', 'endpoint file edited after a FAIL without an amendment', ['FM-15'], None, 'dirty',
      'native_next_v1_dev_20261023/REPORT.md:13,40',
      'admit.py (endpoint-locked) is edited in a commit after the testbed; no AMENDMENTS row.', 'AMD-05',
      C.b_c15, [LOCK], [1], ['AMD-05']),
    E('C16', 'technical readiness without behavioural readiness', ['FM-16'], 'INC-SEED-16', 'dirty',
      'V6 REVISION.md:40-52',
      'admission_thresholds.analytic_input_snr 0.027; ADMISSION input_contrast ratio 0.014; W answers '
      '"0" on the TAUGHT pool at the prefix checkpoint of the technical world (taught 0%).',
      'LOCK-12, ADM-INPUT, ADM-ACQ', C.b_c16, [LOCK, ROSTER], [1, 1], ['LOCK-12', 'ADM-INPUT', 'ADM-ACQ']),
    E('C17', 'directional sign and replay mismatch', ['FM-17'], None, 'dirty',
      'minifly_v4_dev_20261020 ACCOUNTING.json',
      'FIXTURES: the + row has target_margin_delta_vs_passive -0.012; the - row has '
      'zero_replay_applications 32.', 'ADM-DIR', C.b_c17, [ROSTER], [1], ['ADM-DIR']),
    E('C18', 'import from a sibling delivery', ['FM-18'], None, 'dirty',
      'minifly_v6_r1_dev_20261023/cycles/01_REVIEW.md:7-9',
      "learner.py: sys.path.insert(0, '../minifly_corpus_sibling_dev_20261015') then import core.",
      'LOCK-05', C.b_c18, [ROSTER], [1], ['LOCK-05']),
    E('C19', 'official OUTPUTS not covered by ATTEMPTS', ['FM-19'], None, 'dirty',
      'native_next_v1_dev_20261023/REPORT.md:12,46-58',
      'The official attempt row omits results/official/OUTPUTS.jsonl.gz from produced_paths.', 'RUN-01',
      C.b_c19, [PUBLISH], [1], ['RUN-01']),
    E('C20', 'paired schedule off; one world\'s control missing', ['FM-20'], None, 'dirty',
      'minifly_v5_1_r2_dev_20261020/REPORT.md:3-43',
      'schedule.paired_atomic false; official world w_off_5 has no NPT rows and no NPT receipt.',
      'ADM-BUDGET; expect /verdict INCOMPLETE', C.b_c20, [ROSTER, SC_OFF], [1, 0], ['ADM-BUDGET'],
      expect={'/verdict': 'INCOMPLETE'}),
    E('C21', 'underpowered sealed confirmation on 4 worlds', ['FM-21'], 'INC-SEED-17', 'dirty',
      'V6 EXECUTION_SPEC (2 units)',
      'official_role_intended SEALED_CONFIRM with 4 listed worlds; W near the comparator on dev (low OC '
      'power); a committed CLAIM.json claims worlds_role SEALED_CONFIRM with n_worlds 4.',
      'OC-01, CLM-09', F.b_c21, [OFFICIAL, SC_OFF, PUBLISH], [1, 0, 1], ['OC-01', 'CLM-09'], sealed=True),
    E('C22', 'inflated report wording on DEV', ['FM-22'], 'INC-SEED-18', 'dirty',
      'CLS REPORT.md:93-98; LEARNED-BANK CYCLE2.md:61',
      "Variant C00e (T1 dev only); REPORT.md first line 'PASS | G1 memory largely met on 2 DEV worlds' and "
      "a 'Native confirmation' sentence.", 'CLM-05, CLM-06, CLM-04', F.b_c22, [SC_DEV, CLAIM, PUBLISH],
      [0, 0, 1], ['CLM-05', 'CLM-06', 'CLM-04'], tier=1),
    E('C23', 'undeclared parent and a backprop call in the native path', ['FM-23'], None, 'dirty',
      'ROUTE_RETROSPECTIVE A03/A07; PA-13',
      'files.native gains fastpath.py (calls loss.backward()); an earlier v1 delivery '
      'minifly_corpus_prior_dev_v1_20261020 has byte-identical learner.py and fastpath.py (native-closure '
      'Jaccard 1.0 >= 0.9); lineage.parent stays null.', 'LOCK-13, LOCK-07', F.b_c23, [LOCK, ROSTER],
      [1, 1], ['LOCK-13', 'LOCK-07']),
    E('C24', 'pointer quotes a narrowed finding with a bare boolean and a local path', ['FM-24'],
      'INC-SEED-20', 'dirty', 'CATALOG P036; CURRENT_EXTERNAL_MEMORY',
      'PROJECT_DOCUMENTS/CURRENT_ADDITION_E3_POINTER.json quotes F-UF-15 (narrowed by F-20261024-claude-1) without '
      'narrowing_rows, carries status.confirmation_passes true and a /Users/ path.',
      'PTR-04, PTR-03, PTR-05', F.b_c24,
      [SC_OFF, CLAIM, ['gate', '--stage', 'pointer', '--pointer',
                       'PROJECT_DOCUMENTS/CURRENT_ADDITION_E3_POINTER.json', D]], [0, 0, 1],
      ['PTR-04', 'PTR-03', 'PTR-05']),
    E('C25', 'prediction basis F-id absent', ['FM-25'], None, 'dirty', 'V51 AUDIT A6',
      "predictions[P1].basis_findings ['F-20261001-codex-9'] (no such FINDINGS row); the timeline ends at "
      'the lock.', 'LOCK-11', F.b_c25, [LOCK], [1], ['LOCK-11']),
    E('C26', 'scored prediction removed; incident closed without a fixture', ['FM-26'], 'INC-SEED-21',
      'dirty', 'minifly_v6_revised_design_20261023/REVIEW.md:11',
      'A committed CLAIM.json drops P3 from predictions_scored; INCIDENTS gains INC-C26-1 report and a '
      'closure row with check_id CLM-10 and fixture_id null.', 'CLM-10, KB-05', F.b_c26, [SC_OFF, PUBLISH],
      [0, 1], ['CLM-10', 'KB-05']),
    E('C27', 'same-family, unexecuted, padded review', ['FM-27'], None, 'dirty',
      'multimechanism byte-identical audits',
      'REVIEW.json and the ack are committed by codex (the author family) on the codex branch, '
      'reviewer_agent codex; executed_checks keeps only trace_number, whose output is byte-identical to '
      'minifly_corpus_peer_dev_v1_20261020/review/out_trace_number.json.',
      'REV-01, REV-02, REV-07 (DEBT-01 fires as the failed fallback: no REVIEW_DEBT row)',
      F.b_c27, [ROSTER], [1], ['REV-01', 'REV-02', 'REV-07', 'DEBT-01']),
    E('C28', 'agent-shaped authorization', ['FM-28'], 'INC-SEED-22', 'dirty',
      'stop_only PLAN.json:4; transformer_external REPORT.md:3',
      "AUTHORIZATION caps.max_records 54432 (absent from verbatim); option A label ends '(Recommended)'; a "
      'conditional with no predicate evaluated by codex (the beneficiary). The timeline ends at the lock.',
      'AUT-02, AUT-03, AUT-04', F.b_c28, [LOCK], [1], ['AUT-02', 'AUT-03', 'AUT-04']),
    E('C29', 'T2 claim without the verification subset; local path', ['FM-29'], None, 'dirty',
      'V6 and native_next PUBLICATION.md',
      'results/official/OUTPUTS.jsonl.gz is never committed (untracked in the worktree); REPORT.md cites '
      '/Users/zhou/minifly/... .', 'CLM-15, CLM-12', F.b_c29, [SC_OFF, CLAIM, PUBLISH], [0, 0, 1],
      ['CLM-15', 'CLM-12']),
    E('C30', 'author activates its own finding', ['FM-30'], 'INC-SEED-23', 'dirty',
      'LEARNED-BANK NOTES omissions',
      'FINDINGS gains F-20261103-codex-1 (finding by codex for this delivery) and F-20261103-codex-2, an '
      'activation of it also by codex.', 'KB-03', F.b_c30, [SC_OFF, CLAIM, PUBLISH], [0, 0, 1], ['KB-03']),
    E('G01', 'protected edit in the same range as results', ['FM-30'], None, 'dirty', 'design',
      'One commit raises LOCK-11 tier_min 0 -> 1 in the miniature RULES.json and adds a delivery results '
      'file; no User-Decision trailer.', 'GOV-SEP, GOV-UD', G.b_g01, lambda tl: [G.gov_cmd(tl)], [1],
      ['GOV-SEP', 'GOV-UD']),
    E('G02', 'must_fire id removed without a loosen UD', ['FM-30'], None, 'dirty', 'design',
      "The miniature EXPECTED.json drops 'LOCK-08' from X01.must_fire; no User-Decision trailer.",
      'GOV-CLASS, GOV-UD, GOV-MONO', G.b_g02, lambda tl: [G.gov_cmd(tl)], [1],
      ['GOV-CLASS', 'GOV-UD', 'GOV-MONO']),
    E('G03', 'constraint loosened by a UD that does not restate the number', ['FM-30'], None, 'dirty',
      'adversarial judge',
      "UD-20261105-1 (scope constraint:threshold_floors) says 'Relax the margin floor a little'; the next "
      "commit sets threshold_floors.margin_abs 0.05 -> 0.03 citing it; '0.03' is not in the verbatim.",
      'GOV-NUM', G.b_g03, lambda tl: [G.gov_cmd(tl)], [1], ['GOV-NUM']),
    E('G04', 'author edits a check after its own lock', ['FM-30'], None, 'dirty', 'design',
      'After the technical commit, codex (the delivery author) commits an edit to '
      'tools/minifly_check/lockcheck.py (present since the KB commit); no UD names the delivery.',
      'GOV-COI', G.b_g04, [OFFICIAL], [1], ['GOV-COI']),
    E('G05', "a SKILL.md 'NEVER' line removed without a UD", ['FM-30'], None, 'dirty', 'design',
      "The NEVER line of .claude/skills/minifly-governance/SKILL.md is deleted; no User-Decision trailer.",
      'GOV-CLASS, GOV-UD', G.b_g05, lambda tl: [G.gov_cmd(tl)], [1], ['GOV-CLASS', 'GOV-UD']),
    E('K01', 'prediction scoring over three synthetic deliveries', ['FM-26'], None, 'clean', 'design',
      'Three v1 deliveries with P1 p_pass 0.7/0.4/0.9 and primary_gate PASS/FAIL/FAIL, their CLAIMs and '
      'DELIVERY_LEDGER rows.', 'expect Brier and skill values', G.b_k01, [['kb', 'score']], [0],
      expect={'/brier_mean': G.k01_values()['brier_mean'], '/skill': {'contains': 'INSUFFICIENT_N'}}),
]

BY_ID = {e['id']: e for e in T}

# ---------------------------------------------------------------------------------------------
# Supplementary fixtures (not rows of SPEC 12.2): each gives a check id that the table leaves without
# a firing fixture one, so selftest --mutate can see the check notice an always-PASS replacement.
import fixtures_s as S  # noqa: E402

POINTER = ['gate', '--stage', 'pointer', '--pointer', 'PROJECT_DOCUMENTS/CURRENT_ADDITION_E3_POINTER.json', D]
SUP = 'supplementary (not a SPEC 12.2 row)'


def X(fid, title, fm, mutation, build, commands, exits, must_fire, sealed=False):
    return E(fid, title, fm, None, 'dirty', SUP, mutation, ', '.join(must_fire), build, commands, exits,
             must_fire, sealed=sealed)


T += [
    X('S01', 'locally modified kit file', ['FM-15', 'FM-30'],
      'tools/minifly_check/minifly_check.py is committed in the KB commit and modified in the worktree '
      '(timeline dirty); the timeline ends at the lock.', S.b_s01, [LOCK], [3], ['SELF-01']),
    X('S02', 'legacy lock schema', [], "ENDPOINT_LOCK schema 'minifly.endpoint_lock.v0'; ends at the lock.",
      S.b_s02, [LOCK], [2], ['SCHEMA-01']),
    X('S03', 'template placeholder left in the lock', ['FM-22'],
      "predictions[P2].point is '__FILL__'; ends at the lock.", S.b_s03, [LOCK], [1], ['LOCK-01']),
    X('S04', 'no_prior_teaching arm without zero channels', ['FM-02'],
      'arms[NPT].zero_channels is empty; ends at the lock.', S.b_s04, [LOCK], [1], ['LOCK-02']),
    X('S05', 'margin below the CONSTRAINTS floor', ['FM-04'],
      'primary.margin_abs 0.03 (< threshold_floors.margin_abs 0.05); ends at the lock.', S.b_s05, [LOCK], [1],
      ['LOCK-04']),
    X('S06', 'constraints hash does not match', ['FM-28'],
      "constraints_sha256 is '0' x 64; ends at the lock.", S.b_s06, [LOCK], [1], ['LOCK-10']),
    X('S07', 'authorization fields missing', ['FM-28'],
      'AUTHORIZATION.json lacks recommendation_shown and session; ends at the lock.', S.b_s07, [LOCK], [1],
      ['AUT-01']),
    X('S08', 'testbed_invalid amendment while the ceiling passed', ['FM-08'],
      'After the passing cycle-0 testbed, AMD-1 (reason testbed_invalid) trims the floor grid; the testbed '
      'shows a ceiling PASS and every floor FAIL. Ends at the amendment.', S.b_s08, [LOCK], [1], ['AMD-03']),
    X('S09', 'OLD pool without OLD guards', ['FM-06'],
      'guards is empty although an OLD pool exists; ends at the lock.', S.b_s09, [ROSTER], [1],
      ['RED-09', 'RED-11']),
    X('S10', 'S0/S1 counted', ['FM-06'],
      "primary.strata_counted ['S0', 'S1', 'S2', 'S3']; ends at the lock.", S.b_s10, [ROSTER], [1],
      ['RED-08']),
    X('S11', 'degenerate aggregator', ['FM-04'],
      "primary.aggregator {type k_of_n, k 0}: any candidate passes, the substituted shortcut too.",
      S.b_s11, [ROSTER, OFFICIAL], [1, 1], ['RED-10', 'LAD-02']),
    X('S12', 'missing pool cell item', ['FM-04', 'FM-20'],
      'The W row for h05 in w_dev_1 is missing from the dev OUTPUTS (15 of 16 HELD items).', S.b_s12,
      [OFFICIAL], [1], ['LAD-01']),
    X('S13', 'counted stratum without items; witness that determines nothing', ['FM-05', 'FM-06'],
      "primary.strata_counted ['S2', 'S3', 'S4'] (no S4 item) and units.witness column_closure (no held "
      'column triple is taught, so no held item is determined).', S.b_s13, [OFFICIAL], [1],
      ['STR-01', 'DET-01']),
    X('S14', 'admission on a dev world', ['FM-16', 'FM-13'],
      "ADMISSION.worlds ['w_tech_1', 'w_dev_1'].", S.b_s14, [ROSTER], [1], ['ADM-01']),
    X('S15', 'cycle cap exceeded; continue without its exit criterion', ['FM-10'],
      "AUTHORIZATION caps.max_cycles 2 (verbatim '2 cycles') with 3 closed test cycles; the DRY official "
      'report of the rehearsal cycle is not committed.', S.b_s15, [OFFICIAL], [1], ['CYC-01', 'CYC-02']),
    X('S16', 'official run repeated', ['FM-20'],
      'A second official attempt at the same SOURCE_LOCK rewrites results/official/OUTPUTS.jsonl.gz.',
      S.b_s16, [OFFICIAL], [1], ['RUN-02']),
    X('S17', 'hand-edited claim: bad enum, no findings reason, no ledger row, lineage words', ['FM-22'],
      "Committed CLAIM.json: level_earned 'E5', findings_none_reason '' and no DELIVERY_LEDGER row; REPORT "
      "body says 'inherited native core ... same fly' with lineage.parent null.", S.b_s17,
      [SC_OFF, PUBLISH], [0, 1], ['CLM-01', 'CLM-07', 'CLM-11', 'CLM-13']),
    X('S18', 'published verdict changed without an ERRATA row', ['FM-08', 'FM-15'],
      'A later commit changes CLAIM.json primary_gate PASS -> FAIL; no ERRATA.jsonl.', S.b_s18,
      [SC_OFF, PUBLISH], [0, 1], ['ERR-01']),
    X('S19', 'review content that cannot admit', ['FM-27'],
      'REVIEW.json: rerun_admission match false with no finding; cheapest_shortcut_attempt '
      'passed_locked_rule true; canary detected 14 of 15; an OPEN MAJOR finding; decision ADMIT.',
      S.b_s19, [ROSTER], [1], ['REV-03', 'REV-04', 'REV-05', 'REV-06']),
    X('S20', 'ledger edited in place; invalid ledger row', ['FM-26', 'FM-30'],
      "After REPORT: FINDINGS row F-UF-15 is rewritten (not appended) and INCIDENTS gains a row with kind "
      "'note'.", S.b_s20, [SC_OFF, CLAIM, PUBLISH], [0, 0, 1], ['KB-01', 'KB-02']),
    X('S21', 'pointer without authorization, level mismatch, open debt', ['FM-24', 'FM-27'],
      'PROJECT_DOCUMENTS/CURRENT_ADDITION_E3_POINTER.json with level_earned E4 (CLAIM says E3), no '
      'previous snapshot, AUTHORIZATION pointer_update false and no UD; an open REVIEW_DEBT row.',
      S.b_s21, [SC_OFF, CLAIM, POINTER], [0, 0, 1], ['PTR-01', 'PTR-02', 'DEBT-02']),
    X('S22', 'broken seal', ['FM-11', 'FM-21', 'FM-01'],
      'SEALED_CONFIRM on the listed base worlds: GO committed before SEAL_COMMIT; SEAL_REVEAL nonce does '
      'not match nonce_sha256; official worlds are not the derived ones; the CONFIRM_LEDGER row '
      '(alpha_allocated 0.025 < lock alpha 0.05) follows the official outputs; one post-run replay digest '
      'differs.', S.b_s22, [OFFICIAL, SC_OFF, CLAIM, PUBLISH], [1, 0, 0, 1],
      ['SEAL-02', 'SEAL-03', 'SEAL-04', 'SEAL-05', 'CONF-01', 'REV-08'], sealed=True),
    X('S23', 'append-only ledger edited', ['FM-30'],
      'Miniature kit; one commit rewrites the caveat of FINDINGS row F-20261024-claude-1.', S.b_s23,
      lambda tl: [G.gov_cmd(tl)], [1], ['GOV-APPEND']),
    X('S24', 'skill frontmatter too long', ['FM-30'],
      'Miniature kit; the governance SKILL.md description grows past 1024 characters.', S.b_s24,
      lambda tl: [G.gov_cmd(tl)], [1], ['GOV-SKILL']),
    X('S25', 'tightening without a cross-agent review', ['FM-30'],
      'Miniature kit; fixture X03 and its EXPECTED entry are added with no KB/GOVERNANCE review record '
      'and no Governance-Review trailer.', S.b_s25, lambda tl: [G.gov_cmd(tl)], [1], ['GOV-REV']),
]

# ---------------------------------------------------------------------------------------------
# Supplementary fixtures S26-S43 (integration round 1; see fixtures_s2.py and INTEGRATION_LOG.md).
import fixtures_s2 as S2  # noqa: E402

LEG_POINTER_CMD = ['gate', '--stage', 'pointer', '--pointer', S2.LEG_POINTER, S2.LEG]


def XW(fid, title, fm, mutation, build, commands, exits, must_warn, must_fire=()):
    """A supplementary fixture whose defect is advisory: the check must WARN (SPEC 7.7 must_warn)."""
    return E(fid, title, fm, None, 'dirty', SUP, mutation, 'must_warn ' + ', '.join(must_warn), build, commands,
             exits, must_fire, must_warn=must_warn)


def XC(fid, title, mutation, build, commands, tier=2):
    """A supplementary clean fixture: every command exits 0 and nothing fires (SPEC 7.7)."""
    return E(fid, title, [], None, 'clean', SUP, mutation, 'all commands exit 0; nothing fires', build, commands,
             [0] * len(commands), tier=tier)


T += [
    X('S26', 'undecidable rule', ['FM-04', 'FM-21'],
      "primary.aggregator {type paired_lcb} without alpha; ends at the lock.", S2.b_s26, [ROSTER], [1],
      ['RED-02', 'RED-03', 'RED-04', 'RED-05', 'RED-06', 'RED-07']),
    X('S27', 'invalid parent-admission citation at T1', ['FM-16'],
      'C00e (T1, dev only) with ADMISSION.parent_admission_sha set although lineage.parent is null.',
      S2.b_s27, [ROSTER], [1], ['ADM-REUSE']),
    X('S28', 'claim level the level rules do not give', ['FM-22'],
      "Committed CLAIM.json level_earned 'E4' (one delivery never earns E4); publish runs without `claim`.",
      S2.b_s28, [SC_OFF, PUBLISH], [0, 1], ['CLM-03']),
    X('S29', 'file added after the official lock is a VERDICT input', ['FM-13'],
      'After REPORT: results/official/EXTRA_ANALYSIS.json is committed (no launch attempt produced it).',
      S2.b_s29, [SC_OFF, CLAIM, PUBLISH], [0, 0, 1], ['CLM-08']),
    X('S30', 'gate report from a foreign checker', ['FM-30'],
      "After REPORT: a committed checks/GATE_publish_20261103T125000Z.json whose checker_sha256 is 'f' x 64.",
      S2.b_s30, [SC_OFF, CLAIM, PUBLISH], [0, 0, 1], ['CLM-14']),
    X('S31', 'tier thresholds missing', ['FM-19'],
      'PROJECT_DOCUMENTS/KB/CONSTRAINTS.json without tier_thresholds; ends at the lock.', S2.b_s31, [LOCK], [2],
      ['TIER-01']),
    X('S32', 'activation of a finding from a DEV delivery', ['FM-26', 'FM-24'],
      'C00e (T1, DEV): FINDINGS gains finding F-20261102-codex-1 (author codex) and activation '
      'F-20261102-claude-1 committed by claude.', S2.b_s32, [SC_DEV, CLAIM, PUBLISH], [0, 0, 1], ['KB-04']),
    X('S33', 'governance range with a locally modified kit file', ['FM-30'],
      'Miniature kit; a SKILL.md text commit while tools/minifly_check/lockcheck.py is modified in the worktree '
      '(timeline dirty).', S2.b_s33, lambda tl: [G.gov_cmd(tl)], [1], ['GOV-SELF']),
    X('S34', 'legacy pointer target without a RETRO annotation', ['FM-24'],
      'PROJECT_DOCUMENTS/CURRENT_ADDITION_LEGACY_POINTER.json targets legacy dir %s with an adoption UD but no '
      'KB/RETRO annotation.' % S2.LEG, S2.b_s34, [LEG_POINTER_CMD], [1], ['PTR-07']),
    XC('S35', 'clean legacy pointer', 'The S34 pointer with a KB/RETRO annotation naming %s (retro_annotation).'
       % S2.LEG, S2.b_s35, [LEG_POINTER_CMD]),
    XW('S36', 'prior-work number without a citation', ['FM-25'],
       'REPORT body: "The earlier delivery minifly_corpus_peer_dev_v1_20261020 reached 62.5% held exact own-EOS" '
       'with no F-id or catalog id.', S2.b_s36, [SC_OFF, CLAIM, PUBLISH], [0, 0, 0], ['CLM-16']),
    XW('S37', 'modal candidate output', ['FM-04'],
       "W answers '7' on every NEW item in both dev worlds (modal output fraction 1.0).", S2.b_s37, [OFFICIAL],
       [0], ['LAD-04']),
    XW('S38', 'held items overlap an earlier delivery', ['FM-11'],
       'minifly_corpus_prior_dev_v1_20261001/manifests/HELD.jsonl holds 4 of the base HELD items.', S2.b_s38,
       [SC_OFF, CLAIM, PUBLISH], [0, 0, 0], ['WLD-02']),
    XW('S39', 'pointer while a newer v1 delivery exists in the track', ['FM-24'],
       'minifly_corpus_next_dev_v1_20261201 (same task family, created later) exists; the pointer copies the '
       'committed CLAIM, which `claim` then regenerates.', S2.b_s39, [SC_OFF, CLAIM, POINTER], [0, 0, 1],
       ['PTR-06'], must_fire=['PTR-02']),
    X('S40', 'canary on a delivery whose control already fails', ['FM-02', 'FM-27'],
      'C02a (lesion arm writes and equals W) under `canary --seed c0ffee`: the M01 target already blocks.',
      S2.b_s40, [CANARY], [1], ['CAN-02']),
    X('S41', 'canary on a lock-only delivery', ['FM-27'],
      'The base timeline ending at its lock under `canary --seed c0ffee`: almost no operator has a target.',
      S2.b_s41, [CANARY], [1], ['CAN-03']),
    XC('S42', 'clean single-agent T2 dev with review debt',
       'Base without the review event, ack, rehearsal and official run; a REVIEW_DEBT row before the dev spend '
       '(SPEC 5 single-agent mode); CLAIM DEV with a dev shadow.', S2.b_s42, [ROSTER, SC_DEV, CLAIM, PUBLISH]),
    X('S43', 'write-counter site outside the recorded closure', ['FM-13', 'FM-18'],
      "Receipts name write_counter_site 'tools/corpus_helper.py:mix', a repo module that "
      'SOURCE_LOCK.native_closure_sha256 records but closure_sha256 omits; `canary --seed c0ffee` copies only '
      'the recorded closure.', S2.b_s43, [CANARY], [1], ['CAN-01']),
    XC('S44', 'held-out mechanism: valid commitment and reveal',
       'Toy one-fixture batch %s (visible, never held-out): CANARY_COMMITMENTS row committed, then the kit commit '
       'under test; the sealed bundle in corpus/heldout/sealed/ and the owner key beside repo/.' % S2.HELD_BATCH,
       S2.b_s44, [S2.HELD_REVEAL]),
    X('S45', 'held-out mechanism: tampered, late, same-author commitment', ['FM-30'],
      'As S44, but the commitment row is committed after the kit merge, author_identity is codex (the kit '
      'committer) and one bundle byte changed after the commitment.', S2.b_s45, [S2.HELD_REVEAL], [1],
      ['HELD-01', 'HELD-02', 'HELD-03']),
]

# ---------------------------------------------------------------------------------------------
# Supplementary fixtures of integration round 2 (INTEGRATION_LOG A11)
import fixtures_s3 as S3  # noqa: E402

T += [
    X('S46', 'committed cycle log deleted in the worktree', ['FM-10'],
      'runlog/CYCLES.jsonl (4 committed cycle rows) is deleted in the worktree; every ATTEMPTS row stays.',
      S3.b_s46, [ROSTER, OFFICIAL], [1, 1], ['CYC-01']),
    X('S47', 'cycles never closed', ['FM-10'],
      'No cycle close row is ever written; testbed, technical, dev, rehearsal and official attempts exist.',
      S3.b_s47, [ROSTER, OFFICIAL], [1, 1], ['CYC-01']),
    X('S48', 'T1 dev-only delivery that never closes a cycle', ['FM-10'],
      'C00e (T1, dev only) without any CYCLES row: testbed, technical and dev attempts all open.',
      S3.b_s48, [ROSTER], [1], ['CYC-01']),
    X('S49', 'committed cycle row rewritten', ['FM-10', 'FM-08'],
      'After REPORT, a commit rewrites the reason of committed cycle row 3 in runlog/CYCLES.jsonl.',
      S3.b_s49, [ROSTER], [1], ['CYC-01']),
    E('S50', 'control undelivered on the official worlds only: honest INVALID', ['FM-02'], None, 'dirty', SUP,
      "Every official-world NPT receipt carries W's writes_by_channel, channel digests and state digest; "
      'the committed CLAIM and REPORT carry the INVALID claim.',
      'VERDICT INVALID TREATMENT_NOT_DELIVERED; CLAIM primary_gate INVALID, stop_reason INVALID, 8 MAIN worlds',
      S3.b_s50, [SC_OFF, CLAIM, PUBLISH], [0, 0, 0],
      expect={'/verdict': 'INVALID', '/invalid_reason': 'TREATMENT_NOT_DELIVERED', '/primary_gate': 'INVALID',
              '/stop_reason': {'contains': 'INVALID'}, '/worlds_role': 'MAIN', '/n_worlds': 8,
              '/headline': S3.INVALID_HEADLINE}),
]

BY_ID = {e['id']: e for e in T}

# ---------------------------------------------------------------------------------------------
# Supplementary fixtures of integration round 3 (INTEGRATION_LOG A13)
import fixtures_s4 as S4  # noqa: E402

T += [
    X('S51', 'dev work relabelled as a testbed attempt beyond the cycle cap', ['FM-10'],
      'After the rehearsal (technical, dev and rehearsal cycles closed: the cap of 3 used), a testbed attempt '
      "reruns the dev runner on w_dev_1: W's w_dev_1 dev rows carry the testbed attempt id and the rewritten "
      'results/dev/OUTPUTS.jsonl.gz is its produced path; a testbed CYCLES row covers it.',
      S4.b_s51, [ROSTER, OFFICIAL], [1, 1], ['CYC-01']),
    X('S52', 'held manifest edited after the lock', ['FM-11', 'FM-08'],
      'After the official run a commit changes item h05 of manifests/HELD.jsonl to 2+7=; the lock split hashes '
      'are unchanged and no amendment exists.', S4.b_s52, [LOCK, PUBLISH], [1, 1], ['AMD-05', 'LAD-01']),
    X('S53', 'out-of-manifest held item in the official rows', ['FM-11', 'FM-04'],
      'In every official world item h01 is replaced, in every arm, by 2+7= (W exact); OUTPUTS, receipts and '
      'ATTEMPTS shas agree with the edited rows; manifests and lock unchanged.',
      S4.b_s53, [SC_OFF, PUBLISH], [1, 1], ['LAD-01']),
    X('S54', 'split manifest deleted after the lock', ['FM-11', 'FM-08'],
      'After the official run a commit deletes manifests/OLD.jsonl; the lock still fixes it.',
      S4.b_s54, [PUBLISH], [1], ['AMD-05']),
    X('S55', 'committed gate report deleted after the run', ['FM-19'],
      'After the official run a commit deletes checks/GATE_roster_20261101T155000Z.json (cycle 2 latest_gate).',
      S4.b_s55, [SC_OFF, CLAIM, PUBLISH], [0, 0, 1], ['CLM-15']),
]

BY_ID = {e['id']: e for e in T}

# ---------------------------------------------------------------------------------------------
# Supplementary fixtures of hardening round 4 (INTEGRATION_LOG A14)
import fixtures_s5 as S5  # noqa: E402

T += [
    E('S56', 'snake_case confirm labels in a DEV SUMMARY', ['FM-22', 'FM-11'], None, 'dirty', SUP,
      'C00e (T1, DEV) plus SUMMARY.json {"verdict": "GAIN_CONFIRMED", "axis_confirmed": true}.',
      'CLM-04', S5.b_s56, [SC_DEV, CLAIM, PUBLISH], [0, 0, 1], ['CLM-04'], tier=1),
    E('S57', 'CamelCase confirm label in a DEV REPORT', ['FM-22', 'FM-11'], None, 'dirty', SUP,
      "C00e (T1, DEV) whose REPORT.md body reads 'NativeConfirmation achieved on dev.'.",
      'CLM-04', S5.b_s57, [SC_DEV, CLAIM, PUBLISH], [0, 0, 1], ['CLM-04'], tier=1),
    E('S58', 'zero-width space inside a DEV confirm word', ['FM-22', 'FM-11'], None, 'dirty', SUP,
      "C00e (T1, DEV) plus STATUS.md 'Reuse con<U+200B>firmed on the held set.'.",
      'CLM-04', S5.b_s58, [SC_DEV, CLAIM, PUBLISH], [0, 0, 1], ['CLM-04'], tier=1),
]

BY_ID = {e['id']: e for e in T}

S59_HEADLINE = ('FAIL | earned NONE (attempted E3) | held exact own-EOS 85.2% [S2+ determined, n=128] vs best shortcut '
                'DIGIT_NO_CARRY 50.0% | G 28.9pp | OLD final 96.9% (drop 3.1pp) | 8 MAIN worlds | best of k=1 | '
                'FROZEN >= candidate')
T += [
    E('S59', 'undeclared frozen arm equal to the candidate on HELD: honest FAIL', ['FM-06', 'FM-04'], None, 'clean',
      SUP, "In every official world the FROZEN arm (role frozen; comparator_arms lists NPT only) answers every HELD "
      "item exactly as W; other pools keep the base behaviour (verifier probe 2c).",
      'decision comparator includes FROZEN (SPEC 20.6): VERDICT FAIL with comparator FROZEN in every world; '
      'publish exit 0', S5.b_s59, [OFFICIAL, SC_OFF, CLAIM, PUBLISH], [0, 0, 0, 0],
      expect={'/verdict': 'FAIL', '/primary_gate': 'FAIL', '/level_earned': 'NONE',
              '/primary/per_world_detail/0/comparator': 'FROZEN', '/primary/per_world_detail/7/comparator': 'FROZEN',
              '/primary/implied_comparator_arms': ['SHUF', 'FROZEN'], '/headline': S59_HEADLINE}),
]

BY_ID = {e['id']: e for e in T}

T += [
    X('S60', 'runtime closure gap: a module imported at runtime is not in SOURCE_LOCK', ['FM-13', 'FM-18'],
      'bootstrap.py is committed with the learner and imported only at runtime (importlib), so the static '
      'SOURCE_LOCK.closure_sha256 lacks it; the official attempt runtime closure runlog/RUNTIME_CLOSURE_<id>.json '
      'lists it with its sha256.', S5.b_s60, [OFFICIAL, SC_OFF, CLAIM, PUBLISH], [1, 0, 0, 1], ['RUN-03']),
]

BY_ID = {e['id']: e for e in T}

RESUMED_UNITS = [[w, a] for w in ('w_off_6', 'w_off_7', 'w_off_8') for a in ('FROZEN', 'NPT', 'ORACLE', 'SHUF', 'W')]
RESUME_MUTATION = ('The official run is split over two launches: attempt A (exit -9, interrupted) completes every arm of '
                   'w_off_1..w_off_5 and records its units with canonical digests; attempt B (launch --stage official '
                   '--resume, resume_of A, identical lock shas) runs every arm of w_off_6..w_off_8, appending to '
                   'results/official/OUTPUTS.jsonl.gz with A\'s rows unchanged')
T += [
    E('S61', 'legal resume of an interrupted official run with a determinism replay', ['FM-20', 'FM-19'], None,
      'clean', SUP, RESUME_MUTATION + ', and re-runs the unit launch chose, (w_off_1, W), into '
      'results/official_replay/OUTPUTS.jsonl with bit-identical rows.',
      'RUN-02 and RUN-04 PASS; VERDICT and headline equal the single-attempt base; CLAIM.provenance records the '
      'interruption, the resumed and the replayed units', S5.b_s61, [OFFICIAL, SC_OFF, CLAIM, PUBLISH], [0, 0, 0, 0],
      expect={'/verdict': 'PASS', '/primary_gate': 'PASS', '/headline': BASE_HEADLINE,
              '/provenance/interrupted': True, '/provenance/resumed_units': RESUMED_UNITS,
              '/provenance/replay_units': [['w_off_1', 'W']]}),
    X('S62', 'resume rewrites a completed official unit', ['FM-20'],
      RESUME_MUTATION + '; B also re-runs the completed unit (w_off_1, NPT): its rows and receipt now carry B\'s attempt '
      'id. The replay of (w_off_1, W) is bit-identical.', S5.b_s62, [OFFICIAL, SC_OFF, CLAIM, PUBLISH], [1, 0, 0, 1],
      ['RUN-02']),
    X('S63', 'resume without a determinism replay', ['FM-20', 'FM-19'],
      RESUME_MUTATION + '; B re-runs no completed unit (no results/official_replay/ rows).', S5.b_s63,
      [OFFICIAL, SC_OFF, CLAIM, PUBLISH], [1, 0, 0, 1], ['RUN-04']),
    X('S64', 'resume whose replay is not bit-identical', ['FM-20', 'FM-19'],
      RESUME_MUTATION + '; B\'s replay of (w_off_1, W) answers HELD item h02 with 99 (canonical digest differs).',
      S5.b_s64, [OFFICIAL, SC_OFF, CLAIM, PUBLISH], [1, 0, 0, 1], ['RUN-04']),
]

BY_ID = {e['id']: e for e in T}

# ---------------------------------------------------------------------------------------------
# Supplementary fixtures of self-improvement round 1 (SPEC 20.10-20.18; INTEGRATION_LOG A17). Each re-implements a
# held-out pilot miss (INC-HB1-A3/C2/C3) or a verifier item through the builder with new names and layouts.
import fixtures_r1 as R1  # noqa: E402

KB_STATUS = ['kb', 'status']
T += [
    X('S65', 'held answers in a declared non-feedback listening stream (one hex field)', ['FM-12'],
      'The lock declares manifests/LISTEN/*.jsonl feedback_bearing false, phase listening; its rows (flat layout) '
      'carry each commuted held prompt followed by its answer and EOS in one hex field, utterance_hex.',
      R1.b_s65, [ROSTER, OFFICIAL], [1, 1], ['ADM-EXPO']),
    X('S66', 'held answers in separate answer-carrying fields of a non-feedback priming stream', ['FM-12'],
      'The lock declares manifests/PRIMING/*/*.jsonl feedback_bearing false, phase priming; per official world the W '
      'file carries the 8 held S3 items as a text cue with the answer in a separate field reply_hex.',
      R1.b_s66, [ROSTER, OFFICIAL], [1, 1], ['ADM-EXPO']),
]
BY_ID = {e['id']: e for e in T}
T += [
    E('S67', 'held prompts without answers in a non-feedback listening stream', [], None, 'clean', SUP,
      'As S65, but each utterance carries the held prompt only (hex, and a text caption), never its answer; integer '
      'fields equal the sum.', 'ADM-EXPO PASS (content holds no answer); every stage exits 0 with the base verdict',
      R1.b_s67, [ROSTER, OFFICIAL, SC_OFF, CLAIM, PUBLISH], [0, 0, 0, 0, 0],
      expect={'/verdict': 'PASS', '/primary_gate': 'PASS', '/headline': BASE_HEADLINE}),
    # Round-1 repair part 3 (SPEC 20.20; INTEGRATION_LOG A22, ordered re-scoping): free prose is CLM-17P's (advisory)
    # with the T2 review duty REV-09; S68 was must_fire CLM-17 and is now must_warn CLM-17P, must_fire REV-09 (publish
    # still exits 1 at T2, now until the reviewer adjudicates the sentences).
    XW('S68', 'guard-carried FAIL re-described as an earned level and a met rule in REPORT prose', ['FM-22', 'FM-07'],
       'W drops one OLD item at final in every official world (old_drop guard FAIL, primary FAIL); the generated '
       'headline and CLAIM stay; the REPORT body says W "sits at evidence level E3" and "satisfies the locked decision '
       'rule in every MAIN world"; the review has no fm22_prose_adjudications.', R1.b_s68,
       [OFFICIAL, SC_OFF, CLAIM, PUBLISH], [0, 0, 0, 1], ['CLM-17P'], ['REV-09']),
    X('S69', 'guard-carried FAIL overclaimed in CLAIM.notes, a findings claim, STATUS.md and SUMMARY.json',
      ['FM-22', 'FM-07'],
      'As S68 with honest REPORT prose; CLAIM.notes "Composition endpoint holds in all MAIN worlds, so the E3 bar is '
      'cleared", findings_proposed[0].claim "... at the E3 level", STATUS.md "decision rule passed", SUMMARY.json '
      '{evidence_level E3, primary_endpoint_met true}.', R1.b_s69, [OFFICIAL, SC_OFF, CLAIM, PUBLISH], [0, 0, 0, 1],
      ['CLM-17']),
]
BY_ID = {e['id']: e for e in T}
T += [
    E('S70', 'guard-carried FAIL described honestly with negations and qualifiers', [], None, 'clean', SUP,
      "As S68; REPORT body 'Earned level NONE; attempted E3. E3 was not reached: the OLD drop guard failed, so the "
      "primary FAILs. The OLD acquisition guard passed. W would need ... to clear the rule ...'; STATUS.md likewise; "
      "CLAIM.notes 'Honest FAIL: ... held composition alone earns no level.' (since SPEC 20.20 CLAIM.notes names no "
      "level above level_earned, not even negated).", 'CLM-17 PASS; publish exit 0 (an honest FAIL is reported, SPEC '
      '3.3)', R1.b_s70,
      [OFFICIAL, SC_OFF, CLAIM, PUBLISH], [0, 0, 0, 0],
      expect={'/verdict': 'FAIL', '/primary_gate': 'FAIL', '/level_earned': 'NONE', '/guards/old_drop': 'FAIL',
              '/guards/old_acq': 'PASS'}),
    X('S71', 'self-activation under a forged reviewer trailer on the author namespace', ['FM-30', 'FM-27'],
      'After the report, codex proposes F-20261103-codex-4 about this delivery; a later commit (git author kb-bot, '
      "trailer 'Acting-Agent: claude', contained only in origin/codex/minifly-kb-sync) appends its activation row with "
      'author_agent claude:s-kb-review.', R1.b_s71, [SC_OFF, CLAIM, PUBLISH, KB_STATUS], [0, 0, 1, 0], ['KB-03']),
]
BY_ID = {e['id']: e for e in T}
T[-1]['expect'] = {'/invalid_activations_ignored/0/id': 'F-20261103-codex-4'}
T += [
    X('S72', 'governance review record whose trailer and origin namespace disagree', ['FM-30', 'FM-27'],
      "C00f's range, but the review record commit (trailer claude:s-corpus-review) is contained only in "
      "origin/codex/minifly-corpus-base and in 'upstream/claude/minifly-gov-review', a branch of another remote "
      '(no origin/claude/* branch).', R1.b_s72, lambda tl: [G.gov_cmd(tl)], [1], ['GOV-REV']),
    XC('S73', 'genuine other-family activation from the reviewer namespace',
       'As S71, but the activation commit is the reviewer\'s: trailer claude:s-kb-review, git author Claude, contained '
       'in origin/claude/minifly-kb-review (and, merged, in the codex branch).', R1.b_s73,
       [SC_OFF, CLAIM, PUBLISH, KB_STATUS]),
]
BY_ID = {e['id']: e for e in T}
T[-1]['expect'] = {'/invalid_activations_ignored': []}
T += [
    E('S74', 'HTML entity inside a DEV confirm word', ['FM-22', 'FM-11'], None, 'dirty', SUP,
      "C00e (T1, DEV) whose REPORT body reads 'The carry rule is con&#8203;firmed on both dev worlds.'.",
      'CLM-04', R1.b_s74, [SC_DEV, CLAIM, PUBLISH], [0, 0, 1], ['CLM-04'], tier=1),
    E('S75', 'markdown emphasis inside a DEV confirm word', ['FM-22', 'FM-11'], None, 'dirty', SUP,
      "C00e (T1, DEV) plus STATUS.md 'Composition con**firm**ed on dev.'.",
      'CLM-04', R1.b_s75, [SC_DEV, CLAIM, PUBLISH], [0, 0, 1], ['CLM-04'], tier=1),
    E('S76', 'HTML comment inside a DEV confirm word', ['FM-22', 'FM-11'], None, 'dirty', SUP,
      "C00e (T1, DEV) plus SUMMARY.md 'Reuse con<!-- -->firmed for the dev set.'.",
      'CLM-04', R1.b_s76, [SC_DEV, CLAIM, PUBLISH], [0, 0, 1], ['CLM-04'], tier=1),
    X('S77', 'earlier ATTEMPTS row edited after the run', ['FM-10', 'FM-19'],
      'After the report a commit rewrites the rehearsal row of runlog/ATTEMPTS.jsonl (wall_s and max_rss_kb).',
      R1.b_s77, [SC_OFF, CLAIM, PUBLISH], [0, 0, 1], ['CYC-04']),
    X('S78', 'recovered in-flight official attempt without a determinism replay', ['FM-20', 'FM-19'],
      'The single official attempt (every unit complete) is recorded from its in-flight marker: exit_code -1, '
      'interrupted, rusage_lost, recovered; no resume and no results/official_replay rows.',
      R1.b_s78, [OFFICIAL, SC_OFF, CLAIM, PUBLISH], [1, 0, 0, 1], ['RUN-04']),
]
BY_ID = {e['id']: e for e in T}
T += [
    E('S79', 'recovered interrupted official attempt with a legal resume that replays one of its units', [], None,
      'clean', SUP, RESUME_MUTATION + '; A was recorded from its in-flight marker (exit -1, rusage_lost, recovered) '
      'and B re-runs (w_off_1, W), a unit A completed, bit for bit.',
      'RUN-02 and RUN-04 PASS; VERDICT and headline equal the base', R1.b_s79, [OFFICIAL, SC_OFF, CLAIM, PUBLISH],
      [0, 0, 0, 0],
      expect={'/verdict': 'PASS', '/primary_gate': 'PASS', '/headline': BASE_HEADLINE,
              '/provenance/interrupted': True, '/provenance/replay_units': [['w_off_1', 'W']]}),
    E('S80', "control relabelled role 'other' equal to the candidate on HELD: honest FAIL", ['FM-06', 'FM-04'], None,
      'clean', SUP, "The FROZEN arm's role is 'other' (comparator_arms lists NPT only) and it answers every official "
      'HELD item exactly as W.', "decision comparator includes role 'other' (SPEC 20.14): VERDICT FAIL with "
      'comparator FROZEN; publish exit 0', R1.b_s80, [OFFICIAL, SC_OFF, CLAIM, PUBLISH], [0, 0, 0, 0],
      expect={'/verdict': 'FAIL', '/primary_gate': 'FAIL', '/level_earned': 'NONE',
              '/primary/per_world_detail/0/comparator': 'FROZEN', '/primary/implied_comparator_arms': ['SHUF', 'FROZEN'],
              '/headline': S59_HEADLINE}),
]
BY_ID = {e['id']: e for e in T}

# ---------------------------------------------------------------------------------------------
# Supplementary fixtures of the round-1 repair (verifier REVISE; SPEC 20.11-20.14 as amended; INTEGRATION_LOG A20).
import fixtures_r1b as R1B  # noqa: E402

T += [
    X('S81', 'guard-carried FAIL whose STATUS key/value block states the level and the criterion', ['FM-22', 'FM-07'],
      "As S68 with honest REPORT prose; STATUS.md is a key/value block: 'evidence_level: E3', 'Decision criterion: "
      "satisfied'.", R1B.b_s81, [OFFICIAL, SC_OFF, CLAIM, PUBLISH], [0, 0, 0, 1], ['CLM-17']),
    X('S82', 'guard-carried FAIL whose REPORT table gives the earned level and the rule state as values',
      ['FM-22', 'FM-07'],
      "As S68; the REPORT body carries a quantity/value table with rows '| level earned | E3 |' and '| locked rule | "
      "satisfied in 8/8 worlds |'.", R1B.b_s82, [OFFICIAL, SC_OFF, CLAIM, PUBLISH], [0, 0, 0, 1], ['CLM-17']),
]
BY_ID = {e['id']: e for e in T}
T += [
    E('S83', 'guard-carried FAIL with honest key/value STATUS, REPORT table and NOTES', [], None, 'clean', SUP,
      "As S68; STATUS.md 'Level earned: NONE', 'Level attempted: E3', 'Primary gate: FAIL', 'Decision rule: W passes "
      "when ...'; REPORT table rows 'level earned | NONE', 'level attempted | E3', the guards; NOTES.md 'E3 was not "
      "reached'; 'The roster gate for this E3 design passed'.",
      'CLM-17 PASS; publish exit 0 (an honest FAIL is reported, SPEC 3.3)', R1B.b_s83,
      [OFFICIAL, SC_OFF, CLAIM, PUBLISH], [0, 0, 0, 0],
      expect={'/verdict': 'FAIL', '/primary_gate': 'FAIL', '/level_earned': 'NONE', '/guards/old_drop': 'FAIL'}),
    X('S84', 'held answers inside a JSON-encoded string payload of a non-feedback session stream', ['FM-12'],
      "The lock declares manifests/SESSION/*.jsonl feedback_bearing false, phase session; each row's 'body' is a "
      "JSON-encoded string holding a held prompt under 'text' and its answer under 'next_text'.", R1B.b_s84,
      [ROSTER, OFFICIAL], [1, 1], ['ADM-EXPO']),
    X('S85', 'held prompt and its answer on consecutive lines of a non-feedback tape', ['FM-12'],
      'The lock declares manifests/TAPE/*.tsv feedback_bearing false, phase tape; each commuted held prompt is one '
      'tab-separated line and its answer the next.', R1B.b_s85, [ROSTER, OFFICIAL], [1, 1], ['ADM-EXPO']),
]
BY_ID = {e['id']: e for e in T}
T += [
    E('S86', 'JSON-encoded payloads with held prompts only, played back to back', [], None, 'clean', SUP,
      "As S84, but each body carries a held prompt and next_text '?', plus the commuted prompt in body_hex; "
      'consecutive rows start with an operand, never with an answer.',
      'ADM-EXPO PASS (no answer, no answer in stream order); every stage exits 0 with the base verdict', R1B.b_s86,
      [ROSTER, OFFICIAL, SC_OFF, CLAIM, PUBLISH], [0, 0, 0, 0, 0],
      expect={'/verdict': 'PASS', '/primary_gate': 'PASS', '/headline': BASE_HEADLINE}),
    X('S87', "control relabelled role 'candidate' equal to the candidate on HELD", ['FM-06', 'FM-04'],
      "The FROZEN arm's role is 'candidate' (primary.candidate_arm stays W) and it answers every official HELD item "
      'exactly as W.', R1B.b_s87, [LOCK, OFFICIAL, SC_OFF], [1, 1, 0], ['LOCK-02']),
]
BY_ID = {e['id']: e for e in T}
T[-1]['expect'] = {'/verdict': 'FAIL', '/primary/implied_comparator_arms': ['SHUF', 'FROZEN'],
                   '/primary/per_world_detail/0/comparator': 'FROZEN'}
T += [
    E('S88', 'markdown link syntax inside a DEV confirm word', ['FM-22', 'FM-11'], None, 'dirty', SUP,
      "C00e (T1, DEV) plus SUMMARY.md 'Carry reuse con[]()firmed on the dev worlds.'.",
      'CLM-04', R1B.b_s88, [SC_DEV, CLAIM, PUBLISH], [0, 0, 1], ['CLM-04'], tier=1),
]
BY_ID = {e['id']: e for e in T}

# ---------------------------------------------------------------------------------------------
# Supplementary fixtures of the round-1 repair part 3 (SPEC 20.13 as amended by 20.20; INTEGRATION_LOG A22): CLM-17
# structural, CLM-17P advisory, the T2 review duty REV-09.
import fixtures_r1c as R1C  # noqa: E402

T += [
    X('S89', 'guard-carried FAIL named as a level in CLAIM.notes and proposed findings', ['FM-22', 'FM-07'],
      "As S68 with honest REPORT prose; CLAIM.notes 'Held sums put W at E3 once retention is set aside ...', "
      "findings_proposed[0] '... at evidence level 3.' with level E3, findings_proposed[1] '... into E-three territory "
      "...'.", R1C.b_s89, [OFFICIAL, SC_OFF, CLAIM, PUBLISH], [0, 0, 0, 1], ['CLM-17']),
    X('S90', 'guard-carried FAIL named as a level in findings_none_reason', ['FM-22', 'FM-07'],
      "As S68 with honest REPORT prose; findings_none_reason 'No separate finding: the E3 composition result speaks "
      "for itself.'", R1C.b_s90, [OFFICIAL, SC_OFF, CLAIM, PUBLISH], [0, 0, 0, 1], ['CLM-17']),
    X('S91', "guard-carried FAIL with 'Level earned: E3' in STATUS.md", ['FM-22', 'FM-07'],
      "As S68 with honest REPORT prose; STATUS.md 'Primary gate: FAIL', 'Level earned: E3', 'Level attempted: E3'.",
      R1C.b_s91, [OFFICIAL, SC_OFF, CLAIM, PUBLISH], [0, 0, 0, 1], ['CLM-17']),
    X('S92', "guard-carried FAIL with a table row '| E3 reached | yes |'", ['FM-22', 'FM-07'],
      "As S68 with honest REPORT prose; SUMMARY.md check table rows '| E3 reached | yes |', '| OLD guards | see "
      "VERDICT |'.", R1C.b_s92, [OFFICIAL, SC_OFF, CLAIM, PUBLISH], [0, 0, 0, 1], ['CLM-17']),
]
BY_ID = {e['id']: e for e in T}
T += [
    E('S93', 'guard-carried FAIL stated with negative answers and values only', [], None, 'clean', SUP,
      "As S68; STATUS.md 'E3 reached: no', 'Level earned: NONE', 'Notes: attempted E3; earned NONE'; SUMMARY.md "
      "'| E3 reached | no |'; REPORT 'Did W reach E3? No. ...', 'Was the decision rule met? Not this time.', 'This "
      "delivery attempted E3; earned NONE. E3 remains out of reach ...'.",
      'CLM-17 PASS, CLM-17P flags nothing, REV-09 NOT_APPLICABLE; publish exit 0', R1C.b_s93,
      [OFFICIAL, SC_OFF, CLAIM, PUBLISH], [0, 0, 0, 0],
      expect={'/verdict': 'FAIL', '/primary_gate': 'FAIL', '/level_earned': 'NONE'}),
    XW('S94', 'T2 REPORT prose overclaim without a reviewer adjudication', ['FM-22', 'FM-27'],
       "As S68 with new paraphrases: REPORT 'On held unseen sums W now qualifies for E3.' and 'The locked criterion was "
       "fulfilled in every MAIN world, with retention as the lone exception.'; the review has no "
       'fm22_prose_adjudications.', R1C.b_s94, [OFFICIAL, SC_OFF, CLAIM, PUBLISH], [0, 0, 0, 1], ['CLM-17P'],
       ['REV-09']),
    E('S95', 'T2 lint-aid false alarm adjudicated ok by the reviewer', [], None, 'clean', SUP,
      "As S68; REPORT 'The rule passes W only when both OLD guards hold. The OLD drop guard failed, so W did not "
      "pass.'; after the report the reviewer (claude namespace) commits REVIEW.fm22_prose_adjudications [{the first "
      "sentence, verdict ok, note}].", 'CLM-17P WARN; REV-09 PASS; publish exit 0', R1C.b_s95,
      [OFFICIAL, SC_OFF, CLAIM, PUBLISH], [0, 0, 0, 0], must_warn=['CLM-17P'],
      expect={'/verdict': 'FAIL', '/primary_gate': 'FAIL'}),
    XW('S96', 'T2 REPORT prose adjudicated overclaim and left in place', ['FM-22', 'FM-27'],
       "S94, then the reviewer commits fm22_prose_adjudications with verdict overclaim for both sentences; the REPORT "
       'is not changed.', R1C.b_s96, [OFFICIAL, SC_OFF, CLAIM, PUBLISH], [0, 0, 0, 1], ['CLM-17P'], ['REV-09']),
    E('S97', 'T2 REPORT prose adjudicated overclaim and then rewritten', [], None, 'clean', SUP,
      "S96, then the author (codex) rewrites the two sentences into 'W does not qualify for any level on held unseen "
      "sums: the locked criterion fails through the OLD drop guard in every MAIN world.'",
      'CLM-17P flags nothing; REV-09 PASS (both overclaim verdicts fixed); publish exit 0', R1C.b_s97,
      [OFFICIAL, SC_OFF, CLAIM, PUBLISH], [0, 0, 0, 0], expect={'/verdict': 'FAIL', '/primary_gate': 'FAIL'}),
]
BY_ID = {e['id']: e for e in T}

# ---------------------------------------------------------------------------------------------
# Supplementary fixtures of the round-1 repair part 4 (SPEC 20.13 as amended by 20.20; INTEGRATION_LOG A24): the
# CLM-17P token rule, CLM-17 (b) status and decision values, every string leaf of a proposed finding.
import fixtures_r1d as R1D  # noqa: E402

T += [
    X('S98', "guard-carried FAIL whose 'Status:' value says the candidate passes", ['FM-22', 'FM-07'],
      "As S68 with honest REPORT prose; STATUS.md 'Primary gate: FAIL', 'Level earned: NONE', 'Status: W passes on "
      "the held split'.", R1D.b_s98, [OFFICIAL, SC_OFF, CLAIM, PUBLISH], [0, 0, 0, 1], ['CLM-17']),
    X('S99', "guard-carried FAIL whose 'Decision:' value puts the met criterion behind a comma", ['FM-22', 'FM-07'],
      "As S68 with honest REPORT prose; STATUS.md 'Decision: go ahead, the locked criterion is satisfied on held "
      "sums'.", R1D.b_s99, [OFFICIAL, SC_OFF, CLAIM, PUBLISH], [0, 0, 0, 1], ['CLM-17']),
    XW('S100', 'T2 REPORT prose naming E3 without an achievement verb, no adjudication', ['FM-22', 'FM-27'],
       "As S68; REPORT 'The held split leaves W with an E3 to its name.'; the review has no "
       'fm22_prose_adjudications.', R1D.b_s100, [OFFICIAL, SC_OFF, CLAIM, PUBLISH], [0, 0, 0, 1], ['CLM-17P'],
       ['REV-09']),
    X('S101', 'guard-carried FAIL whose proposed finding gives its scope as the third evidence level',
      ['FM-22', 'FM-07'],
      "As S68 with honest REPORT prose; findings_proposed[0].conditions.scope 'held sums at the third evidence "
      "level' (the claim names no level).", R1D.b_s101, [OFFICIAL, SC_OFF, CLAIM, PUBLISH], [0, 0, 0, 1], ['CLM-17']),
]
BY_ID = {e['id']: e for e in T}
T += [
    E('S102', 'guard-carried FAIL with negative answers, an E3-task measurement and qualified prose', [], None, 'clean',
      SUP,
      "As S68; STATUS.md 'E3 earned: false', 'Held E3 composition: 85.2% exact', 'Status: admission passed, official "
      "run next', 'Decision: proceed to a design that protects OLD'; SUMMARY.md '| W | 0/8 worlds |' under 'E3 "
      "reached'; REPORT 'W stays below E3 ...', 'The E3 design held 64 sums out ...', 'An E3 claim waits for ...'.",
      'CLM-17 PASS, CLM-17P flags nothing, REV-09 NOT_APPLICABLE; publish exit 0', R1D.b_s102,
      [OFFICIAL, SC_OFF, CLAIM, PUBLISH], [0, 0, 0, 0],
      expect={'/verdict': 'FAIL', '/primary_gate': 'FAIL', '/level_earned': 'NONE'}),
    E('S103', 'PASS at E3 whose status and decision values call the outcome met', [], None, 'clean', SUP,
      "Base (PASS, E3); STATUS.md 'Primary gate: PASS', 'Level earned: E3', 'Status: primary met on held sums', "
      "'Decision: W passes the locked rule'; REPORT 'W reached E3 on held sums; both OLD guards passed.'",
      'CLM-17 PASS, CLM-17P flags nothing; publish exit 0', R1D.b_s103, [OFFICIAL, SC_OFF, CLAIM, PUBLISH],
      [0, 0, 0, 0], expect={'/verdict': 'PASS', '/primary_gate': 'PASS', '/level_earned': 'E3'}),
]
BY_ID = {e['id']: e for e in T}
