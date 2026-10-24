"""Commit plan of the corpus base delivery (T2, MAIN, PASS) as a Timeline.

Commits (label: what):
  kb         PROJECT_DOCUMENTS/KB overlay (CONSTRAINTS draft values, minimal ledgers)
  lock       ENDPOINT_LOCK, AUTHORIZATION, params, admit.py, manifests, lock gate report
  testbed    C0: CEILING/FLOOR rows on 8 technical worlds, ATTEMPTS, TESTBED.json, cycle 1 testbed
  learner    learner.py, run.py, SOURCE_LOCK.json
  technical  C1: native rows + receipts on w_tech_1, FIXTURES, ADMISSION
  review     cross-family review (claude namespace): REVIEW.json, scripts, outputs, canary report
  roster     roster gate report, cycle 2 technical
  dev        C2: dev rows + receipts (w_dev_1, w_dev_2)
  dev_close  results/dev/SCORES.json, cycle 3 dev
  ack        reviewer acks the final SOURCE_LOCK (claude namespace)
  rehearsal  C3: rehearsal rows (w_reh_1), dry official gate report, cycle 4 rehearsal
  official   official rows + receipts on 8 listed MAIN worlds
  report     REPORT.md whose first line is the hand-computed headline
"""
import copy

import base_artifacts as BA
import base_files as BF
import base_items as BI
import refcalc as RC
from mrk_util import Timeline, commit_sha, dump_compact, dump_json, dump_jsonl, gzip_bytes, sha256_bytes

D = BF.D
P = D + '/'
FAMILY_ID = 'FAM-' + sha256_bytes(D.encode('utf-8'))[:8]
LABELS = ['kb', 'lock', 'testbed', 'learner', 'technical', 'review', 'roster', 'dev', 'dev_close',
          'ack', 'rehearsal', 'official', 'report']
UTC = {'kb': '20261101T080000Z', 'lock': '20261101T090000Z', 'testbed': '20261101T100000Z',
       'learner': '20261101T110000Z', 'technical': '20261101T120000Z', 'review': '20261101T150000Z',
       'roster': '20261101T160000Z', 'dev': '20261102T090000Z', 'dev_close': '20261102T100000Z',
       'ack': '20261102T110000Z', 'rehearsal': '20261102T120000Z', 'official': '20261103T090000Z',
       'report': '20261103T120000Z'}


class BaseBuild(object):
    def __init__(self, opts=None):
        self.o = dict(opts or {})
        self.bh = BI.default_behaviour()
        self.bh.update(copy.deepcopy(self.o.get('behaviour', {})))
        self.tl = Timeline()
        self.files = {}          # current tree, repo paths
        self.attempts = []
        self.cycles = []
        self.notes = {}

    # -- helpers ---------------------------------------------------------------------------------
    def commit(self, label, changes, who='codex', utc=None):
        meta = BF.CODEX if who == 'codex' else BF.CLAUDE
        changes = dict(changes)
        changes.update(self.o.get('extra_files', {}).get(label, {}))
        for path in self.o.get('drop_files', {}).get(label, []):
            changes.pop(path, None)
        self.files.update(changes)
        self.tl.add(label, utc or UTC[label], meta['author'], meta['trailers'], [BF.CODEX_BRANCH],
                    changes)
        for hook in self.o.get('after', {}).get(label, []):
            hook(self)

    def who(self, label, default):
        return self.o.get('who', {}).get(label, default)

    def patch(self, key, obj, *args):
        fn = self.o.get(key)
        if fn is not None:
            res = fn(obj, *args)
            if res is not None:
                return res
        return obj

    def rows_file(self, rows):
        return gzip_bytes(dump_jsonl(rows))

    def runtime_closure(self, aid, label):
        """(delivery-relative path, bytes) of runlog/RUNTIME_CLOSURE_<aid>.json, the file the runner of every
        counted run writes at its end (SPEC 20.7, RUN-03): every repo file in sys.modules, here the SOURCE_LOCK
        closure (the corpus stubs import nothing else), plus opts['runtime_extra'][label] entries; None when
        opts['no_runtime_closure'] names the label."""
        if label in self.o.get('no_runtime_closure', ()):
            return None
        files = [{'path': p, 'sha256': h} for p, h in sorted(self.sl['doc']['closure_sha256'].items())]
        files += [{'path': p, 'sha256': h} for p, h in self.o.get('runtime_extra', {}).get(label, [])]
        doc = {'schema': 'minifly.runtime_closure.v1', 'attempt_id': aid,
               'files': sorted(files, key=lambda f: f['path'])}
        doc = self.patch('runtime_closure_patch', doc, label)
        return 'runlog/RUNTIME_CLOSURE_%s.json' % aid, dump_json(doc)

    def add_runtime_closure(self, aid, label, f, prod):
        rc = self.runtime_closure(aid, label)
        if rc is not None:
            f[P + rc[0]] = rc[1]
            prod.append((rc[0], sha256_bytes(rc[1])))

    # -- build -----------------------------------------------------------------------------------
    def build(self):
        skip = set(self.o.get('skip', []))
        stop = self.o.get('stop_after')
        for label in LABELS:
            if label not in skip:
                getattr(self, 'c_' + label)()
            if label == stop:
                break
        self.fix_branches()
        return self.tl

    def fix_branches(self):
        # Linear history: the reviewer branch contains every commit up to its last commit.
        idx = [i for i, c in enumerate(self.tl.commits)
               if c['trailers'].get('Acting-Agent', [''])[0].startswith('claude')]
        last_claude = max(idx) if idx else -1
        for i, c in enumerate(self.tl.commits):
            c['branches'] = [BF.CODEX_BRANCH] + ([BF.CLAUDE_BRANCH] if i <= last_claude else [])

    def c_kb(self):
        kb = BF.kb_files(**self.o.get('kb', {}))
        self.constraints_sha = sha256_bytes(kb[BF.KB + 'CONSTRAINTS.json'])
        self.commit('kb', kb)

    def c_lock(self):
        f = {P + 'params.json': BF.params_bytes(self.o.get('params')), P + 'admit.py': BF.ADMIT_PY.encode()}
        for pool, name in (('HELD', 'HELD'), ('OLD', 'OLD'), ('NEW', 'NEW')):
            f[P + 'manifests/%s.jsonl' % name] = dump_jsonl(BF.manifest_rows(pool))
        # TAUGHT streams are written with the cycle-0 testbed (after the held set is fixed, SPEC 6.2).
        self.taught_f = BF.taught_files(P)
        for path in list(self.taught_f):
            self.taught_f[path] = self.patch('taught_patch', self.taught_f[path], path)
        split, held = BF.split_shas(f, P)
        self.admit_sha = sha256_bytes(f[P + 'admit.py'])
        lock = BF.endpoint_lock(self.constraints_sha, self.admit_sha, split, held)
        lock = self.patch('lock_patch', lock)
        self.lock = lock
        f[P + 'ENDPOINT_LOCK.json'] = dump_json(lock)
        self.lock_sha = sha256_bytes(f[P + 'ENDPOINT_LOCK.json'])
        auth = self.patch('auth_patch', BF.authorization())
        f[P + 'AUTHORIZATION.json'] = dump_json(auth)
        g = BA.gate_report('lock', False, '20261101T085500Z', self.tl.head_sha(), self.lock_sha, None,
                           'gate', FAMILY_ID)
        f[P + 'checks/GATE_lock_20261101T085500Z.json'] = dump_compact(g)
        f.update(self.o.get('extra_lock_files', {}))
        self.commit('lock', f)

    def c_testbed(self):
        aid = BA.att_id('20261101T093000Z', 'testbed')
        arms = ['CEILING_' + c for c in BI.CEILINGS] + ['FLOOR_' + c for c in BI.FLOORS]
        rows = []
        for w in BI.TECH_WORLDS:
            rows += BA.output_rows(w, 'technical', arms, BI.TESTBED_CELLS, aid, None, None, self.bh,
                                   self.o.get('row_patch', {}).get('testbed'))
        self.testbed_rows = rows
        out = self.rows_file(rows)
        gpath = 'checks/GATE_lock_20261101T093000Z.json'
        g = BA.gate_report('lock', False, '20261101T093000Z', self.tl.head_sha(), self.lock_sha, None,
                           'launch', FAMILY_ID)
        a = BA.attempt(aid, 'testbed', ['python3', '-I', 'admit.py', '--testbed'], '20261101T093000Z',
                       '20261101T095000Z', self.tl.head_sha(), self.lock_sha, None,
                       [('results/admission/OUTPUTS.jsonl.gz', sha256_bytes(out))], gpath, cpu=300.0)
        self.attempts.append(a)
        testbed = self.testbed_summary()
        tb = dump_json(testbed)
        dec, why = self.o.get('cycle1', ('continue', 'A ceiling passes and every floor fails.'))
        cyc = BA.cycle(1, 'testbed', dec, why,
                       '20261101T095500Z', None, [aid], None, False, 0)
        self.cycles.append(cyc)
        tf = dict(self.taught_f)
        tf.update({P + 'results/admission/OUTPUTS.jsonl.gz': out, P + gpath: dump_compact(g),
                   P + 'runlog/ATTEMPTS.jsonl': dump_jsonl(self.attempts),
                   P + 'results/admission/TESTBED.json': tb,
                   P + 'runlog/CYCLES.jsonl': dump_jsonl(self.cycles)})
        self.commit('testbed', tf)

    def testbed_summary(self):
        taught = [(BI.prompt(a, b), BI.truth(a, b)) for (a, b) in BI.TAUGHT_PAIRS]
        res = {}
        for arm in sorted(set(r['arm'] for r in self.testbed_rows)):
            d = []
            for w in BI.TECH_WORLDS:
                cell = [r for r in self.testbed_rows if r['arm'] == arm and r['world'] == w
                        and r['pool'] == 'HELD']
                acc = sum(1 for r in cell if RC.is_exact(dict(r, answer_path='native'))) / 16.0
                comp = max(0.05, RC.best(RC.ladder(taught, cell, 'HELD'))[1])
                d.append(acc - comp)
            mean, sd, lcb = RC.paired_lcb(d)
            res[arm] = {'mean_d': round(mean, 6), 'lcb': round(lcb, 6),
                        'verdict': 'PASS' if lcb > 0.05 else 'FAIL'}
        ceil_ok = any(v['verdict'] == 'PASS' for k, v in res.items() if k.startswith('CEILING_'))
        floor_ok = all(v['verdict'] == 'FAIL' for k, v in res.items() if k.startswith('FLOOR_'))
        return {'schema': 'minifly.testbed.v1', 'delivery': D, 'endpoint_lock_sha256': self.lock_sha,
                'worlds': list(BI.TECH_WORLDS), 'configs': res,
                'verdict': 'TESTABLE' if (ceil_ok and floor_ok) else 'CANNOT_TEST_CLAIM_AS_SPECIFIED',
                'note': 'executor copy written from corpus/_build/refcalc.py; MFC score --role '
                        'admission rewrites it'}

    def c_learner(self):
        f = {P + 'learner.py': self.o.get('learner_py', BF.LEARNER_PY.encode()),
             P + 'run.py': self.o.get('run_py', BF.RUN_PY.encode())}
        f.update(self.o.get('extra_native', {}))
        self.files.update(f)
        native = tuple(self.lock['files']['native'])
        sl = BA.source_lock(self.lock_sha, self.files, native=native)
        sl = self.patch('source_lock_patch', sl)
        f[P + 'SOURCE_LOCK.json'] = dump_json(sl)
        self.sl = {'doc': sl, 'sha': sha256_bytes(f[P + 'SOURCE_LOCK.json'])}
        self.commit('learner', f)

    def native_run(self, worlds, role, aid, label_prefix):
        rows, rec = [], {}
        ncs, psha = self.sl['doc']['native_closure_sha'], self.sl['doc']['params_sha256']
        for w in worlds:
            rows += BA.output_rows(w, role, BI.ARMS, BI.NATIVE_CELLS, aid, ncs, psha, self.bh,
                                   self.o.get('row_patch', {}).get(label_prefix))
            for arm in BI.ARMS:
                r = BA.receipt(w, role, arm, aid, ncs, psha, self.lock_sha)
                r = self.patch('receipt_patch', r, label_prefix)
                if r is not False:
                    rec[P + 'results/receipts/%s/%s.json' % (w, arm)] = dump_compact(r)
        return rows, rec

    def c_technical(self):
        aid = BA.att_id('20261101T113000Z', 'technical-run')
        rows, rec = self.native_run([BI.NATIVE_TECH_WORLD], 'technical', aid, 'technical')
        self.technical_rows = rows
        out = self.rows_file(self.testbed_rows + rows)
        g1p = 'checks/GATE_technical_20261101T113000Z.json'
        g1 = BA.gate_report('technical', False, '20261101T113000Z', self.tl.head_sha(),
                            self.lock_sha, self.sl['sha'], 'launch', FAMILY_ID)
        prod = [('results/admission/OUTPUTS.jsonl.gz', sha256_bytes(out))]
        prod += sorted((p[len(P):], sha256_bytes(b)) for p, b in rec.items())
        rcf = {}
        self.add_runtime_closure(aid, 'technical', rcf, prod)
        a1 = BA.attempt(aid, 'technical', ['python3', '-I', 'run.py', '--worlds', BI.NATIVE_TECH_WORLD],
                        '20261101T113000Z', '20261101T114000Z', self.tl.head_sha(), self.lock_sha,
                        self.sl, prod, g1p, cpu=90.0)
        aid2 = BA.att_id('20261101T114500Z', 'technical-admit')
        params = BF.PARAMS if self.o.get('params') is None else self.o['params']
        fx = self.patch('fixtures_patch', BA.fixtures_rows(params, aid2))
        fxb = dump_jsonl(fx)
        g2p = 'checks/GATE_technical_20261101T114500Z.json'
        g2 = BA.gate_report('technical', False, '20261101T114500Z', self.tl.head_sha(),
                            self.lock_sha, self.sl['sha'], 'launch', FAMILY_ID)
        prod2 = [('results/admission/FIXTURES.jsonl', sha256_bytes(fxb))]
        self.add_runtime_closure(aid2, 'technical-admit', rcf, prod2)
        a2 = BA.attempt(aid2, 'technical', ['python3', '-I', 'admit.py', '--technical'],
                        '20261101T114500Z', '20261101T115000Z', self.tl.head_sha(), self.lock_sha,
                        self.sl, prod2, g2p, cpu=30.0)
        self.attempts += [a1, a2]
        adm = self.patch('admission_patch', BA.admission(self.lock_sha, self.sl['sha'], self.admit_sha))
        f = {P + 'results/admission/OUTPUTS.jsonl.gz': out, P + 'results/admission/FIXTURES.jsonl': fxb,
             P + 'ADMISSION.json': dump_json(adm), P + g1p: dump_compact(g1), P + g2p: dump_compact(g2),
             P + 'runlog/ATTEMPTS.jsonl': dump_jsonl(self.attempts)}
        f.update(rec)
        f.update(rcf)
        self.adm_sha = sha256_bytes(f[P + 'ADMISSION.json'])
        self.tech_attempts = [aid, aid2]
        self.commit('technical', f)

    def review_files(self, acks):
        f = {}
        for kind, script in BA.REVIEW_SCRIPTS.items():
            f[P + script] = BA.review_script(kind)
            f[P + 'review/out_%s.json' % kind] = BA.review_output(kind)
        self.files.update(f)
        doc = BA.review_doc(self.lock_sha, self.adm_sha, self.files, P, acks=acks)
        doc = self.patch('review_patch', doc, 'ack' if acks else 'review')
        f[P + 'review/REVIEW.json'] = dump_json(doc)
        return f

    def c_review(self):
        f = self.review_files(None)
        f[P + 'checks/CANARY_20261101T143000Z.json'] = BA.canary_report(lock_sha=self.lock_sha, sl_sha=self.sl['sha'],
                                                                        git_head=self.tl.head_sha())
        self.commit('review', f, who=self.who('review', 'claude'))

    def c_roster(self):
        gp = 'checks/GATE_roster_20261101T155000Z.json'
        g = dump_compact(BA.gate_report('roster', False, '20261101T155000Z', self.tl.head_sha(),
                                     self.lock_sha, self.sl['sha'], 'gate', FAMILY_ID))
        self.cycles.append(BA.cycle(2, 'technical', 'continue', 'Roster gate passes at the current '
                                    'SOURCE_LOCK.', '20261101T155500Z', self.sl, self.tech_attempts,
                                    BA.gate_ref(gp, g, 'roster', False, '20261101T155000Z',
                                                self.sl['sha']), True, 1))
        self.commit('roster', {P + gp: g, P + 'runlog/CYCLES.jsonl': dump_jsonl(self.cycles)})

    def stage_run(self, label, stage, role, worlds, gate_utc, start, end, cpu):
        aid = BA.att_id(start, label)
        rows, rec = self.native_run(worlds, role, aid, label)
        out = self.rows_file(rows)
        bundle = 'official' if stage == 'official' else 'roster'
        gp = 'checks/GATE_%s_%s.json' % (bundle, gate_utc)
        g = dump_compact(BA.gate_report(bundle, False, gate_utc, self.tl.head_sha(), self.lock_sha,
                                     self.sl['sha'], 'launch', FAMILY_ID))
        prod = [('results/%s/OUTPUTS.jsonl.gz' % role, sha256_bytes(out))]
        prod += sorted((p[len(P):], sha256_bytes(b)) for p, b in rec.items())
        # TAUGHT manifests the runner writes during this run (e.g. seal-derived worlds, C00d) are recorded by
        # launch like its results/ outputs (SPEC 6.2 item 7 runner contract)
        extra = self.o.get('extra_files', {}).get(label, {})
        prod += sorted((p[len(P):], sha256_bytes(b)) for p, b in extra.items() if p.startswith(P + 'manifests/TAUGHT/'))
        rcf = {}
        self.add_runtime_closure(aid, label, rcf, prod)
        drop = set(self.o.get('drop_produced', {}).get(label, []))
        prod = [x for x in prod if x[0] not in drop]
        argv = self.o.get('argv', {}).get(label, ['python3', '-I', 'run.py', '--worlds', ','.join(worlds)])
        a = BA.attempt(aid, stage, argv, start, end, self.tl.head_sha(), self.lock_sha, self.sl, prod,
                       gp, cpu=cpu)
        self.attempts.append(a)
        f = {P + 'results/%s/OUTPUTS.jsonl.gz' % role: out, P + gp: g,
             P + 'runlog/ATTEMPTS.jsonl': dump_jsonl(self.attempts)}
        f.update(rec)
        f.update(rcf)
        return f, rows, aid

    def c_dev(self):
        f, rows, aid = self.stage_run('dev', 'dev', 'dev', self.o.get('dev_worlds', BI.DEV_WORLDS),
                                      '20261102T083000Z', '20261102T083000Z', '20261102T085000Z', 120.0)
        self.dev_rows, self.dev_attempts = rows, [aid]
        self.commit('dev', f)
        # C10: further outcome-informed learner revisions inside the dev cycle, each with a new
        # SOURCE_LOCK and a dev attempt (designs_tried_k counts distinct (closure, params) pairs).
        for n, learner in enumerate(self.o.get('dev_redesigns', [])):
            k = n + 2
            lab = 'dev_r%d' % k
            UTC['%s_lock' % lab] = '20261102T09%d000Z' % (2 * k - 3)
            UTC[lab] = '20261102T09%d500Z' % (2 * k - 3)
            self.files[P + 'learner.py'] = learner
            sl = BA.source_lock(self.lock_sha, self.files, built_utc=UTC['%s_lock' % lab],
                                native=tuple(self.lock['files']['native']))
            slb = dump_json(sl)
            self.sl = {'doc': sl, 'sha': sha256_bytes(slb)}
            self.commit('%s_lock' % lab, {P + 'learner.py': learner, P + 'SOURCE_LOCK.json': slb})
            worlds = self.o.get('dev_redesign_worlds', {}).get(k, BI.DEV_WORLDS)
            f, rows, aid = self.stage_run(lab, 'dev', 'dev', worlds, UTC[lab], UTC[lab],
                                          UTC[lab][:-3] + '800Z', 60.0)
            self.dev_rows = rows
            self.dev_attempts.append(aid)
            self.commit(lab, f)
        self.designs_k = 1 + len(self.o.get('dev_redesigns', []))

    def c_dev_close(self):
        import scores_doc
        sc = scores_doc.scores(self.dev_rows, 'dev', self.bh)
        self.dev_scores = sc
        self.cycles.append(BA.cycle(3, 'dev', 'continue', 'Dev SCORES written; W above the comparator '
                                    'on both dev worlds.', '20261102T095500Z', self.sl,
                                    list(self.dev_attempts), None, True, self.designs_k))
        self.commit('dev_close', {P + 'results/dev/SCORES.json': dump_compact(sc),
                                  P + 'runlog/CYCLES.jsonl': dump_jsonl(self.cycles)})

    def c_ack(self):
        acks = [{'sha': self.sl['sha'], 'utc': '20261102T105000Z',
                 'note': 'Closure diff since the review: none; final SOURCE_LOCK acknowledged.'}]
        self.commit('ack', {P + 'review/REVIEW.json': self.review_files(acks)[P + 'review/REVIEW.json']},
                    who=self.who('ack', 'claude'))

    def c_rehearsal(self):
        f, rows, aid = self.stage_run('rehearsal', 'rehearsal', 'rehearsal', BI.REH_WORLDS,
                                      '20261102T113000Z', '20261102T113000Z', '20261102T114000Z', 60.0)
        dp = 'checks/DRY_official_20261102T115000Z.json'
        d = dump_compact(BA.gate_report('official', True, '20261102T115000Z', self.tl.head_sha(),
                                     self.lock_sha, self.sl['sha'], 'gate', FAMILY_ID))
        f[P + dp] = d
        self.cycles.append(BA.cycle(4, 'rehearsal', 'continue', 'Dry official gate passes on the fresh '
                                    'rehearsal world.', '20261102T115500Z', self.sl, [aid],
                                    BA.gate_ref(dp, d, 'official', True, '20261102T115000Z',
                                                self.sl['sha']), True, getattr(self, 'designs_k', 1)))
        f[P + 'runlog/CYCLES.jsonl'] = dump_jsonl(self.cycles)
        self.commit('rehearsal', f)

    def c_official(self):
        f, rows, aid = self.stage_run('official', 'official', 'official', self.off_worlds(),
                                      '20261103T083000Z', '20261103T083000Z', '20261103T085000Z', 480.0)
        self.official_rows = rows
        self.commit('official', f)

    def off_worlds(self):
        if self.o.get('official_worlds'):
            return list(self.o['official_worlds'])
        return list(self.lock['worlds']['official']['ids']) or list(BI.OFF_WORLDS)

    def c_report(self):
        import scores_doc
        if self.o.get('headline'):
            self.headline = self.o['headline']
        else:
            rows = getattr(self, 'official_rows', None)
            self.headline = scores_doc.headline(rows, self.bh, k=getattr(self, 'designs_k', 1),
                                                role=self.o.get('headline_role', 'MAIN'))
        body = self.o.get('report_body', 'Corpus base delivery for the MRK-1 regression corpus. The '
                          'first line is the generated headline; numbers are hand-computed in '
                          'tools/minifly_check/corpus/_build/base/EXPECTED_VALUES.md.\n')
        first = self.o.get('report_first_line', self.headline)
        f = {P + 'REPORT.md': ('%s\n\n%s' % (first, body)).encode('utf-8')}
        fn = self.o.get('claim_doc', 'default')
        if fn == 'default':
            import mutate
            fn = mutate.claim_doc
        if fn:
            # The executor ran `MFC claim`, filled findings_none_reason and committed CLAIM.json with
            # REPORT.md; claim appended the DELIVERY_LEDGER row at the same time.
            f[P + 'CLAIM.json'] = dump_json(fn(self))
            if self.o.get('ledger_row', True):
                f.update(ledger_append(self.files, f[P + 'CLAIM.json'], '20261103T115500Z'))
        self.commit('report', f)


def ledger_append(files, claim_bytes, utc):
    """The DELIVERY_LEDGER row `MFC claim` appends (SCHEMAS 18), committed with CLAIM.json."""
    import json
    claim = json.loads(claim_bytes.decode('utf-8'))
    om = claim['objective_metric']
    opened = (claim['level_attempted'] == 'E3' and claim['primary_gate'] in ('PASS', 'FAIL')
              and claim['worlds_role'] in ('MAIN', 'SEALED_CONFIRM'))
    row = {'delivery': claim['delivery'], 'utc': utc, 'legacy': False,
           'task_family': 'addition_single_digit', 'level_earned': claim['level_earned'],
           'primary_gate': claim['primary_gate'], 'gate_type': claim['gate_type'],
           'worlds_role': claim['worlds_role'], 'opened_held_E3': opened,
           'e3_basis': ['pool HELD', 'strata S2-S3 determined', 'ADM-EXPO PASS'] if opened else [],
           'objective_metric': {'held': om['held_exact_own_eos'], 'old_final': om['old_final'],
                                'G': om['G']},
           'family_id': claim['family_id'], 'claim_sha256': sha256_bytes(claim_bytes)}
    path = BF.KB + 'DELIVERY_LEDGER.jsonl'
    return {path: files.get(path, b'') + dump_jsonl([row])}


OVERRIDABLE = ['OFF_WORLDS', 'DEV_WORLDS', 'REH_WORLDS', 'TECH_WORLDS', 'HELD_S2', 'HELD_S3',
               'HELD_PAIRS']


def build_base(opts=None):
    """Build the base timeline. opts['bi'] temporarily overrides base_items module lists (for
    example HELD_S3 or OFF_WORLDS) for the duration of the build."""
    opts = dict(opts or {})
    saved = {k: getattr(BI, k) for k in OVERRIDABLE}
    saved_utc = dict(UTC)
    try:
        for k, v in opts.get('bi', {}).items():
            if k != 'HELD_EXTRA':
                setattr(BI, k, v)
        if 'bi' in opts:
            BI.HELD_PAIRS = list(BI.HELD_S2) + list(BI.HELD_S3) + list(opts['bi'].get('HELD_EXTRA', []))
        BA.CTX['tier'] = opts.get('tier', 2)
        BA.CTX['sealed'] = opts.get('sealed', False)
        b = BaseBuild(opts)
        tl = b.build()
        b.items = BI.pool_items()
        return tl, b
    finally:
        for k, v in saved.items():
            setattr(BI, k, v)
        UTC.clear()
        UTC.update(saved_utc)
        BA.CTX['tier'] = 2
        BA.CTX['sealed'] = False
