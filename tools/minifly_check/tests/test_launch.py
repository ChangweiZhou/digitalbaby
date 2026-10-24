"""Unit tests for launch.py (WP1): launch refusal rules, ATTEMPTS rows, cycle close, RUN-01/02, CYC-01..03.

Launch tests use a temporary `git init` repository; checks use synthetic deliveries. The bundles are
replaced by small test bundles so these tests do not depend on other work packages."""
import argparse
import ast
import contextlib
import gzip
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]
if str(KIT) not in sys.path:
    sys.path.insert(0, str(KIT))

import common  # noqa: E402
import launch  # noqa: E402
import minifly_check  # noqa: E402
from common import FAIL, PASS, MissingInput, Result  # noqa: E402

HAVE_GIT = shutil.which('git') is not None
NAME = 'lab_dev_v1_20261101'
CONSTRAINTS = {
    'schema': 'minifly.constraints.v1',
    'tier_thresholds': {'t0_max_cpu_h': 1, 't1_max_cpu_h': 4, 't1_max_records': 20000, 'family_t1_max_cpu_h': 12},
    'default_grant_caps': {'max_cpu_h': 2, 'max_records': 5000, 'max_cycles': 3},
}
STATE = {'gate_fail': False}


@common.check('TL-GATE')
def _tl_gate(ctx):
    return Result('TL-GATE', FAIL if STATE['gate_fail'] else PASS, 'test gate')


MOD = _tl_gate.__module__.split('.')[-1]


def _rule(rid, module, aw='always'):
    return {'id': rid, 'module': module, 'stages': ['lock', 'technical', 'roster', 'official'], 'tier_min': 0,
            'introduced_in': '1.0.0', 'status': 'enforced', 'fm_ids': [], 'applies_when': aw}


TEST_RULES = {
    'rules': [_rule('TL-GATE', MOD), _rule('RUN-02', 'launch')],
    'bundles': {'lock': {'checks': ['TL-GATE']}, 'technical': {'extends': 'lock', 'checks': []},
                'roster': {'extends': 'technical', 'checks': []}, 'official': {'extends': 'roster', 'checks': ['RUN-02']}},
}


def write(path, data):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, (dict, list)):
        data = json.dumps(data, indent=1)
    if isinstance(data, str):
        data = data.encode('utf-8')
    p.write_bytes(data)
    return p


def jsonl(rows):
    return ''.join(json.dumps(r) + '\n' for r in rows)


def git(repo, *args):
    e = dict(os.environ)
    e.update({'GIT_AUTHOR_NAME': 'T', 'GIT_AUTHOR_EMAIL': 't@example.invalid', 'GIT_COMMITTER_NAME': 'T',
              'GIT_COMMITTER_EMAIL': 't@example.invalid', 'GIT_CONFIG_NOSYSTEM': '1'})
    return subprocess.run(['git', '-c', 'commit.gpgsign=false', '-c', 'init.defaultBranch=main'] + list(args),
                          cwd=str(repo), check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=e)


def lock_obj(**over):
    lock = {'schema': common.SCHEMA_ENDPOINT_LOCK, 'delivery': NAME, 'tier_declared': 1, 'intent': 'diagnostic',
            'budget': {'max_records': 1000, 'max_cpu_h': 1},
            'worlds': {'technical': ['wt1'], 'dev': ['wd1', 'wd2'], 'rehearsal': ['wr1'],
                       'official': {'mode': 'listed', 'ids': ['wo1', 'wo2'], 'n': 2}, 'official_role_intended': 'PILOT'},
            'files': {'admit': 'admit.py', 'evaluator': [], 'generators': [], 'runner': ['run.py'],
                      'native': ['native.py'], 'shared': [], 'assets': [], 'params': 'params.json'},
            'endpoint_files_sha256': {}}
    lock.update(over)
    return lock


class LaunchRepoCase(unittest.TestCase):
    """A temporary git repo with KB CONSTRAINTS and a committed, locked v1 delivery."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix='mfc_launch_'))
        self.repo = self.tmp / 'repo'
        self.d = self.repo / NAME
        write(self.repo / 'PROJECT_DOCUMENTS' / 'KB' / 'CONSTRAINTS.json', CONSTRAINTS)
        write(self.repo / '.gitignore', '__pycache__/\n')
        write(self.d / 'native.py', 'def step(x):\n    return x\n')
        write(self.d / 'run.py', 'import native\n')
        write(self.d / 'admit.py', 'print("admit")\n')
        write(self.d / 'params.json', {'lr': 0.1})
        admit_sha = common.sha256_file(self.d / 'admit.py')
        write(self.d / 'ENDPOINT_LOCK.json', lock_obj(endpoint_files_sha256={'admit.py': admit_sha}))
        self.write_source_lock()
        git(self.repo, 'init', '-q')
        self.commit('lock')
        STATE['gate_fail'] = False
        self._saved_load_rules = common.load_rules
        common.load_rules = lambda path=None: TEST_RULES

    def tearDown(self):
        common.load_rules = self._saved_load_rules
        shutil.rmtree(self.tmp, ignore_errors=True)

    def write_source_lock(self):
        clo = {}
        for f in ('native.py', 'run.py', 'admit.py'):
            clo[NAME + '/' + f] = common.sha256_file(self.d / f)
        write(self.d / 'SOURCE_LOCK.json', {'schema': common.SCHEMA_SOURCE_LOCK, 'delivery': NAME,
                                            'checker_version': '1.0.0', 'closure_sha256': clo,
                                            'native_closure_sha': 'n' * 64, 'params_sha256': common.sha256_file(self.d / 'params.json')})

    def commit(self, msg, *paths):
        git(self.repo, 'add', '-A', *paths) if not paths else git(self.repo, 'add', *paths)
        git(self.repo, 'commit', '-q', '-m', msg)
        return git(self.repo, 'rev-parse', 'HEAD').stdout.decode().strip()

    def launch(self, stage, code, *extra_cmd):
        argv = ['--repo-root', str(self.repo), 'launch', '--stage', stage, NAME, '--', sys.executable, '-I', '-c', code]
        argv += list(extra_cmd)
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            rc = minifly_check.main(argv)
        self.last_err = err.getvalue()
        return rc

    def attempts(self):
        p = self.d / 'runlog' / 'ATTEMPTS.jsonl'
        return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []

    def ctx(self, **opts):
        return common.Ctx(repo_root=self.repo, delivery=self.d, git=common.RealGit(self.repo), kb=common.KB(self.repo),
                          rules=TEST_RULES, options=dict(opts))


WRITE_DEV = ("import os, json\n"
             "os.makedirs('results/dev', exist_ok=True)\n"
             "open('results/dev/OUTPUTS.jsonl', 'a').write(json.dumps({'arm': 'W', 'world': 'wd1', 'world_role': 'dev',"
             " 'attempt_id': os.environ['MINIFLY_ATTEMPT_ID'], 'native_closure_sha': os.environ['MINIFLY_NATIVE_CLOSURE_SHA'],"
             " 'params_sha256': os.environ['MINIFLY_PARAMS_SHA256']}) + '\\n')\n"
             "s = 0\n"
             "for i in range(400000): s += i * i\n")
WRITE_OFFICIAL = ("import os, json\n"
                  "os.makedirs('results/official', exist_ok=True)\n"
                  "open('results/official/OUTPUTS.jsonl', 'a').write(json.dumps({'arm': 'W', 'world': 'wo1',"
                  " 'world_role': 'official', 'attempt_id': os.environ['MINIFLY_ATTEMPT_ID']}) + '\\n')\n")
MARKER = "open('ran.marker', 'w').write('x')\n"


# an official runner that honours the resume environment (SPEC 20.8): it skips completed units, writes the missing
# ones and re-runs the units launch chose into MINIFLY_REPLAY_DIR
OFFICIAL_UNITS = ("import json, os, sys\n"
                  "aid = os.environ['MINIFLY_ATTEMPT_ID']\n"
                  "done = [tuple(u) for u in json.loads(os.environ.get('MINIFLY_COMPLETED_UNITS', '[]'))]\n"
                  "replay = [tuple(u) for u in json.loads(os.environ.get('MINIFLY_REPLAY_UNITS', '[]'))]\n"
                  "def rows(w, a):\n"
                  "    return [{'world': w, 'world_role': 'official', 'arm': a, 'pool': 'HELD', 'checkpoint': 'final',\n"
                  "             'item_id': 'h%d' % i, 'stream_hex': '%02x' % (i + len(w)), 'attempt_id': aid}\n"
                  "            for i in range(3)]\n"
                  "def put(rel, rs):\n"
                  "    os.makedirs(os.path.dirname(rel), exist_ok=True)\n"
                  "    with open(rel, 'a') as fh:\n"
                  "        fh.write(''.join(json.dumps(r) + '\\n' for r in rs))\n"
                  "UNITS = [(w, a) for w in ('wo1', 'wo2') for a in ('W', 'NPT')]\n"
                  "STOP = os.environ.get('STOP_AFTER')\n"
                  "if STOP == 'crash':\n"
                  "    sys.exit(5)\n"
                  "for n, u in enumerate(UNITS):\n"
                  "    if STOP is not None and n == int(STOP):\n"
                  "        os.kill(os.getpid(), 9)\n"
                  "    if u in done and os.environ.get('REWRITE') != '%s|%s' % u:\n"
                  "        continue\n"
                  "    put('results/official/OUTPUTS.jsonl', rows(*u))\n"
                  "for u in replay:\n"
                  "    rs = rows(*u)\n"
                  "    if os.environ.get('BREAK_REPLAY'):\n"
                  "        rs[0]['stream_hex'] = 'ff'\n"
                  "    put(os.environ['MINIFLY_REPLAY_DIR'] + '/OUTPUTS.jsonl', rs)\n")


class TestSnapshot(unittest.TestCase):
    def test_snapshot_covers_results_and_runner_written_taught(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            for rel in ('results/dev/OUTPUTS.jsonl', 'manifests/TAUGHT/w1/W.jsonl', 'manifests/HELD.jsonl'):
                (d / rel).parent.mkdir(parents=True, exist_ok=True)
                (d / rel).write_text('{}\n')
            snap = launch.snapshot_results(d)
            self.assertEqual(sorted(snap), ['manifests/TAUGHT/w1/W.jsonl', 'results/dev/OUTPUTS.jsonl'])


@unittest.skipUnless(HAVE_GIT, 'git not installed')
class TestLaunch(LaunchRepoCase):
    def test_clean_launch_records_attempt_row(self):
        head = git(self.repo, 'rev-parse', 'HEAD').stdout.decode().strip()
        rc = self.launch('dev', WRITE_DEV)
        self.assertEqual(rc, 0, self.last_err)
        rows = self.attempts()
        self.assertEqual(len(rows), 1)
        a = rows[0]
        self.assertEqual(a['schema'], 'minifly.attempt.v1')
        self.assertEqual(a['stage'], 'dev')
        self.assertEqual(a['argv'], [sys.executable, '-I', '-c', WRITE_DEV])
        self.assertEqual(a['exit_code'], 0)
        self.assertEqual(a['git_head'], head)
        self.assertIs(a['closure_clean'], True)
        self.assertIs(a['closure_clean_after'], True)
        self.assertEqual(a['endpoint_lock_sha256'], common.sha256_file(self.d / 'ENDPOINT_LOCK.json'))
        self.assertEqual(a['source_lock_sha256'], common.sha256_file(self.d / 'SOURCE_LOCK.json'))
        self.assertEqual(a['native_closure_sha'], 'n' * 64)
        self.assertEqual(a['produced_paths'], [{'path': 'results/dev/OUTPUTS.jsonl',
                                                'sha256': common.sha256_file(self.d / 'results/dev/OUTPUTS.jsonl')}])
        for k in ('utc_start', 'utc_end'):
            self.assertRegex(a[k], r'^\d{8}T\d{6}Z$')
        self.assertGreaterEqual(a['wall_s'], 0)
        self.assertGreater(a['cpu_children_user_s'] + a['cpu_children_sys_s'], 0.0)
        self.assertIsInstance(a['max_rss_kb'], int)
        self.assertGreater(a['max_rss_kb'], 0)
        out = json.loads((self.d / 'results/dev/OUTPUTS.jsonl').read_text().splitlines()[0])
        self.assertEqual(out['attempt_id'], a['attempt_id'])
        rep = json.loads((self.d / a['gate_report']).read_text())
        self.assertEqual((rep['stage'], rep['invoked_by'], rep['verdict']), ('roster', 'launch', 'PASS'))
        # the produced file is covered by RUN-01
        self.assertEqual(launch.check_run_01(self.ctx()).status, PASS)

    def test_refuses_modified_closure_file(self):
        write(self.d / 'native.py', 'def step(x):\n    return x + 1\n')
        rc = self.launch('dev', MARKER)
        self.assertEqual(rc, 1)
        self.assertIn('not committed and clean', self.last_err)
        self.assertIn(NAME + '/native.py', self.last_err)
        self.assertEqual(self.attempts(), [])
        self.assertFalse((self.d / 'ran.marker').exists())

    def test_refuses_modified_source_lock(self):
        sl = json.loads((self.d / 'SOURCE_LOCK.json').read_text())
        sl['params_sha256'] = 'x' * 64
        write(self.d / 'SOURCE_LOCK.json', sl)
        self.assertEqual(self.launch('technical', MARKER), 1)
        self.assertIn('SOURCE_LOCK.json', self.last_err)
        self.assertFalse((self.d / 'ran.marker').exists())

    def test_refuses_untracked_closure_file(self):
        write(self.d / 'helper.py', 'X = 1\n')
        write(self.d / 'ENDPOINT_LOCK.json', lock_obj(files=dict(lock_obj()['files'], native=['native.py', 'helper.py']),
                                                      endpoint_files_sha256={'admit.py': common.sha256_file(self.d / 'admit.py')}))
        self.commit('add helper to the lock only', NAME + '/ENDPOINT_LOCK.json')
        self.assertEqual(self.launch('dev', MARKER), 1)
        self.assertIn('helper.py', self.last_err)
        self.assertFalse((self.d / 'ran.marker').exists())

    def test_refuses_committed_closure_that_differs_from_source_lock(self):
        write(self.d / 'native.py', 'def step(x):\n    return 2 * x\n')
        self.commit('edit native after lock build')
        self.assertEqual(self.launch('dev', MARKER), 1)
        self.assertIn('differs from locked', self.last_err)

    def test_refuses_when_gate_fails(self):
        STATE['gate_fail'] = True
        self.assertEqual(self.launch('dev', MARKER), 1)
        self.assertIn('gate exit 1', self.last_err)
        self.assertEqual(self.attempts(), [])
        reports = list((self.d / 'checks').glob('GATE_roster_*.json'))
        self.assertEqual(len(reports), 1)
        self.assertEqual(json.loads(reports[0].read_text())['verdict'], 'FAIL')

    def close(self, kind, decision='revise'):
        argv = ['--repo-root', str(self.repo), 'cycle', 'close', NAME, '--kind', kind, '--decision', decision,
                '--reason', 'r']
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            rc = minifly_check.main(argv)
        self.last_err = err.getvalue()
        return rc

    def test_cycle_cap(self):
        rows = [{'k': i + 1, 'kind': k, 'decision': 'revise', 'reason': 'r', 'utc': '20261101T00000%dZ' % i,
                 'computed': {'attempts': []}} for i, k in enumerate(['testbed', 'technical', 'dev', 'dev'])]
        write(self.d / 'runlog' / 'CYCLES.jsonl', jsonl(rows))
        self.assertEqual(self.launch('dev', MARKER), 1)
        self.assertIn('cycle cap', self.last_err)
        self.assertEqual(self.launch('rehearsal', MARKER), 1)
        # CYC-01 counts technical cycles against the same cap, so launch refuses a technical attempt too
        self.assertEqual(self.launch('technical', MARKER), 1)
        self.assertIn('cycle cap', self.last_err)
        self.assertFalse((self.d / 'ran.marker').exists())
        write(self.d / 'AUTHORIZATION.json', {'verbatim': 'You may run 5 test cycles.', 'caps': {'max_cycles': 5}})
        self.commit('authorization')
        self.assertEqual(self.launch('technical', 'pass'), 0, self.last_err)   # 3 closed + this open one <= 5
        # the technical attempt is an open cycle: a dev attempt needs its close first (SPEC 3.2)
        self.assertEqual(self.launch('dev', 'pass'), 1)
        self.assertIn('not closed', self.last_err)
        self.assertEqual(self.close('dev'), 1)          # a dev close cannot cover the technical attempt
        self.assertIn('--kind technical', self.last_err)
        self.assertEqual(self.close('technical'), 0, self.last_err)
        self.assertEqual(self.launch('dev', 'pass'), 0, self.last_err)   # 4 closed + this open dev cycle = 5
        self.assertEqual(self.launch('dev', 'pass'), 0, self.last_err)   # joins the open dev cycle
        self.assertEqual(self.launch('rehearsal', 'pass'), 1)            # the dev cycle is still open
        self.assertEqual(self.close('dev'), 0, self.last_err)
        self.assertEqual(self.launch('rehearsal', 'pass'), 1)            # 5 closed + a new rehearsal cycle > 5
        self.assertIn('cycle cap', self.last_err)
        rows = [json.loads(l) for l in (self.d / 'runlog' / 'CYCLES.jsonl').read_text().splitlines()]
        att = {a['attempt_id']: a['stage'] for a in self.attempts()}
        self.assertEqual([[att[x] for x in r['computed']['attempts']] for r in rows[4:]],
                         [['technical'], ['dev', 'dev']])

    def test_never_closing_cycles_cannot_bypass_the_cap(self):
        write(self.d / 'AUTHORIZATION.json', {'verbatim': 'You may run 1 test cycle.', 'caps': {'max_cycles': 1}})
        self.commit('authorization')
        self.assertEqual(self.launch('dev', 'pass'), 0, self.last_err)   # opens the one allowed cycle
        self.assertEqual(self.launch('dev', 'pass'), 0, self.last_err)   # same open cycle
        self.assertEqual(self.launch('rehearsal', 'pass'), 1)            # never closed: refused
        self.assertEqual(self.close('dev'), 0, self.last_err)
        self.assertEqual(self.launch('rehearsal', 'pass'), 1)            # 1 closed + 1 new > 1
        self.assertIn('cycle cap', self.last_err)
        self.assertEqual(self.launch('dev', 'pass'), 1)

    def test_testbed_cannot_carry_dev_outcomes(self):
        # verifier round-3 probe: with the one allowed cycle used, dev work relabelled as testbed (not counted)
        write(self.d / 'AUTHORIZATION.json', {'verbatim': 'You may run 1 test cycle.', 'caps': {'max_cycles': 1}})
        self.commit('authorization')
        self.assertEqual(self.launch('dev', WRITE_DEV), 0, self.last_err)
        self.assertEqual(self.close('dev'), 0, self.last_err)
        self.assertEqual(self.launch('dev', WRITE_DEV), 1)
        self.assertIn('cycle cap', self.last_err)
        # naming a dev world on a testbed command line is refused before anything runs
        self.assertEqual(self.launch('testbed', MARKER, '--worlds', 'wd1'), 1)
        self.assertIn('wd1', self.last_err)
        self.assertFalse((self.d / 'ran.marker').exists())
        n = len(self.attempts())
        # the same dev runner (its world id built at run time) under --stage testbed: recorded, but launch fails
        # and CYC-01 FAILs for good
        self.assertEqual(self.launch('testbed', WRITE_DEV.replace("'wd1'", "'wd' + '1'")), 1)
        self.assertIn('wrote outside the testbed stage', self.last_err)
        self.assertEqual(len(self.attempts()), n + 1)
        tb = self.attempts()[-1]
        self.assertEqual(tb['stage'], 'testbed')
        self.assertEqual([e['path'] for e in tb['produced_paths']], ['results/dev/OUTPUTS.jsonl'])
        self.assertEqual(self.close('testbed'), 0, self.last_err)
        for stage, dry in (('roster', False), ('official', True), ('official', False)):
            ctx = common.Ctx(repo_root=self.repo, delivery=self.d, git=common.RealGit(self.repo),
                             kb=common.KB(self.repo), rules=TEST_RULES, stage=stage, dry=dry, options={})
            r = launch.check_cyc_01(ctx)
            self.assertEqual(r.status, FAIL, (stage, dry, r.detail))
            self.assertIn('a testbed attempt writes results/admission/ only', r.detail)
            self.assertIn('rows of testbed attempt %s under results/dev/' % tb['attempt_id'], r.detail)
        # every dev row is still attributed to an attempt, so RUN-01 alone would not see it
        self.assertEqual(launch.check_run_01(self.ctx()).status, PASS)

    def runtime_helper(self):
        """write_runtime_closure() exactly as templates/admit_skeleton.py ships it (the runner contract helper)."""
        src = (KIT / 'templates' / 'admit_skeleton.py').read_text(encoding='utf-8')
        fn = [n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == 'write_runtime_closure'][0]
        return ast.get_source_segment(src, fn) + '\n'

    def test_runtime_closure_from_the_template_helper(self):
        """RUN-03 end to end (SPEC 20.7): the template helper, run under launch, records the modules imported at
        runtime; launch lists the file in produced_paths; an import outside SOURCE_LOCK is a closure gap."""
        pre = 'import hashlib, json, os, sys\nsys.path.insert(0, os.getcwd())\nimport native\n'
        code = pre + self.runtime_helper() + 'write_runtime_closure()\n' + WRITE_DEV
        self.assertEqual(self.launch('dev', code), 0, self.last_err)
        a = self.attempts()[-1]
        rel = launch.runtime_closure_rel(a['attempt_id'])
        self.assertIn(rel, [e['path'] for e in a['produced_paths']])
        doc = json.loads((self.d / rel).read_text())
        self.assertEqual((doc['schema'], doc['attempt_id']), ('minifly.runtime_closure.v1', a['attempt_id']))
        self.assertIn(NAME + '/native.py', [f['path'] for f in doc['files']])
        r = launch.check_run_03(self.ctx())
        self.assertEqual(r.status, PASS, r.detail)
        # a module that only a runtime import reaches (no static import in the closure): the closure gap
        write(self.d / 'bootstrap.py', 'z = 3\n')
        self.commit('bootstrap')
        code = pre + 'import importlib\nimportlib.import_module("bootstrap")\n' + self.runtime_helper() + \
            'write_runtime_closure()\n' + WRITE_DEV
        self.assertEqual(self.launch('dev', code), 0, self.last_err)
        r = launch.check_run_03(self.ctx())
        self.assertEqual(r.status, FAIL)
        self.assertIn('closure gap', r.detail)
        self.assertIn(NAME + '/bootstrap.py', r.detail)

    def test_runtime_closure_contract_snippet(self):
        """The 12-line helper printed in the runner contract (minifly-e3-design ADMISSION_AND_RUNNER_CONTRACT.md 6.1)
        works as documented under launch."""
        import re
        doc = KIT.parents[1] / '.claude' / 'skills' / 'minifly-e3-design' / 'references' / 'ADMISSION_AND_RUNNER_CONTRACT.md'
        if not doc.is_file():
            self.skipTest('skill reference not present in this checkout')
        m = re.search(r'```python\n(def write_runtime_closure.*?)```', doc.read_text(encoding='utf-8'), re.S)
        self.assertIsNotNone(m)
        self.assertEqual(len(m.group(1).strip().splitlines()), 12)
        code = ('import hashlib, json, os, sys\nsys.path.insert(0, os.getcwd())\nimport native\n' + m.group(1) +
                'write_runtime_closure()\n' + WRITE_DEV)
        self.assertEqual(self.launch('dev', code), 0, self.last_err)
        r = launch.check_run_03(self.ctx())
        self.assertEqual(r.status, PASS, r.detail)

    def test_every_phase_relaunch_rechecks_the_lock(self):
        """SPEC 20.9: a multi-phase run calls launch once per phase, and every launch re-verifies ENDPOINT_LOCK,
        SOURCE_LOCK and every closure sha, so a mid-run fix between phases fails loudly."""
        self.assertEqual(self.launch('dev', WRITE_DEV), 0, self.last_err)          # phase 1
        write(self.d / 'native.py', 'def step(x):\n    return x * 2\n')           # a fix between phases
        self.commit('mid-run fix')
        self.assertEqual(self.launch('dev', WRITE_DEV), 1)                        # phase 2 refused
        self.assertIn('differs from locked', self.last_err)
        self.assertEqual(len(self.attempts()), 1)
        self.write_source_lock()                                                   # a new lock build is a new design
        self.commit('lock build')
        self.assertEqual(self.launch('dev', WRITE_DEV), 0, self.last_err)
        self.assertNotEqual(self.attempts()[0]['source_lock_sha256'], self.attempts()[1]['source_lock_sha256'])

    def test_failed_command_is_still_recorded(self):
        self.assertEqual(self.launch('dev', 'import sys; sys.exit(3)'), 1)
        rows = self.attempts()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['exit_code'], 3)
        argv = ['--repo-root', str(self.repo), 'launch', '--stage', 'dev', NAME, '--', str(self.tmp / 'no_such_binary')]
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(minifly_check.main(argv), 1)
        rows = self.attempts()
        self.assertEqual(rows[-1]['exit_code'], 127)
        self.assertIn('launch_error', rows[-1])

    def test_closure_changed_during_attempt(self):
        rc = self.launch('dev', "open('native.py', 'a').write('# tampered\\n')\n")
        self.assertEqual(rc, 1)
        a = self.attempts()[-1]
        self.assertIs(a['closure_clean_after'], False)

    def test_official_never_rerun(self):
        self.assertEqual(self.launch('official', WRITE_OFFICIAL), 0, self.last_err)
        self.assertEqual(self.launch('official', WRITE_OFFICIAL), 1)
        self.assertIn('RUN-02', self.last_err)
        self.assertEqual([a['stage'] for a in self.attempts()], ['official'])

    def launch_env(self, stage, code, env, resume=False):
        saved = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        try:
            argv = ['--repo-root', str(self.repo), 'launch', '--stage', stage] + (['--resume'] if resume else []) + \
                [NAME, '--', sys.executable, '-I', '-c', code]
            err = io.StringIO()
            with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
                rc = minifly_check.main(argv)
            self.last_err = err.getvalue()
            return rc
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_official_resume_with_replay(self):
        """SPEC 20.8: a killed official attempt is resumed under the identical lock; the resume writes only the
        missing units and re-runs the unit launch chose into results/official_replay/ (RUN-02, RUN-04 PASS)."""
        self.assertEqual(self.launch_env('official', OFFICIAL_UNITS, {'STOP_AFTER': '2'}), 1)
        a = self.attempts()[-1]
        self.assertEqual(a['exit_code'], -9)
        self.assertEqual([(u['world'], u['arm']) for u in a['units']], [('wo1', 'NPT'), ('wo1', 'W')])
        self.assertFalse(list((self.d / 'runlog').glob('INFLIGHT_*.json')))   # marker removed with the row
        # without --resume an official attempt after an assessment-bearing one is refused (RUN-02 as before)
        self.assertEqual(self.launch_env('official', OFFICIAL_UNITS, {}), 1)
        self.assertIn('--resume', self.last_err)
        self.assertEqual(len(self.attempts()), 1)
        self.assertEqual(launch.check_run_04(self.ctx()).status, common.NOT_APPLICABLE)
        rc = self.launch_env('official', OFFICIAL_UNITS, {'STOP_AFTER': 'crash'}, resume=True)
        self.assertEqual(rc, 1, self.last_err)   # the resume itself crashes before writing any row
        self.assertEqual(len(self.attempts()), 2)
        b = self.attempts()[-1]
        self.assertEqual((b['resume_of'], b['exit_code'] != 0, b['units']), (a['attempt_id'], True, []))
        # resume the resume: the latest interrupted attempt is the one to name
        rc = self.launch_env('official', OFFICIAL_UNITS, {}, resume=True)
        self.assertEqual(rc, 0, self.last_err)
        c = self.attempts()[-1]
        self.assertEqual(c['resume_of'], b['attempt_id'])
        self.assertEqual([(u['world'], u['arm']) for u in c['units']], [('wo2', 'NPT'), ('wo2', 'W')])
        self.assertEqual(len(c['replay_requested']), 1)
        self.assertIn(tuple(c['replay_requested'][0]), (('wo1', 'NPT'), ('wo1', 'W')))
        self.assertEqual(c['replay_units'], c['replay_requested'])
        for chk in (launch.check_run_02, launch.check_run_04):
            r = chk(self.ctx())
            self.assertEqual(r.status, PASS, r.detail)
        # the run is complete and exited 0: nothing is left to resume
        self.assertEqual(self.launch_env('official', OFFICIAL_UNITS, {}, resume=True), 1)
        self.assertIn('not interrupted', self.last_err)
        self.assertEqual(len(self.attempts()), 3)

    def test_official_resume_violations(self):
        self.assertEqual(self.launch_env('official', OFFICIAL_UNITS, {'STOP_AFTER': '2'}), 1)
        # a resume that also rewrites a completed unit: launch exits 1, RUN-02 FAILs for good
        rc = self.launch_env('official', OFFICIAL_UNITS, {'REWRITE': 'wo1|W'}, resume=True)
        self.assertEqual(rc, 1)
        self.assertIn('wrote rows for 1 unit(s) an earlier official attempt completed', self.last_err)
        r = launch.check_run_02(self.ctx())
        self.assertEqual(r.status, FAIL)
        self.assertIn('wo1|W', r.detail)

    def test_official_resume_replay_mismatch_and_lock_change(self):
        self.assertEqual(self.launch_env('official', OFFICIAL_UNITS, {'STOP_AFTER': '2'}), 1)
        # the lock changed since the interrupted attempt: no resume
        write(self.d / 'native.py', 'def step(x):\n    return -x\n')
        self.write_source_lock()
        self.commit('relock')
        self.assertEqual(self.launch_env('official', OFFICIAL_UNITS, {}, resume=True), 1)
        self.assertIn('identical lock', self.last_err)
        write(self.d / 'native.py', 'def step(x):\n    return x\n')   # back to the locked bytes
        self.write_source_lock()
        self.commit('lock restored')
        rc = self.launch_env('official', OFFICIAL_UNITS, {'BREAK_REPLAY': '1'}, resume=True)
        self.assertEqual(rc, 1)
        self.assertIn('canonical digest', self.last_err)
        r = launch.check_run_04(self.ctx())
        self.assertEqual(r.status, FAIL)
        self.assertIn('canonical digest', r.detail)
        self.assertEqual(launch.check_run_02(self.ctx()).status, PASS)

    def test_resume_only_for_official(self):
        self.assertEqual(self.launch_env('dev', WRITE_DEV, {}, resume=True), 3)

    def test_inflight_marker_recovery(self):
        """SPEC 20.8: when the launch process itself dies (container restart), its in-flight marker lets the next
        launch record the attempt (interrupted, recovered) instead of leaving its outputs unattributed."""
        self.assertEqual(self.launch_env('official', OFFICIAL_UNITS, {'STOP_AFTER': '2'}), 1)
        rows = self.attempts()
        a = rows[0]
        # rewrite history as if launch had died: drop the row, plant the marker launch writes before the command
        (self.d / 'runlog' / 'ATTEMPTS.jsonl').write_text('')
        proc = subprocess.Popen([sys.executable, '-c', 'pass'])
        proc.wait()
        marker = {'schema': launch.SCHEMA_INFLIGHT, 'attempt_id': a['attempt_id'], 'stage': 'official',
                  'argv': a['argv'], 'cwd': a['cwd'], 'utc_start': a['utc_start'], 'git_head': a['git_head'],
                  'endpoint_lock_sha256': a['endpoint_lock_sha256'], 'source_lock_sha256': a['source_lock_sha256'],
                  'native_closure_sha': a['native_closure_sha'], 'params_sha256': a['params_sha256'],
                  'gate_report': a['gate_report'], 'pid': proc.pid, 'boot_id': launch._boot_id(),
                  'host': launch.socket.gethostname(), 'before': {}}
        write(self.d / launch.inflight_rel(a['attempt_id']), marker)
        r = launch.check_run_01(self.ctx())
        self.assertEqual(r.status, FAIL)
        self.assertIn('in-flight attempt', r.detail)
        rc = self.launch_env('official', OFFICIAL_UNITS, {}, resume=True)
        self.assertEqual(rc, 0, self.last_err)
        self.assertIn('recorded attempt %s' % a['attempt_id'], self.last_err)
        rows = self.attempts()
        rec = rows[0]
        self.assertEqual((rec['attempt_id'], rec['exit_code'], rec['interrupted'], rec['rusage_lost']),
                         (a['attempt_id'], -1, True, True))
        self.assertIn('recovered', rec)
        self.assertEqual([e['path'] for e in rec['produced_paths']], ['results/official/OUTPUTS.jsonl'])
        self.assertEqual(rows[1]['resume_of'], a['attempt_id'])
        self.assertFalse(list((self.d / 'runlog').glob('INFLIGHT_*.json')))
        for chk in (launch.check_run_01, launch.check_run_02, launch.check_run_04):
            r = chk(self.ctx())
            self.assertEqual(r.status, PASS, (chk.__name__, r.detail))
        # a marker of a launch that is still alive is never recovered
        write(self.d / launch.inflight_rel('ATT-live'), dict(marker, attempt_id='ATT-live', pid=os.getpid()))
        self.assertEqual(self.launch_env('dev', WRITE_DEV, {}), 1)
        self.assertIn('still running', self.last_err)

    def test_testbed_without_source_lock(self):
        git(self.repo, 'rm', '-q', NAME + '/SOURCE_LOCK.json')
        self.commit('drop source lock')
        self.assertEqual(self.launch('testbed', 'pass'), 0, self.last_err)
        self.assertIsNone(self.attempts()[-1]['source_lock_sha256'])
        self.assertEqual(self.launch('technical', 'pass'), 2)

    def test_refused_without_command_and_in_timeline_mode(self):
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(minifly_check.main(['--repo-root', str(self.repo), 'launch', '--stage', 'dev', NAME]), 2)
            self.assertEqual(minifly_check.main(['--git', 'timeline:' + str(self.tmp), 'launch', '--stage', 'dev', NAME,
                                                 '--', 'true']), 3)

    def test_cycle_close_rows(self):
        self.assertEqual(self.launch('dev', WRITE_DEV), 0, self.last_err)
        aid = self.attempts()[0]['attempt_id']
        base = ['--repo-root', str(self.repo), 'cycle', 'close', NAME]
        with contextlib.redirect_stderr(io.StringIO()) as err, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(minifly_check.main(base + ['--kind', 'dev', '--decision', 'continue', '--reason', 'go']), 1)
        self.assertIn('exit criterion', err.getvalue())
        self.assertFalse((self.d / 'runlog' / 'CYCLES.jsonl').exists())
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(minifly_check.main(base + ['--kind', 'dev', '--decision', 'revise', '--reason', 'tune']), 0)
        rows = [json.loads(l) for l in (self.d / 'runlog' / 'CYCLES.jsonl').read_text().splitlines()]
        self.assertEqual(rows[0]['k'], 1)
        c = rows[0]['computed']
        self.assertEqual(c['attempts'], [aid])
        self.assertEqual(c['native_closure_sha'], 'n' * 64)
        self.assertIs(c['outcomes_readable'], True)
        self.assertEqual(c['designs_tried_k'], 1)
        self.assertEqual(c['latest_gate']['stage'], 'roster')
        write(self.d / 'results' / 'dev' / 'SCORES.json', {'cells': []})
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(minifly_check.main(base + ['--kind', 'dev', '--decision', 'continue', '--reason', 'ok']), 0)
        rows = [json.loads(l) for l in (self.d / 'runlog' / 'CYCLES.jsonl').read_text().splitlines()]
        self.assertEqual((rows[1]['k'], rows[1]['computed']['attempts']), (2, []))
        # technical continue needs a PASS roster gate report at the current SOURCE_LOCK
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(minifly_check.main(['--repo-root', str(self.repo), 'gate', '--stage', 'roster', NAME]), 0)
            self.assertEqual(minifly_check.main(base + ['--kind', 'technical', '--decision', 'continue', '--reason', 'ok']), 0)
        self.assertEqual(launch.check_cyc_02(self.ctx()).status, PASS)


# ------------------------------------------------------------------------------------------------
class CheckCase(unittest.TestCase):
    """Synthetic delivery with a trivial timeline (no git needed)."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix='mfc_launchchk_'))
        self.fx = self.tmp / 'fx'
        self.root = self.fx / 'repo'
        self.d = self.root / NAME
        write(self.root / 'PROJECT_DOCUMENTS' / 'KB' / 'CONSTRAINTS.json', CONSTRAINTS)
        write(self.d / 'ENDPOINT_LOCK.json', lock_obj())
        write(self.fx / 'timeline.json', {'head': 'a' * 40, 'commits': [
            {'sha': 'a' * 40, 'utc': '20261101T000000Z', 'author': 't', 'trailers': {}, 'branches': [],
             'paths': [NAME + '/ENDPOINT_LOCK.json', 'PROJECT_DOCUMENTS/KB/CONSTRAINTS.json']}]})

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def ctx(self, **opts):
        return common.Ctx(repo_root=self.root, delivery=self.d, git=common.TimelineGit(self.fx), kb=common.KB(self.root),
                          rules=TEST_RULES, options=dict(opts))

    def ctx_stage(self, stage, dry=False, **opts):
        return common.Ctx(repo_root=self.root, delivery=self.d, git=common.TimelineGit(self.fx), kb=common.KB(self.root),
                          rules=TEST_RULES, stage=stage, dry=dry, options=dict(opts))

    def history(self, commits, shallow=False):
        """Rewrite the timeline: the lock commit, then (label, {delivery-relative path: bytes}) commits."""
        base = {'sha': 'a' * 40, 'utc': '20261101T000000Z', 'author': 't', 'trailers': {}, 'branches': [],
                'paths': [NAME + '/ENDPOINT_LOCK.json', 'PROJECT_DOCUMENTS/KB/CONSTRAINTS.json']}
        cs = [base]
        shutil.rmtree(str(self.fx / 'history'), ignore_errors=True)
        for i, (label, files) in enumerate(commits):
            sha = ('%x' % (i + 11)) * 20
            sha = sha[:40]
            for rel, data in files.items():
                write(self.fx / 'history' / sha / NAME / rel, data)
            cs.append({'sha': sha, 'utc': '2026110%dT000000Z' % (i + 2), 'author': 't', 'trailers': {},
                       'branches': [], 'paths': [NAME + '/' + rel for rel in files]})
        write(self.fx / 'timeline.json', {'head': cs[-1]['sha'], 'commits': cs, 'shallow': shallow})

    def attempts(self, rows):
        write(self.d / 'runlog' / 'ATTEMPTS.jsonl', jsonl(rows))

    def att(self, aid, stage, produced=(), clean=True, argv=None, **kw):
        a = {'attempt_id': aid, 'stage': stage, 'argv': argv or ['python3', 'run.py'], 'closure_clean': clean,
             'cpu_children_user_s': 1.0, 'cpu_children_sys_s': 0.0,
             'produced_paths': [{'path': p, 'sha256': common.sha256_file(self.d / p)} for p in produced]}
        a.update(kw)
        return a


class TestRun01(CheckCase):
    def test_coverage(self):
        self.assertEqual(launch.check_run_01(self.ctx()).status, PASS)  # nothing to cover
        write(self.d / 'results/dev/OUTPUTS.jsonl', jsonl([{'attempt_id': 'A1'}]))
        write(self.d / 'results/receipts/wd1/W.json', {'arm': 'W'})
        write(self.d / 'results/dev/SCORES.json', {'checker': 'written'})          # checker-owned: ignored
        write(self.d / 'results/dev/.gitkeep', '')
        r = launch.check_run_01(self.ctx())
        self.assertEqual(r.status, FAIL)
        self.assertIn('results/dev/OUTPUTS.jsonl', r.evidence)
        self.attempts([self.att('A1', 'dev', ['results/dev/OUTPUTS.jsonl', 'results/receipts/wd1/W.json'])])
        self.assertEqual(launch.check_run_01(self.ctx()).status, PASS)
        write(self.d / 'results/dev/OUTPUTS.jsonl', jsonl([{'attempt_id': 'A1'}, {'hand': 'written'}]))
        r = launch.check_run_01(self.ctx())
        self.assertEqual(r.status, FAIL)
        self.assertEqual(r.evidence[0], 'results/dev/OUTPUTS.jsonl')
        self.assertIn('results/dev/OUTPUTS.jsonl:2 attempt_id=None', r.evidence)   # the hand-written row

    def test_rows_must_name_a_launch_attempt(self):
        # the file is launch-recorded, but a row names no attempt: at T2 every counted row comes from launch
        write(self.d / 'results/dev/OUTPUTS.jsonl', jsonl([{'attempt_id': 'A1'}, {'attempt_id': 'A7'}]))
        self.attempts([self.att('A1', 'dev', ['results/dev/OUTPUTS.jsonl'])])
        r = launch.check_run_01(self.ctx())
        self.assertEqual(r.status, FAIL)
        self.assertIn('1 OUTPUTS row(s) whose attempt_id names no ATTEMPTS row', r.detail)
        self.assertIn("results/dev/OUTPUTS.jsonl:2 attempt_id='A7'", r.evidence)

    def test_gz_zip_and_unclean(self):
        write(self.d / 'results/official/OUTPUTS.jsonl', jsonl([{'attempt_id': 'A1'}]))
        self.attempts([self.att('A1', 'official', ['results/official/OUTPUTS.jsonl'])])
        raw = (self.d / 'results/official/OUTPUTS.jsonl').read_bytes()
        (self.d / 'results/official/OUTPUTS.jsonl').unlink()
        write(self.d / 'results/official/OUTPUTS.jsonl.gz', gzip.compress(raw))
        self.assertEqual(launch.check_run_01(self.ctx()).status, PASS)  # decompressed content matches
        (self.d / 'results/official/OUTPUTS.jsonl.gz').unlink()
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w') as zf:
            zf.writestr('results/official/OUTPUTS.jsonl', raw)
        write(self.d / 'RESULT_BUNDLE_official.zip', buf.getvalue())
        self.assertEqual(launch.check_run_01(self.ctx()).status, PASS)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w') as zf:
            zf.writestr('results/official/OUTPUTS.jsonl', raw + b'{"injected": 1}\n')
        write(self.d / 'RESULT_BUNDLE_official.zip', buf.getvalue())
        r = launch.check_run_01(self.ctx())
        self.assertEqual(r.status, FAIL)
        self.assertIn('RESULT_BUNDLE_official.zip!results/official/OUTPUTS.jsonl', r.evidence)
        (self.d / 'RESULT_BUNDLE_official.zip').unlink()
        write(self.d / 'results/official/OUTPUTS.jsonl', raw)
        self.attempts([self.att('A1', 'official', ['results/official/OUTPUTS.jsonl'], clean=False)])
        r = launch.check_run_01(self.ctx())
        self.assertEqual(r.status, FAIL)
        self.assertIn('clean closure', r.detail)


class TestRun03(CheckCase):
    """RUN-03 (SPEC 20.7): the runtime closure of every counted attempt; synthetic deliveries."""

    def setUp(self):
        super().setUp()
        for f, text in (('native.py', 'x = 1\n'), ('run.py', 'import native\n'), ('extra.py', 'y = 2\n'),
                        ('data.bin', 'asset\n'), ('params.json', '{"lr": 0.1}\n')):
            write(self.d / f, text)
        lock = lock_obj()
        lock['files']['assets'] = ['data.bin']
        write(self.d / 'ENDPOINT_LOCK.json', lock)
        self.sha = {f: common.sha256_file(self.d / f) for f in ('native.py', 'run.py', 'extra.py', 'data.bin', 'params.json')}
        write(self.d / 'SOURCE_LOCK.json', {'schema': common.SCHEMA_SOURCE_LOCK, 'closure_sha256': {
            NAME + '/native.py': self.sha['native.py'], NAME + '/run.py': self.sha['run.py']},
            'params_sha256': self.sha['params.json']})
        self.history([('files', {f: (self.d / f).read_bytes() for f in ('native.py', 'run.py', 'extra.py', 'data.bin',
                                                                         'params.json', 'SOURCE_LOCK.json')})])
        self.git_head = json.loads((self.fx / 'timeline.json').read_text())['head']
        write(self.d / 'results/dev/OUTPUTS.jsonl', jsonl([{'attempt_id': 'A1'}]))

    def closure(self, aid, files, over=None):
        doc = {'schema': 'minifly.runtime_closure.v1', 'attempt_id': aid,
               'files': [{'path': NAME + '/' + f, 'sha256': h if h else self.sha[f]} for f, h in files]}
        doc.update(over or {})
        write(self.d / launch.runtime_closure_rel(aid), doc)

    def run03(self, aid='A1', stage='dev', with_closure=True, **kw):
        prod = ['results/dev/OUTPUTS.jsonl'] + ([launch.runtime_closure_rel(aid)] if with_closure else [])
        sl = common.sha256_file(self.d / 'SOURCE_LOCK.json')
        self.attempts([self.att(aid, stage, prod, source_lock_sha256=sl, git_head=self.git_head, **kw)])
        return launch.check_run_03(self.ctx())

    def test_clean_and_closure_gap(self):
        self.closure('A1', [('native.py', None), ('run.py', None), ('data.bin', None), ('params.json', None)])
        r = self.run03()
        self.assertEqual(r.status, PASS, r.detail)
        # a repo .py file imported at runtime but absent from SOURCE_LOCK (the LEARNED-BANK make_lock gap)
        self.closure('A1', [('native.py', None), ('run.py', None), ('extra.py', None)])
        r = self.run03()
        self.assertEqual(r.status, FAIL)
        self.assertIn('closure gap', r.detail)
        self.assertIn(NAME + '/extra.py', r.detail)

    def test_missing_unrecorded_or_edited_closure(self):
        r = self.run03(with_closure=False)
        self.assertEqual(r.status, FAIL)
        self.assertIn('not among its produced_paths', r.detail)
        self.closure('A1', [('native.py', None)])
        r = self.run03(with_closure=False)                       # written, but launch never recorded it
        self.assertEqual(r.status, FAIL)
        self.run03()
        self.closure('A1', [('native.py', None), ('run.py', None)])   # rewritten after the attempt
        r = launch.check_run_03(self.ctx())
        self.assertEqual(r.status, FAIL)
        self.assertIn('no longer holds the bytes', r.detail)
        for bad in ({'schema': 'x'}, {'attempt_id': 'A9'}, {'files': []}):
            self.closure('A1', [('native.py', None)], bad)
            self.assertEqual(self.run03().status, FAIL, bad)

    def test_sha_mismatch_and_undeclared_asset(self):
        self.closure('A1', [('native.py', 'f' * 64)])
        r = self.run03()
        self.assertEqual(r.status, FAIL)
        self.assertIn('SOURCE_LOCK.closure_sha256 has', r.detail)
        write(self.d / 'other.dat', 'z\n')
        self.sha['other.dat'] = common.sha256_file(self.d / 'other.dat')
        self.closure('A1', [('native.py', None), ('other.dat', None)])
        r = self.run03()
        self.assertEqual(r.status, FAIL)
        self.assertIn('data asset', r.detail)
        self.closure('A1', [('native.py', None), ('data.bin', 'e' * 64)])
        self.assertIn('data asset %s/data.bin ran with' % NAME, self.run03().detail)
        self.closure('A1', [('native.py', None), ('params.json', 'e' * 64)])
        self.assertEqual(self.run03().status, FAIL)

    def test_scope_exemptions_and_history_lock(self):
        # testbed attempts are not counted runs; an attempt that produced nothing has nothing to attribute
        self.assertEqual(self.run03(stage='testbed', with_closure=False).status, PASS)
        self.attempts([self.att('A2', 'official', [], exit_code=-9)])
        self.assertEqual(launch.check_run_03(self.ctx()).status, PASS)
        # an attempt may not produce another attempt's runtime closure
        self.closure('A1', [('native.py', None)])
        self.attempts([self.att('A2', 'dev', [launch.runtime_closure_rel('A1')])])
        r = launch.check_run_03(self.ctx())
        self.assertEqual(r.status, FAIL)
        self.assertIn('runtime closure of another attempt', r.detail)
        # the attempt's SOURCE_LOCK is found by sha in history after a later lock build
        self.closure('A1', [('native.py', None), ('run.py', None)])
        r = self.run03()
        self.assertEqual(r.status, PASS, r.detail)
        a = self.att('A1', 'dev', ['results/dev/OUTPUTS.jsonl', launch.runtime_closure_rel('A1')],
                     source_lock_sha256=common.sha256_file(self.d / 'SOURCE_LOCK.json'), git_head=self.git_head)
        write(self.d / 'SOURCE_LOCK.json', {'schema': common.SCHEMA_SOURCE_LOCK, 'closure_sha256': {}})
        self.attempts([a])
        self.assertEqual(launch.check_run_03(self.ctx()).status, PASS)
        a['source_lock_sha256'] = 'd' * 64
        self.attempts([a])
        self.assertEqual(launch.check_run_03(self.ctx()).status, common.MISSING)


class TestRun02(CheckCase):
    def test_official_attempts(self):
        self.attempts([self.att('A1', 'official')])
        self.assertEqual(launch.check_run_02(self.ctx()).status, PASS)  # a crashed attempt with no rows
        write(self.d / 'results/official/OUTPUTS.jsonl', jsonl([{'attempt_id': 'A2', 'world_role': 'official'}]))
        self.attempts([self.att('A1', 'official'), self.att('A2', 'official', ['results/official/OUTPUTS.jsonl'])])
        r = launch.check_run_02(self.ctx())
        self.assertEqual(r.status, PASS, r.detail)
        self.assertEqual(launch.check_run_02(self.ctx(launch_stage='official')).status, FAIL)
        self.attempts([self.att('A1', 'official'), self.att('A2', 'official', ['results/official/OUTPUTS.jsonl']),
                       self.att('A3', 'official')])
        self.assertIn('rerun', launch.check_run_02(self.ctx()).detail)
        write(self.d / 'results/official/OUTPUTS.jsonl', jsonl([{'attempt_id': 'A2'}, {'attempt_id': 'A9'}]))
        self.attempts([self.att('A2', 'official', ['results/official/OUTPUTS.jsonl'])])
        r = launch.check_run_02(self.ctx())
        self.assertEqual(r.status, FAIL)
        self.assertIn('not attributed', r.detail)
        write(self.d / 'results/dev/OUTPUTS.jsonl', jsonl([{'attempt_id': 'D1', 'world_role': 'official'}]))
        write(self.d / 'results/official/OUTPUTS.jsonl', jsonl([{'attempt_id': 'A2'}]))
        self.assertEqual(launch.check_run_02(self.ctx()).status, FAIL)  # official-world rows outside results/official


class TestResumeChecks(CheckCase):
    """RUN-02 as amended and RUN-04 (SPEC 20.8) on synthetic official attempts."""

    def rows(self, aid, units):
        return [{'world': w, 'world_role': 'official', 'arm': a, 'pool': 'HELD', 'checkpoint': 'final',
                 'item_id': 'h%d' % i, 'stream_hex': '%02x' % i, 'attempt_id': aid} for w, a in units for i in range(2)]

    def setup_chain(self, b_extra=None, a_extra=None, replay_aid='B'):
        ua, ub = [('wo1', 'W')], [('wo2', 'W')]
        ra, rb = self.rows('A', ua), self.rows('B', ub)
        write(self.d / 'results/official/OUTPUTS.jsonl', jsonl(ra + rb))
        rep = [dict(r, attempt_id=replay_aid) for r in ra]
        write(self.d / 'results/official_replay/OUTPUTS.jsonl', jsonl(rep))
        a = self.att('A', 'official', ['results/official/OUTPUTS.jsonl'], exit_code=-9, endpoint_lock_sha256='e',
                     source_lock_sha256='s', units=launch.units_record({'wo1|W': ra}))
        b = self.att('B', 'official', ['results/official/OUTPUTS.jsonl', 'results/official_replay/OUTPUTS.jsonl'],
                     exit_code=0, endpoint_lock_sha256='e', source_lock_sha256='s', resume_of='A',
                     replay_requested=[['wo1', 'W']], units=launch.units_record({'wo2|W': rb}))
        a.update(a_extra or {})
        b.update(b_extra or {})
        self.attempts([a, b])

    def test_legal_chain(self):
        self.setup_chain()
        for chk in (launch.check_run_02, launch.check_run_04):
            r = chk(self.ctx())
            self.assertEqual(r.status, PASS, r.detail)

    def test_resume_conditions(self):
        cases = [({'source_lock_sha256': 't'}, None, 'source_lock_sha256'),         # (a) another lock
                 ({'resume_of': 'Z'}, None, 'resume_of'),                          # (b) not the interrupted attempt
                 (None, {'units': None}, 'records no units')]
        lock = json.loads((self.d / 'ENDPOINT_LOCK.json').read_text())
        lock['arms'] = [{'id': 'W', 'role': 'candidate'}]
        write(self.d / 'ENDPOINT_LOCK.json', lock)   # every (world, arm) unit known, so A's unit set is judged
        for b_extra, a_extra, want in cases:
            self.setup_chain(b_extra, a_extra)
            r = launch.check_run_02(self.ctx())
            self.assertEqual(r.status, FAIL, want)
            self.assertIn(want, r.detail)
        # an incomplete unit set makes an exit-0 attempt resumable (A ran wo1 only; the lock lists wo1, wo2)
        self.setup_chain(None, {'exit_code': 0})
        self.assertEqual(launch.check_run_02(self.ctx()).status, PASS)
        # (b) exit 0 with every unit complete: nothing was interrupted, so a resume is a rerun
        lock['worlds']['official'] = {'mode': 'listed', 'ids': ['wo1'], 'n': 1}
        write(self.d / 'ENDPOINT_LOCK.json', lock)
        self.setup_chain(None, {'exit_code': 0})
        r = launch.check_run_02(self.ctx())
        self.assertEqual(r.status, FAIL)
        self.assertIn('not interrupted', r.detail)
        # a second official attempt without resume_of is still a rerun
        self.setup_chain({'resume_of': None})
        r = launch.check_run_02(self.ctx())
        self.assertEqual(r.status, FAIL)
        self.assertIn('rerun', r.detail)

    def test_recovered_attempt_needs_a_replay_of_its_own_units(self):
        # SPEC 20.17 (round 1): an official attempt recovered from an in-flight marker (rusage lost) vouches for its
        # rows only through a determinism replay of one of its OWN units, with the identical digest
        rec = {'exit_code': -1, 'interrupted': True, 'rusage_lost': True,
               'recovered': {'utc': '20261103T000000Z', 'reason': 'test: launch process ended'}}
        self.setup_chain(a_extra=dict(rec))           # B replays A's unit wo1|W
        r = launch.check_run_04(self.ctx())
        self.assertEqual(r.status, PASS, r.detail)
        self.setup_chain(b_extra=dict(rec))           # B itself was recovered: nobody replayed B's unit wo2|W
        r = launch.check_run_04(self.ctx())
        self.assertEqual(r.status, FAIL)
        self.assertIn('recovered official attempt B', r.detail)
        # a lone recovered attempt that completed every unit, no resume and no replay
        ra = self.rows('A', [('wo1', 'W')])
        write(self.d / 'results/official/OUTPUTS.jsonl', jsonl(ra))
        (self.d / 'results/official_replay/OUTPUTS.jsonl').unlink()
        a = self.att('A', 'official', ['results/official/OUTPUTS.jsonl'], endpoint_lock_sha256='e',
                     source_lock_sha256='s', units=launch.units_record({'wo1|W': ra}), **rec)
        self.attempts([a])
        r = launch.check_run_04(self.ctx())
        self.assertEqual(r.status, FAIL)
        self.assertIn('none was replayed', r.detail)
        self.assertEqual(launch.replay_analysis(self.ctx())['recovered_unvouched'], {'A': ['wo1|W']})
        # the same attempt not recovered (an ordinary completed official run): RUN-04 does not apply
        self.attempts([dict(a, exit_code=0, interrupted=False, rusage_lost=False, recovered=None)])
        self.assertEqual(launch.check_run_04(self.ctx()).status, common.NOT_APPLICABLE)

    def test_replay_rows_only_from_resume_attempts(self):
        self.setup_chain(replay_aid='A')
        r = launch.check_run_04(self.ctx())
        self.assertEqual(r.status, FAIL)
        self.assertIn('only a resume attempt', r.detail)
        # replay rows land in results/official_replay/ only from an official attempt (stage scope, CYC-01)
        why = launch.path_scope_problem(self.ctx(), 'dev', 'results/official_replay/OUTPUTS.jsonl', None, {}, False)
        self.assertIn('official resume attempt', why)


class TestCycles(CheckCase):
    def cycles(self, rows):
        write(self.d / 'runlog' / 'CYCLES.jsonl', jsonl(rows))

    def report(self, name, stage, verdict='PASS', dry=False, utc='20261101T000000Z', sl=None):
        status = PASS if verdict == 'PASS' else FAIL
        rc = 0 if verdict == 'PASS' else 1
        write(self.d / 'checks' / name, {'schema': common.SCHEMA_GATE_REPORT, 'stage': stage, 'dry': dry,
                                         'delivery': NAME, 'utc': utc, 'git_mode': 'timeline', 'source_lock_sha256': sl,
                                         'results': [{'id': 'X', 'status': status}], 'exit_code': rc, 'verdict': verdict})

    def rows(self, kinds_attempts):
        return [{'k': i + 1, 'kind': k, 'decision': 'revise', 'computed': {'attempts': list(a)}}
                for i, (k, a) in enumerate(kinds_attempts)]

    def test_cyc_01(self):
        self.cycles(self.rows([(k, []) for k in ('testbed', 'technical', 'dev', 'rehearsal')]))
        self.assertEqual(launch.check_cyc_01(self.ctx()).status, PASS)
        self.cycles(self.rows([(k, []) for k in ('technical', 'dev', 'dev', 'rehearsal')]))
        self.assertEqual(launch.check_cyc_01(self.ctx()).status, FAIL)
        self.cycles([{'kind': 'warmup', 'decision': 'revise', 'computed': {'attempts': []}}])
        self.assertEqual(launch.check_cyc_01(self.ctx()).status, FAIL)
        self.cycles([{'kind': 'dev', 'decision': 'revise'}])           # not written by `cycle close`
        r = launch.check_cyc_01(self.ctx())
        self.assertEqual(r.status, FAIL)
        self.assertIn('computed.attempts', r.detail)

    def test_cyc_01_open_cycles_count_against_the_cap(self):
        self.attempts([self.att('T1', 'technical'), self.att('D1', 'dev'), self.att('D2', 'dev'),
                       self.att('R1', 'rehearsal')])
        self.cycles(self.rows([('technical', ['T1']), ('dev', ['D1']), ('dev', ['D2'])]))
        r = launch.check_cyc_01(self.ctx())
        self.assertEqual(r.status, FAIL)        # 3 closed + the open rehearsal cycle > 3
        self.assertIn('open: rehearsal', r.detail)
        write(self.d / 'AUTHORIZATION.json', {'verbatim': 'Four test cycles are fine.', 'caps': {'max_cycles': 4}})
        self.assertEqual(launch.check_cyc_01(self.ctx()).status, FAIL)   # '4' is not in the verbatim text
        write(self.d / 'AUTHORIZATION.json', {'verbatim': 'Up to 4 test cycles.', 'caps': {'max_cycles': 4}})
        r = launch.check_cyc_01(self.ctx())
        self.assertEqual(r.status, PASS, r.detail)
        r = launch.check_cyc_01(self.ctx_stage('official', dry=True))   # the rehearsal cycle may be open at --dry
        self.assertEqual(r.status, PASS, r.detail)
        r = launch.check_cyc_01(self.ctx_stage('official'))
        self.assertEqual(r.status, FAIL)
        self.assertIn('R1 (rehearsal)', r.detail)
        r = launch.check_cyc_01(self.ctx(launch_stage='official'))
        self.assertEqual(r.status, FAIL)

    def test_cyc_01_cycles_never_closed(self):
        self.attempts([self.att('B1', 'testbed'), self.att('T1', 'technical'), self.att('D1', 'dev'),
                       self.att('R1', 'rehearsal'), self.att('O1', 'official')])
        r = launch.check_cyc_01(self.ctx())
        self.assertEqual(r.status, FAIL)
        self.assertIn('was not closed', r.detail)
        self.assertIn('once an official attempt exists', r.detail)
        self.attempts([self.att('D1', 'dev'), self.att('D2', 'dev')])
        self.assertEqual(launch.check_cyc_01(self.ctx()).status, PASS)     # one open dev cycle within the cap
        self.assertEqual(launch.check_cyc_01(self.ctx_stage('official')).status, FAIL)

    def test_cyc_01_rows_must_agree_with_attempts(self):
        self.attempts([self.att('T1', 'technical'), self.att('D1', 'dev')])
        self.cycles(self.rows([('technical', ['T1', 'D1'])]))
        r = launch.check_cyc_01(self.ctx())
        self.assertEqual(r.status, FAIL)
        self.assertIn('covers dev attempt D1', r.detail)
        self.cycles(self.rows([('technical', ['T1']), ('dev', ['D1', 'D9'])]))
        r = launch.check_cyc_01(self.ctx())
        self.assertEqual(r.status, FAIL)
        self.assertIn("'D9'", r.detail)
        self.cycles(self.rows([('technical', ['T1']), ('technical', ['T1'])]))
        self.assertIn('covered by cycles 1 and 2', launch.check_cyc_01(self.ctx()).detail)
        self.cycles(self.rows([('dev', ['D1'])]))
        r = launch.check_cyc_01(self.ctx())
        self.assertEqual(r.status, FAIL)
        self.assertIn('precede a later cycle close', r.detail)
        self.cycles(self.rows([('technical', ['T1']), ('dev', ['D1'])]))
        self.assertEqual(launch.check_cyc_01(self.ctx()).status, PASS)

    def test_cyc_01_stage_scope(self):
        # each stage writes its own results dir and its own worlds' receipts/TAUGHT (SPEC 3.1, 3.2)
        write(self.d / 'results/admission/OUTPUTS.jsonl', jsonl([
            {'attempt_id': 'B1', 'arm': 'CEILING_X', 'world': 'wt1'}, {'attempt_id': 'T1', 'arm': 'W', 'world': 'wt1'}]))
        write(self.d / 'results/dev/OUTPUTS.jsonl', jsonl([{'attempt_id': 'D1', 'arm': 'W', 'world': 'wd1'}]))
        write(self.d / 'results/receipts/wd1/W.json', {'arm': 'W'})
        write(self.d / 'manifests/TAUGHT/wd1/W.jsonl', '{}\n')
        good = [self.att('B1', 'testbed', ['results/admission/OUTPUTS.jsonl', 'manifests/TAUGHT/wd1/W.jsonl']),
                self.att('T1', 'technical', ['results/admission/OUTPUTS.jsonl']),
                self.att('D1', 'dev', ['results/dev/OUTPUTS.jsonl', 'results/receipts/wd1/W.json'])]
        self.attempts(good)
        self.cycles(self.rows([('testbed', ['B1']), ('technical', ['T1']), ('dev', ['D1'])]))
        r = launch.check_cyc_01(self.ctx())
        self.assertEqual(r.status, PASS, r.detail)
        self.assertEqual(launch.stage_scope_problems(self.ctx()), [])

        def scope(attempts, rows=None):
            self.attempts(attempts)
            self.cycles(self.rows([(a['stage'], [a['attempt_id']]) for a in attempts]))
            return ' | '.join(launch.stage_scope_problems(self.ctx()))
        # a testbed attempt that writes results/dev (the cap bypass) or a learner-arm receipt
        self.assertIn('wrote results/dev/OUTPUTS.jsonl (a testbed attempt writes results/admission/ only)',
                      scope([self.att('B1', 'testbed', ['results/dev/OUTPUTS.jsonl'])]))
        self.assertIn('learner-arm receipt results/receipts/wd1/W.json',
                      scope([self.att('B1', 'testbed', ['results/receipts/wd1/W.json'])]))
        # dev rows that name a testbed attempt; learner rows from a testbed attempt under results/admission
        write(self.d / 'results/dev/OUTPUTS.jsonl', jsonl([{'attempt_id': 'B1', 'arm': 'W', 'world': 'wd1'}]))
        self.assertIn('rows of testbed attempt B1 under results/dev/', scope([self.att('B1', 'testbed')]))
        write(self.d / 'results/admission/OUTPUTS.jsonl', jsonl([{'attempt_id': 'B1', 'arm': 'W', 'world': 'wt1'}]))
        self.assertIn('learner-arm rows (arm W) from testbed attempt B1', scope([self.att('B1', 'testbed')]))
        # a technical attempt on a dev world; a dev attempt on the rehearsal or official world
        write(self.d / 'results/admission/OUTPUTS.jsonl', jsonl([{'attempt_id': 'T1', 'arm': 'W', 'world': 'wd2'}]))
        self.assertIn('rows of technical attempt T1 on world wd2, which is not a technical world',
                      scope([self.att('T1', 'technical')]))
        write(self.d / 'results/receipts/wr1/W.json', {'arm': 'W'})
        self.assertIn('receipt results/receipts/wr1/W.json of world wr1, which is not a dev world',
                      scope([self.att('D1', 'dev', ['results/receipts/wr1/W.json'])]))
        write(self.d / 'manifests/TAUGHT/wo1/W.jsonl', '{}\n')
        self.assertIn('TAUGHT manifest manifests/TAUGHT/wo1/W.jsonl of world wo1',
                      scope([self.att('D1', 'dev', ['manifests/TAUGHT/wo1/W.jsonl'])]))
        # a testbed attempt may generate any world's TAUGHT streams in cycle 0, never after learner attempts
        write(self.d / 'results/dev/OUTPUTS.jsonl', jsonl([{'attempt_id': 'D1', 'arm': 'W', 'world': 'wd1'}]))
        write(self.d / 'results/admission/OUTPUTS.jsonl', jsonl([{'attempt_id': 'T1', 'arm': 'W', 'world': 'wt1'}]))
        self.assertEqual(scope([self.att('B1', 'testbed', ['manifests/TAUGHT/wd1/W.jsonl'])]), '')
        self.assertIn('after learner attempts exist',
                      scope([self.att('T1', 'technical'), self.att('B2', 'testbed', ['manifests/TAUGHT/wd1/W.jsonl'])]))
        # a result bundle outside results/<role>/ is judged by its members
        (self.d / 'results/dev/OUTPUTS.jsonl').unlink()
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w') as zf:
            zf.writestr('results/dev/OUTPUTS.jsonl', jsonl([{'attempt_id': 'B1', 'arm': 'W', 'world': 'wd1'}]))
        write(self.d / 'results/RESULT_BUNDLE_y.zip', buf.getvalue())
        out = scope([self.att('B1', 'testbed', ['results/RESULT_BUNDLE_y.zip'])])
        self.assertIn('(member of results/RESULT_BUNDLE_y.zip)', out)
        self.assertIn('rows of testbed attempt B1 under results/dev/', out)
        r = launch.check_cyc_01(self.ctx())
        self.assertEqual(r.status, FAIL)
        # the pending argv check used by launch
        self.assertEqual(launch.argv_world_problems('testbed', ['run.py', '--worlds', 'wd1,wt1'], lock_obj()), ['wd1'])
        self.assertEqual(launch.argv_world_problems('official', ['run.py', '--worlds', 'wo1,wo2'], lock_obj()), [])
        self.assertEqual(launch.argv_world_problems('dev', ['run.py', 'wr1'], lock_obj()), ['wr1'])

    def test_cyc_01_runlog_append_only(self):
        rows = self.rows([('technical', ['T1']), ('dev', ['D1'])])
        att = [self.att('T1', 'technical'), self.att('D1', 'dev')]
        v1 = jsonl(rows).encode()
        a1 = jsonl(att).encode()
        self.history([('c1', {'runlog/CYCLES.jsonl': v1, 'runlog/ATTEMPTS.jsonl': a1})])
        self.cycles(rows)
        self.attempts(att)
        self.assertEqual(launch.check_cyc_01(self.ctx()).status, PASS)
        self.cycles(rows[:1])                                           # a committed row deleted in the worktree
        r = launch.check_cyc_01(self.ctx())
        self.assertEqual(r.status, FAIL)
        self.assertIn('runlog/CYCLES.jsonl: the worktree removes or edits 1 committed row', r.detail)
        (self.d / 'runlog' / 'CYCLES.jsonl').unlink()                   # the whole file deleted (probe P4)
        r = launch.check_cyc_01(self.ctx())
        self.assertEqual(r.status, FAIL)
        self.assertIn('deletes the file', r.detail)
        self.attempts(att[1:])                                          # an attempt row deleted
        self.cycles(rows)
        self.assertIn('runlog/ATTEMPTS.jsonl', launch.check_cyc_01(self.ctx()).detail)
        self.attempts(att)
        edited = [dict(rows[0], reason='rewritten later'), rows[1]]
        self.history([('c1', {'runlog/CYCLES.jsonl': v1, 'runlog/ATTEMPTS.jsonl': a1}),
                      ('c2', {'runlog/CYCLES.jsonl': jsonl(edited).encode()})])
        self.cycles(edited)
        r = launch.check_cyc_01(self.ctx())
        self.assertEqual(r.status, FAIL)                                # a committed row edited in a later commit
        self.assertIn('removes or edits 1 row(s) committed at', r.detail)
        self.history([('c1', {'runlog/CYCLES.jsonl': v1, 'runlog/ATTEMPTS.jsonl': a1})], shallow=True)
        self.cycles(rows)
        self.assertEqual(common.run_check(self.ctx(), {'id': 'CYC-01', 'module': 'launch', 'status': 'enforced',
                                                       'applies_when': 'always'})[0].status, common.MISSING)

    def test_cyc_04_append_only_at_publish(self):
        # SPEC 20.16 (round 1): the publish-stage check of the same append-only rule; an ATTEMPTS row edited after the
        # run (here the earlier dev attempt's cpu, in a later commit) cannot pass publish
        rows = self.rows([('technical', ['T1']), ('dev', ['D1'])])
        att = [self.att('T1', 'technical'), self.att('D1', 'dev')]
        v1, a1 = jsonl(rows).encode(), jsonl(att).encode()
        self.history([('c1', {'runlog/CYCLES.jsonl': v1, 'runlog/ATTEMPTS.jsonl': a1})])
        self.cycles(rows)
        self.attempts(att)
        r = launch.check_cyc_04(self.ctx_stage('publish'))
        self.assertEqual(r.status, PASS, r.detail)
        self.assertIn('runlog/ATTEMPTS.jsonl 2 rows', r.detail)
        edited = [att[0], dict(att[1], cpu_children_user_s=0.01)]
        self.history([('c1', {'runlog/CYCLES.jsonl': v1, 'runlog/ATTEMPTS.jsonl': a1}),
                      ('c2', {'runlog/ATTEMPTS.jsonl': jsonl(edited).encode()})])
        self.attempts(edited)
        r = launch.check_cyc_04(self.ctx_stage('publish'))
        self.assertEqual(r.status, FAIL)
        self.assertIn('runlog/ATTEMPTS.jsonl: commit', r.detail)
        self.attempts(att[:1] + edited[1:])                              # worktree edit only
        self.history([('c1', {'runlog/CYCLES.jsonl': v1, 'runlog/ATTEMPTS.jsonl': a1})])
        self.assertIn('the worktree removes or edits', launch.check_cyc_04(self.ctx_stage('publish')).detail)
        self.history([('c1', {'runlog/CYCLES.jsonl': v1, 'runlog/ATTEMPTS.jsonl': a1})], shallow=True)
        self.attempts(att)
        self.assertEqual(common.run_check(self.ctx(), {'id': 'CYC-04', 'module': 'launch', 'status': 'enforced',
                                                       'applies_when': 'always'})[0].status, common.MISSING)

    def test_cyc_02_exit_criteria(self):
        self.cycles([{'k': 1, 'kind': 'testbed', 'decision': 'continue', 'utc': '20261102T000000Z'}])
        self.assertEqual(launch.check_cyc_02(self.ctx()).status, FAIL)
        write(self.d / 'results/admission/TESTBED.json', {'x': 1})
        self.assertEqual(launch.check_cyc_02(self.ctx()).status, PASS)
        self.cycles([{'k': 1, 'kind': 'technical', 'decision': 'continue', 'utc': '20261102T000000Z',
                      'computed': {'source_lock_sha256': 's' * 64}}])
        self.report('GATE_roster_20261101T000000Z.json', 'roster', sl='t' * 64)
        self.assertEqual(launch.check_cyc_02(self.ctx()).status, FAIL)  # wrong SOURCE_LOCK
        self.report('GATE_roster_20261103T000000Z.json', 'roster', sl='s' * 64, utc='20261103T000000Z')
        self.assertEqual(launch.check_cyc_02(self.ctx()).status, FAIL)  # after the close
        self.report('GATE_roster_20261101T000001Z.json', 'roster', sl='s' * 64, utc='20261101T000001Z')
        self.assertEqual(launch.check_cyc_02(self.ctx()).status, PASS)
        self.cycles([{'k': 1, 'kind': 'rehearsal', 'decision': 'continue', 'utc': '20261102T000000Z'},
                     {'k': 2, 'kind': 'dev', 'decision': 'stop', 'utc': '20261102T000000Z'}])
        self.report('GATE_official_20261101T000000Z.json', 'official')          # not dry: does not count
        self.assertEqual(launch.check_cyc_02(self.ctx()).status, FAIL)
        self.report('DRY_official_20261101T000000Z.json', 'official', dry=True, verdict='FAIL')
        self.assertEqual(launch.check_cyc_02(self.ctx()).status, FAIL)
        self.report('DRY_official_20261101T000002Z.json', 'official', dry=True, utc='20261101T000002Z')
        self.assertEqual(launch.check_cyc_02(self.ctx()).status, PASS)
        self.cycles([{'k': 1, 'kind': 'dev', 'decision': 'continue', 'utc': '20261102T000000Z'}])
        self.assertEqual(launch.check_cyc_02(self.ctx()).status, FAIL)
        write(self.d / 'results/dev/SCORES.json', {})
        self.assertEqual(launch.check_cyc_02(self.ctx()).status, PASS)

    def test_argv_template(self):
        ids = ['wr1', 'wo1', 'wo2']
        self.assertEqual(launch.argv_template(['python3', 'run.py', '--worlds', 'wo1,wo2', '--seed', '3'], ids),
                         ['python3', 'run.py', '--seed', '3'])
        self.assertEqual(launch.argv_template(['python3', 'run.py', '--only-world=wr1', 'wr1'], ids), ['python3', 'run.py'])
        self.assertEqual(launch.argv_template(['python3', 'run.py', '--arms', 'W'], ids), ['python3', 'run.py', '--arms', 'W'])

    def test_cyc_03(self):
        r = launch.check_cyc_03(self.ctx())
        self.assertEqual(r.status, FAIL)
        self.assertIn('no rehearsal attempt', r.detail)
        self.attempts([self.att('R1', 'rehearsal', argv=['python3', 'run.py', '--worlds', 'wr1'])])
        r = launch.check_cyc_03(self.ctx())
        self.assertEqual(r.status, PASS, r.detail)
        self.assertIn('enforced at launch', r.detail)
        self.assertEqual(launch.check_cyc_03(self.ctx(launch_stage='official',
                                                      pending_argv=['python3', 'run.py', '--worlds', 'wo1,wo2'])).status, PASS)
        r = launch.check_cyc_03(self.ctx(launch_stage='official', pending_argv=['python3', 'run.py', '--fast', '--worlds', 'wo1']))
        self.assertEqual(r.status, FAIL)
        self.assertIn('template', r.detail)
        self.attempts([self.att('D1', 'dev', argv=['python3', 'run.py', '--only-world', 'wr1']),
                       self.att('R1', 'rehearsal', argv=['python3', 'run.py', '--worlds', 'wr1'])])
        self.assertEqual(launch.check_cyc_03(self.ctx()).status, FAIL)
        self.attempts([self.att('R1', 'rehearsal', argv=['python3', 'run.py', '--worlds', 'wr1'])])
        write(self.d / 'results/dev/OUTPUTS.jsonl', jsonl([{'world': 'wr1', 'world_role': 'dev'}]))
        self.assertEqual(launch.check_cyc_03(self.ctx()).status, FAIL)
        (self.d / 'results/dev/OUTPUTS.jsonl').unlink()
        write(self.d / 'results/receipts/wr1/W.json', {'world': 'wr1', 'world_role': 'technical'})
        self.assertEqual(launch.check_cyc_03(self.ctx()).status, FAIL)
        (self.d / 'results/receipts/wr1/W.json').unlink()
        write(self.d / 'ENDPOINT_LOCK.json', lock_obj(worlds=dict(lock_obj()['worlds'], dev=['wd1', 'wr1'])))
        self.assertEqual(launch.check_cyc_03(self.ctx()).status, FAIL)
        write(self.d / 'ENDPOINT_LOCK.json', lock_obj(worlds=dict(lock_obj()['worlds'], rehearsal=[])))
        self.assertEqual(launch.check_cyc_03(self.ctx()).status, FAIL)


if __name__ == '__main__':
    unittest.main()
