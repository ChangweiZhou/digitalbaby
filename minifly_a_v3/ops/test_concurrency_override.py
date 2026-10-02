"""Isolated operational acceptance tests for the frozen six-worker override.

The frozen launcher and its fake test environment are imported unchanged.
Every receipt and ledger here lives in a temporary synthetic test directory.
"""
import contextlib
import hashlib
import io
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests'))
import test_launcher as fixture

ds = fixture.ds
ARMS = ('R1', 'R0', 'R0_signed', 'R3', 'R3_randtarget', 'Z2',
        'Z0_resource', 'Z2_rand', 'R1_rand', 'P0', 'P1', 'P2', 'P4')


class TrackingDriver(ds.Driver):
    def __init__(self, env):
        super().__init__(env)
        self.peak_workers = 0

    def status(self, locked, done, running):
        self.peak_workers = max(self.peak_workers, len(running))
        super().status(locked, done, running)


def setup():
    ds.ARMS, ds.WORLDS = ARMS, (1, 2)
    ds.DEPS, ds.PERSIST_EVERY = {'R1_rand': 'R1', 'Z2_rand': 'Z2'}, 4
    return Path(tempfile.mkdtemp(prefix='a3-six-workers-'))


def test_six_workers_preserve_roster_yokes_and_budget_definition():
    base = setup()
    try:
        env = fixture.FakeEnv(base, hard={'workers': 4, 'per_world_arm_life_s_max': 10000})
        before = env.budget()
        driver = TrackingDriver(env)
        result = driver.run(workers=6)
        assert result['state'] == 'complete' and result['validated'] == 26
        assert driver.peak_workers == 6
        assert env.budget() == before and before['hard_budget']['workers'] == 4
        jobs = fixture.joblog(base)
        assert len(jobs) == len(set(jobs)) == 26
        for world in ('1', '2'):
            for child, parent in ds.DEPS.items():
                assert jobs.index((parent, world)) < jobs.index((child, world))
    finally:
        shutil.rmtree(base)


def test_six_worker_seconds_are_charged_and_exhaustion_remains_final():
    base = setup()
    try:
        env = fixture.FakeEnv(base, job='hang', hard={'workers': 4, 'core_hours': 100 / 3600})
        driver = TrackingDriver(env)
        result = driver.run(workers=6)
        assert driver.peak_workers == 6
        assert result['failure']['type'] == 'budget_exhausted'
        assert result['failure']['detail']['worker_s'] == 120
        fixture.expect_exit(lambda: ds.Driver(fixture.FakeEnv(base)).run(workers=6), 'final')
        assert not list((base / 'science').rglob('*.json.gz'))
    finally:
        shutil.rmtree(base)


def test_persistence_stop_drains_then_resume_six_never_repeats_receipts():
    base = setup()
    try:
        stopped = TrackingDriver(fixture.FakeEnv(base, push_ok=False,
                  hard={'workers': 4, 'per_world_arm_life_s_max': 10000})).run(workers=4)
        assert stopped['state'] == 'stopped_on_failure'
        assert stopped['failure']['type'] == 'infrastructure'
        status = json.loads((base / 'science/RUN_STATUS.json').read_text())
        assert status['running'] == []
        hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in (base / 'science').rglob('*.json.gz')}
        old = json.loads((base / 'science/RUN_LEDGER.json').read_text())
        resumed = TrackingDriver(fixture.FakeEnv(base, hard={'workers': 4,
                                'per_world_arm_life_s_max': 10000}))
        assert resumed.run(workers=6)['state'] == 'complete'
        assert resumed.peak_workers == 6
        jobs = fixture.joblog(base)
        assert len(jobs) == len(set(jobs)) == 26
        assert all(hashlib.sha256(p.read_bytes()).hexdigest() == h for p, h in hashes.items())
        new = json.loads((base / 'science/RUN_LEDGER.json').read_text())
        assert all(new[k] >= old[k] for k in ('active_wall_s', 'worker_s'))
        assert new['cleared'][-1]['failure'] == old['failure']
    finally:
        shutil.rmtree(base)


def test_failed_checkpoint_finishes_three_inflight_jobs_without_dispatch():
    import time
    base = setup()
    ds.PERSIST_EVERY = 1
    class DrainEnv(fixture.FakeEnv):
        def __init__(self):
            super().__init__(base, hard={'workers': 4, 'per_world_arm_life_s_max': 10000})
            self.observed_inflight_at_failure = 0
        def job_target(self):
            def job(arm, world, result):
                (base / f'{arm}-{world}.started').write_text('started')
                if arm == 'R1' and world == 1:
                    while len(list(base.glob('*.started'))) < 4:
                        time.sleep(0.005)
                else:
                    while not (base / 'failed-checkpoint').exists():
                        time.sleep(0.005)
                fixture._write_receipt(self.sci, arm, world, fixture.fake_receipt(arm, world))
                Path(result).write_text(json.dumps({'ok': True, 'arm': arm, 'world': world,
                    'life_s': 1, 'peak_rss_bytes': 1, 'receipt_bytes': 1}))
            return job
        def persist(self, message):
            if not (base / 'failed-checkpoint').exists():
                self.observed_inflight_at_failure = 4 - len(list(self.sci.glob('*/*.json.gz')))
                assert self.observed_inflight_at_failure == 3
                (base / 'failed-checkpoint').write_text('intentional test persistence failure')
            raise RuntimeError('synthetic durability pause')
    try:
        env = DrainEnv()
        result = TrackingDriver(env).run(workers=4)
        assert env.observed_inflight_at_failure == 3
        assert result['state'] == 'stopped_on_failure' and result['validated'] == 4
        assert result['failure']['type'] == 'infrastructure'
        assert len(list(base.glob('*.started'))) == 4, 'Dispatch continued after checkpoint failure'
        assert len(list(env.sci.glob('*/*.json.gz'))) == 4, 'An in-flight job was killed or lost'
        assert all(json.loads(p.read_text())['ok'] is True for p in env.jobs.glob('*.json'))
        assert json.loads((env.sci / 'RUN_STATUS.json').read_text())['running'] == []
    finally:
        shutil.rmtree(base)


def test_drain_guard_refuses_a_budget_failure_masked_by_infrastructure():
    from concurrency_transition import check_drain
    failure = {'type': 'infrastructure', 'detail': {'stage': 'persist', 'error': 'RuntimeError(\"Durability backlog reached8 receipts without either verified GitHub or private Library backup\")'}}
    status = {'state': 'stopped_on_failure', 'running': [], 'failure': failure, 'validated_receipts': 1}
    ledger = {'failure': failure, 'active_wall_s': 1, 'worker_s': 6}
    observation = {'first_failure_seen_unix_s': 1, 'observed_running_pairs': [['R0', 1]]}
    jobs = {('R0', 1): {'ok': True, 'arm': 'R0', 'world': 1, 'life_s': 1, 'peak_rss_bytes': 1}}
    assert check_drain(status, ledger, observation, {('R0', 1): 'hash'}, jobs, fixture.HARD)
    for change in ('missing_receipt', 'missing_job', 'failed_job', 'rss', 'deadline', 'active_workers', 'not_terminal', 'budget'):
        import copy
        st, le, ob, ha, jo = map(copy.deepcopy, (status, ledger, observation, {('R0', 1): 'hash'}, jobs))
        if change == 'missing_receipt': ha.clear()
        elif change == 'missing_job': jo.clear()
        elif change == 'failed_job': jo[('R0', 1)]['ok'] = False
        elif change == 'rss': jo[('R0', 1)]['peak_rss_bytes'] = 1001
        elif change == 'deadline': jo[('R0', 1)]['life_s'] = 101
        elif change == 'active_workers': st['running'] = [['R0', 1]]
        elif change == 'not_terminal': st.pop('state')
        elif change == 'budget': le['worker_s'] = fixture.HARD['core_hours'] * 3600 + 1
        try:
            check_drain(st, le, ob, ha, jo, fixture.HARD)
        except AssertionError:
            continue
        raise AssertionError('Drain guard accepted invalid case: ' + change)


if __name__ == '__main__':
    tests = sorted((name, value) for name, value in globals().items()
                   if name.startswith('test_') and callable(value))
    for name, test in tests:
        with contextlib.redirect_stdout(io.StringIO()):
            test()
        print('PASS', name, flush=True)
    print(json.dumps({'pass': True, 'tests': len(tests),
                      'launcher_sha256': hashlib.sha256((ROOT / 'src/drive_science.py').read_bytes()).hexdigest()}))
