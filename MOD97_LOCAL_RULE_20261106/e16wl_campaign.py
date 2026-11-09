"""Acquire the shared machine lock, qualify, then supervise six E16-WL lives."""
import math
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from e16wr2_common import OUT, ROOT, SEEDS, producers, read, save, sha
from e16wl import DOSES

STATUS = ROOT / 'E16WL_STATUS.json'
LOCK = ROOT / 'E16WL_LOCK.json'
MACHINE = OUT / 'MACHINE_LOCK'
LOGS = OUT / 'logs'
TASKS = ('add97', 'sub97', 'max97', 'min97', 'add73', 'sub73', 'max73', 'min73', 'compose')
PAIRS = ('+-', '-+', '><', '<>')
JOBS = [(arm, seed) for arm in ('C', 'D') for seed in SEEDS]


def receipt_path(arm, seed):
    tag = '_observed' if arm == 'C' else ''
    return OUT / f'e16wl_e8_life_{arm}{tag}_{seed}.json'


def control_path(arm, seed):
    return OUT / (f'e16wr2_B_{seed}_e8_life_C_observed_{seed}.json' if arm == 'C'
                  else f'e8_life_D_{seed}.json')


def verify():
    lock = read(LOCK)
    for path, digest in lock['hashes'].items():
        assert sha(ROOT / path) == digest, path
    return lock


def prepare():
    files = [ROOT / name for name in (
        'E16WL_SOURCE_BRIEF.md', 'E16WL_PREDICTIONS.md', 'e16wl.py', 'e16wl_checks.py',
        'e16wl_campaign.py', 'exp8_e16wr2.py', 'e16wr2_common.py', 'e16wr_reservoir.py',
        'e14_fast_inference.py', 'exp45.py', 'exp3_state.py', 'mod97_local.py',
        'E16WR2_B_INPUT_REFERENCE.json')]
    files += [OUT / f'e8_dev_{seed}.pt' for seed in SEEDS]
    files += [control_path(arm, seed) for arm, seed in JOBS]
    with LOCK.open('x') as stream:
        import json
        json.dump(dict(brief_commit='24180a8', jobs=JOBS, updates_per_life=sum(DOSES),
                       python=sys.version, executable=sys.executable,
                       hashes={str(path.relative_to(ROOT)): sha(path) for path in files}), stream, indent=2)
        stream.write('\n')


def legacy_handoff():
    # The existing Claude waiter predates the common lock and launches the
    # all-byte cell twin as soon as the previous workers exit.
    ps = subprocess.run(['ps', '-axo', 'command='], check=True, capture_output=True, text=True)
    return any(line.lstrip().startswith(('/bin/zsh ', '/bin/bash ')) and
               'E16_LOSS=all nohup python3 exp16_integrated.py' in line for line in ps.stdout.splitlines())


def acquire(token):
    while True:
        if not producers() and not legacy_handoff():
            try:
                fd = os.open(MACHINE, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            except FileExistsError:
                pass
            else:
                with os.fdopen(fd, 'w') as stream:
                    stream.write(token)
                if not producers() and not legacy_handoff():
                    return
                if MACHINE.read_text() == token:
                    MACHINE.unlink()
        time.sleep(15)


def validate(arm, seed):
    path = receipt_path(arm, seed)
    data = read(path)
    assert data['arm'] == arm and data['seed'] == seed
    assert len(data['stages']) == 10 and [r['task'] for r in data['stages'][1:]] == list(TASKS)
    for stage in data['stages']:
        assert all(stage['scores'][op]['n'] == 600 and 0 <= stage['scores'][op]['exact'] <= 600 for op in TASKS[:-1])
    assert all(row['n'] == 600 and 0 <= row['exact'] <= row['final'] <= 600
               for row in data['stages'][-1]['scores']['compose'].values())
    meta = read(path.with_suffix('.loss.json'))
    assert meta['arm'] == arm and meta['seed'] == seed and meta['loss'] == 'all'
    assert [r['updates'] for r in meta['stages']] == DOSES
    assert all(math.isfinite(r[k]) and r[k] >= 0 for r in meta['stages'] for k in ('prompt_nll', 'answer_nll'))
    return data, meta


def finalize(results, wall):
    lines = ['# E16-WL results', '', 'All six runs passed receipt checks.', '',
             'Each loss comparison uses its arm’s answer-loss reference. C uses observed replay and a Store. D samples the joint pool of observed examples and has no Store.', '',
             '| Arm | Seed | Stage 8 mean % | Add97 % | Sub97 % | Final held % | Reference held % | Held loss pp | P1/P2 | P3 |',
             '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    stage_lines = ['| Arm | Seed | Stage | Task | Prompt NLL | Answer NLL | Prompt tokens | Answer tokens |',
                   '| --- | --- | --- | --- | --- | --- | --- | --- |']
    costs = []
    for arm, seed in JOBS:
        data, meta = results[arm, seed]
        ref = read(control_path(arm, seed))
        ops = data['stages'][8]['scores']
        count = sum(ops[name]['exact'] for name in TASKS[:-1])
        add, sub = (ops[name]['exact'] for name in ('add97', 'sub97'))
        def held(receipt):
            return sum(receipt['stages'][-1]['scores']['compose'][pair]['final'] for pair in PAIRS)
        actual, baseline = held(data), held(ref)
        diff = baseline - actual
        operation_pass = (count < 2880 and add < 120 and sub < 120) if arm == 'C' else count >= 4080
        composition_pass = diff >= 360 if arm == 'C' else diff <= 240
        lines.append(f'| {arm} | {seed} | {count/48:.2f} | {add/6:.2f} | {sub/6:.2f} | {actual/24:.2f} | {baseline/24:.2f} | {diff/24:+.2f} | {operation_pass} | {composition_pass} |')
        for row, task in zip(meta['stages'], TASKS):
            stage_lines.append(f"| {arm} | {seed} | {row['stage']} | {task} | {row['prompt_nll']:.5f} | {row['answer_nll']:.5f} | {row['prompt_bytes']} | {row['answer_bytes']} |")
        costs.append(dict(arm=arm, seed=seed, cpu=meta['worker_cpu_seconds'],
                          peak_rss=meta['peak_rss_raw'], units=meta['peak_rss_units']))
    summary = '\n'.join(lines)
    summary += f'\n\nThe six worker timers sum to {sum(row["cpu"] for row in costs)/3600:.3f} CPU-hours. Qualification and training took {wall/60:.2f} minutes after lock acquisition.\n'
    (ROOT / 'E16WL_REVIEW_SECTION_9_50.md').write_text(summary.replace('# E16-WL results', '# 9.50 E16-WL'))
    (ROOT / 'E16WL_REPORT.md').write_text(summary + '\nPer-stage token-weighted training losses in nats.\n\n' + '\n'.join(stage_lines) + '\n')
    save(ROOT / 'E16WL_VERIFICATION.json', dict(status='PASS', lives=6, checks=read(ROOT/'E16WL_CHECKS.json'),
         controller_wall_seconds=wall, costs=costs,
         receipts={receipt_path(arm, seed).name: sha(receipt_path(arm, seed)) for arm, seed in JOBS}))


def main():
    (OUT / 'e16wl_campaign.guard').mkdir()
    lock = verify()
    token = f'E16-WL pid={os.getpid()} time={datetime.now(timezone.utc).isoformat()}\n'
    save(STATUS, dict(status='WAITING_FOR_MACHINE', pid=os.getpid(), jobs=6))
    acquire(token)
    started = time.monotonic()
    workers = {}
    state = dict(pid=os.getpid(), jobs=6, completed=0)

    def interrupt(signum, frame):
        raise InterruptedError(f'Signal {signum}')

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    try:
        save(STATUS, dict(state, status='QUALIFYING'))
        with (LOGS / 'e16wl_checks.log').open('x') as log:
            subprocess.run([sys.executable, 'e16wl_checks.py'], stdout=log, stderr=subprocess.STDOUT, check=True)
        assert read(ROOT / 'E16WL_CHECKS.json')['status'] == 'PASS'
        assert verify() == lock and MACHINE.read_text() == token
        assert not producers() and not legacy_handoff()
        for arm, seed in JOBS:
            cmd = [sys.executable, 'e16wl.py', arm, str(seed)]
            with (LOGS / f'e16wl_{arm}_{seed}.log').open('x') as log:
                workers[arm, seed] = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                                      env=dict(os.environ, E8_LOSS='all'))
        save(STATUS, dict(state, status='RUNNING', workers={f'{a}_{s}': p.pid for (a, s), p in workers.items()}))
        results = {}
        pending = workers.copy()
        failures = []
        while pending:
            for (arm, seed), worker in list(pending.items()):
                code = worker.poll()
                if code is None:
                    continue
                del pending[arm, seed]
                job = dict(arm=arm, seed=seed, returncode=code, command=worker.args)
                try:
                    assert code == 0
                    results[arm, seed] = validate(arm, seed)
                    job['receipt_sha256'] = sha(receipt_path(arm, seed))
                except Exception as exc:
                    job['error'] = repr(exc)
                    failures.append(job)
                save(OUT / f'E16WL_JOB_{arm}_{seed}.json', job)
            state.update(completed=len(results), running=len(pending), failures=failures)
            save(STATUS, dict(state, status='RUNNING'))
            if pending:
                time.sleep(10)
        if failures:
            raise RuntimeError('One or more lives failed. See exact job receipts.')
        verify()
        finalize(results, time.monotonic()-started)
        save(STATUS, dict(state, status='COMPLETE_VERIFIED'))
    except BaseException as exc:
        save(STATUS, dict(state, status='FAILED', error=repr(exc)))
        raise
    finally:
        for worker in workers.values():
            if worker.poll() is None:
                worker.terminate()
        for worker in workers.values():
            worker.wait()
        if MACHINE.exists() and MACHINE.read_text() == token:
            MACHINE.unlink()


if __name__ == '__main__':
    (prepare if sys.argv[1:] == ['--prepare'] else main)()
