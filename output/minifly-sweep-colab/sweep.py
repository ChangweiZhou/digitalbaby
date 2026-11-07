"""Run the MiniFly development sweep with durable per-job receipts."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
import traceback

ROOT = Path(__file__).resolve().parent
SEEDS = (284813, 516197, 873411)
BLOCKS = {
    'cortical': ('sign', 'dfa', 'bp'),
    'attention': ('attention', 'register', 'uniform'),
    'central_complex': ('native', 'rewired', 'displaced'),
    'cerebellar': ('grouped_lags', 'ordinary', 'shuffled_lags'),
    'thalamocortical': ('loop', 'feedforward', 'one_pass'),
    'working_memory_gate': ('learned', 'ungated', 'yoked'),
    'actor_critic': ('trace', 'immediate', 'shifted'),
    'mushroom_body': ('closed', 'open', 'permuted'),
    'fast_slow': ('coupled', 'single', 'uncoupled'),
    'sleep_replay': ('blocked', 'interleaved', 'pause'),
    'grid_head_direction': ('attractor', 'diffuse', 'reset'),
}
PROFILES = ('smoke', 'pilot', 'screen')


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def verify_bundle():
    manifest = ROOT / 'SHA256SUMS'
    for row in manifest.read_text().splitlines():
        expected, name = row.split('  ', 1)
        if digest(ROOT / name) != expected:
            raise ValueError(f'Bundle integrity check failed for {name}')
    return digest(manifest)


def runtime():
    return dict(python=platform.python_version(), numpy=version('numpy'),
                torch=version('torch'), platform=platform.system(),
                machine=platform.machine())


def jobs(profile, blocks, bundle):
    seeds = SEEDS[:1] if profile == 'smoke' else SEEDS
    env = runtime()
    for block in blocks:
        for seed in seeds:
            for arm in BLOCKS[block]:
                spec = dict(block=block, arm=arm, seed=seed, profile=profile, bundle=bundle, runtime=env)
                key = sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]
                yield f'{block}-{arm}-{seed}-{key}', spec


def write_atomic(path, record):
    with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False, suffix='.tmp') as stream:
        scratch = Path(stream.name)
        try:
            json.dump(record, stream, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
            os.replace(scratch, path)
        finally:
            scratch.unlink(missing_ok=True)


def worker(job, spec, output):
    from additional import (actor_critic, cerebellar, grid_head_direction, mushroom_body,
                            retention, thalamocortical, working_memory_gate)
    from blocks import attention, central_complex, cortical

    runners = dict(attention=attention, central_complex=central_complex, cortical=cortical,
                   actor_critic=actor_critic, cerebellar=cerebellar, grid_head_direction=grid_head_direction,
                   mushroom_body=mushroom_body, thalamocortical=thalamocortical,
                   working_memory_gate=working_memory_gate)
    cpu, wall = time.process_time(), time.monotonic()
    path = Path(output) / f'{job}.json'
    try:
        args = spec['arm'], spec['seed'], spec['profile']
        if spec['block'] in ('fast_slow', 'sleep_replay'):
            result = retention(*args, mechanism=spec['block'])
        else:
            result = runners[spec['block']](*args)
        receipt = dict(spec=spec, status='complete', result=result)
    except Exception:
        receipt = dict(spec=spec, status='failed', error=traceback.format_exc())
    receipt.update(cpu_seconds=time.process_time()-cpu, wall_seconds=time.monotonic()-wall)
    write_atomic(path, receipt)
    return job, receipt['status']


def read_receipt(path, spec):
    try:
        record = json.loads(path.read_text())
    except FileNotFoundError:
        return None
    if record.get('spec') != spec:
        raise ValueError(f'Unexpected receipt identity {path}')
    return record


def launch(job, spec, output):
    command = ('import json,sys; from sweep import worker; '
               'print(*worker(sys.argv[1], json.loads(sys.argv[2]), sys.argv[3]))')
    subprocess.run([sys.executable, '-B', '-c', command, job, json.dumps(spec), str(output)],
                   cwd=ROOT, check=True)


def report(roster, output):
    counts = {'complete': 0, 'failed': 0, 'pending': 0}
    panels = {}
    for job, spec in roster:
        r = read_receipt(output / f'{job}.json', spec)
        if r is None:
            counts['pending'] += 1
            continue
        counts[r['status']] += 1
        if r['status'] != 'complete':
            print(job, r['status'], r['error'].splitlines()[-1])
            continue
        result = r['result']
        key = spec['block'], spec['seed']
        signature = result['held_sha256']
        if panels.setdefault(key, signature) != signature:
            raise ValueError(f'Paired arms used different test inputs for {key}')
        score = result.get('exact_pairs', result.get('within_20_deg'))
        print(f"{spec['block']:16} {spec['arm']:10} seed={spec['seed']} "
              f"{score:2}/{result['n']}  CPU {r['cpu_seconds']:.1f}s")
    print('Roster status', counts, flush=True)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('plan', 'run', 'report'))
    parser.add_argument('--profile', choices=PROFILES, default='smoke')
    parser.add_argument('--blocks', nargs='+', choices=tuple(BLOCKS))
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--out', type=Path, default=Path('receipts'))
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('workers must be positive')
    bundle = verify_bundle()
    selected = args.blocks or list(BLOCKS)
    print('Blocks:', ', '.join(selected), flush=True)
    if args.command == 'plan':
        print(f'{3 * len(selected) * (1 if args.profile == "smoke" else len(SEEDS))} jobs in {args.profile}')
        return
    if len(selected) != len(set(selected)):
        parser.error('List each block once')
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[key] = '1'
    os.environ.setdefault("KMP_USE_SHM", "0")
    roster = list(jobs(args.profile, selected, bundle))
    out = args.out.resolve() / args.profile
    if args.command == 'run':
        out.mkdir(parents=True, exist_ok=True)
        pending = [(job, spec) for job, spec in roster
                   if (read_receipt(out / f'{job}.json', spec) or {}).get('status') != 'complete']
        print(f'{len(pending)} to run, {len(roster) - len(pending)} reused', flush=True)
        if pending:
            with ThreadPoolExecutor(max_workers=min(args.workers, len(pending))) as pool:
                futures = [pool.submit(launch, job, spec, out) for job, spec in pending]
                for future in as_completed(futures):
                    future.result()
    counts = report(roster, out)
    if args.command == 'run' and (counts['failed'] or counts['pending']):
        sys.exit(1)


if __name__ == '__main__':
    main()
