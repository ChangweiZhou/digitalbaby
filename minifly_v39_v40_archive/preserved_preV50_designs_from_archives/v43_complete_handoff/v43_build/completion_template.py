#!/usr/bin/env python3
"""V43 complete runner: frozen scientific core plus corrected diagnostic decoding.

Run or resume: python3 elm_v43_fast_slow_authority_complete.py --out-dir v43_results
Add --install-deps if required. All original runner options are supported.
"""
from __future__ import annotations
import argparse
import base64
import contextlib
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import types
import zlib

PAYLOADS = __COMPLETION_PAYLOADS__
DRIVER_PATH = Path(__file__).resolve()
DRIVER_BYTES = DRIVER_PATH.read_bytes()
DRIVER_SHA256 = hashlib.sha256(DRIVER_BYTES).hexdigest()
CORE_NAME = 'elm_v43_fast_slow_authority.py'


def unpack(name):
    item = PAYLOADS[name]
    raw = zlib.decompress(base64.b85decode(item['data']))
    if hashlib.sha256(raw).hexdigest() != item['sha256']:
        raise ValueError('Embedded payload checksum mismatch: ' + name)
    return raw


def guard_driver():
    if hashlib.sha256(DRIVER_PATH.read_bytes()).hexdigest() != DRIVER_SHA256:
        raise RuntimeError('Completion driver changed during execution')


def bootstrap_core(root):
    """Serialize materialization with both the original and completion runners."""
    lock = root / 'locks' / 'runner.lock'
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open('a+') as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another writer holds ' + str(lock))
        try:
            guard_driver()
            path = root / CORE_NAME
            raw = unpack('core')
            if path.exists():
                if path.read_bytes() != raw:
                    raise ValueError('Existing frozen scientific core differs: ' + str(path))
            else:
                temporary = path.with_name(path.name + '.tmp-' + str(os.getpid()))
                with temporary.open('wb') as stream:
                    stream.write(raw)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, path)
                fd = os.open(root, os.O_RDONLY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
            return path
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def prepare_core(root, install=False):
    path = bootstrap_core(root)
    spec = importlib.util.spec_from_file_location('v43_frozen_scientific_runner', path)
    core = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = core
    spec.loader.exec_module(core)
    if core.BOOT_SOURCE_SHA != PAYLOADS['core']['sha256']:
        raise ValueError('Scientific core identity changed while loading')
    core.dependencies(install)
    core.load_modules()
    diagnostic = types.ModuleType('v43_diagnostics')
    diagnostic.__file__ = str(root / 'diagnostics_corrected.py')
    sys.modules[diagnostic.__name__] = diagnostic
    diagnostic.np = core.np
    from scipy.stats import t
    diagnostic.student_t = t
    exec(compile(unpack('diagnostics'), '<embedded:diagnostics-corrected>', 'exec'), diagnostic.__dict__)
    # Keep the corrected module's function globals untouched so its cache
    # implementation hash remains identical to the independently tested source.
    def guarded_diagnostic_run(root, authority_events=None, should_stop=None):
        def guarded_stop():
            guard_driver()
            return bool(should_stop is not None and should_stop())
        guarded_stop()
        result = diagnostic.run(root, authority_events, should_stop=guarded_stop)
        guard_driver()
        return result
    core.MODULES['diagnostics'] = types.SimpleNamespace(
        self_test=diagnostic.self_test, run=guarded_diagnostic_run)

    original_source_sha = core.source_sha
    def guarded_source_sha():
        guard_driver()
        return original_source_sha()
    core.source_sha = guarded_source_sha

    provenance = {
        'format': 'V43-analysis-implementation-1',
        'scientific_core_sha256': PAYLOADS['core']['sha256'],
        'completion_driver_sha256': DRIVER_SHA256,
        'original_diagnostics_sha256': core.PAYLOADS['diagnostics']['sha256'],
        'corrected_diagnostics_sha256': PAYLOADS['diagnostics']['sha256'],
        'diagnostic_implementation_sha256': diagnostic._implementation_hash(),
        'scientific_plan_and_checkpoint_identity': 'unchanged frozen core',
        'change': 'Decode never-revised revision_age=-1 as missing; reject contradictory markers.',
        'physical_learning_changes': False,
        'policy_scoring_changes': False,
        'model_or_hyperparameter_changes': False,
        'gate_changes': False,
        'status_scope': 'Implementation provenance only; completion is recorded in PREDICTABILITY_REPORT.json.',
        'pause_exit_status': '75 when an interruption is requested, including during diagnostics',
    }
    original_lock = core.writer_lock
    @contextlib.contextmanager
    def guarded_lock(path, wait=False):
        with original_lock(path, wait=wait):
            if Path(path).resolve() == (root / 'locks' / 'runner.lock').resolve():
                guard_driver()
                if (root / 'PLAN.json').exists():
                    plan = json.loads((root / 'PLAN.json').read_bytes())
                    if plan != core.make_plan(plan.get('mode') == 'engineering_smoke'):
                        raise ValueError('Existing plan/source mismatch')
                prior = root / 'ANALYSIS_IMPLEMENTATION.json'
                if prior.exists() and json.loads(prior.read_bytes()) != provenance:
                    raise ValueError('Existing analysis implementation identity mismatch')
                core.write_json(prior, provenance)
                core.atomic(root / 'elm_v43_fast_slow_authority_complete.py', DRIVER_BYTES)
                for name, filename in (
                    ('diagnostics', 'diagnostics_corrected.py'),
                    ('fix_md', 'DIAGNOSTIC_ENCODING_FIX.md'),
                    ('fix_json', 'DIAGNOSTIC_ENCODING_FIX.json'),
                    ('erratum', 'PROTOCOL_ERRATUM.md'),
                ):
                    core.atomic(root / filename, unpack(name))
            yield
    core.writer_lock = guarded_lock
    return core


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--out-dir', '--out', default='v43_results')
    parser.add_argument('--install-deps', action='store_true')
    args, _ = parser.parse_known_args()
    root = Path(args.out_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    core = prepare_core(root, args.install_deps)
    status = core.main()
    return 75 if core.STOP else status


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (KeyboardInterrupt, InterruptedError):
        raise SystemExit(75)
