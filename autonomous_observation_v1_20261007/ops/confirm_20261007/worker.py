"""Run exactly one authorized world once; native runner remains unchanged."""
import argparse
import builtins
import os
import threading
import time
import traceback
from ops_common import OPS, RESULTS, EVIDENCE, WORLDS, atomic, authorized_worlds, read, verify_sources


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--world', type=int, required=True)
    world = parser.parse_args().world
    if world not in WORLDS:
        raise ValueError('unauthorized world')
    path = RESULTS / 'receipts' / f'WORLD_{world}.json.gz'
    if path.exists():
        raise ValueError('refusing to repeat a committed world')
    source = verify_sources()
    active = RESULTS / 'active' / f'{world}.json'
    state = dict(world=world, pid=os.getpid(), stage='birth', completed_inflight_training_bytes=0,
                 complete_lives_committed=False, source_lock_sha256=source)
    mutex, stop = threading.Lock(), threading.Event()

    def heartbeat():
        with mutex:
            atomic(active, {**state, 'heartbeat_unix': time.time()})

    def beats():
        while not stop.wait(5):
            heartbeat()

    heartbeat()
    thread = threading.Thread(target=beats, daemon=True)
    thread.start()
    try:
        import runner
        from science_audit import audit_full
        original_consume = runner.consume
        batch = 0

        def progress_print(message, **kwargs):
            nonlocal batch
            stage = message.split()[-1]
            with mutex:
                state['stage'] = stage
            batch = 0
            heartbeat()
            builtins.print(message.replace('DEV ', 'CONFIRM ', 1), **kwargs)

        def consume(model, raw):
            nonlocal batch
            rows = original_consume(model, raw)
            with mutex:
                if state['stage'] in ('old', 'new', 'revised') and batch < 3:
                    state['completed_inflight_training_bytes'] += len(raw)
                batch += 1
            heartbeat()
            return rows

        runner.print = progress_print
        runner.consume = consume  # Calls the frozen implementation once, then reports completion.
        with authorized_worlds():
            receipt = runner.run_world(world)
        receipt['evidence'] = EVIDENCE
        receipt['confirmation_source_lock_sha256'] = source
        with mutex:
            state['stage'] = 'independent_audit'
        heartbeat()
        plan = read(OPS / 'INPUT_MANIFEST.json')['worlds'][str(world)]
        accepted = audit_full(receipt, plan, source)
        verify_sources()
        receipt['independent_audit'] = accepted
        atomic(path, receipt, exclusive=True)  # Only a complete, accepted three-life job commits.
        active.unlink(missing_ok=True)
        builtins.print(f'COMMITTED world={world} lives=3 training_bytes=3840', flush=True)
    except BaseException as exc:
        atomic(RESULTS / 'failures' / f'{world}.json', dict(world=world, error=repr(exc),
               traceback=traceback.format_exc(), progress=state, time_unix=time.time()), exclusive=True)
        raise
    finally:
        stop.set()
        thread.join(timeout=6)
        if path.exists():
            active.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
