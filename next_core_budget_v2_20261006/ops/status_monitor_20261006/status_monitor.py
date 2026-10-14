"""Display-only observer. Never starts, stops, or validates science trajectories."""
import datetime
import fcntl
import json
import os
from pathlib import Path
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
OPS = Path(__file__).resolve().parent
TERMINAL = {'COMPLETE', 'FAILED', 'STOP_NEEDS_REVIEW', 'PAUSED_SESSION_LIMIT'}


def read(path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {}


def alive(pid):
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def write(path, value):
    tmp = path.with_name(path.name + f'.pending-monitor-{os.getpid()}')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(tmp, path)


def main():
    lock = (OPS / 'observer.lock').open('a+')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    previous = None
    while True:
        current = read(ROOT / 'results/STATUS.json')
        if current.get('state') in TERMINAL or (ROOT / 'results/FAILURE.json').exists():
            final = read(ROOT / 'results/FAILURE.json') or current
            write(ROOT / 'results/LIVE_STATUS.json', final)
            with (ROOT / 'RUN_STATUS.log').open('a') as log:
                local = datetime.datetime.now(ZoneInfo('America/New_York')).isoformat(timespec='seconds')
                log.write(f"{local} | final {json.dumps(final, ensure_ascii=False)}\n")
            return
        launch = read(ROOT / 'results/LAUNCH.json')
        if not alive(launch.get('pid')):
            # Do not fabricate a scientific failure or overwrite final status.
            write(OPS / 'OBSERVER_STOP.json', {'state': 'SUPERVISOR_NOT_ALIVE',
                                               'updated_unix': time.time()})
            return
        screen = read(ROOT / 'science/screen/STATUS.json')
        confirm = read(ROOT / 'science/confirm/STATUS.json')
        stage = 'confirm' if confirm else 'screen'
        s = confirm or screen
        state = s.get('state', 'STARTING')
        if state == 'STAGE_COMPLETE':
            stage, state = 'selection_or_analysis', 'ANALYSIS'
            if (ROOT / 'results/FINAL_AUDIT.json').exists():
                stage, state = 'packaging', 'PACKAGING'
            elif (ROOT / 'results/FINAL_AUDIT_PROGRESS.json').exists():
                stage = 'final_audit'
        active = s.get('active_jobs', [])
        t = time.time()
        total_jobs = sum(x.get('completed_jobs', 0) for x in (screen, confirm))
        total_lives = sum(x.get('completed_lives', 0) for x in (screen, confirm))
        out = dict(s, state=state, stage=stage, updated_unix=t,
                   supervisor_pid=launch['pid'], supervisor_alive=True,
                   stage_updated_unix=s.get('updated_unix'),
                   current_stage_status=f'science/{"confirm" if confirm else "screen"}/STATUS.json',
                   screen_completed_jobs=screen.get('completed_jobs', 0),
                   screen_planned_jobs=screen.get('planned_jobs', 176),
                   confirm_completed_jobs=confirm.get('completed_jobs', 0),
                   confirm_planned_jobs=confirm.get('planned_jobs'),
                   total_committed_jobs=total_jobs, total_committed_lives=total_lives,
                   observer_pid=os.getpid(), display_only=True,
                   counts_are_verification_snapshot=False)
        # Final pipeline status is authoritative; check again before display update.
        if read(ROOT / 'results/STATUS.json').get('state') in TERMINAL:
            return
        # Write only the observer's own file. STATUS.json initially links to it;
        # the pipeline atomically replaces that link with its final status, so
        # a display update can never overwrite the final scientific outcome.
        write(ROOT / 'results/LIVE_STATUS.json', out)
        signature = (stage, state, total_jobs, total_lives, len(active))
        if signature != previous:
            local = datetime.datetime.fromtimestamp(t, ZoneInfo('America/New_York'))
            line = (f"{local.isoformat(timespec='seconds')} | {stage} {state} | "
                    f"screen {out['screen_completed_jobs']}/176 jobs | "
                    f"confirm {out['confirm_completed_jobs']}/{out['confirm_planned_jobs'] or 'pending'} jobs | "
                    f"total committed {total_jobs} jobs, {total_lives} lives | "
                    f"active {len(active)} | supervisor {launch['pid']} alive\n")
            with (ROOT / 'RUN_STATUS.log').open('a') as log:
                log.write(line)
            previous = signature
        if s.get('state') in TERMINAL:
            return
        time.sleep(2)


if __name__ == '__main__':
    main()
