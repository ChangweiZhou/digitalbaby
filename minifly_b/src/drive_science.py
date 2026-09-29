"""Science roster driver: 13 arms x 64 worlds, 4 workers, write-once receipts.

* Verifies SOURCE_LOCK.json before starting and in every worker (runner does).
* Skips world-arms whose receipt is already committed; never reruns or replaces one.
* Srand_x for a world starts only after S{x} for that world is committed.
* The first failed world-arm stops scheduling; running jobs finish; the actual
  error and committed count are written to results/science/RUN_STATUS.json.
* Every BATCH new receipts are git-committed and pushed (append-only).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402

ROOT = paths.ROOT
REPO = ROOT.parent
STATUS = ROOT / "results" / "science" / "RUN_STATUS.json"
ARMS = ("T1", "T0_1", "T3", "T0_3", "S0", "S1", "S2", "S3", "S4",
        "Srand_1", "Srand_2", "Srand_3", "Srand_4")
WORLDS = tuple(range(190001, 190065))
BATCH = 16
BRANCH = "claude/charming-clarke-u9g850"
GIT_ENV = {**os.environ, "GIT_AUTHOR_NAME": os.environ.get("GIT_AUTHOR_NAME", "Claude"),
           "GIT_COMMITTER_NAME": os.environ.get("GIT_COMMITTER_NAME", "Claude")}


def path(arm, world):
    return ROOT / "results" / "science" / arm / f"{world}.json.gz"


def job(arm: str, world: int) -> dict:
    import runner
    try:
        return {"ok": True, **runner.run(arm, world, technical=False)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "arm": arm, "world": world, "error": repr(exc),
                "traceback": traceback.format_exc()[-4000:]}


def committed():
    return {(a, w) for a in ARMS for w in WORLDS if path(a, w).exists()}


def git_push(message: str) -> str:
    cmds = [["git", "add", "minifly_b/results/science"],
            ["git", "commit", "-q", "-m", message + "\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n"
             "Claude-Session: https://claude.ai/code/session_01Mt6g5a5H7My1LFSbErXdCh"]]
    for c in cmds:
        subprocess.run(c, cwd=REPO, env=GIT_ENV, check=False, capture_output=True)
    for delay in (0, 2, 4, 8, 16):
        time.sleep(delay)
        p = subprocess.run(["git", "push", "-q", "-u", "origin", BRANCH], cwd=REPO, env=GIT_ENV,
                           capture_output=True, text=True)
        if p.returncode == 0:
            return "pushed"
    return "push-failed: " + p.stderr[-300:]


def write_status(doc):
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATUS.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, STATUS)


def main() -> None:
    import lock
    lk = lock.verify_lock()
    budget = json.loads((ROOT / "RESOURCE_BUDGET.json").read_text())["hard_budget"]
    active_s = 0.0          # driver wall time actually spent computing (VM-running time)
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    done = committed()
    queue = [(a, w) for w in WORLDS for a in ARMS if (a, w) not in done]
    failure = None
    since_push = 0
    started = time.time()
    running = {}
    with ProcessPoolExecutor(workers) as ex:
        while (queue or running) :
            if failure is None:
                for item in list(queue):
                    if len(running) >= workers:
                        break
                    arm, world = item
                    if arm.startswith("Srand") and (f"S{arm[-1]}", world) not in done:
                        continue
                    queue.remove(item)
                    running[ex.submit(job, arm, world)] = item
            if not running:
                break
            t_wait = time.monotonic()
            finished, _ = wait(running, return_when=FIRST_COMPLETED)
            active_s += time.monotonic() - t_wait
            for fut in finished:
                item = running.pop(fut)
                res = fut.result()
                if res["ok"] and (res["life_s"] > budget["per_world_arm_learner_s_max"] or
                                  res["peak_rss_bytes"] > budget["peak_rss_bytes_per_worker"]):
                    done.add(item)
                    if failure is None:
                        failure = {**res, "ok": False, "error": "RESOURCE_BUDGET_EXCEEDED (receipt kept)"}
                    print(json.dumps(failure), flush=True)
                elif res["ok"]:
                    done.add(item)
                    since_push += 1
                    print(json.dumps(res), flush=True)
                elif failure is None:
                    failure = res
                    print(json.dumps({k: res[k] for k in ("arm", "world", "error")}), flush=True)
            write_status({"lock_digest": lk["lock_digest"], "committed": len(committed()),
                          "roster": len(ARMS) * len(WORLDS), "failure": failure,
                          "elapsed_driver_s": time.time() - started})
            if failure is None and active_s > budget["wall_hours_active_compute"] * 3600:
                failure = {"ok": False, "error": "RESOURCE_BUDGET_EXCEEDED: active wall hours"}
            if since_push >= BATCH:
                print(git_push(f"minifly_b: science receipts ({len(committed())}/832 committed)"), flush=True)
                since_push = 0
    n = len(committed())
    write_status({"lock_digest": lk["lock_digest"], "committed": n, "roster": len(ARMS) * len(WORLDS),
                  "failure": failure, "elapsed_driver_s": time.time() - started,
                  "state": "stopped_on_failure" if failure else ("complete" if n == 832 else "incomplete")})
    print(git_push(f"minifly_b: science receipts ({n}/832 committed)"), flush=True)


if __name__ == "__main__":
    main()
