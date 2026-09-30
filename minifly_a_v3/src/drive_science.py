"""Package A V3-CLAUDE science launcher: 13 arms x 64 worlds, write-once receipts. NOT to be run before the
RESOURCE_BUDGET.json approval field is set and SOURCE_LOCK.json verifies.

* Verifies SOURCE_LOCK.json (runner re-verifies in every worker) and refuses to start unless the budget is approved.
* Skips committed world-arms; never reruns or replaces one (resume = only uncommitted world-arms, same lock).
* R1_rand/Z2_rand for a world start only after R1/Z2 for that world is committed.
* The first failed world-arm, or any budget breach, stops scheduling; running jobs finish; RUN_STATUS.json records
  the actual error and committed count. No automatic retry of a scientific failure.
* Every BATCH new receipts are git-committed and pushed (append-only; never force).
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
ARMS = ("R1", "R0", "R0_signed", "R3", "R3_randtarget", "Z2", "Z0_resource", "Z2_rand", "R1_rand",
        "P0", "P1", "P2", "P4")
DEPS = {"R1_rand": "R1", "Z2_rand": "Z2"}
WORLDS = tuple(range(190001, 190065))
BATCH = 16
BRANCH = "claude/charming-clarke-u9g850"
TRAILER = ("\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n"
           "Claude-Session: https://claude.ai/code/session_012L5f7WRCtxBkKWVj4mCZJ7")


def path(arm, world):
    return ROOT / "results" / "science" / arm / f"{world}.json.gz"


def committed():
    return {(a, w) for a in ARMS for w in WORLDS if path(a, w).exists()}


def job(arm, world):
    import runner
    try:
        return {"ok": True, **runner.run(arm, world, kind="science")}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "arm": arm, "world": world, "error": repr(exc),
                "traceback": traceback.format_exc()[-4000:]}


def git_push(message: str) -> str:
    subprocess.run(["git", "add", "minifly_a_v3/results/science"], cwd=REPO, capture_output=True)
    subprocess.run(["git", "commit", "-q", "-m", message + TRAILER], cwd=REPO, capture_output=True)
    p = None
    for delay in (0, 2, 4, 8, 16):
        time.sleep(delay)
        p = subprocess.run(["git", "push", "-q", "-u", "origin", BRANCH], cwd=REPO, capture_output=True, text=True)
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
    budget = json.loads((ROOT / "RESOURCE_BUDGET.json").read_text())
    if budget.get("approval", {}).get("approved") is not True:
        raise SystemExit("RESOURCE_BUDGET.json is not approved; science may not start")
    hard = budget["hard_budget"]
    workers = int(hard["workers"])
    done = committed()
    queue = [(a, w) for w in WORLDS for a in ARMS if (a, w) not in done]
    failure, since_push, active_s, started, running = None, 0, 0.0, time.time(), {}
    with ProcessPoolExecutor(workers) as ex:
        while queue or running:
            if failure is None:
                for item in list(queue):
                    if len(running) >= workers:
                        break
                    arm, world = item
                    if arm in DEPS and (DEPS[arm], world) not in done:
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
                if res["ok"]:
                    done.add(item)
                    since_push += 1
                    print(json.dumps(res), flush=True)
                    if (res["life_s"] > hard["per_world_arm_life_s_max"] or
                            res["peak_rss_bytes"] > hard["peak_rss_bytes_per_worker"]) and failure is None:
                        failure = {**res, "ok": False, "error": "RESOURCE_BUDGET_EXCEEDED (receipt kept)"}
                elif failure is None:
                    failure = res
                    print(json.dumps({k: res[k] for k in ("arm", "world", "error")}), flush=True)
            if failure is None and active_s > hard["active_wall_hours"] * 3600:
                failure = {"ok": False, "error": "RESOURCE_BUDGET_EXCEEDED: active wall hours"}
            write_status({"lock_digest": lk["lock_digest"], "committed": len(committed()),
                          "roster": len(ARMS) * len(WORLDS), "failure": failure, "active_driver_s": active_s,
                          "elapsed_driver_s": time.time() - started})
            if since_push >= BATCH:
                print(git_push(f"minifly_a_v3: science receipts ({len(committed())}/832 committed)"), flush=True)
                since_push = 0
    n = len(committed())
    write_status({"lock_digest": lk["lock_digest"], "committed": n, "roster": len(ARMS) * len(WORLDS),
                  "failure": failure, "active_driver_s": active_s, "elapsed_driver_s": time.time() - started,
                  "state": "stopped_on_failure" if failure else ("complete" if n == 832 else "incomplete")})
    print(git_push(f"minifly_a_v3: science receipts ({n}/832 committed)"), flush=True)


if __name__ == "__main__":
    main()
