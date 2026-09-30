"""Run every Package A V3-CLAUDE technical arm on world 190000 (4 workers; R1_rand after R1, Z2_rand after Z2).
No science."""
from __future__ import annotations

import json
import sys
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import runner  # noqa: E402


KIND = "technical_final"


def job(arm):
    try:
        return {"ok": True, **runner.run(arm, kind=KIND)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "arm": arm, "error": repr(exc), "traceback": traceback.format_exc()[-3000:]}


def main():
    deps = {"R1_rand": "R1", "Z2_rand": "Z2"}
    have = lambda a: runner.receipt_path(KIND, a, 190000).exists()  # noqa: E731
    queue = [a for a in ("R1", "Z2") + tuple(x for x in runner.ARMS if x not in ("R1", "Z2")) if not have(a)]
    failed = False
    with ProcessPoolExecutor(4) as ex:
        running = {}
        while queue or running:
            for a in list(queue):
                if len(running) >= 4 or failed:
                    break
                if a in deps and not have(deps[a]):
                    continue
                queue.remove(a)
                running[ex.submit(job, a)] = a
            if not running:
                break
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for f in done:
                running.pop(f)
                res = f.result()
                failed = failed or not res["ok"]
                print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
