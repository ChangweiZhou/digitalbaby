"""Run every Package A V3-CLAUDE technical arm on world 190000 (4 workers; R1_rand after R1). No science."""
from __future__ import annotations

import json
import sys
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import runner  # noqa: E402


def job(arm):
    try:
        return {"ok": True, **runner.run(arm)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "arm": arm, "error": repr(exc), "traceback": traceback.format_exc()[-3000:]}


def main():
    todo = [a for a in runner.ARMS if a != "R1_rand" and not runner.receipt_path("technical", a, 190000).exists()]
    order = ["R1"] + [a for a in todo if a != "R1"] if "R1" in todo else todo
    pending_rand = not runner.receipt_path("technical", "R1_rand", 190000).exists()
    with ProcessPoolExecutor(4) as ex:
        running = {ex.submit(job, a): a for a in order[:4]}
        queue = order[4:]
        while running:
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for f in done:
                arm = running.pop(f)
                res = f.result()
                print(json.dumps(res), flush=True)
                if pending_rand and runner.receipt_path("technical", "R1", 190000).exists():
                    queue.insert(0, "R1_rand")
                    pending_rand = False
                if queue:
                    a = queue.pop(0)
                    running[ex.submit(job, a)] = a
    if pending_rand and runner.receipt_path("technical", "R1", 190000).exists():
        print(json.dumps(job("R1_rand")), flush=True)


if __name__ == "__main__":
    main()
