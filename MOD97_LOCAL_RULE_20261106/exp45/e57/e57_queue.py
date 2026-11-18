"""E57's queue: the nine runs, at most three processes at once, one thread each, every run with --resume and its output appended to a log
of its own. A run whose final weights exist is finished and is skipped. Start and end times and each run's real exit code go to
logs/queue.log. Run from anywhere: python3 e57_queue.py
"""
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent            # exp45/e57
M = HERE.parents[1]                               # MOD97_LOCAL_RULE_20261106
LOG = HERE / "logs" / "queue.log"
JOBS = [  # (network, scale, rule, extra options, the run's name): the first three are started first, as the brief asks
    ("in16", "matrix", "S", ["--dose", "49152"], "e57_in16_matrix_S_u49152_73129"),
    ("in16", "row", "S", ["--dose", "49152"], "e57_in16_row_S_u49152_73129"),
    ("in16", "rowcells", "S", ["--dose", "49152"], "e57_in16_rowcells_S_u49152_73129"),
    ("in16", "rowcol", "S", ["--dose", "49152"], "e57_in16_rowcol_S_u49152_73129"),
    ("in16", "matrix", "S", ["--dose", "49152", "--code-decay", "0"], "e57_in16_matrix_S_cd0_u49152_73129"),
    ("in16", "none", "B", [], "e57_in16_none_B_73129"),
    ("F10", "row", "S", [], "e57_F10_row_S_73129"),
    ("F10", "rowcol", "S", [], "e57_F10_rowcol_S_73129"),
    ("G", "row", "S", [], "e57_G_row_S_73129"),
]


def log(msg):
    with LOG.open("a") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}\n")


def main():
    env = {k: v for k, v in os.environ.items() if not k.startswith(("E11_", "E16_", "EXP45_"))}
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", E57_BESIDE="3")
    (HERE / "logs").mkdir(exist_ok=True)
    todo = [j for j in JOBS if not (HERE / f"{j[4]}.pt").exists()]
    log(f"queue start: {len(todo)} runs to do")
    running = {}
    while todo or running:
        for name, (p, f) in list(running.items()):
            rc = p.poll()
            if rc is not None:
                f.close(); del running[name]; log(f"end {name} exit={rc}")
        while todo and len(running) < 3:
            net, scale, rule, extra, name = todo.pop(0)
            lf = open(HERE / "logs" / f"{name}.log", "a")
            p = subprocess.Popen([sys.executable, "exp57_scale.py", "run", net, scale, rule, "73129", *extra, "--resume"], cwd=M, env=env,
                                 stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            running[name] = (p, lf); log(f"start {name} pid {p.pid}")
        time.sleep(10)
    log("queue end")


if __name__ == "__main__":
    main()
