"""E53: keeps at most three runs going, one thread each, from exp45/e53/queues/all.txt (one "arm rule" a line, taken in order).
Every run is started with --resume and its output appended to exp45/e53/logs/<receipt's stem>.log. A run whose receipt is complete is
skipped. A run that has a starting file but no saved state (interrupted before update 1,024) has its starting file moved into
exp45/e53/interrupted/ before it is started again (the brief's rule).
    python3 exp45/e53/scheduler.py     (from MOD97_LOCAL_RULE_20261106; ends when exp45/e53/queues/STOP exists and nothing runs)"""
import json, os, subprocess, time
from pathlib import Path

OUT = Path("exp45/e53"); Q = OUT / "queues" / "all.txt"; PY = "/home/user/e58venv/bin/python"
running = {}


def stem(arm, rule):
    return f"e53_32_{arm}_{rule}_73129"


def others():
    o = subprocess.run(["pgrep", "-af", "exp53_local_start.py run"], capture_output=True, text=True).stdout.splitlines()
    return [l for l in o if "pgrep" not in l]


while True:
    for k in [k for k, p in running.items() if p.poll() is not None]:
        del running[k]
    lines = [l.split() for l in Q.read_text().splitlines() if l.strip()] if Q.exists() else []
    for arm, rule in lines:
        if len(others()) >= 3:
            break
        s = stem(arm, rule)
        if s in running or any(f"run {arm} {rule} 73129" in l for l in others()):
            continue
        rec = OUT / f"{s}.json"
        if rec.exists() and json.loads(rec.read_text()).get("complete"):
            continue
        start, ckpt = OUT / f"{s}_start.pt", OUT / f"{s}_ckpt.pt"
        if start.exists() and not ckpt.exists():
            d = OUT / "interrupted"; d.mkdir(exist_ok=True)
            tag = time.strftime("%Y%m%dT%H%M%S")
            for p in [start, rec]:
                if p.exists():
                    p.rename(d / f"{p.name}.{tag}")
        beside = ", ".join(sorted(l.split("run ", 1)[1] for l in others())) or "none"
        env = {k: v for k, v in os.environ.items() if not k.startswith(("E11_", "E16_"))}
        env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", E53_BESIDE=f"at start: {beside}")
        log = open(OUT / "logs" / f"{s}.log", "a")
        running[s] = subprocess.Popen([PY, "exp53_local_start.py", "run", arm, rule, "73129", "--resume"], stdout=log, stderr=subprocess.STDOUT, env=env)
        time.sleep(2)
    if (OUT / "queues" / "STOP").exists() and not running:
        break
    time.sleep(10)
