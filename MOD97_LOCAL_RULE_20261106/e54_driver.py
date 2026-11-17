"""E54: the launcher of the twenty runs (brief revision 3). It keeps at most three processes running, each with one thread, in the brief's
order (parts 1, 2, 3, 4; within parts 1 and 3 the two runs of G first), applies the checks that are read from the runs, and stops what
the brief says to stop when one of them fails. It trains nothing itself: every run is `exp54_fraction.py run ...` (parts 1 to 3) or
`exp41_scale.py run ...` (part 4), started as its own process without any time limit.

    python3 e54_driver.py            # started again after a stop of the host, it resumes: complete runs are skipped, runs of parts 1 to 3
                                     # continue from their saved state, a run of part 4 is started again once from its beginning

Gates. No run of part 2, 3 or 4 is launched before check 3 has passed at f = 0.25 under both rules (every row to update 12,288 and the
hash). Check 3 is read at row 256 (printed line) and then at every saved state (every 1,024 updates); a difference stops every run. Check 2
is read at row 4,096 of the first run of each part; a difference stops that part. In part 4 check 3 is read at row 256 and at every
writing of the receipt; a difference stops part 4 only.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "exp45" / "e54"
Q, LOGS = OUT / "qual", OUT / "logs"
SEED = 73129
STATE = OUT / "driver_state.json"
PY = sys.executable
ENV = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", E54_BESIDE="2", E41_BESIDE="2",
           PATH=str(Path(PY).parent) + os.pathsep + os.environ["PATH"])

JOBS = []
for part, fp in ((1, 25), (2, 15), (3, 10)):
    for cfg in ("G", "F10", "P8"):
        for rule in "SB":
            JOBS.append(dict(part=part, name=f"e54_{cfg}_{rule}_f{fp}", cfg=cfg, rule=rule, fp=fp, receipt=OUT / f"e54_{cfg}_{rule}_f{fp}_{SEED}.json",
                             cmd=["python3", "exp54_fraction.py", "run", cfg, rule, str(SEED), "--fraction", str(fp)] + (["--theta", "0.5"] if cfg == "P8" else [])))
for rule in "SB":
    JOBS.append(dict(part=4, name=f"part4_{rule}", cfg="H", rule=rule, fp=10, receipt=OUT / f"e41_{rule}_h256_f10_dry122880_{SEED}.json",
                     cmd=["python3", "exp41_scale.py", "run", rule, "256", "10", str(SEED), "--dose", "122880", "--out", "exp45/e54"]))


def say(*a):
    line = time.strftime("%FT%TZ ", time.gmtime()) + " ".join(str(x) for x in a)
    with open(LOGS / "driver.log", "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def complete(job):
    return job["receipt"].exists() and json.loads(job["receipt"].read_text()).get("complete") is True


def check(*args):
    r = subprocess.run([PY, "e54_checks.py", *args], cwd=HERE, env=ENV, capture_output=True, text=True)
    say("check", " ".join(args), "->", (r.stdout.strip().splitlines() or [r.stderr.strip()[-300:]])[-1])
    return r.returncode == 0


def passed(name):
    p = Q / f"{name}.json"
    return p.exists() and json.loads(p.read_text())["pass"] is True


def main():
    LOGS.mkdir(parents=True, exist_ok=True)
    st = json.loads(STATE.read_text()) if STATE.exists() else dict(stopped_parts=[], stop_all=None, failed=[], part4_interrupted={}, seen={})
    save = lambda: STATE.write_text(json.dumps(st, indent=1))
    running = {}
    say("driver start", "state", json.dumps(st))
    while True:
        # ---- processes that ended
        for name, (job, p) in list(running.items()):
            rc = p.poll()
            if rc is not None:
                del running[name]
                say("ended", name, "exit", rc)
                if rc != 0 and name not in st["failed"]:
                    st["failed"].append(name); save()
        # ---- checks read from the runs
        for job in JOBS:
            name, rule, fp = job["name"], job["rule"], job["fp"]
            log = LOGS / f"{name}.log"
            if job["cfg"] == "G" and fp in (25, 10) and not passed(f"check3_G_{rule}_f{fp}") and log.exists():
                if not (Q / f"check3_G_{rule}_f{fp}_row256.json").exists() and f" {SEED} 256:" in log.read_text():
                    if not check("row256", "G", rule, str(fp)):
                        st["stop_all"] = f"check 3 failed at row 256 of {name}"
                ck = OUT / "ckpt" / f"{job['receipt'].stem}.ckpt.pt"
                mark = max([p.stat().st_mtime for p in (ck, job["receipt"]) if p.exists()], default=None)
                if mark is not None and st["seen"].get(name) != mark:
                    st["seen"][name] = mark
                    if not check("check3", rule, str(fp)):
                        st["stop_all"] = f"check 3 failed in {name}"
                    elif passed(f"check3_G_{rule}_f{fp}"):
                        (Q / f"check3_G_{rule}_f{fp}_partial.json").unlink(missing_ok=True)
                    save()
            if job["cfg"] == "G" and rule == "S" and not (Q / f"check2_f{fp}.json").exists() and log.exists() and f" {SEED} 4096:" in log.read_text():
                time.sleep(5)                                    # the receipt and the saved state follow the printed line at once
                if not check("check2", str(fp)) and job["part"] not in st["stopped_parts"]:
                    st["stopped_parts"].append(job["part"]); save()
            if job["part"] == 4 and name not in st["failed"] and log.exists():
                if not (Q / f"check3_part4_{rule}_row256.json").exists() and f" {SEED} 256:" in log.read_text():
                    if not check("row256", "P4", rule) and 4 not in st["stopped_parts"]:
                        st["stopped_parts"].append(4); save()
                if job["receipt"].exists() and not passed(f"check3_part4_{rule}") and st["seen"].get(name) != job["receipt"].stat().st_mtime:
                    st["seen"][name] = job["receipt"].stat().st_mtime
                    if not check("check3p4", rule) and 4 not in st["stopped_parts"]:
                        st["stopped_parts"].append(4)
                    elif passed(f"check3_part4_{rule}"):
                        (Q / f"check3_part4_{rule}_partial.json").unlink(missing_ok=True)
                    save()
        # ---- stops
        if st["stop_all"]:
            save()
            for name, (job, p) in running.items():
                p.terminate()
            say("STOP EVERY RUN:", st["stop_all"]); return 1
        for name, (job, p) in list(running.items()):
            if job["part"] in st["stopped_parts"]:
                p.terminate(); del running[name]; say("stopped", name, "(its part is stopped)")
        # ---- launches
        gate = passed("check3_G_S_f25") and passed("check3_G_B_f25")
        for job in JOBS:
            if len(running) >= 3:
                break
            name = job["name"]
            if name in running or name in st["failed"] or job["part"] in st["stopped_parts"] or complete(job):
                continue
            if job["part"] > 1 and not gate:
                continue
            if job["part"] == 4 and job["receipt"].exists():     # an interrupted run of part 4 cannot resume
                n = st["part4_interrupted"].get(name, 0)
                (OUT / "interrupted").mkdir(exist_ok=True)
                job["receipt"].rename(OUT / "interrupted" / f"{job['receipt'].stem}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json")
                st["part4_interrupted"][name] = n + 1; save()
                if n + 1 >= 2:
                    st["stopped_parts"].append(4); save(); say("part 4 interrupted a second time:", name, "- part 4 is stopped"); continue
                say("part 4 interrupted once:", name, "- started again from its beginning")
                (Q / f"check3_part4_{job['rule']}_row256.json").unlink(missing_ok=True)
            elif job["part"] == 4 and (LOGS / f"{name}.log").exists() and (LOGS / f"{name}.log").stat().st_size > 0:
                n = st["part4_interrupted"].get(name, 0)             # interrupted before its first receipt
                st["part4_interrupted"][name] = n + 1; save()
                if n + 1 >= 2:
                    st["stopped_parts"].append(4); save(); say("part 4 interrupted a second time:", name, "- part 4 is stopped"); continue
                say("part 4 interrupted once (before its first receipt):", name, "- started again from its beginning")
                (Q / f"check3_part4_{job['rule']}_row256.json").unlink(missing_ok=True)
            if job["part"] == 4:
                wp = job["receipt"].with_suffix(".pt")
                for x in (wp, Path(str(wp) + ".tmp")):
                    x.unlink(missing_ok=True)
                lg = LOGS / f"{name}.log"
                if lg.exists() and lg.stat().st_size > 0:
                    lg.rename(LOGS / f"{name}_interrupted_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.log")
            f = open(LOGS / f"{name}.log", "a")
            running[name] = (job, subprocess.Popen(job["cmd"], cwd=HERE, env=ENV, stdout=f, stderr=subprocess.STDOUT, start_new_session=True))
            say("launched", name, " ".join(job["cmd"]))
        if not running:
            left = [j["name"] for j in JOBS if not complete(j)]
            say("driver done; not complete:", left); return 0
        time.sleep(30)


if __name__ == "__main__":
    sys.exit(main())
