"""E60 (brief TASK_E60_THE_SWEEPS_LONG_LIFE_FROM_BACK_PROPAGATIONS_BIRTH_20261118.md, revision 3, on claude/vibrant-cerf-sje726 at 9bb87033):
the four long lives of the arm `sweep-after-naive` (the birth of rule B, a life by the sweep) as jobs for the main worker's driver
l2_driver.py (unchanged), and the saving of receipts, event files, logs and states to the helper's branch while lives run.

    python3 l2_san.py jobs               # writes exp45/e44/L2/sweep_after_naive/san_jobs.json
    python3 l2_san.py save [message]     # commits and pushes everything under sweep_after_naive/ except the event and list files of an
                                         #   unfinished life, and the latest state of every running life
    python3 l2_san.py saveloop [minutes] # `save` every so many minutes (default 25) until sweep_after_naive/STOP_SAVING exists

Each life (section 4 of the brief), from MOD97_LOCAL_RULE_20261106/smallworld:
  python3 sw_run.py life sweep-after-naive <w> 1024 --world lw --H 256 --U 128 --learner <abs. path of e41_B_h256_f50_<seed>.pt>
      --visit-updates 256 --wd 0.25 --lr 0.001 --list-every 8 --decay-distances none --save-states 64,128,256,384,512,640,768,896,1024
      --out <abs. path of exp45/e44/L2/sweep_after_naive/lives>
with the seed of world w: 73129 (worlds 1 and 4), 93683 (2), 116089 (3). Nothing here trains.
"""
import json
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MDIR = HERE.parent
ROOT = MDIR.parent
SAN = MDIR / "exp45" / "e44" / "L2" / "sweep_after_naive"
LIVES, CHK = SAN / "lives", SAN / "checks"
SEEDS = {1: 73129, 2: 93683, 3: 116089, 4: 73129}
STATES = "64,128,256,384,512,640,768,896,1024"
BRANCH = "claude/e44-l2-sweep-long-lives"
AGENT = "Acting-Agent: worker session_017dSEpTpaat6RxRKpJXsZWQ"


def birth(w, rule="B"):
    return MDIR / "exp45" / "e41" / f"e41_{rule}_h256_f50_{SEEDS[w]}.pt"


def stem(w):
    return f"lw_sweep-after-naive_w{w}_v1024"


def life_args(w, visits=1024, out=LIVES):
    return ["life", "sweep-after-naive", str(w), str(visits), "--world", "lw", "--H", "256", "--U", "128", "--learner", str(birth(w)),
            "--visit-updates", "256", "--wd", "0.25", "--lr", "0.001", "--list-every", "8", "--decay-distances", "none",
            "--save-states", STATES, "--out", str(out)]


def roster():
    # l2_driver.py appends --out itself; the args here carry it as well, and the runner takes the last --out given (the same path)
    return [dict(id=stem(w), out=str(LIVES), stem=stem(w), args=life_args(w)[:-2]) for w in (1, 2, 3, 4)]


def jobs():
    SAN.mkdir(parents=True, exist_ok=True)
    (SAN / "san_jobs.json").write_text(json.dumps(roster(), indent=1))
    print(len(roster()), "lives")


def states_of(s):
    out = {}
    for p in LIVES.glob(f"{s}_state_v*.pt"):
        m = re.fullmatch(re.escape(s) + r"_state_v(\d+)\.pt", p.name)
        if m:
            out[int(m.group(1))] = p
    return out


def git(*a, check=True):
    r = subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True)
    if check and r.returncode:
        raise RuntimeError(f"git {' '.join(a[:3])}: {r.stderr.strip()[-400:]}")
    return r


def save(message=None):
    assert git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip() == BRANCH, "not on the helper's branch"
    keep = []
    for j in roster():
        st = states_of(j["stem"])
        if st and not (LIVES / f"{j['stem']}.json").exists():
            keep.append(st[max(st)])
    rel = str(SAN.relative_to(ROOT))
    tracked = [x for x in git("ls-files", rel).stdout.splitlines() if x.endswith(".pt") and "/lives/" in x]
    gone = [x for x in tracked if not any(str(p.relative_to(ROOT)) == x for p in keep)]
    if gone:
        git("rm", "-q", "--cached", "--ignore-unmatch", *gone)
    unfinished = [p for p in LIVES.glob("*.npz") if not (LIVES / (p.name.replace("_list.npz", ".npz")[:-4] + ".json")).exists()] if LIVES.exists() else []
    git("add", "-A", rel)
    if unfinished:
        git("reset", "-q", "--", *[str(p.relative_to(ROOT)) for p in unfinished])
    if keep:
        git("add", "-f", *[str(p.relative_to(ROOT)) for p in keep])
    if git("diff", "--cached", "--quiet", check=False).returncode == 0:
        print(time.strftime("%H:%M:%S"), "nothing to save", flush=True); return
    git("commit", "-q", "-m", (message or "E60: receipts, event files, logs and the latest states of the running lives (saved while the lives run)") + "\n\n" + AGENT)
    for k in range(5):
        r = git("push", "-q", "origin", BRANCH, check=False)
        if r.returncode == 0:
            print(time.strftime("%H:%M:%S"), "saved", git("rev-parse", "--short", "HEAD").stdout.strip(), flush=True); return
        time.sleep(2 ** (k + 1))
    print(time.strftime("%H:%M:%S"), "push failed:", r.stderr.strip()[-300:], flush=True)


def saveloop(minutes=25):
    while not (SAN / "STOP_SAVING").exists():
        t0 = time.time()
        while time.time() - t0 < 60 * minutes and not (SAN / "STOP_SAVING").exists():
            time.sleep(20)
        try:
            save()
        except Exception as e:      # noqa: BLE001
            print(time.strftime("%H:%M:%S"), "save failed:", e, flush=True)


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "saveloop":
        saveloop(float(sys.argv[2]) if len(sys.argv) > 2 else 25)
    elif c == "save":
        save(sys.argv[2] if len(sys.argv) > 2 else None)
    else:
        {"jobs": jobs}[c]()
