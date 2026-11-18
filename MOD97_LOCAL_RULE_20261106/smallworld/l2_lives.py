"""E44, job L2 (roster note OPUS_TASK_E44_L2_ROSTER_FIRST_LONG_LIVES_20261117.md, revision 2): the roster of release 1 as jobs for
l2_driver.py, the weights' sizes, and the saving of receipts, event files and states to the branch while lives run.

    python3 l2_lives.py jobs               # writes exp45/e44/L2/roster_jobs.json: the learning probe's 24 branches (each starts when its state
                                           #   exists), the trial, the six long lives, the eight short ones, the sixteen lives without decay
    python3 l2_lives.py norms              # the norm of every weight tensor in every state on disk -> exp45/e44/L2/norms/<stem>_norms.json
    python3 l2_lives.py save               # norms; removes states no longer needed; commits and pushes receipts, event and list files of
                                           #   finished lives, logs, and the latest state of every running life
    python3 l2_lives.py saveloop [minutes] # `save` every so many minutes (default 90) until exp45/e44/L2/STOP_SAVING exists

Every life: --world lw --learner <E41's half-width backpropagation birth of the world's seed> --list-every 8 --decay-distances none.
Long lives (1,024 visits; worlds 1 to 4; naive, replay): --save-states 64,128,256,384,512,640,768,896,1024.
Short lives (256 visits; worlds 5 to 8; naive, replay): --save-states 64,128,192,256.
Lives without decay (256 visits; worlds 1 to 8; naive, replay; --wd 0): --save-states 8,16,...,256, so that the weights' sizes can be read at
every eighth visit, where the whole list is answered (the runner is not changed for this).
The probe: `branch <state after 64, 256, 1,024 visits of a long life> --groups 52,53,54,55`, into exp45/e44/L2/probe/.

States kept on disk: the latest of every life; those after 64, 256 and 1,024 visits of a long life until its probe branch from them is
done; world 1's after 256 and 1,024 visits (committed, as the note says). Every other state is removed once its weights' sizes are read.
Nothing here trains.
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
L2 = MDIR / "exp45" / "e44" / "L2"
LIVES, PROBE, NORMS = L2 / "lives", L2 / "probe", L2 / "norms"
SEEDS = (73129, 93683, 116089)
LONG_STATES = "64,128,256,384,512,640,768,896,1024"
PROBE_AGES = (64, 256, 1024)
PROBE_GROUPS = "52,53,54,55"
BRANCH = "claude/e44-l2-first-long-lives"


def birth(w):
    return MDIR / "exp45" / "e41" / f"e41_B_h256_f50_{SEEDS[(w - 1) % 3]}.pt"


def stem(arm, w, visits, wd0=False):
    return f"lw_{arm}_w{w}_v{visits}" + ("_rho0.5" if arm == "replay" else "") + ("_wd0" if wd0 else "")


def life_job(arm, w, visits, wd0=False):
    a = ["life", arm, str(w), str(visits)] + (["--rho", "0.5"] if arm == "replay" else []) + (["--wd", "0"] if wd0 else [])
    a += ["--world", "lw", "--learner", str(birth(w)), "--list-every", "8", "--decay-distances", "none"]
    a += ["--save-states", ",".join(str(k) for k in range(8, 257, 8)) if wd0 else LONG_STATES if visits == 1024 else "64,128,192,256"]
    s = stem(arm, w, visits, wd0)
    return dict(id=s, out=str(LIVES), stem=s, args=a)


def roster():
    lives = [life_job(arm, 1, 1024) for arm in ("naive", "replay")]                                   # the trial
    lives += [life_job(arm, w, 1024) for w in (2, 3, 4) for arm in ("naive", "replay")]              # long (world 2's naive beside the trial)
    lives += [life_job(arm, w, 256) for w in (5, 6, 7, 8) for arm in ("naive", "replay")]            # short
    lives += [life_job(arm, w, 256, True) for w in range(1, 9) for arm in ("naive", "replay")]       # without decay
    probes = []
    for j in lives[:8]:
        for n in PROBE_AGES:
            st = LIVES / f"{j['stem']}_state_v{n}.pt"
            probes.append(dict(id=f"{j['stem']}_br{n}_g{PROBE_GROUPS.replace(',', '-')}", out=str(PROBE), needs=[str(st)],
                               args=["branch", str(st), "--groups", PROBE_GROUPS]))
    return probes + lives


def jobs():
    L2.mkdir(parents=True, exist_ok=True)
    (L2 / "roster_jobs.json").write_text(json.dumps(roster(), indent=1))
    print(len(roster()), "jobs")


# ---------------------------------------------------------------- the weights' sizes
def norms():
    import torch as T
    NORMS.mkdir(parents=True, exist_ok=True)
    for p in sorted(LIVES.glob("lw_*_state_v*.pt")):
        m = re.fullmatch(r"(lw_.*)_state_v(\d+)\.pt", p.name)
        out = NORMS / f"{m.group(1)}_norms.json"
        d = json.loads(out.read_text()) if out.exists() else dict(
            life=m.group(1), note="The Euclidean norm of every weight tensor of the network in the state after the named number of visits "
                                  "(state['model']); 0: the born weights, read from the birth file. Read by l2_lives.py norms.", visits={})
        if m.group(2) in d["visits"]:
            continue
        try:
            s = T.load(p, weights_only=False)
        except Exception as e:      # noqa: BLE001  (a state still being written is read at the next call)
            print("not read:", p.name, e); continue
        if "0" not in d["visits"]:
            b = T.load(MDIR / s["cfg"]["learner"]["file"], weights_only=True)
            d["visits"]["0"] = dict(norms={k: float(v.double().norm()) for k, v in b.items()}, weights_tensors_sha256=s["cfg"]["learner"]["tensors_sha256"])
        d["visits"][m.group(2)] = dict(norms={k: float(v.double().norm()) for k, v in s["model"].items()}, weights_tensors_sha256=s["weights_tensors_sha256"],
                                       store=dict(items=len(s["buffer"]), bytes=s["buffer_bytes"], offered=s["offered"]), done_updates=s["done_updates"],
                                       state_file_sha256=subprocess.run(["sha256sum", str(p)], capture_output=True, text=True).stdout.split()[0])
        tmp = Path(str(out) + ".tmp"); tmp.write_text(json.dumps(d, indent=1)); tmp.replace(out)


# ---------------------------------------------------------------- saving to the branch
def states_of(s):
    out = {}
    for p in LIVES.glob(f"{s}_state_v*.pt"):
        m = re.fullmatch(re.escape(s) + r"_state_v(\d+)\.pt", p.name)
        if m:
            out[int(m.group(1))] = p
    return out


def prune():
    """Removes the states that are no longer needed (see the module's text); returns the states to commit."""
    keep = []
    done_norms = lambda s, n: (NORMS / f"{s}_norms.json").exists() and str(n) in json.loads((NORMS / f"{s}_norms.json").read_text())["visits"]
    for j in roster():
        if "stem" not in j:
            continue
        s = j["stem"]; st = states_of(s)
        if not st:
            continue
        long_ = "_v1024" in s
        finished = (LIVES / f"{s}.json").exists()
        latest = max(st)
        for n, p in sorted(st.items()):
            probe_pending = long_ and n in PROBE_AGES and not (PROBE / f"{s}_br{n}_g{PROBE_GROUPS.replace(',', '-')}.json").exists()
            world1 = s in ("lw_naive_w1_v1024", "lw_replay_w1_v1024_rho0.5") and n in (256, 1024)
            if world1 or (n == latest and not finished):
                keep.append(p)
            elif probe_pending or (n == latest and finished and long_ and probe_pending):
                pass                                            # stays on disk for the probe; not committed
            elif done_norms(s, n) and not (n == latest and not finished):
                p.unlink()
    return keep


def git(*a, check=True):
    r = subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True)
    if check and r.returncode:
        raise RuntimeError(f"git {' '.join(a[:3])}: {r.stderr.strip()[-400:]}")
    return r


def save(message=None):
    norms()
    keep = prune()
    rel = str(L2.relative_to(ROOT))
    tracked = [x for x in git("ls-files", rel).stdout.splitlines() if x.endswith(".pt")]
    gone = [x for x in tracked if not any(str(p.relative_to(ROOT)) == x for p in keep) and "/lives/" in x]
    if gone:
        git("rm", "-q", "--cached", "--ignore-unmatch", *gone)
    # npz files of a life are added only once its receipt exists (the receipt is written last)
    unfinished = [p for d in (LIVES, PROBE) for p in d.glob("*.npz") if not (d / (p.name.replace("_list.npz", ".npz")[:-4] + ".json")).exists()]
    git("add", "-A", rel)
    if unfinished:
        git("reset", "-q", "--", *[str(p.relative_to(ROOT)) for p in unfinished])
    if keep:
        git("add", "-f", *[str(p.relative_to(ROOT)) for p in keep])
    if git("diff", "--cached", "--quiet", check=False).returncode == 0:
        print(time.strftime("%H:%M:%S"), "nothing to save", flush=True); return
    git("commit", "-q", "-m", (message or "E44 L2: receipts, event files, logs and the latest states of the running lives (saved while the lives run)")
        + "\n\nActing-Agent: worker session_017dSEpXu6oc38q4xQZ5sPv3")
    for k in range(5):
        r = git("push", "-q", "origin", BRANCH, check=False)
        if r.returncode == 0:
            print(time.strftime("%H:%M:%S"), "saved", git("rev-parse", "--short", "HEAD").stdout.strip(), flush=True); return
        time.sleep(2 ** (k + 1))
    print(time.strftime("%H:%M:%S"), "push failed:", r.stderr.strip()[-300:], flush=True)


def saveloop(minutes=90):
    while not (L2 / "STOP_SAVING").exists():
        t0 = time.time()
        while time.time() - t0 < 60 * minutes and not (L2 / "STOP_SAVING").exists():
            time.sleep(20)
        try:
            save()
        except Exception as e:      # noqa: BLE001
            print(time.strftime("%H:%M:%S"), "save failed:", e, flush=True)


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "saveloop":
        saveloop(float(sys.argv[2]) if len(sys.argv) > 2 else 90)
    elif c == "save":
        save(sys.argv[2] if len(sys.argv) > 2 else None)
    else:
        {"jobs": jobs, "norms": norms}[c]()
