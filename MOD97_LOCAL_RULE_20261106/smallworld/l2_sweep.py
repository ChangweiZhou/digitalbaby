"""E44, job L2, the sweep's long lives (the reviewer's instructions WORKER_E44_L2_SWEEP_LONG_LIVES.md of 17 November 2026, on
claude/vibrant-cerf-sje726): the check lives and the eight lives as jobs for l2_driver.py, and the saving of receipts, event files and
states to the helper's branch while lives run. The runner (sw_run.py), the scorer (l2_score.py) and the driver (l2_driver.py) are the
main worker's, unchanged.

    python3 l2_sweep.py jobs               # writes exp45/e44/L2/sweep/checks/check_jobs.json and exp45/e44/L2/sweep/sweep_jobs.json
    python3 l2_sweep.py save [message]     # commits and pushes receipts, event and list files of finished lives, logs, and the latest
                                           #   state of every running life; states after 256 and 1,024 visits of world 1 are kept
    python3 l2_sweep.py saveloop [minutes] # `save` every so many minutes (default 45) until exp45/e44/L2/sweep/STOP_SAVING exists

The eight lives (section 4 of the instructions): arms `sweep` and `sweep+replay` (rho = 0.5), worlds 1 to 4, 1,024 visits, from E41's
half-width birth under the sweep of the world's seed (worlds 1 and 4: 73129; 2: 93683; 3: 116089), with release 1's settings:
--world lw --H 256 --U 128, visits of 256 updates, weight decay 0.25, learning rate 0.001, --list-every 8 --decay-distances none,
--save-states 64,128,256,384,512,640,768,896,1024. Order: world 1's two lives and world 2's `sweep` first; then the rest.

The check lives (section 5), in exp45/e44/L2/sweep/checks/:
  c2_calib                 world 1, 8 visits in turn, 256 updates, `sweep`, the defaults of the two new arguments (E44's receipt
                           exp45/e44/F1/lw_sweep_w1_v8_turn.json)
  c3_sweep, c3_sweep_replay_rho0   world 1, one visit: `sweep`; `sweep+replay` with rho = 0
  c4_whole, c4_cut         world 1, 8 visits, `sweep+replay`, rho 0.5, --list-every 2 --decay-distances 1,2,4: whole; stopped after 4
                           visits and resumed from its state
  c6_sweep, c6_sweep_replay   world 1, 4 visits, the lives' settings, one process alone on the host: the two arms' cost
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
SW = MDIR / "exp45" / "e44" / "L2" / "sweep"
LIVES, CHK = SW / "lives", SW / "checks"
SEEDS = (73129, 93683, 116089)
LONG_STATES = "64,128,256,384,512,640,768,896,1024"
BRANCH = "claude/e44-l2-sweep-long-lives"
AGENT = "Acting-Agent: worker session_017dSEpTpaat6RxRKpJXsZWQ"
KEEP = {("lw_sweep_w1_v1024", 256), ("lw_sweep_w1_v1024", 1024), ("lw_sweep+replay_w1_v1024_rho0.5", 256), ("lw_sweep+replay_w1_v1024_rho0.5", 1024)}


def birth(w):
    return MDIR / "exp45" / "e41" / f"e41_S_h256_f50_{SEEDS[(w - 1) % 3]}.pt"


def stem(arm, w, visits):
    return f"lw_{arm}_w{w}_v{visits}" + ("_rho0.5" if arm == "sweep+replay" else "")


def lw(w):
    return ["--world", "lw", "--H", "256", "--U", "128", "--learner", str(birth(w))]


def life_job(arm, w, visits=1024):
    a = ["life", arm, str(w), str(visits)] + (["--rho", "0.5"] if arm == "sweep+replay" else [])
    a += lw(w) + ["--visit-updates", "256", "--wd", "0.25", "--lr", "0.001", "--list-every", "8", "--decay-distances", "none", "--save-states", LONG_STATES]
    s = stem(arm, w, visits)
    return dict(id=s, out=str(LIVES), stem=s, args=a)


def roster():
    order = [("sweep", 1), ("sweep+replay", 1), ("sweep", 2), ("sweep+replay", 2), ("sweep", 3), ("sweep+replay", 3), ("sweep", 4), ("sweep+replay", 4)]
    return [life_job(arm, w) for arm, w in order]


def check_jobs():
    c4 = ["life", "sweep+replay", "1", "8", "--rho", "0.5", *lw(1), "--list-every", "2", "--decay-distances", "1,2,4"]
    cost = ["--list-every", "8", "--decay-distances", "none", "--save-states", "4"]
    return [dict(id="c2_calib", out=str(CHK / "c2_calib"), args=["life", "sweep", "1", "8", *lw(1), "--in-turn", "--visit-updates", "256", "--save-states", "8"]),
            dict(id="c3_sweep", out=str(CHK / "c3_sweep"), args=["life", "sweep", "1", "1", *lw(1)]),
            dict(id="c3_sweep_replay_rho0", out=str(CHK / "c3_sweep_replay_rho0"), args=["life", "sweep+replay", "1", "1", "--rho", "0", *lw(1)]),
            dict(id="c4_whole", out=str(CHK / "c4_whole"), args=c4),
            dict(id="c4_cut", out=str(CHK / "c4_cut"), args=c4 + ["--stop-after", "4"]),
            dict(id="c4_resumed", out=str(CHK / "c4_cut"), after=["c4_cut"],
                 args=["life", "--resume", str(CHK / "c4_cut" / "lw_sweep+replay_w1_v8_rho0.5_state_v4.pt")]),
            dict(id="c6_sweep", out=str(CHK / "c6_sweep"), alone=True, args=["life", "sweep", "1", "4", *lw(1), *cost]),
            dict(id="c6_sweep_replay", out=str(CHK / "c6_sweep_replay"), alone=True, args=["life", "sweep+replay", "1", "4", "--rho", "0.5", *lw(1), *cost])]


def jobs():
    CHK.mkdir(parents=True, exist_ok=True)
    (CHK / "check_jobs.json").write_text(json.dumps(check_jobs(), indent=1))
    (SW / "sweep_jobs.json").write_text(json.dumps(roster(), indent=1))
    print(len(check_jobs()), "check jobs;", len(roster()), "lives")


# ---------------------------------------------------------------- saving to the branch
def states_of(s):
    out = {}
    for p in LIVES.glob(f"{s}_state_v*.pt"):
        m = re.fullmatch(re.escape(s) + r"_state_v(\d+)\.pt", p.name)
        if m:
            out[int(m.group(1))] = p
    return out


def keep_states():
    """The states to commit: the latest of every unfinished life, and world 1's after 256 and 1,024 visits. No state is deleted from disk."""
    keep = []
    for j in roster():
        s = j["stem"]; st = states_of(s)
        if not st:
            continue
        finished = (LIVES / f"{s}.json").exists()
        for n, p in sorted(st.items()):
            if (s, n) in KEEP or (n == max(st) and not finished):
                keep.append(p)
    return keep


def git(*a, check=True):
    r = subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True)
    if check and r.returncode:
        raise RuntimeError(f"git {' '.join(a[:3])}: {r.stderr.strip()[-400:]}")
    return r


def save(message=None):
    assert git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip() == BRANCH, "not on the helper's branch"
    keep = keep_states()
    rel = str(SW.relative_to(ROOT))
    tracked = [x for x in git("ls-files", rel).stdout.splitlines() if x.endswith(".pt") and "/lives/" in x]
    gone = [x for x in tracked if not any(str(p.relative_to(ROOT)) == x for p in keep)]
    if gone:
        git("rm", "-q", "--cached", "--ignore-unmatch", *gone)
    # npz files of a life are added only once its receipt exists (the receipt is written last)
    unfinished = [p for p in LIVES.glob("*.npz") if not (LIVES / (p.name.replace("_list.npz", ".npz")[:-4] + ".json")).exists()]
    git("add", "-A", rel)
    if unfinished:
        git("reset", "-q", "--", *[str(p.relative_to(ROOT)) for p in unfinished])
    if keep:
        git("add", "-f", *[str(p.relative_to(ROOT)) for p in keep])
    if git("diff", "--cached", "--quiet", check=False).returncode == 0:
        print(time.strftime("%H:%M:%S"), "nothing to save", flush=True); return
    git("commit", "-q", "-m", (message or "E44 L2 sweep: receipts, event files, logs and the latest states of the running lives (saved while the lives run)")
        + "\n\n" + AGENT)
    for k in range(5):
        r = git("push", "-q", "-u", "origin", BRANCH, check=False)
        if r.returncode == 0:
            print(time.strftime("%H:%M:%S"), "saved", git("rev-parse", "--short", "HEAD").stdout.strip(), flush=True); return
        time.sleep(2 ** (k + 1))
    print(time.strftime("%H:%M:%S"), "push failed:", r.stderr.strip()[-300:], flush=True)


def saveloop(minutes=45):
    while not (SW / "STOP_SAVING").exists():
        t0 = time.time()
        while time.time() - t0 < 60 * minutes and not (SW / "STOP_SAVING").exists():
            time.sleep(20)
        try:
            save()
        except Exception as e:      # noqa: BLE001
            print(time.strftime("%H:%M:%S"), "save failed:", e, flush=True)


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "saveloop":
        saveloop(float(sys.argv[2]) if len(sys.argv) > 2 else 45)
    elif c == "save":
        save(sys.argv[2] if len(sys.argv) > 2 else None)
    else:
        {"jobs": jobs}[c]()
