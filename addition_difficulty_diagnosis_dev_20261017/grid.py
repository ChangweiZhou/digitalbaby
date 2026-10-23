"""Run the diagnostic grid (4 parallel single-thread workers). Resumable: an existing result file is skipped."""
import itertools
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "runs"
WORLDS = (202610143901, 202610143902, 202610143903)
SEEDS = (0, 1, 2)


def jobs():
    for w, s in itertools.product(WORLDS, SEEDS):
        yield dict(world=w, seed=s, inp="digit", out="dec", regime="online", steps=16384, wd=0.0, log=1024)
    for inp, out, w, s in itertools.product(("digit", "unary"), ("dec", "sym"), WORLDS, SEEDS):
        yield dict(world=w, seed=s, inp=inp, out=out, regime="batch", steps=20000, wd=0.0, log=250)
    for out, w, s in itertools.product(("dec", "sym"), WORLDS, SEEDS):
        yield dict(world=w, seed=s, inp="digit", out=out, regime="batch", steps=20000, wd=1.0, log=250)
    # Stage 2: counting experience (and its permuted control) on the registered task format.
    for aux, w, s in itertools.product(("count", "count_shuf"), WORLDS, SEEDS):
        yield dict(world=w, seed=s, inp="digit", out="dec", regime="online", steps=16384, wd=0.0, log=1024, aux=aux)
        for wd in (0.0, 1.0):
            yield dict(world=w, seed=s, inp="digit", out="dec", regime="batch", steps=20000, wd=wd, log=250, aux=aux)
    # Stage 3: smaller recurrent width (less room to memorise), registered task format, no weight decay.
    for width, w, s in itertools.product((16, 32), WORLDS, SEEDS):
        yield dict(world=w, seed=s, inp="digit", out="dec", regime="batch", steps=20000, wd=0.0, log=250, width=width)


def name(j):
    aux = "" if j.get("aux", "none") == "none" else f"_{j['aux']}"
    width = "" if j.get("width", 64) == 64 else f"_w{j['width']}"
    return f"{j['regime']}_wd{j['wd']:g}_{j['inp']}_{j['out']}{aux}{width}_{j['world']}_s{j['seed']}.json"


def one(j):
    path = OUT / name(j)
    if path.exists():
        return f"skip {path.name}"
    tmp = path.with_suffix(".tmp")
    cmd = [sys.executable, str(HERE / "diag.py"), "--world", str(j["world"]), "--seed", str(j["seed"]),
           "--inp", j["inp"], "--out", j["out"], "--regime", j["regime"], "--steps", str(j["steps"]),
           "--wd", str(j["wd"]), "--log-every", str(j["log"]), "--aux", j.get("aux", "none"),
           "--width", str(j.get("width", 64)), "--save", str(tmp)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode:
        return f"FAIL {path.name}: {p.stderr[-300:]}"
    tmp.rename(path)
    return f"done {path.name} {p.stdout.strip()}"


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(4) as ex:
        for msg in ex.map(one, list(jobs())):
            print(msg, flush=True)
