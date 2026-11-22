"""Architecture search by fast iteration (22 November 2026). Each candidate is a set of net10 options (the architecture); it is scored
on a fixed battery, the same for every candidate, of which two streams were never used in design:
  moby  - Moby Dick, read 16,000 bytes, held-out 10,000 (learning off)          (t17_general.py)
  code  - the repository's Python files outside NEWNET, the same                (t17_general.py)
  keep  - A on letters (G=8, 4,000 bytes), then 16,000 bytes of text: A after B  (t13_life8.py life=ctx)
  mem   - context stream G=16, 40,000 bytes: right at the byte after the gap     (t11_gate.py)
Score = the mean of the four. Results are appended to search_results.tsv (one line a test), so that every candidate ever run is kept.
Usage: python3 search_arch.py NAME 'opt=val opt=val ...' [tests=moby,code,keep,mem]"""
import sys, subprocess, re, os, time
name, opts = sys.argv[1], sys.argv[2]
tests = dict(a.split("=") for a in sys.argv[3:]).get("tests", "moby,code,keep,mem").split(",")
here = os.path.dirname(os.path.abspath(__file__))
cmd = {"moby": f"python3 t17_general.py stream=moby {opts}",
       "code": f"python3 t17_general.py stream=code {opts}",
       "keep": f"python3 t13_life8.py net=10 life=ctx {opts}",
       "mem": f"python3 t11_gate.py net=10 G=16 T=40000 {opts}"}
pat = {"moby": r"held-out ([0-9.]+)", "code": r"held-out ([0-9.]+)", "keep": r"after B ([0-9.]+) /", "mem": r"after the gap \(last quarter\) ([0-9.]+)"}
env = dict(os.environ, OMP_NUM_THREADS="1")
procs = {t: (subprocess.Popen(cmd[t], shell=True, cwd=here, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True), time.time()) for t in tests}
res = {}
for t, (p, t0) in procs.items():
    out = p.communicate()[0]; m = re.search(pat[t], out)
    res[t] = float(m.group(1)) if m else float("nan")
    with open(os.path.join(here, "search_results.tsv"), "a") as f:
        f.write(f"{name}\t{opts}\t{t}\t{res[t]:.3f}\t{time.time() - t0:.0f}\n")
    if not m: print(f"{name} {t}: no result\n{out[-600:]}")
score = sum(res.values()) / len(res)
print(f"{name}: " + "  ".join(f"{t} {v:.3f}" for t, v in res.items()) + f"  | score {score:.3f}  | {opts}")
