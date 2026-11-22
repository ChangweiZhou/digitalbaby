"""Driver for BATTERY12_REPORT.md: runs the battery (search_arch.py, one test per job) and the English/Chinese lives
(t13_life8.py life=enzh/zhen, n_values=256, NT=10000, as LIFE_EN_ZH.md) for net12 candidates, 4 jobs at a time (OMP_NUM_THREADS=1).
Battery lines go to search_results_net12.tsv (via search_arch.py); life outputs to battery12_runs/<name>.log."""
import subprocess, os, sys
from concurrent.futures import ThreadPoolExecutor
here = os.path.dirname(os.path.abspath(__file__)); env = dict(os.environ, OMP_NUM_THREADS="1", SEARCH_OUT="search_results_net12.tsv")
K8 = "L=16 cC=64 joint=1 mesh=1 mesh_base=0 mesh_dag=5 c_k_win=4 c_th_on=0.9 c_th_match=0.75"
C = {"B0": "net=12 " + K8, "B1": "net=12 " + K8 + " lagro=1", "B2": "net=12 " + K8 + " lagro=1 relay=1",
     "B3": "net=12 " + K8 + " lagro=1 relay=1 ctxg=16", "B4": "net=12 " + K8 + " lagro=1 relay=1 ctxg=16 ctxg_err=1.0"}
T = "moby,code,keep,keep4,keep2,mem".split(",")
jobs = []
def bat(n, s):
    o = C[n] + ("" if s == 1 else f" seed={s}"); nm = n if s == 1 else f"{n}-s{s}"
    for t in T: jobs.append((f"{nm}.{t}", f"python3 search_arch.py {nm} '{o}' tests={t}"))
def life(n, s):
    o = C[n] + f" seed={s}"
    for lf in ("enzh", "zhen"): jobs.append((f"life_{n}_s{s}_{lf}", f"python3 t13_life8.py net=10 life={lf} n_values=256 NT=10000 {o}"))
for n in ("B0", "B1", "B2", "B3", "B4"): bat(n, 1)
for n in ("B0", "B2", "B3"): life(n, 1)
for s in (2, 3):
    for n in ("B2", "B3"): bat(n, s)
for s in (2, 3):
    for n in ("B2", "B3"): life(n, s)
def go(j):
    nm, c = j; log = os.path.join(here, "battery12_runs", nm + ".log")
    if os.path.exists(log) and "done" in open(log).read()[-200:]: return
    with open(log, "w") as f:
        f.write(c + "\n"); f.flush(); r = subprocess.run(c, shell=True, cwd=here, env=env, stdout=f, stderr=subprocess.STDOUT)
        f.write(f"\nexit {r.returncode} done\n")
with ThreadPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 4) as ex: list(ex.map(go, jobs))
