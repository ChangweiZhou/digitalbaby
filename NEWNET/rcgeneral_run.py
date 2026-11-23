"""Driver for RCGENERAL_REPORT.md (copy of battery12_run.py's scheme): the battery (search_arch.py, one test per job) and the
English/Chinese lives (t13_life8.py life=enzh/zhen n_values=256 NT=10000) for net12_rc (net=121 alias) rc=64 rc_gate=1 lagro=1
relay=1 + K8, seeds 1-3. Battery lines -> search_results_rc.tsv; logs -> rcgeneral_runs/<name>.log."""
import subprocess, os, sys
from concurrent.futures import ThreadPoolExecutor
here = os.path.dirname(os.path.abspath(__file__)); env = dict(os.environ, OMP_NUM_THREADS="1", SEARCH_OUT="search_results_rc.tsv")
K8 = "L=16 cC=64 joint=1 mesh=1 mesh_base=0 mesh_dag=5 c_k_win=4 c_th_on=0.9 c_th_match=0.75"
C = {"RC": "net=121 " + K8 + " lagro=1 relay=1 rc=64 rc_gate=1"}
T = "moby,code,keep,keep4,keep2,mem".split(",")
jobs = []
def bat(n, s):
    o = C[n] + ("" if s == 1 else f" seed={s}"); nm = n if s == 1 else f"{n}-s{s}"
    for t in T: jobs.append((f"{nm}.{t}", f"python3 search_arch.py {nm} '{o}' tests={t}"))
def life(n, s):
    o = C[n] + f" seed={s}"
    for lf in ("enzh", "zhen"): jobs.append((f"life_{n}_s{s}_{lf}", f"python3 t13_life8.py net=10 life={lf} n_values=256 NT=10000 {o}"))
for s in (1, 2, 3): life("RC", s)
for s in (1, 2, 3): bat("RC", s)
os.makedirs(os.path.join(here, "rcgeneral_runs"), exist_ok=True)
def go(j):
    nm, c = j; log = os.path.join(here, "rcgeneral_runs", nm + ".log")
    if os.path.exists(log) and "done" in open(log).read()[-200:]: return
    with open(log, "w") as f:
        f.write(c + "\n"); f.flush(); r = subprocess.run(c, shell=True, cwd=here, env=env, stdout=f, stderr=subprocess.STDOUT)
        f.write(f"\nexit {r.returncode} done\n")
with ThreadPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 4) as ex: list(ex.map(go, jobs))
