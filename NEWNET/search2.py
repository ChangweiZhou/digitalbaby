"""Architecture iteration on an extended, fixed battery (23 November 2026). Owner: iterate the general architecture, judged on the whole
battery, not on one test; no devices for single phenomena (MEMO_20261123.md, working rule).
A candidate is a set of options of net13 (or net=10/12 for references) along GENERAL dimensions only (sizes, topology, sparseness,
rates, which of the existing cell kinds are present and their sizes). Every candidate is run on the same tests:
  dev (used to choose):
    moby   - t17_general.py stream=moby: next-byte right share on never-read Moby Dick after 16 kB
    keep, keep4, keep2 - t13_life8.py life=ctx G=8/4/2: a context kept through 16 kB of text (right share after B)
    mem    - t11_gate.py G=16 T=40000: the byte after a gap (right share, last quarter)
    life23 - t23_life_rules.py cue=1: one life of rules and text; score 1 - bits / bits of storage (1,984,668 at seed 1)
    enzh   - t13_life8.py life=enzh n_values=256 NT=10000: English kept through Chinese (after B / after A)
  held-out (reported, never used to choose):
    code   - t17_general.py stream=code
    life27 - t27_newlife.py cue=1 (a life never used in design); score 1 - bits / 2,002,585 (storage, seed 1)
    zhen   - t13_life8.py life=zhen n_values=256 NT=10000: Chinese kept through English
dev score = mean of the dev tests; held score = mean of the held-out tests. Results are appended (one line a test) to SEARCH2_OUT
(default search2_results.tsv). SMOKE=1 shrinks every test (only to check the harness; never compare smoke numbers).
Usage: python3 search2.py NAME 'opt=val ...' [tests=...] [net=13]"""
import sys, subprocess, re, os, time, json
name, opts = sys.argv[1], sys.argv[2]
extra = dict(a.split("=", 1) for a in sys.argv[3:])
net = extra.get("net", "13")
DEV = ["moby", "keep", "keep4", "keep2", "mem", "life23", "enzh"]
HELD = ["code", "life27", "zhen"]
tests = extra.get("tests", ",".join(DEV + HELD)).split(",")
here = os.path.dirname(os.path.abspath(__file__))
K8 = "L=16 cC=64 joint=1 mesh=1 mesh_base=0 mesh_dag=5 c_k_win=4 c_th_on=0.9 c_th_match=0.75"
o = f"net={net} {K8} {opts}"                                    # later options override K8's
sm = os.environ.get("SMOKE") == "1"
S = dict(t17="T=2000 NT=1000" if sm else "", keep="TA=1000 TB=2000" if sm else "", mem="T=4000" if sm else "T=40000",
         t23="N0=300 N1=300 TXT=2000 N3=300 N4=150 N5=150 PROBE=300" if sm else "", t27="NB=150 TXT=2000 PROBE=300" if sm else "",
         lang="TA=2000 TB=2000 NT=1000" if sm else "NT=10000")
cmd = {"moby": f"python3 t17_general.py stream=moby {S['t17']} {o}",
       "code": f"python3 t17_general.py stream=code {S['t17']} {o}",
       "keep": f"python3 t13_life8.py life=ctx {S['keep']} {o}",
       "keep4": f"python3 t13_life8.py life=ctx G=4 {S['keep']} {o}",
       "keep2": f"python3 t13_life8.py life=ctx G=2 {S['keep']} {o}",
       "mem": f"python3 t11_gate.py G=16 {S['mem']} {o}",
       "life23": f"python3 t23_life_rules.py cue=1 struct=K8 {S['t23']} {o}",
       "life27": f"python3 t27_newlife.py cue=1 struct=K8 {S['t27']} {o}",
       "enzh": f"python3 t13_life8.py life=enzh n_values=256 {S['lang']} {o}",
       "zhen": f"python3 t13_life8.py life=zhen n_values=256 {S['lang']} {o}"}
REF = {"life23": 1984668.0, "life27": 2002585.0}
def parse(t, out):
    if t in ("moby", "code"):
        m = re.search(r"held-out ([0-9.]+)", out); return float(m.group(1)) if m else None
    if t.startswith("keep"):
        m = re.search(r"after B ([0-9.]+) /", out); return float(m.group(1)) if m else None
    if t == "mem":
        m = re.search(r"after the gap \(last quarter\) ([0-9.]+)", out); return float(m.group(1)) if m else None
    if t in ("enzh", "zhen"):
        m = re.search(r"after A ([0-9.]+) / [0-9.]+\s+after B ([0-9.]+) /", out)
        return float(m.group(2)) / float(m.group(1)) if m and float(m.group(1)) > 0 else None
    if t in ("life23", "life27"):
        for line in reversed(out.strip().splitlines()):
            if line.startswith("{") and '"bits"' in line:
                return 1.0 - json.loads(line)["bits"]["total"] / REF[t]
    return None
env = dict(os.environ, OMP_NUM_THREADS="1")
def tenv(t):                                                       # long tests save their state and go on after a container restart (resume.py)
    if sm or t not in ("life23", "life27", "enzh", "zhen", "keep", "keep4", "keep2", "mem"): return env
    return dict(env, CKPT=os.path.join(here, "search2_ckpt", f"{name}.{t}.pkl.gz"))
procs = {t: (subprocess.Popen(cmd[t], shell=True, cwd=here, env=tenv(t), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True), time.time()) for t in tests}
res = {}
os.makedirs(os.path.join(here, "search2_logs"), exist_ok=True)
for t, (p, t0) in procs.items():
    out = p.communicate()[0]; v = parse(t, out); res[t] = v
    open(os.path.join(here, "search2_logs", f"{name}.{t}.log"), "w").write(cmd[t] + "\n" + out)
    with open(os.path.join(here, os.environ.get("SEARCH2_OUT", "search2_results.tsv")), "a") as f:
        f.write(f"{name}\t{net}\t{opts}\t{t}\t{'nan' if v is None else f'{v:.4f}'}\t{time.time() - t0:.0f}\t{'smoke' if sm else 'full'}\n")
    if v is None: print(f"{name} {t}: no result\n{out[-800:]}")
def mean(ks):
    v = [res[k] for k in ks if k in res and res[k] is not None]; return sum(v) / len(v) if v else float("nan")
print(f"{name}: " + "  ".join(f"{t} {res[t]:.3f}" if res[t] is not None else f"{t} nan" for t in tests)
      + f"  | dev {mean(DEV):.4f}  held {mean(HELD):.4f}  | net={net} {opts}")
