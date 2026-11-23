"""check_net13_m.py: net13_m rc_match=0 against net13 (same seed, K8), each in its own process, on check_net13.py's stream.
Also runs rc_match=4 to show it differs. Usage: python3 check_net13_m.py [n_lines]"""
import sys, subprocess, numpy as np
K8 = dict(L=16, cC=64, joint=1, mesh=1, mesh_base=0, mesh_dag=5, c_k_win=4, c_th_on=0.9, c_th_match=0.75)
def run(which, n, out):
    import net13, net13_m
    net = net13.Tissue(seed=1, n_values=128, **K8) if which == "13" else net13_m.Tissue(seed=1, n_values=128, rc_match=int(which[1:]), **K8)
    r = np.random.default_rng(5); sym = [c for c in range(33, 127) if chr(c) not in "=~+"][:12]
    SH = open("/home/user/digitalbaby/PROJECT_DOCUMENTS/MINIFLY_OUTSIDE_ADVICE_20261118/nano/data/tinyshakespeare.txt", "rb").read()[:800]
    st = []
    for k in range(n):
        w = [sym[int(r.integers(12))] for _ in range(3)]; d = 61 if k % 2 == 0 or k < n // 3 else 126
        st += [(b, True) for b in w + [d] + (w if d == 61 else w[::-1]) + [10]]
    st += [(b, True) for b in SH] + [(b, False) for b, _ in st[:400]]
    Os, gs = [], []
    for b, lrn in st: gs.append(net.answer()); Os.append(net._osum().copy()); net.step(b, lrn)
    np.savez(out, O=np.array(Os), g=np.array(gs))
if len(sys.argv) > 2: run(sys.argv[2], int(sys.argv[1]), sys.argv[3]); sys.exit()
n = sys.argv[1] if len(sys.argv) > 1 else "300"; d = "/tmp/claude-0/check_net13_m"; subprocess.run(["mkdir", "-p", d])
ps = [subprocess.Popen([sys.executable, __file__, n, w, f"{d}/{w}.npz"]) for w in ("13", "m0", "m4")]; [p.wait() for p in ps]
a, b, c = (np.load(f"{d}/{w}.npz") for w in ("13", "m0", "m4"))
print(f"bytes {len(a['g'])}  rc_match=0 vs net13: max|dO| {np.abs(a['O'] - b['O']).max():.3g}  answers differing {(a['g'] != b['g']).sum()}")
print(f"rc_match=4 vs net13: max|dO| {np.abs(a['O'] - c['O']).max():.3g}  answers differing {(a['g'] != c['g']).sum()}")
