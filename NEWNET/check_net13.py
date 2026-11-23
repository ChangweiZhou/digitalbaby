"""check_net13.py: net13 (rc_all=0) against net12_rc (lagro=1 relay=1 rc=64) with the same seed (K8, n_values=128).
Each net runs in its own process (net7's context cells draw from a process-wide random state in their C library, so two nets in one
process disturb each other). Stream: copy lines (=), rev lines (~), a stretch of tinyshakespeare, then 400 bytes with learning off.
Usage: python3 check_net13.py [n_lines]; prints the largest difference of the summed outputs and the number of different answers."""
import sys, subprocess, numpy as np
K8 = dict(L=16, cC=64, joint=1, mesh=1, mesh_base=0, mesh_dag=5, c_k_win=4, c_th_on=0.9, c_th_match=0.75)
def run(which, n, out):
    import net12_rc, net13
    net = net12_rc.Tissue(seed=1, n_values=128, lagro=1, relay=1, rc=64, **K8) if which == "12rc" else net13.Tissue(seed=1, n_values=128, **K8)
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
n = sys.argv[1] if len(sys.argv) > 1 else "300"; d = "/tmp/claude-0/check_net13"; subprocess.run(["mkdir", "-p", d])
ps = [subprocess.Popen([sys.executable, __file__, n, w, f"{d}/{w}.npz"]) for w in ("12rc", "13")]; [p.wait() for p in ps]
a, b = np.load(f"{d}/12rc.npz"), np.load(f"{d}/13.npz")
print(f"bytes {len(a['g'])}  max|dO| {np.abs(a['O'] - b['O']).max():.3g}  answers differing {(a['g'] != b['g']).sum()}")
