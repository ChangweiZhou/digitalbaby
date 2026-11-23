"""dev diagnostic: candidate scores of net12_rec on the t24 col stream (no scoring of answers)."""
import sys, numpy as np
sys.path.insert(0, ".")
import net12_rec
a = dict(x.split("=", 1) for x in sys.argv[1:]); f = lambda k, d: type(d)(a.get(k, d))
fmt = a.get("fmt", "col"); NL = f("NL", 1500)
net = net12_rec.Tissue(seed=1, n_values=128, L=4, cC=64, joint=1, mesh=1, mesh_base=0, mesh_dag=5, c_k_win=4, c_th_on=0.9, c_th_match=0.75,
                       lagro=1, pairs=32768, rec=f("rec", 16), rec_age=f("rec_age", 10**9), rec_mode=a.get("rec_mode", "w"), rec_score=a.get("rec_score", "gain"),
                       eta_rt=f("eta_rt", 0.2), eta_rw=f("eta_rw", 0.05), rec_init=f("rec_init", 1.0))
D = [ord(c) for c in "0123456789"]; rng = np.random.default_rng(5); nb = 0; H = []
for ln in range(NL):
    n = int(rng.integers(1, 5)); x, y = rng.integers(0, 10, n).tolist() + [0], rng.integers(0, 10, n).tolist() + [0]; c = 0
    for i in range(n + 1):
        s = (x[i] + y[i] + c) % 10
        for b, sc in ((D[x[i]], 0), (D[y[i]], 0), (D[s], 1)):
            if sc: H.append([c] + list(net.h[:net.K]))
            net.answer(); net.step(b, True); nb += 1
        c = (x[i] + y[i] + c) // 10
    net.answer(); net.step(10, True); nb += 1
    if (ln + 1) % 250 == 0:
        g = net.G[:net.K] / net.E; Hh = np.array(H[-600:]); cc = [abs(np.corrcoef(Hh[:, 0], Hh[:, k])[0, 1]) if Hh[:, k].std() > 1e-9 else 0 for k in range(1, Hh.shape[1])]
        print(ln + 1, nb, "|corr carry| max %.2f mean %.2f" % (max(cc), np.mean(cc)), "E=%.3f" % net.E, "score max %.4f mean %.4f" % (g.max(), g.mean()), "top", np.round(np.sort(g)[-4:], 4), flush=True)
H = np.array(H[-3000:]); cc = [np.corrcoef(H[:, 0], H[:, k])[0, 1] if H[:, k].std() > 0 else 0 for k in range(1, H.shape[1])]
print("corr(h, carry) per candidate", np.round(cc, 2)); print("scores", np.round(net.G[:net.K] / net.E, 4))
