"""Late degradation in lt7 (scale 1, Moby): one change each; held-out loss on 3 never-read Moby windows every 32k bytes up to 256k."""
import sys, json, time, numpy as np
L = "/Users/pencilbard/Downloads/digitalbaby_newnet/NEWNET/LOCAL/lt7"; sys.path.insert(0, L)
from lt7 import Net7, biasonly_test, outstats
name = sys.argv[1]; opt = json.loads(sys.argv[2]); freeze = opt.pop("_freeze_after", None)
x = np.frombuffer(open("/Users/pencilbard/Downloads/digitalbaby_newnet/minifly_v155_keys_dev_v1_20261026/data/melville-moby_dick.txt", "rb").read(), np.uint8)
W = [x[a:a + 4000] for a in (600000, 800000, 1000000)]
n = Net7(dict(dict(ntis=2048, ninh=512, nrly=512, seed=1), **opt)); rows = []; t = time.time()
for k in range(8):
    learn = not (freeze is not None and k * 32000 >= freeze)
    n.run(x[k * 32000:(k + 1) * 32000], learn)
    o = outstats(n); a = n.audit()
    rows.append(dict(at=(k + 1) * 32, full=round(float(np.mean([n.test(w)[0] for w in W])), 4), bias=round(float(np.mean([biasonly_test(n, w)[0] for w in W])), 4),
                     out_syn=o["syn"], mean_w=o["mean_w_x1000"] / 1000, rate=round(a["tis_rate_mean_x1e6"] / 1e6, 3), fanin=a["fanin_tis_mean_x1000"] / 1000, syn=sum(a[k] for k in ("syn_exc","syn_inh","syn_dis","syn_out","syn_rly","syn_rd"))))
print(json.dumps(dict(name=name, rows=rows, secs=round(time.time() - t))), flush=True)
