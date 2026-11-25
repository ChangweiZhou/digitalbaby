"""Why is a bigger lt7 worse? One probe: name, scale, bytes, option overrides -> held-out Moby loss + output-end and tissue statistics."""
import sys, json, time, numpy as np
L = "/Users/pencilbard/Downloads/digitalbaby_newnet/NEWNET/LOCAL/lt7"; sys.path.insert(0, L)
from lt7 import Net7, biasonly_test, outstats
name, s, D = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); opt = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}
x = np.frombuffer(open("/Users/pencilbard/Downloads/digitalbaby_newnet/minifly_v155_keys_dev_v1_20261026/data/melville-moby_dick.txt", "rb").read(), np.uint8)
m = len(x) // 2; held = x[m:m + 10000]
n = Net7(dict(dict(ntis=2048, ninh=512, nrly=512, seed=1), **opt), scale=s); t = time.time(); curve = []
step = D // 4
for k in range(4): n.run(x[k * step:(k + 1) * step]); curve.append(round(n.test(held)[0], 4))
a = n.audit(); o = outstats(n)
print(json.dumps(dict(name=name, s=s, D=D, opt=opt, curve=curve, bias_only=round(biasonly_test(n, held)[0], 4), out=o,
                      fanin_tis_mean=a["fanin_tis_mean_x1000"] / 1000, rate=a["tis_rate_mean_x1e6"] / 1e6, syn_rd=a["syn_rd"], syn_exc=a["syn_exc"],
                      secs=round(time.time() - t))), flush=True)
