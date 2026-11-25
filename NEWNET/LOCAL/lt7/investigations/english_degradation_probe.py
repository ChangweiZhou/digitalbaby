"""Late English degradation in lt7: one change at a time, held-out loss on 5 windows of Shakespeare never read in the life, every 16k bytes."""
import sys, json, numpy as np, ctypes
L = "/Users/pencilbard/Downloads/digitalbaby_newnet/NEWNET/LOCAL/lt7"; sys.path.insert(0, L); sys.path.insert(0, L + "/checks")
from lt7 import Net7, biasonly_test, outstats
import checks as C
WIN = [C.SHAK[a:a + 4000] for a in (300000, 400000, 500000, 700000, 900000)]
def ev(n): return round(float(np.mean([n.test(w)[0] for w in WIN])), 4), round(float(np.mean([biasonly_test(n, w)[0] for w in WIN])), 4)
name = sys.argv[1]; opt = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
freeze = opt.pop("_freeze_after", None); nr = opt.pop("_norelay", 0)
n = Net7(dict(C.SMALL, **opt))
if nr: n.norelay()
row = []
for k in range(4):
    learn = not (freeze is not None and k * 16000 >= freeze)
    n.run(C.SHAK[k * 16000:(k + 1) * 16000], learn)
    f, b = ev(n); a = n.audit(); o = outstats(n); beta = None
    row.append(dict(at=(k + 1) * 16, full=f, bias=b, rate=round(a["tis_rate_mean_x1e6"] / 1e6, 3), out_syn=o["syn"], driven=o["driven"], bias_mean=o["mean_bias_x1000"] / 1000))
print(json.dumps(dict(name=name, rows=row)), flush=True)
