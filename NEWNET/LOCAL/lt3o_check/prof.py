import sys, time
sys.path.insert(0, "."); from harness import Lib, Net; from cands import CANDS
from ltb import S
so, pre, name, scale, nb = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
c = dict(CANDS)[name]; net = Net(Lib(so, pre), c, scale)
x = S["moby"]; net.run(x[:2000]); t = time.time(); net.run(x[2000:2000 + nb]); print(name, scale, "us/byte", round((time.time() - t) / nb * 1e6), flush=True)
