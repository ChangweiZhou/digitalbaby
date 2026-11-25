"""Same battery as search_mech/search_pop (life -> cl; Moby ladder -> slope), for one (candidate, code, seed); prints one JSON line."""
import sys, json, time, numpy as np
sys.path.insert(0, "."); from harness import Lib, Net; from cands import CANDS
from ltb import S, held
name, code, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
lib = Lib("./old.so", "lt3") if code == "old" else Lib("./new.so", "lt3o"); c = dict(CANDS)[name]
T, NT, T0 = 64000, 10000, 32000; t0 = time.time()
net = Net(lib, c, 1, seed); order = ["shak", "emma", "code", "moby", "shak"]; tests = ["shak", "emma", "code", "moby"]; pos = {k: 0 for k in S}; mat = []
for k in order:
    net.run(S[k][pos[k]:pos[k] + T]); pos[k] += T; mat.append([round(net.test(held(j, NT)), 4) for j in tests])
ys = []
for s in (1, 2, 4):
    n2 = Net(lib, c, s, seed); n2.run(S["moby"][:T0 * s]); ys.append(round(n2.test(held("moby", NT)), 4))
print(json.dumps(dict(name=name, code=code, seed=seed, cl=round(float(np.mean(mat[-1])), 4), mat=mat, ladder=ys,
                      slope=round(float(np.polyfit(np.log2([1, 2, 4]), ys, 1)[0]), 4), secs=round(time.time() - t0, 1))), flush=True)
