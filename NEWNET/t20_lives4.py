"""One life that alternates four very different byte streams (22 November 2026), for the two best structures of the search, unchanged
(net10, n_values=256). Streams: en = tinyshakespeare; zh = Hong Lou Meng, UTF-8 (NEWNET/data/hongloumeng.txt, from
gutenberg.pglaf.org/2/4/2/6/24264/24264-0.txt, not tracked); code = the repository's tracked .py files outside NEWNET, concatenated in
sorted path order (git ls-files); num = lines "a+b=c\\n", a and b uniform in 0..999 (numpy default_rng(12345)).
Life (mode=life): blocks of B=8,000 bytes in the order en, zh, code, num, for R=3 rounds, nothing reset, no task signal; each block
continues where its stream stopped. After every block: the right share of the next byte on each stream's held-out 5,000 predictions
(5,001 bytes never read), learning off, each test on a copy of the network (the life itself is not touched by the tests).
Control (mode=ctl stream=X): stream X alone for the same R blocks of B bytes, its own test after each block.
Usage: python3 t20_lives4.py mode=life|ctl struct=K8|N1 seed=1 [stream=en|zh|code|num] [B=8000 R=3 NT=5000]
Writes one JSON line to NEWNET/lives4_runs/<mode>_<struct>_<stream>_s<seed>.json"""
import sys, os, json, time, copy, subprocess, numpy as np
from net10 import Tissue
a = dict(x.split("=") for x in sys.argv[1:])
mode, struct, seed = a.get("mode", "life"), a.get("struct", "K8"), int(a.get("seed", 1))
B, R, NT = int(a.get("B", 8000)), int(a.get("R", 3)), int(a.get("NT", 5000))
STRUCTS = {"K8": dict(L=16, cC=64, joint=1, mesh=1, mesh_base=0, mesh_dag=5, c_k_win=4, c_th_on=0.9, c_th_match=0.75),
           "N1": dict(L=8, cC=64, M=32, joint=1, mesh=2, mesh_base=0, mesh_dag=2, c_k_win=4)}
here = "/home/user/digitalbaby/"
def u8(b): return np.frombuffer(b, np.uint8).astype(int)
EN = u8(open(here + "PROJECT_DOCUMENTS/MINIFLY_OUTSIDE_ADVICE_20261118/nano/data/tinyshakespeare.txt", "rb").read())
ZH = u8(open(here + "NEWNET/data/hongloumeng.txt", "rb").read())
files = sorted(f for f in subprocess.run(["git", "-C", here, "ls-files", "*.py"], capture_output=True, text=True).stdout.split()
               if not f.startswith("NEWNET/"))
CODE = u8(b"".join(open(here + f, "rb").read() for f in files))
nrng = np.random.default_rng(12345); lines = []; n = 0
while n < 200000:
    x, y = nrng.integers(0, 1000, 2); s = f"{x}+{y}={x + y}\n"; lines.append(s); n += len(s)
NUM = u8("".join(lines).encode())
# (life start, test start): the life reads at most R*B+1 bytes from its start; the test region lies far beyond it
OFF = {"en": (0, 600000), "zh": (100000, 1500000), "code": (0, 16000000), "num": (0, 100000)}
DATA = {"en": EN, "zh": ZH, "code": CODE, "num": NUM}
for k, (r0, t0) in OFF.items(): assert r0 + R * B + 1 <= t0 and t0 + NT + 1 <= len(DATA[k]), k
TEST = {k: DATA[k][t0:t0 + NT + 1] for k, (r0, t0) in OFF.items()}
net = Tissue(seed=seed, n_values=256, **STRUCTS[struct])
def run(nt, stream, learn):
    r = np.zeros(len(stream) - 1, bool)
    for t in range(len(stream) - 1):
        nt.step(int(stream[t]), learn); r[t] = nt.answer() == stream[t + 1]
    return r.mean()
def test(names): return {k: round(float(run(copy.deepcopy(net), TEST[k], False)), 4) for k in names}
pos = {k: OFF[k][0] for k in OFF}; rows = []; t0 = time.time()
order = ["en", "zh", "code", "num"] if mode == "life" else [a["stream"]]
for rnd in range(R):
    for k in order:
        run(net, DATA[k][pos[k]:pos[k] + B + 1], True); pos[k] += B
        rows.append(dict(round=rnd + 1, block=k, scores=test(order)))
        print(rnd + 1, k, rows[-1]["scores"], f"{time.time() - t0:.0f}s", flush=True)
os.makedirs(here + "NEWNET/lives4_runs", exist_ok=True)
out = dict(mode=mode, struct=struct, seed=seed, stream=a.get("stream", "all"), B=B, R=R, NT=NT, options=STRUCTS[struct], offsets=OFF,
           code_bytes=len(CODE), code_files=len(files), num_bytes=len(NUM), rows=rows, seconds=round(time.time() - t0))
open(here + f"NEWNET/lives4_runs/{mode}_{struct}_{out['stream']}_s{seed}.json", "w").write(json.dumps(out) + "\n")
