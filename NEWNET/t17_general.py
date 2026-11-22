"""Streams never used in designing the learner, for any net with step/answer (net=10 by default). stream=moby: Melville's Moby Dick;
stream=code: the Python files of the repository outside NEWNET, concatenated in sorted order. Reads T bytes from the stream's start
(learning on), then scores the next byte, learning off, on NT bytes from the middle of the stream (never read). Also the count tables
of the last 1/2/3 bytes fitted on the same T bytes and scored on the same NT bytes (reference only).
Usage: python3 t17_general.py stream=moby|code [T=16000 NT=10000 net=10 ...net parameters]"""
import sys, glob, importlib, os
import numpy as np
from collections import Counter, defaultdict
def num(v):
    for f in (int, float):
        try: return f(v)
        except ValueError: pass
    return v
a = {k: num(v) for k, v in (x.split("=") for x in sys.argv[1:])}
kind, T, NT, netn = a.pop("stream", "moby"), a.pop("T", 16000), a.pop("NT", 10000), a.pop("net", 10)
root = "/home/user/digitalbaby/"
if kind == "moby":
    raw = open(root + "minifly_v155_keys_dev_v1_20261026/data/melville-moby_dick.txt", "rb").read()
else:
    fs = sorted(f for f in glob.glob(root + "**/*.py", recursive=True) if "/NEWNET/" not in f and "/.git/" not in f)
    raw = b"".join(open(f, "rb").read() for f in fs)
s = np.frombuffer(raw, np.uint8).astype(int); s = np.minimum(s, 127)        # bytes above 127 (rare) read as 127
mid = len(s) // 2; test = s[mid:mid + NT + 1]
net = importlib.import_module("net%d" % netn).Tissue(**a)
for t in range(T): net.step(int(s[t])); net.answer()
r = np.zeros(NT, bool)
for t in range(NT): net.step(int(test[t]), False); r[t] = net.answer() == test[t + 1]
tabs = [defaultdict(Counter) for _ in range(3)]
for t in range(2, T):
    for o in range(3): tabs[o][tuple(s[t - o:t + 1])][s[t + 1]] += 1
ref = []
for o in range(3):
    ok = [bool(tabs[o][tuple(test[t - o:t + 1])]) and tabs[o][tuple(test[t - o:t + 1])].most_common(1)[0][0] == test[t + 1] for t in range(2, NT)]
    ref.append(np.mean(ok))
print(f"stream={kind} T={T} {' '.join(f'{k}={v}' for k, v in a.items())}: held-out {r.mean():.3f} | count tables fitted on the same bytes, last 1/2/3: " + "/".join(f"{x:.3f}" for x in ref) + f" | stream length {len(s)}")
