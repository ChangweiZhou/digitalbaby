"""Trace one region of net5 on the context stream: per byte, S = surprise, then the cells chosen in the first three winning columns.
Usage: python3 trace_region.py SEED REGION G"""
import sys; sys.path.insert(0, "/home/user/digitalbaby/NEWNET")
import numpy as np
from net5 import Tissue
G = int(sys.argv[3]) if len(sys.argv) > 3 else 8; seed = int(sys.argv[1]) if len(sys.argv) > 1 else 3
R = int(sys.argv[2]) if len(sys.argv) > 2 else 0; net = Tissue(seed=seed, L=R + 1, n_grow=8); rng = np.random.default_rng(seed + 1000); V = 32
u = rng.choice(V, G + 4, replace=False); mid = list(u[:G]); st, en = u[G:G+2], u[G+2:]
names = {int(x): n for x, n in zip(list(st) + mid + list(en), ["A", "E"] + [f"m{i+1}" for i in range(G)] + ["D", "F"])}
line = ""; n = 0
for seq in range(400):
    b = int(rng.integers(2)); line = f"{seq:3d} {'AE'[b]}: "
    for x in [st[b]] + mid + [en[b]]:
        cells, cols = net.step(x)
        sur = net.surprise[R]; ch = np.argmax(cells[R], 1)[cols[R] > 0]
        line += ("S" if sur else ".") + "".join(map(str, ch[:3])) + " "
    if seq % 40 == 0 or seq > 390: print(line)
