"""Comparison of the outputs at the reading step of every byte, cell by cell, between two recordings (float32, bytes x 2048).

    python3 compare.py <mine.rec> <theirs.npy> <nbytes> [first_byte]
"""
import sys
import numpy as np
me = np.fromfile(sys.argv[1], dtype=np.float32).reshape(-1, 2048)
lo = int(sys.argv[4]) if len(sys.argv) > 4 else 0
rv = np.load(sys.argv[2], mmap_mode="r")[lo:lo + me.shape[0]]
first = first6 = None; mx = 0.0
for b0 in range(0, me.shape[0], 1024):
    d = np.abs(me[b0:b0 + 1024].astype(np.float64) - rv[b0:b0 + 1024]); mx = max(mx, float(d.max()))
    w = np.argwhere(d > 0); w6 = np.argwhere(d > 1e-6)
    if first is None and len(w): first = (lo + b0 + int(w[0][0]), int(w[0][1]), float(d[w[0][0], w[0][1]]))
    if first6 is None and len(w6): first6 = (lo + b0 + int(w6[0][0]), int(w6[0][1]), float(d[w6[0][0], w6[0][1]]))
print(dict(max_difference=mx, first_difference_byte_cell_size=first, first_above_1e6=first6))
