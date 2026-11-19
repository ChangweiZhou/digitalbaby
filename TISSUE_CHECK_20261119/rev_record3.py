"""Run the reviewer's program (through tis.make and tis_run, as far.py and diag.py do; its source is not read here) and record the
outputs of every cell at phi = 31 of every byte from 0 to nbytes, as float32 (the only form tis_run gives).

    python3 rev_record.py <nbytes> <out.npy> key=value ...      (the options of drive.py)
"""
import ctypes
import os
import sys

R = "/home/user/rev3/PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/COORDINATION_20261117/reviewer_tissue_20261119"
sys.path.insert(0, R); os.chdir(R)
import numpy as np   # noqa: E402
import tis           # noqa: E402

nbytes, outp = int(sys.argv[1]), sys.argv[2]
kw = dict(a.split("=") for a in sys.argv[3:])
kw["nbytes"] = str(nbytes)
T = tis.make(kw); t, L, b, N = T["t"], T["L"], T["b"], T["N"]
rec = np.zeros((nbytes, N), np.float32)
L.tis_run(ctypes.byref(t), ctypes.c_void_p(b.ctypes.data), ctypes.c_int64(0), ctypes.c_int64(nbytes), ctypes.c_int64(0), ctypes.c_int64(nbytes),
          ctypes.c_void_p(rec.ctypes.data))
np.save(outp, rec)
A = T["A"]
print(dict(bytes=nbytes, deg_total=int(np.asarray(A["deg"]).sum()), keys=sorted(A.keys()) if hasattr(A, "keys") else None))
