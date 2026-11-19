# Compares the reviewer's tissue.c with E57's worker's program on the unchanging tissue (no random draws): outputs after each of the first bytes.
import sys, os, ctypes, numpy as np
FAST = os.environ['FASTDIR']; sys.path.insert(0, FAST + '/TISSUE_STEP1'); sys.path.insert(0, FAST + '/TISSUE_EVENT')
import tissue_step1 as ts, evt
src = open('drive.py').read().split("b, _ = ts.stream(SEED, NB)")[0]
sys.argv = ['drive.py', 'cmp', 'rule=fixed', 'nbytes=64']
exec(src)
b, _ = ts.stream(SEED, 64); b = np.ascontiguousarray(b, np.uint8)
T = evt.EvTissue(SEED, 'fixed')
rec = np.zeros((1, N), np.float32)
for i in range(40):
    T.run_bytes(b[i:i + 1]); xe = T.x.copy()
    L.tis_run(ctypes.byref(t), ctypes.c_void_p(b.ctypes.data), ctypes.c_int64(i), ctypes.c_int64(i + 1), ctypes.c_int64(i), ctypes.c_int64(i + 1), ctypes.c_void_p(rec.ctypes.data))
    d = np.abs(rec[0] - xe)
    if i < 6 or i % 8 == 0: print(i, 'byte', b[i], 'max |diff|', float(d.max()), 'at cell', int(d.argmax()), 'mine', float(rec[0][d.argmax()]), 'theirs', float(xe[d.argmax()]), 'nonzero mine/theirs', int((rec[0] != 0).sum()), int((xe != 0).sum()))
