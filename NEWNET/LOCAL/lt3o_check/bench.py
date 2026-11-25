"""Speed: old lt3 vs lt3o, per byte on Moby after 2,000 warm-up bytes, one process at a time; then lt3o's per-byte time at scale 1, 2, 4."""
import sys, time, ctypes, numpy as np
sys.path.insert(0, "."); from harness import Lib, Net; from cands import CANDS
from ltb import S
old, new = Lib("./old.so", "lt3"), Lib("./new.so", "lt3o")
st = getattr(ctypes.CDLL("./new.so"), "lt3o_stats"); st.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
x = S["moby"]
def per_byte(lib, c, scale, nb):
    net = Net(lib, c, scale); net.run(x[:2000]); t = time.time(); net.run(x[2000:2000 + nb]); return (time.time() - t) / nb * 1e6, net
mode = sys.argv[1]
for name, c in CANDS:
    if mode == "speed":
        nb = 1500 if name == "p2_102" else 4000
        sc = 4 if name == "p2_102" else 1
        to, _ = per_byte(old, c, sc, nb); tn, net = per_byte(new, c, sc, nb)
        a = np.zeros(13, np.int64); st(net.h, a.ctypes.data)
        print(f"{name:7s} x{sc} old {to:8.0f} us/byte  new {tn:7.0f} us/byte  speed-up {to/tn:6.1f}x  | grow calls {a[0]} draws/call {a[1]/max(a[0],1):.1f} fallback {a[2]/max(a[0],1):.3f} lists ok {a[3]}", flush=True)
    else:
        ts = [per_byte(new, c, s, 3000)[0] for s in (1, 2, 4)]
        ex = np.polyfit(np.log2([1, 2, 4]), np.log2(ts), 1)[0]
        print(f"{name:7s} lt3o us/byte x1 {ts[0]:6.0f} x2 {ts[1]:6.0f} x4 {ts[2]:6.0f}  exponent per byte in size {ex:+.2f}", flush=True)
