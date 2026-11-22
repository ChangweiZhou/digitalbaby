"""ms a byte of net10's context stack (net7 / net7e Tissue.step at L=8, C=128, S=64, lines_in=16, n_values=128, ctx=2, hold=0,
n_grow=8, mesh=2 mesh_base=0 mesh_dag=1, seed 301), tinyshakespeare from its start: bytes [W, W+N) timed after reading W bytes,
also the time inside the C kernel. Each run alone. A hash of the cells at every byte is kept so that net7 and net7e can be compared.
Usage: OMP_NUM_THREADS=1 python3 time_net7e.py net7|net7e M W N"""
import sys, time, os, hashlib, json, numpy as np
which, M, W, N = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
mod = __import__(which)
real = mod._lib
class _Timed:                                                                       # the C kernel's own time
    t = 0.0
    def __getattr__(self, k): return getattr(real, k)
    def step(self, *a):
        t0 = time.perf_counter(); real.step(*a); _Timed.t += time.perf_counter() - t0
mod._lib = _Timed()
here = os.path.dirname(os.path.abspath(__file__)) + "/../"
SH = np.frombuffer(open(here + "PROJECT_DOCUMENTS/MINIFLY_OUTSIDE_ADVICE_20261118/nano/data/tinyshakespeare.txt", "rb").read(), np.uint8)[:W + N].astype(int)
t = mod.Tissue(seed=301, L=8, C=128, M=M, S=64, lines_in=16, n_values=128, ctx=2, hold=0, n_grow=8, mesh=2, mesh_base=0, mesh_dag=1)
h = hashlib.sha256(); t0 = time.perf_counter()
for x in SH[:W]: c, _ = t.step(int(x)); h.update(np.packbits(c > 0).tobytes())
tw = time.perf_counter() - t0
_Timed.t = 0.0; t0 = time.perf_counter()
for x in SH[W:W + N]: c, _ = t.step(int(x)); h.update(np.packbits(c > 0).tobytes())
tt = time.perf_counter() - t0
r = dict(which=which, M=M, W=W, N=N, ms_byte=1e3 * tt / N, ms_byte_kernel=1e3 * _Timed.t / N, ms_byte_warmup=1e3 * tw / max(W, 1),
         slots_in_use=float((t.rw > 0).mean()), cells_hash=h.hexdigest()[:16], rw_hash=hashlib.sha256(t.rw.tobytes()).hexdigest()[:16])
print(json.dumps(r), flush=True)
