# Throughput of independent tissues run side by side, one process each (ev2 kernel, first row of ev2_time.py, 8,192 bytes each).
import subprocess, sys, time, os
one = r'''
import sys; sys.argv=['x', sys.argv[1]]
import builtins
src=open('ev2_time.py').read().split("go(label='32 steps a byte, all connections')")[0]
exec(src)
import time
t0=time.perf_counter(); go(nbytes=8192,label='p'); 
'''
open('one.py','w').write(one)
env=dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
for k in (1,2,4,8):
    if k>os.cpu_count(): break
    t0=time.perf_counter()
    ps=[subprocess.Popen([sys.executable,'-I','one.py',sys.argv[1]],env=env,stdout=subprocess.DEVNULL) for _ in range(k)]
    for p in ps: p.wait()
    dt=time.perf_counter()-t0
    print(f"{k} tissues at once: wall {dt:5.2f} s for {k}x8,192 bytes -> {k*8192/dt:9.0f} bytes a second in all (includes start-up of Python and loading)")
