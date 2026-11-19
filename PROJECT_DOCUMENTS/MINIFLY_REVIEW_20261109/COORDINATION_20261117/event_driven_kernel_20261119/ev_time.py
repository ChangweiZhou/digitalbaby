# Times the event-driven kernel ev.c on the real connections of E63's saved state (first form, byte 4,096).
# The kernel's cell dynamics are a simplified form of the definition's (state, prediction, threshold, output floor),
# so this measures the cost of the design at a measured level of activity; it is not a run of the tissue.
import ctypes, sys, time, numpy as np
d=np.load(sys.argv[1], allow_pickle=False)
N=2048
alive=d['alive']; snd=d['snd'][alive]; rcv=d['rcv'][alive]; w=d['w'][alive]; e=d['e'][alive]
cc=snd<N                                   # connections from cells (lines are driven separately)
snd,rcv,w,e=snd[cc],rcv[cc],w[cc].copy(),e[cc].copy()
o=np.argsort(rcv,kind='stable'); snd,rcv,w,e=snd[o].astype(np.int32),rcv[o].astype(np.int32),w[o],e[o]
n=len(snd)
in_ptr=np.zeros(N+1,np.int64); np.add.at(in_ptr,rcv+1,1); in_ptr=np.cumsum(in_ptr)
oo=np.argsort(snd,kind='stable').astype(np.int32)
out_ptr=np.zeros(N+1,np.int64); np.add.at(out_ptr,snd+1,1); out_ptr=np.cumsum(out_ptr)
rs=np.random.default_rng(1)
sig=np.where(rs.random(N)<0.25,-1.0,1.0)
line_cells=np.unique(d['rcv'][alive][d['snd'][alive]>=N])      # cells that read an input line
print(f"connections between cells: {n:,} ({n/N:.1f} a cell); cells that read a line: {len(line_cells)}")
class T(ctypes.Structure):
    _fields_=[('N',ctypes.c_int64),('n',ctypes.c_int64)]+[(k,ctypes.c_void_p) for k in
      ('snd','rcv','in_ptr','out_ptr','out_conn','w','e','te','sig','v','p','th','abar','x','tc','A','touched','tl','act','act2')]+[('na',ctypes.c_int64),('dec_v',ctypes.c_void_p),('dec_p',ctypes.c_void_p),('floor_x',ctypes.c_double)]
lib=ctypes.CDLL('./ev.so'); lib.ev_run.restype=ctypes.c_int64
def go(floor, per_byte, nbytes=2048):
    A=dict(te=np.zeros(n,np.int32),v=np.zeros(N),p=np.zeros(N),th=np.zeros(N),abar=np.zeros(N),x=np.zeros(N),tc=np.zeros(N,np.int32),
           A=np.zeros(N),touched=np.zeros(N,np.uint8),tl=np.zeros(N,np.int32),act=np.zeros(N+64,np.int32),act2=np.zeros(N+64,np.int32),
           dec_v=0.5**np.arange(4096),dec_p=(1-1/32)**np.arange(4096.0),w=w.copy(),e=e.copy())
    t=T(); t.N=N; t.n=n; t.na=0; t.floor_x=floor
    keep=dict(snd=snd,rcv=rcv,in_ptr=in_ptr,out_ptr=out_ptr,out_conn=oo,sig=sig,**A)
    for k,a in keep.items(): setattr(t,k,a.ctypes.data)
    drive=rs.choice(line_cells,size=(nbytes,per_byte)).astype(np.int32)
    vis=ctypes.c_int64(0); asum=ctypes.c_int64(0)
    t0=time.process_time()
    lib.ev_run(ctypes.byref(t),ctypes.c_int64(nbytes),ctypes.c_void_p(drive.ctypes.data),ctypes.c_int64(per_byte),ctypes.byref(vis),ctypes.byref(asum))
    dt=time.process_time()-t0
    steps=nbytes*32
    print(f"output floor {floor:g}, {per_byte} cells driven a byte: mean {asum.value/steps:7.1f} cells active a step; "
          f"{vis.value/steps:9.0f} connection visits a step ({100*vis.value/steps/(2*n):5.1f}% of two passes over all); "
          f"{dt/nbytes*1e3:7.3f} ms a byte; 262,144 bytes in {dt/nbytes*262144/60:6.1f} min")
for floor in (0.01, 1/128, 0.001, 0.0):
    go(floor, 5)
go(0.01, 20)
