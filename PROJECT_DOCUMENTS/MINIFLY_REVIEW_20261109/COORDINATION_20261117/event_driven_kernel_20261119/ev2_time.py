# Times ev2.c on the real connections of E63's saved state. A timing of the design; the cell's formulas are shortened.
import ctypes, sys, time, numpy as np
d=np.load(sys.argv[1], allow_pickle=False); N=2048
alive=d['alive']; snd0=d['snd'][alive]; rcv0=d['rcv'][alive]; w0=d['w'][alive]
cc=snd0<N; snd0,rcv0,w0=snd0[cc],rcv0[cc],w0[cc]
line_cells=np.unique(d['rcv'][alive][d['snd'][alive]>=N])
rs=np.random.default_rng(1); sig=np.where(rs.random(N)<0.25,-1.0,1.0).astype(np.float32)
class T2(ctypes.Structure):
    _fields_=[('N',ctypes.c_int64),('n',ctypes.c_int64),('S',ctypes.c_int64)]+[(k,ctypes.c_void_p) for k in
      ('snd','rcv','in_ptr','out_ptr','out_conn','w','sig','A','v','p','th','x','xs','acc','e','inW','inD','W','W2','D')]+\
      [('nW',ctypes.c_int64),('nD',ctypes.c_int64),('floor_x',ctypes.c_double),('delta',ctypes.c_double),
       ('visits_push',ctypes.c_int64),('visits_learn',ctypes.c_int64),('cell_updates',ctypes.c_int64)]
lib=ctypes.CDLL('./ev2.so')
def build(cap=None):
    snd,rcv,w=snd0,rcv0,w0
    if cap is not None:                       # keep, for each receiving cell, its `cap` strongest connections
        o=np.lexsort((-w,rcv)); snd,rcv,w=snd[o],rcv[o],w[o]
        first=np.r_[0,np.nonzero(np.diff(rcv))[0]+1]; rank=np.arange(len(rcv))-np.repeat(first,np.diff(np.r_[first,len(rcv)]))
        k=rank<cap; snd,rcv,w=snd[k],rcv[k],w[k]
    o=np.argsort(rcv,kind='stable'); snd,rcv,w=snd[o].astype(np.uint16),rcv[o].astype(np.uint16),w[o].astype(np.float32)
    n=len(snd)
    ip=np.zeros(N+1,np.int64); np.add.at(ip,rcv.astype(np.int64)+1,1); ip=np.cumsum(ip)
    oo=np.argsort(snd,kind='stable').astype(np.int32)
    op=np.zeros(N+1,np.int64); np.add.at(op,snd.astype(np.int64)+1,1); op=np.cumsum(op)
    return snd,rcv,w,ip,op,oo,n
def go(S=32, floor=1/128, delta=1/1024, cap=None, per_byte=5, nbytes=4096, label=''):
    snd,rcv,w,ip,op,oo,n=build(cap)
    z=lambda dt=np.float64: np.zeros(N,dt)
    A=dict(snd=snd,rcv=rcv,in_ptr=ip,out_ptr=op,out_conn=oo,w=w.copy(),sig=sig,A=z(),v=z(),p=z(),th=z(),x=z(),xs=z(),acc=z(),e=z(),
           inW=z(np.uint8),inD=z(np.uint8),W=np.zeros(N+64,np.int32),W2=np.zeros(N+64,np.int32),D=np.zeros(N+64,np.int32))
    t=T2(); t.N=N; t.n=n; t.S=S; t.nW=0; t.nD=0; t.floor_x=floor; t.delta=delta
    for k,a in A.items(): setattr(t,k,a.ctypes.data)
    drive=rs.choice(line_cells,size=(nbytes,per_byte)).astype(np.int32)
    asum=ctypes.c_double(0)
    t0=time.process_time()
    lib.ev2_run(ctypes.byref(t),ctypes.c_int64(nbytes),ctypes.c_void_p(drive.ctypes.data),ctypes.c_int64(per_byte),ctypes.byref(asum))
    dt=time.process_time()-t0
    print(f"{label:34s} conns/cell {n/N:5.1f} | cells sending at byte end {asum.value/nbytes:6.1f} | a byte: {t.cell_updates/nbytes:8.0f} cell updates, "
          f"{t.visits_push/nbytes:8.0f} + {t.visits_learn/nbytes:7.0f} connection visits | {dt/nbytes*1e3:6.3f} ms | 262,144 bytes: {dt/nbytes*262144/60:5.2f} min", flush=True)
go(label='32 steps a byte, all connections')
go(S=8, label='8 steps a byte')
go(cap=32, label='32 steps, 32 connections a cell')
go(cap=16, label='32 steps, 16 connections a cell')
go(S=8, cap=16, label='8 steps, 16 connections a cell')
go(delta=1/128, label='32 steps, coarser delta 1/128')
go(per_byte=20, label='32 steps, 20 cells driven a byte')
