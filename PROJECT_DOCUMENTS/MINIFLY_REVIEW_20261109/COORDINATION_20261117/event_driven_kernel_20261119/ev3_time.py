# Compares ev2.c and ev3.c (same algorithm, different data structures) on the real connections of E63's saved state.
import ctypes, sys, time, numpy as np
d=np.load(sys.argv[1], allow_pickle=False); N=2048
alive=d['alive']; snd0=d['snd'][alive]; rcv0=d['rcv'][alive]; w0=d['w'][alive]
cc=snd0<N; snd0,rcv0,w0=snd0[cc].astype(np.int64),rcv0[cc].astype(np.int64),w0[cc]
line_cells0=np.unique(d['rcv'][alive][d['snd'][alive]>=N])
sig0=np.where(np.random.default_rng(1).random(N)<0.25,-1.0,1.0).astype(np.float32)
NB=8192; PB=5
drive0=np.random.default_rng(7).choice(line_cells0,size=(NB,PB))
def order(kind):
    if kind=='as numbered': return np.arange(N)
    if kind=='breadth-first from the input end':          # cells that are connected get nearby numbers
        adj=[[] for _ in range(N)]
        for s,r in zip(snd0,rcv0): adj[s].append(r); adj[r].append(s)
        seen=np.zeros(N,bool); out=[]; q=list(line_cells0)
        for c in q: seen[c]=True
        while len(out)<N:
            if not q:
                rest=np.nonzero(~seen)[0]; q=[int(rest[0])]; seen[q[0]]=True
            nq=[]
            for c in q:
                out.append(c)
                for t in adj[c]:
                    if not seen[t]: seen[t]=True; nq.append(t)
            q=nq
        perm=np.empty(N,np.int64); perm[np.array(out)]=np.arange(N); return perm
class T2(ctypes.Structure):
    _fields_=[('N',ctypes.c_int64),('n',ctypes.c_int64),('S',ctypes.c_int64)]+[(k,ctypes.c_void_p) for k in
      ('snd','rcv','in_ptr','out_ptr','out_conn','w','sig','A','v','p','th','x','xs','acc','e','inW','inD','W','W2','D')]+\
      [('nW',ctypes.c_int64),('nD',ctypes.c_int64),('floor_x',ctypes.c_double),('delta',ctypes.c_double),
       ('visits_push',ctypes.c_int64),('visits_learn',ctypes.c_int64),('cell_updates',ctypes.c_int64)]
class T3(ctypes.Structure):
    _fields_=[('N',ctypes.c_int64),('n',ctypes.c_int64),('S',ctypes.c_int64)]+[(k,ctypes.c_void_p) for k in
      ('out_ptr','out_tgt','out_w','in_ptr','in_src','in_q','c','xs','e','inW','inD','W','W2','D')]+\
      [('nW',ctypes.c_int64),('nD',ctypes.c_int64),('floor_x',ctypes.c_double),('delta',ctypes.c_double),
       ('visits_push',ctypes.c_int64),('visits_learn',ctypes.c_int64),('cell_updates',ctypes.c_int64)]
l2=ctypes.CDLL('./ev2.so'); l3=ctypes.CDLL('./ev3.so')
def run(which, kind, sort_out=False, reps=3):
    perm=order(kind); snd,rcv,w=perm[snd0],perm[rcv0],w0
    sig=np.empty(N,np.float32); sig[perm]=sig0
    drive=perm[drive0].astype(np.int32)
    best=None
    for _ in range(reps):
        z=lambda dt=np.float64: np.zeros(N,dt)
        lists=dict(inW=z(np.uint8),inD=z(np.uint8),W=np.zeros(N+64,np.int32),W2=np.zeros(N+64,np.int32),D=np.zeros(N+64,np.int32))
        if which==2:
            o=np.argsort(rcv,kind='stable'); s_,r_,w_=snd[o].astype(np.uint16),rcv[o].astype(np.uint16),w[o].astype(np.float32); n=len(s_)
            ip=np.zeros(N+1,np.int64); np.add.at(ip,r_.astype(np.int64)+1,1); ip=np.cumsum(ip)
            oo=np.argsort(s_,kind='stable').astype(np.int32); op=np.zeros(N+1,np.int64); np.add.at(op,s_.astype(np.int64)+1,1); op=np.cumsum(op)
            A=dict(snd=s_,rcv=r_,in_ptr=ip,out_ptr=op,out_conn=oo,w=w_,sig=sig,A=z(),v=z(),p=z(),th=z(),x=z(),xs=z(),acc=z(),e=z(),**lists)
            t=T2(); lib=l2.ev2_run
        else:
            key=(rcv if sort_out else np.zeros_like(rcv)); o=np.lexsort((key,snd))        # by sending cell (then by target)
            s_,r_,w_=snd[o],rcv[o],w[o].astype(np.float32); n=len(s_)
            op=np.zeros(N+1,np.int64); np.add.at(op,s_+1,1); op=np.cumsum(op)
            oi=np.argsort(r_,kind='stable')                                               # incoming lists point at the records
            ip=np.zeros(N+1,np.int64); np.add.at(ip,r_+1,1); ip=np.cumsum(ip)
            cells=np.zeros(N,dtype=[('A','f8'),('v','f8'),('p','f8'),('th','f8'),('x','f8'),('acc','f8'),('sig','f4'),('pad','i4'),('pad2','f8')])
            cells['sig']=sig
            A=dict(out_ptr=op,out_tgt=r_.astype(np.uint16),out_w=w_,in_ptr=ip,in_src=s_[oi].astype(np.uint16),in_q=oi.astype(np.int32),c=cells,xs=z(),e=z(),**lists)
            t=T3(); lib=l3.ev3_run
        t.N=N; t.n=n; t.S=32; t.nW=0; t.nD=0; t.floor_x=1/128; t.delta=1/1024
        for k,a in A.items(): setattr(t,k,a.ctypes.data)
        asum=ctypes.c_double(0); t0=time.process_time()
        lib(ctypes.byref(t),ctypes.c_int64(NB),ctypes.c_void_p(drive.ctypes.data),ctypes.c_int64(PB),ctypes.byref(asum))
        dt=time.process_time()-t0
        res=(dt,t.cell_updates/NB,(t.visits_push+t.visits_learn)/NB)
        best=res if best is None or res[0]<best[0] else best
    return best
rows=[("ev2: connections by receiver, index by sender; one array for each quantity",2,'as numbered',False),
      ("ev3: connections by sender as (target, strength); one record a cell",3,'as numbered',False),
      ("ev3, each cell's targets in ascending order",3,'as numbered',True),
      ("ev3, cells numbered breadth-first from the input end",3,'breadth-first from the input end',False),
      ("ev3, breadth-first numbering and targets in order",3,'breadth-first from the input end',True)]
base=None
for label,which,kind,so in rows:
    dt,cu,cv=run(which,kind,so)
    base=base or dt
    print(f"{label:76s} {cu:7.0f} cell updates, {cv:7.0f} connection visits a byte | {dt/NB*1e3:6.3f} ms a byte | x{base/dt:4.2f} | 262,144 bytes {dt/NB*262144:5.1f} s",flush=True)
