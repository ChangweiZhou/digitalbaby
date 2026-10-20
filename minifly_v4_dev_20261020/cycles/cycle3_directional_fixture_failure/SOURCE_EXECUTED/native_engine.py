"""Pinned native event equations, with one declared alpha-allocation intervention."""
import math
from v4_common import np,NativeBank,bootstrap,query_batch,predict_drive
from numba import njit

@njit(cache=True)
def event_v4(fast, slow, adapt, x, dt, coeff, plastic, Q, T, F, MM, k0,
                fw0, fwd, ta, tg, tf, ts, fraction, alpha_scale, genes, sf, ss, confirm, confirm_at, last_record, record_id, now, gated):
    nc, n = slow.shape
    dg, df, ds, da = math.exp(-dt/tg), math.exp(-dt/tf), math.exp(-dt/ts), math.exp(-dt/ta)
    rm_all = np.zeros((nc, 4))
    applied = 0.0
    has_genes = np.any(genes[:4] != 0)
    for c in range(nc):
        dx = np.empty(n)
        end = np.empty(n)
        agg, inp = np.zeros(4), np.zeros(4)
        for i in range(n):
            end[i] = 1 - (1 - adapt[c, i]*math.exp(-.05*dt*x[i]))*da
            dx[i] = .5*(adapt[c, i]+end[i])*x[i]
            for j in range(4):
                agg[j] += Q[i, j]*dx[i]
                value = fast[c, i, 0] if j < 2 else fast[c, i, 1]+slow[c, i]
                inp[j] += Q[i, j]*dx[i]*(k0[3 if j < 2 else 5]+value)
        rm = np.empty(4)
        for j in range(2):
            rm[j] = min(max(inp[j]+35.2, 0), 71.66)-35.2
        for j in range(2):
            val = inp[j+2]
            for h in range(2):
                val += MM[h, j]*rm[h]
            rm[j+2] = min(max(val+11.2, 0), 31.16)-11.2
        rm_all[c] = rm
        p0, p1 = np.empty(4), np.empty(4)
        for j in range(4):
            d0 = k0[0 if j < 2 else 2]*agg[j]
            for h in range(4):
                d0 += F[h, j]*rm[h]
            p0[j] = fw0*d0
            p1[j] = fw0*d0+fwd*(27.85 if j < 2 else 11.38)
        for i in range(n):
            f0, f1, sl = fast[c, i, 0], fast[c, i, 1], slow[c, i]
            if plastic and coeff[c] != 0 and dx[i] != 0:
                g0 = (T[i, 0]*p0[0]+T[i, 1]*p0[1])*dx[i]*dt/90
                g1 = (T[i, 0]*p1[0]+T[i, 1]*p1[1])*dx[i]*dt/90
                a0 = (T[i, 2]*p0[2]+T[i, 3]*p0[3])*dx[i]*dt/90*alpha_scale
                a1 = (T[i, 2]*p1[2]+T[i, 3]*p1[3])*dx[i]*dt/90*alpha_scale
                q0, q1 = fraction, fraction
                if has_genes:
                    base = math.log(fraction/(1-fraction))+genes[0]+genes[1]*(2*adapt[c, i]-1)
                    influence = genes[2]*math.tanh(f1/sf)+genes[3]*math.tanh(sl/ss)
                    q0 = 1/(1+math.exp(-(base+influence*np.sign(a0))))
                    q1 = 1/(1+math.exp(-(base+influence*np.sign(a1))))
                if gated:
                    if last_record[c, i] != record_id:
                        confirm[c, i] = confirm[c, i]*math.exp(-(now-confirm_at[c, i])/86400.)+np.sign(coeff[c]*(a1-a0))
                        confirm_at[c, i] = now
                        last_record[c, i] = record_id
                    q = confirm[c, i]*math.exp(-(now-confirm_at[c, i])/86400.)
                    q0 = .5/(1+math.exp(-(abs(q)-4)))
                    q1 = q0
                fast[c, i, 0] = f0*dg+coeff[c]*((f0+g1)*dg-(f0+g0)*dg)
                fast[c, i, 1] = f1*df+coeff[c]*((f1+(1-q1)*a1)*df-(f1+(1-q0)*a0)*df)
                slow[c, i] = sl*ds+coeff[c]*((sl+q1*a1)*ds-(sl+q0*a0)*ds)
                applied += abs(coeff[c]*(a1-a0))
            else:
                fast[c, i, 0] = f0*dg
                fast[c, i, 1] = f1*df
                slow[c, i] = sl*ds
            adapt[c, i] = end[i]
    return rm_all, applied



class Engine:
 def __init__(self,n,seed,gated=False):
  parent=bootstrap.upstream_fastmem().FastMemory();bank=NativeBank(parent);self.bank=bank;self.gated=gated
  old_n=bank.slow.shape[1];rng=np.random.default_rng(seed)
  rows=np.arange(n) if n==old_n else rng.integers(0,old_n,n)
  for key in ('Q','T','reader_Q'):
   a=np.array(getattr(bank,key)[rows],copy=True)
   if key!='T':a*=old_n/n
   setattr(bank,key,a)
  bank.fast=np.zeros((9,n,2));bank.slow=np.zeros((9,n));bank.adapt=np.ones((9,n))
  self.confirm=np.zeros((9,n));self.confirm_at=np.zeros((9,n));self.last_record=np.full((9,n),-1,dtype=np.int64)
  self.time=0.;self.events=0;self.write_l1=0.
 def rest(self,dt):self.bank.rest(dt);self.time+=dt
 def values(self,x,dt=30/14):return self.bank.values(x,dt)
 def event(self,x,coeff=None,dt=30/14,record_id=-1):
  b=self.bank;c=np.zeros(9) if coeff is None else np.asarray(coeff,dtype=float)
  rm,l1=event_v4(b.fast,b.slow,b.adapt,x,dt,c,coeff is not None,b.Q,b.T,b.F,b.MM,b.k0,b.fw0,b.fwd,b.ta,b.tg,b.tf,b.ts,b.fraction,b.alpha_scale,b.genes,b.sf,b.ss,self.confirm,self.confirm_at,self.last_record,record_id,self.time+dt,self.gated)
  self.time+=dt;self.events+=1;self.write_l1+=l1
  return rm,l1
 def stats(self):
  return dict(fast_l1=float(abs(self.bank.fast).sum()),slow_l1=float(abs(self.bank.slow).sum()),q_nonzero=int(np.count_nonzero(self.confirm)),q_max=float(abs(self.confirm).max()),native_write_l1=self.write_l1,
   native_state_bytes=sum(a.nbytes for a in (self.bank.fast,self.bank.slow,self.bank.adapt,self.confirm,self.confirm_at,self.last_record)))
