"""Forward-only module eligibility, no autograd or backward-time sweep."""
import numpy as np
from numba import njit
@njit(cache=True,fastmath=False)
def gradients(p,o,e,feedback,src,x,y,mode):
 batch=len(x);gp=np.zeros_like(p);go=np.zeros_like(o);loss=0.
 for b in range(batch):
  h=np.zeros(128);trace=np.zeros((16,8,528))
  for t in range(x.shape[1]):
   old=h.copy();a=np.empty(128);g=np.empty(128);dv=np.empty(128);dq=np.empty(128);phi=np.empty((128,33))
   for j in range(128):
    v=p[j,32];q=p[j,65]
    for k in range(16):v+=p[j,k]*e[x[b,t],k];q+=p[j,k+33]*e[x[b,t],k];phi[j,k]=e[x[b,t],k]
    for k in range(16):v+=p[j,k+16]*old[src[j,k]];q+=p[j,k+49]*old[src[j,k]];phi[j,k+16]=old[src[j,k]]
    phi[j,32]=1.;a[j]=np.tanh(v);g[j]=1/(1+np.exp(-q));dv[j]=g[j]*(1-a[j]*a[j]);dq[j]=(a[j]-old[j])*g[j]*(1-g[j]);h[j]=(1-g[j])*old[j]+g[j]*a[j]
   new=np.empty_like(trace)
   for m in range(16):
    jm=np.empty((8,8))
    for j in range(8):
     jj=m*8+j
     for k in range(8):
      jm[j,k]=dv[jj]*p[jj,16+k]+dq[jj]*p[jj,49+k]
      if k==j:jm[j,k]+=1-g[jj]
    for j in range(8):
     jj=m*8+j
     for c in range(528):
      val=0.
      for k in range(8):
       val+=jm[j,k]*trace[m,k,c]
      new[m,j,c]=val
     for k in range(33):new[m,j,j*66+k]+=dv[jj]*phi[jj,k];new[m,j,j*66+33+k]+=dq[jj]*phi[jj,k]
   trace=new;z=np.empty(257)
   for c in range(257):
    z[c]=o[c,128]
    for j in range(128):z[c]+=o[c,j]*h[j]
   maximum=np.max(z);denom=np.exp(z-maximum).sum();loss+=maximum+np.log(denom)-z[y[b,t]];err=np.exp(z-maximum)/denom;err[y[b,t]]-=1
   for c in range(257):
    for j in range(128):go[c,j]+=err[c]*h[j]
    go[c,128]+=err[c]
   for m in range(16):
    delta=np.zeros(8)
    for j in range(8):
     for c in range(257):delta[j]+=(o[c,m*8+j] if mode==0 else feedback[m*8+j,c])*err[c]
    for c in range(528):
     val=0.
     for j in range(8):val+=trace[m,j,c]*delta[j]
     gp[m*8+c//66,c%66]+=val
 return gp,go,loss
