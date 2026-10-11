"""Pure accessed-row time evolution; never receives labels or trial identities."""
import numpy as np
import settings

def advance(f,s,dt,mode):
    dt=np.asarray(dt,dtype=float)
    if not np.isfinite(dt).all() or np.any(dt<0):raise ValueError('negative/nonfinite elapsed time')
    if mode=='DECAY':
        return f*np.exp(-dt[:,None]/settings.FAST_TAU),s*np.exp(-dt[:,None]/settings.SLOW_TAU)
    if mode=='STABLE':return f.copy(),s.copy()
    if mode=='TRANSFER':
        u=-np.expm1(-dt[:,None]/settings.FAST_TAU)*f
        accepted=np.clip(s+u,-settings.ROW_BOUND,settings.ROW_BOUND)-s
        return f-accepted,s+accepted
    raise ValueError('unknown time rule')
