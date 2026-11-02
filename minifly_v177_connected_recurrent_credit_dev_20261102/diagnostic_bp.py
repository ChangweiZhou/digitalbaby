"""Registered frozen-checkpoint gradient instrument; no parameter update."""
import numpy as np
import core as C
from bp import gradients
for w in (202611027711,202611027712):
 a=np.load(C.H/f'local/BP_{w}.npz');x,y,_=C.data(w,'train',64);g=gradients(a['p'],a['o'],a['e'],x[:8],y[:8]);np.savez(C.H/f'local/diagnostic_BP_{w}.npz',gp=g[0],go=g[1])
