"""Single fixed label-blind causal geometry; evaluator utilities live separately."""
from __future__ import annotations
import copy, hashlib, json, math, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
A3=ROOT.parent/'minifly_a_v3'
sys.path.insert(0,str(A3/'src'))
import paths
import brain_byte as bb
from fixture import DT, RECORD_SECONDS, make_world
BETA=1/16
ETA=.25
TAU=86400.
GAIN=.4237781016501581
NU_FLOOR=(1-.8**(1/96))/.25

class Geometry:
    """API accepts only cue bytes, boundary and clock; no labels/IDs/positions."""
    def __init__(self,centered=True):
        self.centered=centered
        self.mu_a=np.zeros(88);self.mu_r=np.zeros(88)
        self.recent=np.zeros(88);self.elig=np.zeros((88,88));self.last=None
    def clone(self):return copy.deepcopy(self)
    def begin_cue(self):
        self.recent.fill(0);self.elig.fill(0);self.last=None
    def byte(self,b,t):
        dt=0. if self.last is None else t-self.last
        if dt<0:raise ValueError('clock reversal')
        d=math.exp(-dt/10.)
        self.recent*=d;self.elig*=d
        a=bb.bc.R_TABLE[b]
        if self.centered:
            self.elig+=np.outer(self.recent-self.mu_r,a-self.mu_a)
            self.mu_a+=(a-self.mu_a)*BETA
            self.mu_r+=(self.recent-self.mu_r)*BETA
        else:self.elig+=np.outer(self.recent,a)
        self.recent+=a;self.last=float(t)
    def feature(self,t):
        return (self.elig*math.exp(-(t-self.last)/10.)).ravel().copy()
    def feed_cue(self,cue,at):
        self.begin_cue()
        for k,b in enumerate(cue):self.byte(b,at+k*DT)
        return self.feature(at+12*DT)
    def panel(self,cues,at):
        return np.array([self.clone().feed_cue(cue,at) for cue in cues])

def source_verify():
    lock=json.loads((A3/'SOURCE_LOCK.json').read_text())
    rows={}
    for p,v in lock['files'].items():
        if p.startswith(('src/','package/')):
            h=hashlib.sha256((A3/p).read_bytes()).hexdigest()
            if h!=v:raise AssertionError(p)
            rows[p]=h
    return rows

def history(world):
    d=make_world(world);out=[]
    for stage in ('old','new'):
        rr=[r for r in d['records'] if r['stage']==stage and r['domain']=='fact'][:96]
        for r in rr:
            r=dict(r);r['index']=len(out);r['at']=r['index']*RECORD_SECONDS+(86400 if stage=='new' else 0)
            out.append(r)
    return d,out

def int_basis():
    # Orthonormal basis for 4-level contrasts, with known dimension nine.
    q=np.linalg.qr(np.eye(4)[:,:3]-np.ones((4,3))/4)[0]
    return np.kron(q,q)

def components(f):
    x=np.asarray(f).reshape(4,4,-1)
    b=x.mean((0,1));a=x.mean(1)-b;c=x.mean(0)-b;h=x-b-a[:,None]-c[None,:]
    return b,a,c,h

def spectrum(f):
    z=f/np.sqrt(np.maximum(1,(f*f).sum(1)))[:,None]
    k=z@z.T/16;q=int_basis();p=q@q.T
    eig=np.linalg.eigvalsh(q.T@k@q)
    b,a,c,h=components(f);den=float(np.mean(f*f))
    energies={'common':float(np.mean(b*b)/den),'first':float(np.mean(a*a)/den),'second':float(np.mean(c*c)/den),'interaction':float(np.mean(h*h)/den)}
    return {'eigenvalues':eig.tolist(),'minimum':float(eig[0]),'maximum':float(eig[-1]),'nu_floor':NU_FLOOR,'spectrum_gate':bool(eig[0]>=NU_FLOOR),'cross_subspace_frobenius':float(np.linalg.norm(p@k@(np.eye(16)-p))),'energies':energies,'feature_norm_mean':float(np.linalg.norm(f,axis=1).mean()),'feature_sha256':hashlib.sha256(f.tobytes()).hexdigest()}
