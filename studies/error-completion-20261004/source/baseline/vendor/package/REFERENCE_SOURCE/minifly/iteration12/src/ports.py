"""Teaching-port intervention kernel; GPL-3.0-or-later, derived from frozen hybrid.py.

All four payloads use the same pre-update state. No evaluator labels enter here.
"""
from pathlib import Path
import sys
BASE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(BASE/'iteration10/src'))
from smallfly import SmallFly
from hybrid import njit
import numpy as np

@njit(cache=True)
def port_step(x,dt,pun,Q,T,F,MM,k0,fw0,fwd,ta,tg,tf,ts,fraction,fast,slow,adapt,port,plastic,scale):
    N=len(x);dx=np.empty(N);agg=np.zeros(4);inp=np.zeros(4);base=np.zeros(4)
    for i in range(N):
        end=adapt[i]*np.exp(-.05*dt*x[i]);end=1-(1-end)*np.exp(-dt/ta);dx[i]=.5*(adapt[i]+end)*x[i];adapt[i]=end
        for j in range(4):
            agg[j]+=Q[i,j]*dx[i]
            w=fast[i,0] if j<2 else fast[i,1]+slow[i]
            inp[j]+=Q[i,j]*dx[i]*(k0[3 if j<2 else 5]+w)
            base[j]+=Q[i,j]*dx[i]*k0[3 if j<2 else 5]
    rm=np.zeros((4,4));payload=np.zeros((4,4));rd=np.zeros((4,4))
    for a in range(4):
        for j in range(2):
            v=base[j] if a==1 or a==3 else inp[j]
            rm[a,j]=min(max(v+35.2,0),71.66)-35.2
        for j in range(2):
            v=base[j+2] if a==2 or a==3 else inp[j+2]
            for h in range(2):v+=MM[h,j]*rm[a,h]
            rm[a,j+2]=min(max(v+11.2,0),31.16)-11.2
        for j in range(4):
            d0=k0[0 if j<2 else 2]*agg[j]
            for h in range(4):d0+=F[h,j]*rm[a,h]
            shock=27.85 if j<2 else 11.38
            rd[a,j]=d0+shock*pun;payload[a,j]=fw0*d0+fwd*shock*pun
    raw=np.zeros((N,3));routed=np.zeros((N,2))
    eg=np.exp(-dt/tg);ef=np.exp(-dt/tf);es=np.exp(-dt/ts)
    for i in range(N):
        routed[i,0]=T[i,0]*payload[port,0]+T[i,1]*payload[port,1]
        routed[i,1]=T[i,2]*payload[port,2]+T[i,3]*payload[port,3]
        if plastic:
            dg=routed[i,0]*dx[i]*dt/90.
            da=routed[i,1]*dx[i]*dt/90.
            raw[i,0]=dg*scale;raw[i,1]=(1-fraction)*da*scale;raw[i,2]=fraction*da*scale
            fast[i,0]+=raw[i,0];fast[i,1]+=raw[i,1];slow[i]+=raw[i,2]
        fast[i,0]*=eg;fast[i,1]*=ef;slow[i]*=es
    return np.concatenate((rd[0],rm[0])),payload,rd,rm,dx,raw,routed

class PortFly(SmallFly):
    def __init__(self,path,arm='INTACT'):
        super().__init__(path);self.arm=arm
    def step(self,seconds,pn_activity=None,punishment=0.):
        if seconds<0 or not np.isfinite(seconds) or not np.isfinite(punishment):raise ValueError('Invalid duration/consequence')
        x=self.encode(np.zeros(self.input_channels) if pn_activity is None else pn_activity);k=self.kernel
        port={'INTACT':0,'G0':1,'A0':2,'GA0':3,'NO_WRITE':0,'HALF_WRITE':0}[self.arm]
        result=port_step(x,float(seconds),float(punishment),self.Q,self.T,self.F,self.MM,self.k0,self.fw0,self.fwd,self.ta,self.tg,
            k['fast_tau'],k['slow_tau'],k['fraction'],self.fast,self.slow,self.adapt,port,self.mode!='no_learning' and self.arm!='NO_WRITE',.5 if self.arm=='HALF_WRITE' else 1.)
        self.elapsed=np.float64(self.elapsed+seconds);self.event_count+=np.uint64(1);self.presentation_count+=np.uint64(np.any(x>0))
        rates,ps,rd,rm,dx,raw,routed=result
        return dict(rates=rates,teaching_payload=ps[port],all_payloads=ps,counterfactual_DAN_drive=rd,all_MBON_counterfactuals=rm,
            sensory_activity=dx,raw_increment=raw,routed_teacher=routed,active_KCs=int(x.sum()))
    def save(self,path):
        np.savez_compressed(path,fast=self.fast,slow=self.slow,adapt=self.adapt,elapsed=self.elapsed,event_count=self.event_count,
            presentation_count=self.presentation_count,model_sha=self.model_sha,mode=self.mode,arm=self.arm)
    @classmethod
    def restore(cls,path,model_path):
        with np.load(path,allow_pickle=False) as f:
            m=cls(model_path,str(f['arm']));m.mode=str(f['mode'])
            if str(f['model_sha'])!=m.model_sha:raise ValueError('Wrong model')
            for key in ('fast','slow','adapt'):
                if f[key].shape!=getattr(m,key).shape:raise ValueError('Wrong shape')
                setattr(m,key,f[key].copy())
            m.elapsed=np.float64(f['elapsed']);m.event_count=np.uint64(f['event_count']);m.presentation_count=np.uint64(f['presentation_count'])
        return m
