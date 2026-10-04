"""Actually reduced KC state, with unchanged parent dynamics and fixed RBF head.

GPL-3.0-or-later. No cue-indexed cache, hidden parent state, or reader refit.
"""
from pathlib import Path
import sys,json,copy,hashlib
import numpy as np
from scipy.sparse import csr_matrix
BASE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(BASE/'iteration5/src'))
from hybrid import advance_step
from prepare_interface import encode
sys.path.insert(0,str(BASE/'iteration8/src'))
from readout import predict_drive

class SmallFly:
    def __init__(self,path,mode='full'):
        if mode not in ('full','no_learning'):raise ValueError('Unsupported mode')
        self.model_path=Path(path).resolve();self.model_sha=hashlib.sha256(self.model_path.read_bytes()).hexdigest()
        with np.load(self.model_path,allow_pickle=False) as f:
            for key in ('Q','T','F','MM','k0','pn_type_index','kc_side','ids','gains'):setattr(self,key,f[key].copy())
            self.B=csr_matrix((f['B_data'],f['B_indices'],f['B_indptr']),shape=tuple(f['B_shape']))
            self.kernel=json.loads(str(f['kernel_json']));self.fw0=float(f['fw0']);self.fwd=float(f['fwd'])
            self.ta=float(f['ta']);self.tg=float(f['tg']);self.input_channels=int(f['input_channels'])
            self.model_name=str(f['model_name'])
        for key in ('Q','T','F','MM','k0','pn_type_index','kc_side','ids','gains'):getattr(self,key).flags.writeable=False
        for a in (self.B.data,self.B.indices,self.B.indptr):a.flags.writeable=False
        self.mode=mode;N=len(self.ids);self.fast=np.zeros((N,2));self.slow=np.zeros(N);self.adapt=np.ones(N)
        self.elapsed=np.float64(0);self.event_count=np.uint64(0);self.presentation_count=np.uint64(0)

    def encode(self,p):
        p=np.asarray(p,float)
        if p.shape!=(self.input_channels,) or np.any(~np.isfinite(p)) or np.any(p<0):raise ValueError('Expected fixed finite nonnegative PN vector')
        return encode(self.B,self.pn_type_index,self.kc_side,p)

    def step(self,seconds,pn_activity=None,punishment=0.):
        if not np.isfinite(seconds) or seconds<0 or not np.isfinite(punishment):raise ValueError('Finite nonnegative duration and finite punishment required')
        p=np.zeros(self.input_channels) if pn_activity is None else np.asarray(pn_activity,float);x=self.encode(p)
        # The exposed activity is exactly the effective activity used inside advance_step.
        end=1-(1-self.adapt*np.exp(-.05*seconds*x))*np.exp(-seconds/self.ta)
        dx=.5*(self.adapt+end)*x;k=self.kernel
        rates,payload=advance_step(x,float(seconds),float(punishment),self.Q,self.T,self.F,self.MM,self.k0,self.fw0,self.fwd,
            self.ta,self.tg,k['fast_tau'],k['slow_tau'],k['fraction'],self.fast,self.slow,self.adapt,self.mode!='no_learning',True)
        self.elapsed=np.float64(self.elapsed+seconds);self.event_count+=np.uint64(1);self.presentation_count+=np.uint64(np.any(p>0))
        return dict(rates=rates,teaching_payload=payload,sensory_activity=dx,active_KCs=int(x.sum()))

    def clone(self,reset_plastic=False):
        m=copy.copy(self)
        for key in ('fast','slow','adapt'):setattr(m,key,getattr(self,key).copy())
        if reset_plastic:m.fast[:]=0.;m.slow[:]=0.
        return m
    def mutable_bytes(self):return self.fast.nbytes+self.slow.nbytes+self.adapt.nbytes+24
    def fixed_numeric_bytes(self):
        arrays=[getattr(self,k) for k in ('Q','T','F','MM','k0','pn_type_index','kc_side','ids','gains')]
        return int(sum(a.nbytes for a in arrays)+self.B.data.nbytes+self.B.indices.nbytes+self.B.indptr.nbytes+8*(5+len(self.kernel)))
    def save(self,path):
        np.savez_compressed(path,fast=self.fast,slow=self.slow,adapt=self.adapt,elapsed=self.elapsed,event_count=self.event_count,
            presentation_count=self.presentation_count,model_sha=self.model_sha,mode=self.mode)
    @classmethod
    def restore(cls,checkpoint,model_path):
        with np.load(checkpoint,allow_pickle=False) as f:
            m=cls(model_path,str(f['mode']))
            if str(f['model_sha'])!=m.model_sha:raise ValueError('Checkpoint belongs to a different fixed model')
            for key in ('fast','slow','adapt'):
                if f[key].shape!=getattr(m,key).shape:raise ValueError('Checkpoint shape mismatch')
                setattr(m,key,f[key].copy())
            for key,dtype in [('elapsed',np.float64),('event_count',np.uint64),('presentation_count',np.uint64)]:setattr(m,key,dtype(f[key]))
        return m

class FrozenReader:
    def __init__(self,Q,head_path=None):
        self.Q=Q  # shared fixed projection, not a duplicate or full-size padded array
        path=Path(head_path) if head_path else BASE/'iteration8/models/current_RBF64.npz'
        with np.load(path,allow_pickle=False) as f:
            self.head={k:f[k].copy() for k in ('input_mean','input_scale','centers','width','feature_mean','feature_scale','coefficient','intercept','bounds')}
            self.cue_threshold=float(f['cue_threshold']);self.pair_threshold=float(f['pair_threshold'])
        for a in self.head.values():a.flags.writeable=False
    def predict(self,dx):
        a=np.asarray(dx,float)
        if a.shape[-1:]!=(len(self.Q),) or np.any(~np.isfinite(a)) or np.any(a<0):raise ValueError('Invalid reduced KC activity')
        return predict_drive(a@self.Q,self.head)
    def mutable_bytes(self):return 0
    def extra_fixed_numeric_bytes(self):return sum(a.nbytes for a in self.head.values())+16
