"""Fixed sensory observer. No parent, reset, plastic state, label or cue lookup."""
from pathlib import Path
import numpy as np


def basis(drive,model):
    x=(drive-model['input_mean'])/model['input_scale']
    if len(model['centers']):
        squared=np.maximum(np.sum(x*x,axis=-1)[...,None]+np.sum(model['centers']**2,axis=1)-2*x@model['centers'].T,0)
        z=np.concatenate([x,np.exp(-squared/(2*float(model['width'])**2))],axis=-1)
    else:z=x
    return (z-model['feature_mean'])/model['feature_scale']


def predict_drive(drive,model):
    return np.clip(basis(drive,model)@model['coefficient']+model['intercept'],*model['bounds'])


def analytic_baseline(drive,k0,MM):
    """Equation-derived ceiling; never a fitted-reader scientific success."""
    gamma=np.clip(drive[...,:2]*k0[3]+35.2,0,71.66)-35.2
    return np.clip(drive[...,2:]*k0[5]+gamma@MM+11.2,0,31.16)-11.2


class SensoryReference:
    def __init__(self,path):
        with np.load(Path(path),allow_pickle=False) as f:self.model={k:f[k].copy() for k in f.files}
        for a in self.model.values():a.flags.writeable=False
        self.name=str(self.model['name']);self.source=str(self.model['source'])
        self.cue_threshold=float(self.model['cue_threshold']);self.pair_threshold=float(self.model['pair_threshold'])

    def predict(self,sensory_activity,unadapted_activity=None):
        activity=np.asarray(sensory_activity,float)
        if activity.shape[-1:]!=(self.model['projection'].shape[0],) or np.any(~np.isfinite(activity)) or np.any(activity<0):
            raise ValueError('Expected finite nonnegative current KC activity')
        if self.source in ('unadapted','scalar'):
            x=np.asarray(unadapted_activity,float)
            if x.shape!=activity.shape or np.any(~np.isfinite(x)) or np.any(x<0):raise ValueError('Unadapted control requires matched nonnegative encoding')
            if self.source=='scalar':x=x*np.divide(activity.sum(axis=-1,keepdims=True),x.sum(axis=-1,keepdims=True),out=np.zeros_like(activity.sum(axis=-1,keepdims=True)),where=x.sum(axis=-1,keepdims=True)>0)
            activity=x
        return predict_drive(activity@self.model['projection'],self.model)

    def score(self,sensory_activity,alpha3_response,unadapted_activity=None):
        predicted=self.predict(sensory_activity,unadapted_activity);observed=np.asarray(alpha3_response,float)
        if observed.shape!=predicted.shape or np.any(~np.isfinite(observed)):raise ValueError('Expected two observed alpha3 responses per input')
        return (observed-predicted).mean(axis=-1)

    def choose(self,activity_A,response_A,activity_B,response_B,unadapted_A=None,unadapted_B=None):
        delta=float(self.score(activity_A,response_A,unadapted_A)-self.score(activity_B,response_B,unadapted_B))
        return dict(choice='B' if delta<=-self.pair_threshold else 'A' if delta>=self.pair_threshold else 'abstain',score_A_minus_B=delta)

    def mutable_bytes(self):return 0
