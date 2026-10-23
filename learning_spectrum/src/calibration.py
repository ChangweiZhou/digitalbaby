"""One frozen causal readout surrogate; no labels or evaluator panel inputs."""
from dataclasses import dataclass
import numpy as np
from geometry import bb,TAU

@dataclass(frozen=True)
class Observation:
    cue: bytes
    at: float
    raw: tuple

def embedding(cue):return np.mean(bb.bc.R_TABLE[np.frombuffer(cue,dtype=np.uint8)],axis=0)

def predict_baseline(observations,query_cue,query_at):
    past=[o for o in observations if o.at<query_at]
    if not past:return np.zeros(4)
    values=np.array([o.raw for o in past],float);anchor=values[:min(16,len(past))].mean(0)
    x=np.array([embedding(o.cue) for o in past]);q=embedding(query_cue)
    denom=np.linalg.norm(x,axis=1)*np.linalg.norm(q)
    cos=np.divide(x@q,denom,out=np.zeros(len(x)),where=denom>0);weights=cos**2
    if weights.sum()==0:return anchor
    age=query_at-np.array([o.at for o in past]);prop=anchor+np.exp(-age[:,None]/TAU)*(values-anchor)
    return np.average(prop,axis=0,weights=weights)
