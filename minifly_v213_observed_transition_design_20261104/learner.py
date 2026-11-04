"""V213 bounded local predictor. No simulator labels enter this module."""
from dataclasses import dataclass
import numpy as np

ARMS=("ALIGNED","SHUFFLED-OP","FROZEN-GATE","NO-PRIOR-WRITE")

def sigmoid(x):
    return 1/(1+np.exp(-np.clip(x,-60,60)))

def softmax(x):
    exp=np.exp(x-x.max())
    return exp/exp.sum()

@dataclass(frozen=True)
class Prediction:
    prob: np.ndarray
    self_prob: np.ndarray
    other_prob: np.ndarray
    old_gate: float
    gate: float
    ends: np.ndarray
    opcode: int
    rows: tuple

def snapshot(theta,old,rows,opcode,logits):
    ends=sigmoid(theta[opcode].astype(np.float64))
    g=float((1-old)*ends[0]+old*ends[1])
    a,b=(softmax(logits[i,row].astype(np.float64)) for i,row in enumerate(rows))
    return Prediction((1-g)*a+g*b,a,b,old,g,ends,opcode,rows)

def increments(s,target):
    a,b=s.self_prob,s.other_prob
    den=max(float(s.prob[target]),1e-8)
    y=np.eye(4)[target]
    e=np.stack(((1-s.gate)*a[target]/den*(y-a),s.gate*b[target]/den*(y-b)))
    credit=float(np.clip((b[target]-a[target])/den,-4,4))
    g=credit*np.array([1-s.old_gate,s.old_gate])*s.ends*(1-s.ends)
    return e,g

class Learner:
    def __init__(self,positions,opcodes,seed_value,arm):
        assert arm in ARMS
        self.arm=arm
        self.positions={p:i for i,p in enumerate(positions)}
        self.opcodes={o:i for i,o in enumerate(opcodes)}
        self.weights=np.random.default_rng(seed_value).normal(0,.01,(2,4,4)).astype(np.float32)
        self.theta=np.zeros((2,2),np.float32)
        self.emission_updates=0
        self.gate_updates=[0,0]
        self.reset()

    def reset(self):
        self.last={}
        self.pairs={}
        self.active=None
        self.pending=None

    def consume(self,record):
        tag,*fields=record
        if tag=="obs":
            assert self.pending is None
            entity,pos=fields
            self.last[entity]=self.positions[pos]
        elif tag=="event":
            assert self.pending is None
            token,a,b=fields
            opcode=self.opcodes[token]
            old=self.pairs.get((a,b),.5)
            ends=sigmoid(self.theta[opcode].astype(np.float64))
            self.pairs[a,b]=float((1-old)*ends[0]+old*ends[1])
            self.active=(a,b,opcode,old)
        else:
            assert tag=="move" and self.pending is None
            mover,pos=fields
            a,b,op,old=self.active
            assert a==mover
            self.last[a]=self.positions[pos]
            self.pending=snapshot(self.theta,old,(self.last[b],self.last[a]),op,self.weights)
            assert self.pending.gate==self.pairs[a,b]
            return self.pending.prob.copy()

    def reveal(self,record,eta,sham_opcode=None):
        tag,entity,pos=record
        a,b,_,_=self.active
        assert tag=="obs" and entity==b and self.pending is not None
        s=self.pending
        target=self.positions[pos]
        if self.arm!="NO-PRIOR-WRITE":
            e,g=increments(s,target)
            for branch,row in enumerate(s.rows):
                self.weights[branch,row]+=eta*e[branch]
            self.emission_updates+=1
            if self.arm in ("ALIGNED","SHUFFLED-OP"):
                slot=s.opcode if self.arm=="ALIGNED" else self.opcodes[sham_opcode]
                self.theta[slot]+=eta*g
                self.gate_updates[slot]+=1
        assert self.pairs[a,b]==s.gate
        self.pending=None
        self.consume(record)
