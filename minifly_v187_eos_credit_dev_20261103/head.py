"""V187 local factorized byte/EOS head."""
import math
from pathlib import Path
import sys
sys.path.insert(1,str(Path(__file__).resolve().parents[1]/'minifly_v186_sensor_decay_dev_20261103'))
from qualify import Learner, T

def joint(raw):
    stop=raw[:,256:]
    return T.cat((raw[:,:256].log_softmax(1)+T.nn.functional.logsigmoid(-stop),
                  T.nn.functional.logsigmoid(stop)),1)

def error(raw, labels):
    payload = labels != 256
    e = raw[:,:256].softmax(1)
    e[~payload] = 0
    rows = T.where(payload)[0]
    e[rows,labels[rows]] -= 1
    stop = raw[:,256].sigmoid() - (~payload).to(raw.dtype)
    return T.cat((e,stop[:,None]),1)/len(labels)

class Factorized(Learner):
    def __init__(self,world='M1'):
        super().__init__(world)
        self.p['Ob'][256] -= math.log(256)

    def forward(self,x,training=False):
        raw,state=super().forward(x,training)
        return joint(raw),state

    def gradients(self,x,y,ids):
        raw,(xx,ah,h,at,mask,z)=super().forward(x,True)
        e=error(raw,y)
        dt=(e@self.ft)*(at>0)*mask
        dh=(e@self.fh)*(ah>0)
        gradients={'H':T.einsum('nj,njr->jr',dh,xx),'Hb':dh.sum(0),
                   'T':dt.T@h,'Tb':dt.sum(0),'O':e.T@z,'Ob':e.sum(0)}
        return gradients,{k:T.ones(64) for k in ('H','T')},{'factorized_local_credit':True}
