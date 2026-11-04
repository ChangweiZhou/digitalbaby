"""Prospective evaluation-only episodic address wrapper."""
import numpy as np

ARMS=("RECENT32","RECENT96","SPLIT-RECENT","SPLIT-ANCHOR")

def segment(h):
    h=bytes(h)
    tail=h[:-2]
    cut=max(tail.rfind(b"."),tail.rfind(b"?"),tail.rfind(b"\n"))
    return h[cut+1:] if cut>=0 and cut<len(h)-1 else h

def make(original, arm):
    def lane(raw, step, offset, quota):
        ids,v=original(raw,step=step)
        merged={}
        for i,value in zip(ids,v):
            row=int(i)%32768+offset
            merged[row]=max(merged.get(row,0),value)
        keys=sorted(merged,key=lambda row: ((row*2654435761+206)%2**32,row))[:quota]
        return np.array(keys,dtype=np.int64),np.array([merged[k] for k in keys],dtype=float)

    def address(h,update=False,per=4,step=0):
        if arm=="RECENT32":
            return original(h,update,per,step)
        if arm=="RECENT96":
            ids,v=original(h,update,per,step)
            if len(ids)<=360:
                return ids,v
            top=np.argsort((ids*2654435761+206)%2**32,kind="stable")[:360]
            return ids[top],v[top]
        recent=h[-32:]
        earlier=segment(h)[:16] if arm=="SPLIT-ANCHOR" else recent
        a,x=lane(recent,step,0,240)
        b,y=lane(earlier,step,32768,120)
        return np.concatenate((a,b)),np.concatenate((x,y))
    return address
