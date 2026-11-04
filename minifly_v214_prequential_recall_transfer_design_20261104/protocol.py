"""V214 driver and metrics, with a test double for engineering qualification."""
from collections import Counter
import hashlib
import json
import math
import statistics as st

ARMS=("BASE","RECALL","CUE-SHAM")
CHECKPOINTS=(0,2,8,24)

def sha(b):
    return hashlib.sha256(b).hexdigest()

class ByteCounter:
    """Test double only. No Round 20 association or mixer implementation."""
    def __init__(self):
        self.count=Counter()
        self.total=0

    def predict(self):
        return tuple((self.count.get(x,0)+1)/(self.total+256) for x in range(256))

    def observe(self,x):
        self.count[x]+=1
        self.total+=1

    def snapshot(self):
        return json.dumps(sorted(self.count.items()),separators=(",",":")).encode()

    @classmethod
    def restore(cls,b):
        m=cls()
        m.count=Counter(dict(json.loads(b)))
        m.total=sum(m.count.values())
        assert m.snapshot()==b
        return m

def score(model,stream,spans=(),capture=False):
    labels=[None]*len(stream)
    for start,end,label in spans:
        assert 0<=start<end<=len(stream) and all(x is None for x in labels[start:end])
        labels[start:end]=[label]*(end-start)
    losses=[]
    predictions=[]
    for value in stream:
        p=model.predict()
        assert len(p)==256 and math.isclose(sum(p),1,abs_tol=1e-12)
        assert 0<p[value]<=1
        losses.append(-math.log2(p[value]))
        if capture:
            predictions.append(p)
        model.observe(value)
    sections={name:math.fsum(x for x,t in zip(losses,labels) if t==name)
              for name in set(labels)-{None}}
    sections["nonanswer"]=math.fsum(x for x,t in zip(losses,labels) if t is None)
    assert math.isclose(math.fsum(sections.values()),math.fsum(losses))
    return dict(bytes=len(stream),sha256=sha(stream),bits=math.fsum(losses),
                first_quarter_bits=math.fsum(losses[:len(stream)//4]),sections=sections,
                spans={name:sum(label==name for _,_,label in spans) for name in set(labels)-{None}},
                predictions=predictions)

def efficiency(fresh,score,history_bytes,history_bits):
    gain=fresh-score
    return dict(gain_bits=gain,bits_saved_per_kib=gain*1024/history_bytes if history_bytes else None,
                own_code_ratio=gain/history_bits if history_bits else None)

def interval(values, critical):
    center=st.mean(values)
    half=critical*st.stdev(values)/len(values)**.5
    return [center-half,center+half]

def decision(families,prose):
    assert len(families)==len({r["id"] for r in families})==16
    assert len(prose)==8
    for f in families:
        assert set(f["arms"])==set(ARMS)
        cells=[r for arm in ARMS for r in f["arms"][arm].values()]
        assert all(set(f["arms"][arm])==set(CHECKPOINTS) for arm in ARMS)
        assert len({(r["sha256"],r["bytes"]) for r in cells})==1
    def bits(f,arm,k):
        return f["arms"][arm][k]["bits"]
    def gain(f,arm,k):
        return bits(f,arm,0)-bits(f,arm,k)
    t=2.131449546
    contrasts={}
    for arm in ("BASE","CUE-SHAM"):
        d=[(bits(f,arm,24)-bits(f,"RECALL",24))/f["arms"]["RECALL"][24]["bytes"]
           for f in families]
        contrasts[arm]=dict(mean=st.mean(d),interval=interval(d,t))
    g=[gain(f,"RECALL",24) for f in families]
    late=[gain(f,"RECALL",24)-gain(f,"RECALL",2) for f in families]
    prose_diff=[r["RECALL"]-r["BASE"] for r in prose]
    guard=interval(prose_diff,2.364624251)[1]<.02
    recall=all(x["mean"]>=.02 and x["interval"][0]>0 for x in contrasts.values())
    transfer=interval(g,t)[0]>0
    return dict(recall_pass=recall,positive_transfer=transfer,prose_guard=guard,
                late_decline=interval(late,t)[1]<0,
                eligible_to_design_stage_b=recall and transfer and guard and interval(late,t)[1]<0,
                contrasts=contrasts,gain_interval=interval(g,t),late_interval=interval(late,t))

def uniform_word_bits(answer, candidates):
    assert len(set(candidates))==len(candidates)==6
    assert answer in candidates and all(x.endswith(b"|") and x.count(b"|")==1 for x in candidates)
    remaining=candidates
    bits=0.
    for i,value in enumerate(answer):
        matched=[word for word in remaining if word[i]==value]
        assert matched
        bits-=math.log2(len(matched)/len(remaining))
        remaining=matched
    return bits
