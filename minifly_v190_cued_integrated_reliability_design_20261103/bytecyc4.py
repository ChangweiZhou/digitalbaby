"""New V190 implementation of signed successor populations and local reliability."""
import hashlib
import numpy as np

EOS=256
def address(data, width):
    return int.from_bytes(hashlib.blake2b(data,digest_size=8).digest(),'big')%width

def probabilities(logits):
    values=np.exp(logits-np.max(logits))
    return values/values.sum()

class Relations:
    def __init__(self, symbols, tags):
        self.symbols=tuple(sorted(symbols))
        self.index={b:i for i,b in enumerate(self.symbols)}
        self.tags=tuple(tags)
        self.maps=np.zeros((2,len(symbols),len(symbols)),np.float32)
        self.weights=np.zeros((len(tags),2,256,257),np.float32)
        self.correct=np.ones((2,len(tags),2),np.int32)
        self.total=np.full_like(self.correct,2)
        self.seen={}
        self.frozen=False

    def background(self, raw):
        assert not self.frozen
        for left,right in zip(raw,raw[1:]):
            i,j=self.index[left],self.index[right]
            for expert,gamma in enumerate((.8,-.8)):
                row=self.maps[expert,i]
                delta=gamma*self.maps[expert,j]-row
                delta[i]+=1
                row+=.5*delta

    def feature(self, history, expert):
        # Fixed raw-byte slots. The generic relation frame has two bytes plus '='.
        # Its suffix remains in the window when the candidate emits its answer.
        x=np.zeros(256,np.float32)
        x[0]=1
        for lag,byte in enumerate(reversed(history[-8:]),1):
            x[1+address(bytes((lag,byte)),255)]+=1
        observed=[self.index[b] for b in history[-8:] if b in self.index]
        if len(observed)>=2:
            a,b=observed[:2]
            for k,v in enumerate(self.maps[expert,a]):
                x[1+address(bytes((21,k)),255)]+=v
            for k,v in enumerate(self.maps[expert,b]):
                x[1+address(bytes((22,k)),255)]+=v
            # Shift-invariant successor comparisons. No distance labels enter here.
            for order,(u,v) in enumerate(((a,b),(b,a))):
                x[250+order]+=self.maps[expert,u,v]
        return x

    def read(self, tag, history):
        t=self.tags.index(tag)
        features=[self.feature(history,i) for i in range(2)]
        experts=np.array([probabilities(x@self.weights[t,i]) for i,x in enumerate(features)])
        acc=self.correct[:,t]/self.total[:,t]
        mixture=acc**4
        mixture/=mixture.sum(axis=1,keepdims=True)
        return features, experts, {
            'PREQ':mixture[0]@experts,'REPEAT':mixture[1]@experts,
            'UNIFORM':experts.mean(axis=0),'POS':experts[0],'NEG':experts[1]}

    def teach(self, tag, prompt, value):
        assert self.frozen
        t=self.tags.index(tag)
        occurrence=self.seen.get((tag,prompt),0)
        assert occurrence<2 and len(self.seen)<321
        self.seen[tag,prompt]=occurrence+1
        for history,target in ((prompt,value),(prompt+bytes([value]),EOS)):
            features,experts,_=self.read(tag,history)
            if target!=EOS:
                self.correct[occurrence,t]+=experts.argmax(axis=1)==target
                self.total[occurrence,t]+=1
            for i,x in enumerate(features):
                error=-experts[i].copy()
                error[target]+=1
                active=np.flatnonzero(x)
                self.weights[t,i,active]+=4*x[active,None]*error/(1+float(x@x))
