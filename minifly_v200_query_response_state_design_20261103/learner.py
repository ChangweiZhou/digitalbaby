"""Bounded V200 byte-address learner."""
from functools import lru_cache
from hashlib import blake2b
import numpy as np

ARMS = ("ROLL", "SPLIT-ROLL", "SPLIT-HOLD")
EOS = 256
CAP = 8

def hash64(payload, domain):
    return int.from_bytes(blake2b(payload, digest_size=8, person=domain).digest(), "big")

@lru_cache(maxsize=1024)
def band(history, size, offset, limit, domain):
    grams = [(bytes([k])+history[i:i+k]) for k in (2,3,4)
             for i in range(len(history)-k+1)]
    grams += [b"S"+history[-k:] for k in (1,2) if len(history)>=k]
    if domain == b"V200-response" and not history:
        grams = [b""]
    rows = {(hash64(g, domain)+40503*j)%size+offset for g in grams for j in range(4)}
    selected = sorted(rows, key=lambda row: (hash64(row.to_bytes(2,"big"),b"V200-rank"), row))[:limit]
    return tuple(sorted(selected))

def rows(arm, query, response):
    q = query[-16:]
    if arm == "ROLL":
        return band((q+response)[-16:], 65536, 0, 48, b"V200-roll")
    if arm == "SPLIT-ROLL":
        q = q[-(16-len(response)):]
    return (band(q,32768,0,32,b"V200-query")
            +band(response,32768,32768,16,b"V200-response"))

class Learner:
    def __init__(self, arm):
        self.arm = arm
        self.weights = np.zeros((65536,257), np.float32)
        self.importance = np.zeros(65536)
        self.read_rows = self.write_rows = self.decisions = self.sites = 0

    def score(self, query, response):
        ix = np.array(rows(self.arm,query,response),dtype=np.intp)
        self.decisions += 1
        self.read_rows += len(ix)
        return ix, self.weights[ix].sum(axis=0)

    def teach(self, query, answer, rate):
        for t,target in enumerate((*answer,EOS)):
            ix,scores = self.score(query,answer[:t])
            if len(ix):
                error = -scores
                error[target] += 1
                delta = rate*error[None,:]/(len(ix)*(1+self.importance[ix,None]/5))
                self.weights[ix] += delta.astype(np.float32)
                self.importance[ix] += 1/len(ix)
                self.write_rows += len(ix)
            self.sites += 1

    def generate(self, query):
        out = bytearray()
        for _ in range(CAP):
            choice = int(self.score(query,bytes(out))[1].argmax())
            if choice == EOS:
                return bytes(out),True
            out.append(choice)
        return bytes(out),False

def chain(generate, form, key):
    first = generate(form(key))
    second = generate(form(first[0])) if first[1] else None
    return first,second

def exact_chain(first, second, intermediate, digit):
    return first == (intermediate, True) and second == (digit,True)
