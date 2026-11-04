import json
import random
from pathlib import Path

import numpy as np
from learner import ARMS, EOS, Learner, chain, exact_chain, rows

rng = random.Random(200)
for _ in range(128):
    query = rng.randbytes(24)
    assert rows("SPLIT-ROLL",query,b"") == rows("SPLIT-HOLD",query,b"")
    held = [r for r in rows("SPLIT-HOLD",query,b"") if r<32768]
    for t in range(8):
        response = rng.randbytes(t)
        for arm in ARMS:
            ids = rows(arm,query,response)
            assert len(ids) <= 48 and sorted(set(ids)) == list(ids)
        assert held == [r for r in rows("SPLIT-HOLD",query,response) if r<32768]
        assert (set(rows("SPLIT-HOLD",query,response)) & set(range(32768,65536))
                == {r for r in rows("SPLIT-ROLL",query,response) if r>=32768})
q=b"the code of abcde is "
for arm in ARMS:
    m=Learner(arm)
    ix=rows(arm,q,b"")
    m.weights[list(ix),EOS]=1
    assert m.generate(q)==(b"",True)
    m.weights.fill(0)
    assert m.generate(q)==(bytes(8),False)
    for k in (3,7):
        payload=b"x"*k
        calls=[]
        def gen(query):
            calls.append(query)
            return (payload,True) if len(calls)==1 else (b"0",True)
        first,second=chain(gen,lambda key: b"Q"+key,b"abcde")
        assert calls==[b"Qabcde",b"Q"+payload]
        assert exact_chain(first,second,payload,b"0")
        assert not exact_chain(first,second,payload+b"x",b"0")
assert chain(lambda _: (b"abc",False),bytes,b"a")[1] is None
Path("CYCLE2_TEST.json").write_text(json.dumps(dict(status="PASS", prefixes=128, sites=1024))+"\n")
