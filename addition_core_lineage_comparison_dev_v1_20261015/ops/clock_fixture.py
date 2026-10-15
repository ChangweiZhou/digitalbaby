import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import core
import numpy as np
from io_utils import write,now


def run():
    rows=[]
    for version in core.VERSIONS[:3]:
        a=core.create(version);a.rest(86400.);b=a.clone()
        a.train(b'\x00\xff?',b'\x80',True)
        b.train(b'\x00\xff?',b'\x80',False)
        assert a.time==b.time
        for aa,bb in zip([s for bank in a.banks for s in bank],[s for bank in b.banks for s in bank]):
            assert np.array_equal(aa.fly.m.adapt,bb.fly.m.adapt)
            assert aa.fly.m.elapsed==bb.fly.m.elapsed==a.time
        rows.append(dict(version=version,accepted=True,record_applications=2))
    result=dict(utc=now(),science=False,accepted=True,rows=rows,training_record_applications=6)
    write(core.bootstrap.HERE/'cycles/cycle2/LONG_CLOCK_FIXTURE.json',result)
    return result


if __name__=='__main__':print(run(),flush=True)
